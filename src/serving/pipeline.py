"""The selected RAG pipeline as one object: question -> retrieved chunks -> generation request
-> parsed answer with citations and stated confidence.

Retrieval is the `retrieved` condition's own call (`ScopedFaissIndex.search_corpus` at
`search_depth`, then the top k), and the request is built by the same `build_request` the
Phase 4 traces were, so for a benchmark question the pipeline produces the traced request
byte for byte (same cache key), which the tests and `scripts/12_serving.py check` verify.
The configuration is the one registered in MLflow (`results/metrics/pipeline_selection.json`,
`pipeline_config`).

Generation goes through the response cache. An API model with no spend ledger can only be
served from the cache: `generate_cached` refuses an uncached API call without a ledger.

`index_dir` and `model_path` point the pipeline at the serving bundle (the image's copy of
the index and the pinned embedding weights) instead of the DVC checkout's paths.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.generation.llm import ResponseCache, SpendLedger, generate_cached
from src.generation.parsing import parse_output
from src.generation.prompts import load_registry
from src.generation.requests import build_request, make_backend
from src.generation.trace import trace_chunks


def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000


def bundled_embedding_model(model_name: str, model_path: str, device: str | None = None):
    """An `EmbeddingModel` whose weights load from a local directory (the serving bundle's
    pinned copy) instead of by Hugging Face id. The spec (query prefix, dimension) and the
    encoding are the pipeline's own. Kept here, not in src/retrieval, so the served image's
    needs do not touch the offline pipeline's stages."""
    from sentence_transformers import SentenceTransformer

    from src.retrieval.embeddings import MODEL_REGISTRY, EmbeddingModel, _default_device

    class BundledEmbeddingModel(EmbeddingModel):
        def __init__(self) -> None:
            self.spec = MODEL_REGISTRY[model_name]
            self.device = device or _default_device()
            self.batch_size = 64
            self._model = SentenceTransformer(model_path, device=self.device)

    return BundledEmbeddingModel()


class RAGPipeline:
    def __init__(
        self,
        config: dict[str, Any],
        root: Path = Path("."),
        *,
        backend: Any = None,
        cache: ResponseCache | None = None,
        ledger: SpendLedger | None = None,
        embed_model: Any = None,
        index: Any = None,
        index_dir: Path | None = None,
        model_path: str | None = None,
        device: str | None = None,
    ):
        self.config = config
        self.root = Path(root)
        rc = config["retrieval"]
        if rc["method"] != "dense" or rc["backend"] != "faiss":
            raise NotImplementedError("the pipeline is wired for dense FAISS retrieval only")
        if rc["search_depth"] < rc["k"]:
            raise ValueError("retrieval.search_depth must be >= retrieval.k")
        self.k, self.search_depth = rc["k"], rc["search_depth"]
        if embed_model is None and model_path is not None:
            embed_model = bundled_embedding_model(rc["embedding"], model_path, device)
        elif embed_model is None:
            from src.retrieval.embeddings import EmbeddingModel

            embed_model = EmbeddingModel(rc["embedding"], device=device)
        if index is None:
            from src.retrieval.scoped import ScopedFaissIndex

            index = ScopedFaissIndex(
                index_dir
                or self.root / "data" / "indices" / f"{rc['chunking']}__{rc['embedding']}__faiss"
            )
        self.embed_model, self.index = embed_model, index

        self.model_cfg = config["generator"]["model"]
        self.template = load_registry(self.root / "prompts")[config["prompt"]["id"]]
        if self.template.file_sha256 != config["prompt"]["file_sha256"]:
            raise ValueError(
                f"prompt {config['prompt']['id']} has changed since the pipeline was selected"
            )
        self.max_tokens = config["max_tokens"]
        self.backend = backend if backend is not None else make_backend(self.model_cfg)
        self.cache = cache if cache is not None else ResponseCache(self.root / "data/cache/llm")
        self.ledger = ledger

    def retrieve(self, question: str, timings: dict[str, float] | None = None) -> list[dict]:
        t0 = time.perf_counter()
        qv = self.embed_model.encode_queries([question])
        t1 = time.perf_counter()
        results = self.index.search_corpus(qv, self.search_depth)[: self.k]
        t2 = time.perf_counter()
        if timings is not None:
            timings["embed"], timings["search"] = _ms(t0, t1), _ms(t1, t2)
        return trace_chunks(results)

    def answer(self, question: str) -> dict[str, Any]:
        timings: dict[str, float] = {}
        chunks = self.retrieve(question, timings)
        t1 = time.perf_counter()
        request, _ = build_request(
            self.model_cfg, self.max_tokens, self.template, {"question": question}, chunks
        )
        response, was_cached, cost = generate_cached(self.backend, request, self.cache, self.ledger)
        t2 = time.perf_counter()
        timings["generate"] = _ms(t1, t2)
        parsed = parse_output(response.text, self.template.output_fields, len(chunks))
        return {
            "question": question,
            "answer": parsed.answer,
            "status": parsed.status,
            "citations": parsed.citations,
            "confidence": parsed.confidence,
            "abstained": parsed.abstained_token,
            "parsed": parsed.to_dict(),
            "chunks": chunks,
            "raw_output": response.text,
            "model": response.model_reported,
            "prompt_id": self.template.id,
            "cache_key": request.cache_key,
            "from_cache": was_cached,
            "cost_usd": cost,
            # the model call's own latency: when it was made (so, for a cached response, the
            # original call's), not this request's cache read
            "model_latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "timings_ms": timings,
        }
