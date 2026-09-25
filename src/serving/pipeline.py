"""The selected RAG pipeline as one object: question -> retrieved chunks -> generation request
-> parsed answer with citations and stated confidence.

Retrieval is the `retrieved` condition's own call (`ScopedFaissIndex.search_corpus` at
`search_depth`, then the top k), and the request is built by the same `build_request` the
Phase 4 traces were, so for a benchmark question the pipeline produces the traced request
byte for byte (same cache key), which the tests check. The configuration is the one
registered in MLflow (`results/metrics/pipeline_selection.json`, `pipeline_config`).

Generation goes through the response cache. An API model with no spend ledger can only be
served from the cache: `generate_cached` refuses an uncached API call without a ledger.
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
    ):
        self.config = config
        self.root = Path(root)
        rc = config["retrieval"]
        if rc["method"] != "dense" or rc["backend"] != "faiss":
            raise NotImplementedError("the pipeline is wired for dense FAISS retrieval only")
        if rc["search_depth"] < rc["k"]:
            raise ValueError("retrieval.search_depth must be >= retrieval.k")
        self.k, self.search_depth = rc["k"], rc["search_depth"]
        if embed_model is None:
            from src.retrieval.embeddings import EmbeddingModel

            embed_model = EmbeddingModel(rc["embedding"])
        if index is None:
            from src.retrieval.scoped import ScopedFaissIndex

            index = ScopedFaissIndex(
                self.root / "data" / "indices" / f"{rc['chunking']}__{rc['embedding']}__faiss"
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

    def retrieve(self, question: str) -> list[dict]:
        qv = self.embed_model.encode_queries([question])
        return trace_chunks(self.index.search_corpus(qv, self.search_depth)[: self.k])

    def answer(self, question: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        chunks = self.retrieve(question)
        t1 = time.perf_counter()
        request, _ = build_request(
            self.model_cfg, self.max_tokens, self.template, {"question": question}, chunks
        )
        response, was_cached, cost = generate_cached(self.backend, request, self.cache, self.ledger)
        t2 = time.perf_counter()
        parsed = parse_output(response.text, self.template.output_fields, len(chunks))
        return {
            "question": question,
            "answer": parsed.answer,
            "status": parsed.status,
            "citations": parsed.citations,
            "confidence": parsed.confidence,
            "abstained": parsed.abstained_token,
            "chunks": chunks,
            "raw_output": response.text,
            "model": response.model_reported,
            "prompt_id": self.template.id,
            "cache_key": request.cache_key,
            "from_cache": was_cached,
            "cost_usd": cost,
            "timings_ms": {
                "retrieve": (t1 - t0) * 1000,
                "generate": (t2 - t1) * 1000,
            },
        }
