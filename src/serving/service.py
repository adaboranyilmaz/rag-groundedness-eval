"""The query service: the registered pipeline plus the groundedness judge, loaded from a
serving bundle (scripts/12_serving.py bundle).

Bundle layout (`RAG_BUNDLE_DIR`; /app/serving in the image):
  manifest.json              sha256 of every file below, and where each came from
  pipeline_selection.json    the registered pipeline (`pipeline_config`)
  judge_agreement.json       the judge's validation against the author's labels
  index/                     the selected FAISS index (index.faiss, meta.json with chunk text)
  model/                     the embedding model's weights at the pinned revision
  cache/                     cached responses for the selected arm's 150 benchmark questions:
                             each generation and its two judge calls

Responses are read from the bundle's cache first, then from a writable cache in the state
directory (`RAG_STATE_DIR`), where new API responses are written, so a repeated question is
never paid for twice. API spend is capped per state directory (`serve_cap_usd`, overridden
by `RAG_SERVE_MAX_USD`). With no API key, only cached questions can be answered; with
`RAG_REPLAY_ONLY=1`, a cache miss is an error even with a key (the load test's mode).
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from src.generation.llm import (
    AnthropicBackend,
    BudgetExceeded,
    CacheMiss,
    GenerationRequest,
    GenerationResponse,
    ResponseCache,
    SpendLedger,
    replay_only,
)
from src.serving.grounding import GroundednessJudge, load_validation
from src.serving.pipeline import RAGPipeline

SERVING_CONFIG_PATH = Path("configs/serving.yaml")
SNIPPET_CHARS = 280
PHASE = "serving"


class ApiKeyMissing(RuntimeError):
    """An uncached question, and no API key to answer it with."""


class NoKeyBackend:
    name = "anthropic"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise ApiKeyMissing(
            "this question is not in the response cache and no ANTHROPIC_API_KEY is set"
        )


class LayeredCache:
    """Read the bundle's cache, then the writable one; write only to the writable one."""

    def __init__(self, read_only: ResponseCache, writable: ResponseCache):
        self.read_only, self.writable = read_only, writable

    def has(self, key: str) -> bool:
        return self.read_only.has(key) or self.writable.has(key)

    def get(self, key: str) -> GenerationResponse | None:
        return self.read_only.get(key) or self.writable.get(key)

    def put(self, request: GenerationRequest, response: GenerationResponse) -> None:
        self.writable.put(request, response)


def make_api_backend(cfg: dict) -> Any:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return NoKeyBackend()
    import anthropic

    client = anthropic.Anthropic(
        timeout=float(cfg["api_timeout_s"]), max_retries=int(cfg["api_max_retries"])
    )
    return AnthropicBackend(client=client)


def upstream_errors() -> tuple[type[BaseException], ...]:
    """Failures of a model call that a request can report instead of crashing on."""
    import anthropic

    return (CacheMiss, BudgetExceeded, ApiKeyMissing, anthropic.APIError)


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 1] + "…"


def _chunk_view(c: dict, with_text: bool) -> dict:
    out = {
        "label": c["label"],
        "chunk_id": c["chunk_id"],
        "doc_id": c["doc_id"],
        "page": c["page"],
        "section": c["section"],
        "score": c["score"],
        "snippet": _snippet(c["text"]),
    }
    if with_text:
        out["text"] = c["text"]
    return out


class QueryService:
    def __init__(
        self,
        pipeline: RAGPipeline,
        judge: GroundednessJudge,
        selection: dict,
        manifest: dict | None = None,
    ):
        self.pipeline, self.judge = pipeline, judge
        self.selection = selection
        self.manifest = manifest or {}
        cfg = selection["pipeline_config"]
        self.pipeline_info = {
            "selected_arm": selection["winner"],
            "chunking": cfg["retrieval"]["chunking"],
            "embedding": cfg["retrieval"]["embedding"],
            "retriever": cfg["retrieval"]["method"],
            "k": cfg["retrieval"]["k"],
            "generator": cfg["generator"]["model"]["model"],
            "prompt_id": cfg["prompt"]["id"],
        }

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: Path,
        state_dir: Path,
        root: Path = Path("."),
        device: str | None = None,
        serving_config: Path = SERVING_CONFIG_PATH,
        *,
        use_bundle_cache: bool = True,
        ledger: SpendLedger | None = None,
    ) -> QueryService:
        """`use_bundle_cache=False` and `ledger` are for measuring live calls (the load
        test's live sample): responses only from the state directory's cache, and spend
        charged to the given ledger (the project's) instead of the state directory's."""
        import json

        bundle_dir, state_dir, root = Path(bundle_dir), Path(state_dir), Path(root)
        scfg = yaml.safe_load((root / serving_config).read_text(encoding="utf-8"))
        ecfg = yaml.safe_load((root / scfg["evaluation_config"]).read_text(encoding="utf-8"))
        selection = json.loads((bundle_dir / "pipeline_selection.json").read_text(encoding="utf-8"))
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))

        state_dir.mkdir(parents=True, exist_ok=True)
        writable = ResponseCache(state_dir / "cache")
        cache = (
            LayeredCache(ResponseCache(bundle_dir / "cache"), writable)
            if use_bundle_cache
            else writable
        )
        if ledger is None:
            cap = float(os.environ.get("RAG_SERVE_MAX_USD") or scfg["serve_cap_usd"])
            b = ecfg["budget"]
            ledger = SpendLedger(
                state_dir / "spend.json",
                b["prices_usd_per_mtok"],
                cap,
                PHASE,
                cap,
                b["batch_discount"],
            )
        backend = make_api_backend(scfg)
        pipeline = RAGPipeline(
            selection["pipeline_config"],
            root=root,
            backend=backend,
            cache=cache,
            ledger=ledger,
            index_dir=bundle_dir / "index",
            model_path=str(bundle_dir / "model"),
            device=device,
        )
        judge = GroundednessJudge(
            ecfg,
            backend=backend,
            cache=cache,
            ledger=ledger,
            prompts_dir=root / "prompts" / "judge",
            validation=load_validation(bundle_dir / "judge_agreement.json"),
            recoverable=upstream_errors(),
        )
        return cls(pipeline, judge, selection, manifest)

    def status(self) -> dict[str, Any]:
        ledger = self.pipeline.ledger
        return {
            "replay_only": replay_only(),
            "api_key_configured": not isinstance(self.pipeline.backend, NoKeyBackend),
            "bundle_sha256": self.manifest.get("bundle_sha256"),
            "spend_usd": round(ledger.state["total_usd"], 6) if ledger else None,
            "spend_cap_usd": ledger.project_cap_usd if ledger else None,
            "pipeline": self.pipeline_info,
        }

    def query(self, question: str, include_context: bool = False) -> dict[str, Any]:
        """Answer + groundedness. A failed generation call raises (one of
        `upstream_errors()`); a failed judge call leaves the score undefined, with the reason."""
        t0 = time.perf_counter()
        ans = self.pipeline.answer(question)
        assessed = self.judge.assess(question, ans["parsed"], ans["chunks"])
        latency = {**ans["timings_ms"], **assessed["timings_ms"]}
        latency["total"] = (time.perf_counter() - t0) * 1000

        by_label = {c["label"]: c for c in ans["chunks"]}
        calls = [
            {
                "purpose": "generate",
                "cache_key": ans["cache_key"],
                "cached": ans["from_cache"],
                "cost_usd": ans["cost_usd"],
                "model_latency_ms": ans["model_latency_ms"],
                "model": ans["model"],
                "input_tokens": ans["input_tokens"],
                "output_tokens": ans["output_tokens"],
            },
            *assessed["llm_calls"],
        ]
        g = assessed["groundedness"]
        return {
            "request_id": uuid.uuid4().hex,
            "question": question,
            "answer": ans["answer"],
            "status": assessed["status"],
            "confidence": ans["confidence"],
            "citations": [
                _chunk_view(by_label[x], False) for x in ans["citations"] if x in by_label
            ],
            "context": [_chunk_view(c, True) for c in ans["chunks"]] if include_context else None,
            "chunk_ids": [c["chunk_id"] for c in ans["chunks"]],
            "groundedness": {
                **g,
                "judge_model": assessed["model"],
                "judge_prompts": {
                    "decompose": assessed["decompose_prompt"],
                    "verify": assessed["verify_prompt"],
                },
                "validation": self.judge.validation,
            },
            "latency_ms": {k: round(v, 3) for k, v in latency.items()},
            "llm_calls": calls,
            "cost_usd": sum(c["cost_usd"] for c in calls),
            "pipeline": self.pipeline_info,
        }
