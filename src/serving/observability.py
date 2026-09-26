"""Prometheus metrics and the JSONL request log.

Metrics live in their own registry (not the client's global one), so a test can build a
fresh set per app instance. Latencies are recorded per pipeline stage, so `/metrics` shows
where time goes as well as how much there is.

The request log is one JSON line per `POST /query`, appended to `<dir>/requests-<UTC
date>.jsonl`: the question, the answer, citations, the groundedness summary, the per-stage
latencies, cache hits and cost; or the error. Writes are serialised by a lock, since the
endpoint runs in a thread pool.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
)

STAGES = ("embed", "search", "generate", "judge_decompose", "judge_verify", "total")
LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0,
)  # fmt: skip


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)  # CPU and memory, on Linux
        r = self.registry
        self.requests = Counter(
            "rag_requests", "POST /query requests by outcome", ["outcome"], registry=r
        )
        self.in_flight = Gauge("rag_requests_in_flight", "requests being served", registry=r)
        self.stage_latency = Histogram(
            "rag_stage_latency_seconds",
            "time per pipeline stage of a successful request",
            ["stage"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.llm_calls = Counter(
            "rag_llm_calls", "model calls by purpose and cache result", ["purpose", "cache"],
            registry=r,
        )  # fmt: skip
        self.spend = Counter("rag_api_spend_usd", "API spend by this process (USD)", registry=r)
        self.answers = Counter("rag_answers", "answers by final status", ["status"], registry=r)
        self.groundedness = Histogram(
            "rag_groundedness_score",
            "groundedness of answered questions (undefined scores are not observed)",
            buckets=tuple(i / 10 for i in range(1, 11)),
            registry=r,
        )
        self.groundedness_unavailable = Counter(
            "rag_groundedness_unavailable", "answers with no score, by reason", ["reason"],
            registry=r,
        )  # fmt: skip

    def observe(self, result: dict[str, Any]) -> None:
        for stage, ms in result["latency_ms"].items():
            self.stage_latency.labels(stage).observe(ms / 1000)
        for call in result["llm_calls"]:
            self.llm_calls.labels(call["purpose"], "hit" if call["cached"] else "miss").inc()
        self.spend.inc(result["cost_usd"])
        self.answers.labels(result["status"]).inc()
        g = result["groundedness"]
        if g["score"] is not None:
            self.groundedness.observe(g["score"])
        else:
            reason = g["unavailable_reason"] or "unknown"
            self.groundedness_unavailable.labels(reason.split(":")[0]).inc()

    def exposition(self) -> bytes:
        return generate_latest(self.registry)


class RequestLog:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path(self, now: datetime) -> Path:
        return self.dir / f"requests-{now:%Y-%m-%d}.jsonl"

    def write(self, record: dict[str, Any]) -> None:
        now = datetime.now(UTC)
        line = json.dumps({"ts_utc": now.isoformat(timespec="milliseconds"), **record})
        with self._lock, self.path(now).open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


def log_record(result: dict[str, Any]) -> dict[str, Any]:
    """The request-log line for a served answer: everything but the chunk texts and claims'
    reasons, which the answer's own record (the response) carries."""
    g = result["groundedness"]
    return {
        "request_id": result["request_id"],
        "outcome": "ok",
        "question": result["question"],
        "answer": result["answer"],
        "status": result["status"],
        "confidence": result["confidence"],
        "citations": [c["label"] for c in result["citations"]],
        "chunk_ids": result["chunk_ids"],
        "groundedness": {
            k: g.get(k)
            for k in (
                "score",
                "unavailable_reason",
                "fully_grounded",
                "n_document_claims",
                "n_supported",
                "n_unsupported",
                "n_contradicted",
            )
        },
        "latency_ms": result["latency_ms"],
        "llm_calls": [
            {k: c[k] for k in ("purpose", "cached", "cost_usd", "cache_key")}
            for c in result["llm_calls"]
        ],
        "cost_usd": result["cost_usd"],
    }
