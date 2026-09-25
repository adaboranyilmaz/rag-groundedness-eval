"""Message Batches: the same requests as `generate_cached`, sent through the Batches API at
half price, with results landing in the same on-disk response cache.

Guarantees:
  - Nothing is paid twice. Requests already in the response cache are not sent, and every
    submitted batch is recorded on disk (`BATCH_DIR/<batch_id>.json`, holding its requests)
    before anything else happens, so a run that is interrupted re-attaches to its batches
    instead of resubmitting them. Collecting skips results whose key is already cached.
  - The caps hold. Each batch reserves its worst case (every request at its full
    `max_tokens`, at the batch price) before it is submitted; requests are packed into a
    batch only while the reservation fits under both caps, and further batches are
    submitted as earlier ones settle. While batches are in flight, a batch smaller than
    `min_requests_per_batch` (or than the rest of the queue) waits for headroom instead of
    being sent (the Phase 5 run sent slivers of 5-18 requests). Re-attached batches are
    re-reserved without a cap
    check, since their cost is already committed.
  - Failures are reported, never hidden: errored, expired or cancelled requests stay
    uncached and are listed, so a re-run retries exactly those.
A batch response has no measured latency (`latency_ms` = 0.0, `extra.service` = "batch").
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.generation.llm import (
    AnthropicBackend,
    BudgetExceeded,
    GenerationRequest,
    ResponseCache,
    SpendLedger,
    response_from_message,
)

BATCH_DIR = Path("data/cache/llm_batches")


@dataclass
class BatchOutcome:
    n_requested: int = 0  # unique requests
    n_cached_before: int = 0
    n_submitted: int = 0
    n_succeeded: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    new_cost_usd: float = 0.0
    batch_ids: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_record(batch_dir: Path, rec: dict) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    path = batch_dir / f"{rec['batch_id']}.json"
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def pending_records(batch_dir: Path = BATCH_DIR) -> list[dict]:
    if not batch_dir.exists():
        return []
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(batch_dir.glob("*.json"))]
    return [r for r in recs if r["status"] != "collected"]


def run_batch_cached(
    requests: list[GenerationRequest],
    cache: ResponseCache,
    ledger: SpendLedger,
    client: Any,
    *,
    max_requests_per_batch: int = 500,
    min_requests_per_batch: int = 50,
    poll_seconds: float = 30.0,
    batch_dir: Path = BATCH_DIR,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> BatchOutcome:
    out = BatchOutcome()
    unique = {r.cache_key: r for r in requests}
    out.n_requested = len(unique)
    todo = {k: r for k, r in unique.items() if not cache.has(k)}
    out.n_cached_before = out.n_requested - len(todo)

    in_flight = pending_records(batch_dir)
    for rec in in_flight:
        ledger.reserve_amount(rec["reserved_usd"], force=True)
        for k in rec["requests"]:
            todo.pop(k, None)
        log(f"re-attached to batch {rec['batch_id']} ({len(rec['requests'])} requests)")
    queue = list(todo.values())

    while queue or in_flight:
        while queue:
            chunk, reserved = [], 0.0
            headroom = ledger.headroom()
            for r in queue[:max_requests_per_batch]:
                worst = ledger.worst_case(r, batch=True)
                if reserved + worst > headroom:
                    break
                chunk.append(r)
                reserved += worst
            if not chunk:
                break
            if in_flight and len(chunk) < min(min_requests_per_batch, len(queue)):
                break  # wait for in-flight batches to settle rather than send a sliver
            ledger.reserve_amount(reserved)
            try:
                batch = client.messages.batches.create(
                    requests=[
                        {"custom_id": r.cache_key, "params": AnthropicBackend.call_params(r)}
                        for r in chunk
                    ]
                )
            except BaseException:
                ledger.release(reserved)
                raise
            rec = {
                "batch_id": batch.id,
                "created_utc": _now(),
                "phase": ledger.phase,
                "status": "submitted",
                "reserved_usd": reserved,
                "requests": {r.cache_key: asdict(r) for r in chunk},
            }
            _write_record(batch_dir, rec)
            in_flight.append(rec)
            out.batch_ids.append(batch.id)
            out.n_submitted += len(chunk)
            queue = queue[len(chunk) :]
            log(
                f"submitted batch {batch.id}: {len(chunk)} requests, "
                f"worst case ${reserved:.4f} reserved; {len(queue)} still queued"
            )
        if not in_flight:
            raise BudgetExceeded(
                f"{len(queue)} requests remain but none fits under the caps "
                f"(headroom ${ledger.headroom():.4f})"
            )

        sleep(poll_seconds)
        for rec in list(in_flight):
            status = client.messages.batches.retrieve(rec["batch_id"])
            if status.processing_status != "ended":
                counts = getattr(status, "request_counts", None)
                done = getattr(counts, "succeeded", "?") if counts is not None else "?"
                log(f"  batch {rec['batch_id']}: {status.processing_status} ({done} succeeded)")
                continue
            _collect(rec, client, cache, ledger, out)
            ledger.release(rec["reserved_usd"])
            rec["status"] = "collected"
            rec["collected_utc"] = _now()
            _write_record(batch_dir, rec)
            in_flight.remove(rec)
            log(f"  batch {rec['batch_id']}: collected; spend this run ${out.new_cost_usd:.4f}")
    return out


def _collect(rec: dict, client: Any, cache: ResponseCache, ledger: SpendLedger, out: BatchOutcome):
    for result in client.messages.batches.results(rec["batch_id"]):
        key = result.custom_id
        if key not in rec["requests"]:
            out.failures.append({"cache_key": key, "type": "unknown_custom_id"})
            continue
        if cache.has(key):  # collected before an interruption; already settled
            continue
        request = GenerationRequest(**rec["requests"][key])
        kind = result.result.type
        if kind != "succeeded":
            error = getattr(result.result, "error", None)
            out.failures.append(
                {
                    "cache_key": key,
                    "batch_id": rec["batch_id"],
                    "type": kind,
                    "error": str(getattr(error, "type", error)) if error is not None else None,
                }
            )
            continue
        msg = result.result.message
        response = response_from_message(
            msg, 0.0, None, extra={"service": "batch", "batch_id": rec["batch_id"]}
        )
        out.new_cost_usd += ledger.settle(
            0.0,
            request.model,
            response.input_tokens or 0,
            response.output_tokens or 0,
            batch=True,
        )
        cache.put(request, response)
        out.n_succeeded += 1
