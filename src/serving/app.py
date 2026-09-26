"""The service.

  POST /query   question -> answer, citations, groundedness score (the Phase 5 judge, with
                its validation against the author's labels) and per-stage latency
                (src/serving/service.py; bodies in schemas.py)
  GET  /health  liveness, readiness, and whether Qdrant and the MLflow server answer. The
                status is "ok" when the service is ready and both answer, "degraded" when
                one does not (the API itself still serves), "not_ready" before the bundle
                has loaded. Always HTTP 200: it reports, it does not gate.
  GET  /ready   200 once the bundle has loaded, else 503 (the container and k8s probes)
  GET  /metrics Prometheus exposition (src/serving/observability.py)
  GET  /config  the pipeline registered in MLflow, as selected by
                scripts/09_track_experiments.py (results/metrics/pipeline_selection.json)

Every `POST /query` is logged as a JSON line under `<RAG_STATE_DIR>/logs/`.

Environment: RAG_BUNDLE_DIR (default build/serving), RAG_STATE_DIR (default state),
ANTHROPIC_API_KEY (optional: without it only cached questions are answered),
RAG_SERVE_MAX_USD, RAG_REPLAY_ONLY, QDRANT_HOST/QDRANT_PORT, MLFLOW_TRACKING_URI.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST

from src.generation.llm import BudgetExceeded, CacheMiss
from src.serving.observability import Metrics, RequestLog, log_record
from src.serving.schemas import QueryRequest, QueryResponse
from src.serving.service import ApiKeyMissing, QueryService, upstream_errors

SELECTION_PATH = Path(
    os.environ.get("RAG_SELECTION_PATH", "results/metrics/pipeline_selection.json")
)
log = logging.getLogger("uvicorn.error")


class State:
    service: QueryService | None = None
    load_error: str | None = None
    metrics: Metrics = Metrics()
    request_log: RequestLog | None = None


state = State()


def load_service() -> None:
    bundle = Path(os.environ.get("RAG_BUNDLE_DIR", "build/serving"))
    state_dir = Path(os.environ.get("RAG_STATE_DIR", "state"))
    state.metrics = Metrics()
    try:
        state.request_log = RequestLog(state_dir / "logs")
        state.service = QueryService.from_bundle(bundle, state_dir)
        state.load_error = None
        log.info("serving bundle loaded from %s", bundle)
    except Exception as e:  # noqa: BLE001 - reported by /health and /ready, not fatal
        state.service = None
        state.load_error = f"{type(e).__name__}: {e}"
        log.error("serving bundle not loaded: %s", state.load_error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("RAG_SKIP_LOAD") != "1":  # tests install their own service
        load_service()
    yield


app = FastAPI(title="RAG groundedness evaluation", version="0.2.0", lifespan=lifespan)


def _probe(url: str, timeout: float = 2.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return {"status": "ok", "http_status": r.status}
    except Exception as e:  # noqa: BLE001 - a health probe reports, it does not raise
        return {"status": "unreachable", "error": f"{type(e).__name__}: {e}"}


@app.get("/health")
def health() -> dict:
    host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant = f"http://{host}:{os.environ.get('QDRANT_PORT', '6333')}"
    mlflow = os.environ.get("MLFLOW_TRACKING_URI", "")
    deps = {"qdrant": _probe(f"{qdrant}/healthz")}
    deps["mlflow"] = (
        _probe(f"{mlflow.rstrip('/')}/health")
        if mlflow.startswith("http")
        else {"status": "not configured", "tracking_uri": mlflow or None}
    )
    ready = state.service is not None
    if not ready:
        status = "not_ready"
    else:
        status = "ok" if all(d["status"] == "ok" for d in deps.values()) else "degraded"
    body = {"status": status, "ready": ready, "dependencies": deps}
    if ready:
        body["service"] = state.service.status()
    else:
        body["load_error"] = state.load_error
    return body


@app.get("/ready")
def ready() -> Response:
    if state.service is None:
        return JSONResponse({"ready": False, "load_error": state.load_error}, status_code=503)
    return JSONResponse({"ready": True})


@app.get("/metrics")
def metrics() -> Response:
    return Response(state.metrics.exposition(), media_type=CONTENT_TYPE_LATEST)


# upstream failure -> (HTTP status, error code) for a failed generation call
ERROR_MAP = (
    (CacheMiss, 503, "not_cached"),
    (ApiKeyMissing, 503, "no_api_key"),
    (BudgetExceeded, 503, "budget_exhausted"),
)


def _error(e: BaseException) -> tuple[int, str]:
    for cls, code, name in ERROR_MAP:
        if isinstance(e, cls):
            return code, name
    return 502, "upstream_error"  # anthropic.APIError: the model API failed


@app.post("/query", response_model=QueryResponse, response_model_exclude_none=False)
def query(body: QueryRequest):
    service = state.service
    if service is None:
        raise HTTPException(503, {"error": "not_ready", "detail": state.load_error})
    m = state.metrics
    m.in_flight.inc()
    try:
        result = service.query(body.question, include_context=body.include_context)
    except upstream_errors() as e:
        code, name = _error(e)
        m.requests.labels(name).inc()
        if state.request_log is not None:
            state.request_log.write(
                {"outcome": name, "question": body.question, "error": f"{type(e).__name__}: {e}"}
            )
        return JSONResponse({"error": name, "detail": str(e)}, status_code=code)
    except Exception as e:
        m.requests.labels("internal_error").inc()
        if state.request_log is not None:
            state.request_log.write(
                {
                    "outcome": "internal_error",
                    "question": body.question,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
        raise
    finally:
        m.in_flight.dec()
    m.requests.labels("ok").inc()
    m.observe(result)
    if state.request_log is not None:
        state.request_log.write(log_record(result))
    result.pop("chunk_ids")
    return result


@app.get("/config")
def config() -> dict:
    if not SELECTION_PATH.exists():
        raise HTTPException(503, f"{SELECTION_PATH} not found: run 09_track_experiments.py select")
    sel = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    winner = sel["arms"][sel["winner"]]
    return {
        "pipeline": sel["pipeline_config"],
        "selected_arm": sel["winner"],
        "rule": sel["meta"]["rule"],
        "correct_and_grounded": winner["correct_and_grounded"],
        "winner_minus_runner_up": sel["winner_minus_runner_up"],
    }
