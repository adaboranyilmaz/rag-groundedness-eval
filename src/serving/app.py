"""The service. Phase 7 scope: the process, its health check and the selected pipeline's
configuration; `POST /query` (answer + citations + groundedness score) is Phase 8.

  GET /health   the API is up, and whether Qdrant and the MLflow server answer. The status
                is "ok" only when both do, "degraded" otherwise (the API itself still serves).
  GET /config   the pipeline registered in MLflow, as selected by
                scripts/09_track_experiments.py (results/metrics/pipeline_selection.json).
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException

SELECTION_PATH = Path(
    os.environ.get("RAG_SELECTION_PATH", "results/metrics/pipeline_selection.json")
)

app = FastAPI(title="RAG groundedness evaluation", version="0.1.0")


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
    status = "ok" if all(d["status"] == "ok" for d in deps.values()) else "degraded"
    return {"status": status, "dependencies": deps}


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
