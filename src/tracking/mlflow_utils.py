"""Shared MLflow setup so every phase logs to the same local tracking store."""

import os

import mlflow
from dotenv import load_dotenv

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "rag-groundedness-eval"


def tracking_uri() -> str:
    """MLFLOW_TRACKING_URI from the environment or .env, else the local SQLite store."""
    load_dotenv()
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def init_tracking(experiment_name: str = DEFAULT_EXPERIMENT) -> None:
    """Point MLflow at the tracking store and select the experiment.

    Call this once at the top of any script that logs runs, before the first
    `mlflow.start_run()`.
    """
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment_name)
