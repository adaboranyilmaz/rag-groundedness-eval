"""The selected pipeline as an MLflow pyfunc model, so the model registry entry is the
runnable pipeline and not only a record of its configuration.

The registered model carries the pipeline configuration and the prompt file as artefacts and
this package as code. Indices, chunks and the response cache are data, not model artefacts
(about 1.5 GB, DVC-versioned): they are read from `RAG_DATA_ROOT` (default: the working
directory, i.e. a checkout with `dvc pull` done). Input: a DataFrame with a `question`
column, or a list of question strings. Output: one answer record per question
(src/serving/pipeline.py).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow.pyfunc


class RAGPipelineModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        from src.serving.pipeline import RAGPipeline

        config = json.loads(Path(context.artifacts["pipeline_config"]).read_text(encoding="utf-8"))
        self.pipeline = RAGPipeline(config, root=Path(os.environ.get("RAG_DATA_ROOT", ".")))

    # model_input is left unannotated: MLflow reads a predict() type hint as an input schema.
    def predict(self, context, model_input, params: dict | None = None) -> list[dict]:
        if hasattr(model_input, "columns"):
            questions = list(model_input["question"])
        else:
            questions = list(model_input)
        return [self.pipeline.answer(str(q)) for q in questions]
