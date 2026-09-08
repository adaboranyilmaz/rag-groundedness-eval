"""sentence-transformers wrapper with the three models named in PROJECT_SPEC.md §4.

`encode_corpus`/`encode_queries` are separate methods (not one `encode` with a flag)
because BGE models require a query-instruction prefix for correct asymmetric retrieval
that must NOT be applied to documents — separating the methods makes that impossible to
get backwards by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class ModelSpec:
    name: str  # short name, used in file/collection naming
    hf_id: str
    dim: int
    query_prefix: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "bge-small-en-v1.5": ModelSpec(
        "bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", 384, BGE_QUERY_PREFIX
    ),
    "all-MiniLM-L6-v2": ModelSpec(
        "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", 384, ""
    ),
    "bge-base-en-v1.5": ModelSpec(
        "bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", 768, BGE_QUERY_PREFIX
    ),
}


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingModel:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown embedding model: {model_name!r}. Known: {list(MODEL_REGISTRY)}"
            )
        self.spec = MODEL_REGISTRY[model_name]
        self.device = device or _default_device()
        self.batch_size = batch_size
        self._model = SentenceTransformer(self.spec.hf_id, device=self.device)

    @property
    def dim(self) -> int:
        return self.spec.dim

    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prefixed = [self.spec.query_prefix + t for t in texts]
        return self._encode(prefixed)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)
