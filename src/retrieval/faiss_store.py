"""FAISS backend for VectorStore: exact cosine search via IndexFlatIP over
unit-normalized vectors (inner product of unit vectors == cosine similarity)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.retrieval.vectorstore import SearchResult


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._metadata: list[dict[str, Any]] = []

    def add(self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        if len(ids) != vectors.shape[0] or len(ids) != len(metadata):
            raise ValueError("ids, vectors, and metadata must have the same length")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(vectors)
        self._ids.extend(ids)
        self._metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self._index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:  # FAISS pads with -1 when k exceeds the number of stored vectors
                continue
            results.append(
                SearchResult(
                    chunk_id=self._ids[idx], score=float(score), metadata=self._metadata[idx]
                )
            )
        return results

    def persist(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        meta = {"dim": self.dim, "ids": self._ids, "metadata": self._metadata}
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self, path: Path) -> None:
        path = Path(path)
        self._index = faiss.read_index(str(path / "index.faiss"))
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self.dim = meta["dim"]
        self._ids = meta["ids"]
        self._metadata = meta["metadata"]
