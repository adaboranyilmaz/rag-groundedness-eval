"""Qdrant backend for VectorStore: one collection per (chunking_strategy, embedding_model),
Distance.DOT over unit-normalized vectors (== cosine similarity), search forced to exact
(brute-force) mode so results are directly comparable to FaissVectorStore's IndexFlatIP.

Qdrant's own storage is server-managed (the Docker volume), so `persist()`/`load()` here
just write/read a small local manifest recording which collection this instance points at
— the vectors themselves never leave the server.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.retrieval.vectorstore import SearchResult


class QdrantVectorStore:
    def __init__(self, dim: int, collection_name: str, recreate: bool = False):
        load_dotenv()
        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", "6333"))
        self.dim = dim
        self.collection_name = collection_name
        self._client = QdrantClient(host=host, port=port)

        exists = self._client.collection_exists(collection_name)
        if recreate or not exists:
            if exists:
                self._client.delete_collection(collection_name)
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.DOT),
            )
            self._next_id = 0
        else:
            self._next_id = self._client.count(collection_name, exact=True).count

    def add(self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        if len(ids) != vectors.shape[0] or len(ids) != len(metadata):
            raise ValueError("ids, vectors, and metadata must have the same length")
        # Qdrant point ids must be int/UUID, not our string chunk_ids, so the payload
        # carries the real chunk_id and a per-instance counter supplies the point id.
        base = self._next_id
        points = [
            models.PointStruct(
                id=base + i, vector=vectors[i].tolist(), payload={"chunk_id": ids[i], **metadata[i]}
            )
            for i in range(len(ids))
        ]
        self._next_id += len(ids)
        for start in range(0, len(points), 256):
            self._client.upsert(
                collection_name=self.collection_name, points=points[start : start + 256]
            )

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        hits = self._client.query_points(
            collection_name=self.collection_name,
            query=np.asarray(query_vector, dtype=np.float32).tolist(),
            limit=k,
            search_params=models.SearchParams(exact=True),
            with_payload=True,
        ).points
        results = []
        for hit in hits:
            payload = dict(hit.payload)
            chunk_id = payload.pop("chunk_id")
            results.append(
                SearchResult(chunk_id=chunk_id, score=float(hit.score), metadata=payload)
            )
        return results

    def persist(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        manifest = {"backend": "qdrant", "collection_name": self.collection_name, "dim": self.dim}
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def load(self, path: Path) -> None:
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["backend"] != "qdrant":
            raise ValueError(f"{path} is not a Qdrant manifest")
        self.collection_name = manifest["collection_name"]
        self.dim = manifest["dim"]
        if not self._client.collection_exists(self.collection_name):
            raise RuntimeError(
                f"Manifest points at collection {self.collection_name!r}, which no longer "
                "exists on the Qdrant server — the manifest and server have diverged."
            )
        self._next_id = self._client.count(self.collection_name, exact=True).count
