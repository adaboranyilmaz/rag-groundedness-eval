"""Backend-agnostic vector store contract.

Every backend stores unit-normalized vectors and does exact (brute-force) cosine search
via inner product — see DECISIONS.md's Phase 2 entry for why exact search is used at this
corpus scale instead of an approximate index. `get_vectorstore` is the single place that
knows how to construct a given backend, so swapping backends elsewhere in the codebase is
a config change (which string you pass), not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    """`add` is append-only. `persist`/`load` round-trip the store's full state through
    `path` so a later process can search it without rebuilding from raw chunks."""

    def add(
        self,
        ids: list[str],
        vectors: Any,  # np.ndarray, shape (n, dim), rows already unit-normalized
        metadata: list[dict[str, Any]],
    ) -> None: ...

    def search(self, query_vector: Any, k: int) -> list[SearchResult]: ...

    def persist(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


def get_vectorstore(backend: str, dim: int, **kwargs: Any) -> VectorStore:
    """Factory keyed by backend name. `kwargs` are passed through to the chosen
    implementation's constructor (e.g. `collection_name` for Qdrant)."""
    if backend == "faiss":
        from src.retrieval.faiss_store import FaissVectorStore

        return FaissVectorStore(dim=dim, **kwargs)
    if backend == "qdrant":
        from src.retrieval.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(dim=dim, **kwargs)
    raise ValueError(f"Unknown vector store backend: {backend!r} (expected 'faiss' or 'qdrant')")
