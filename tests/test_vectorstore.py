"""Unit tests for the VectorStore backends in src/retrieval/.

Qdrant tests need the Docker service from docker-compose.yml running; they skip cleanly
(rather than fail) if it isn't reachable, so `pytest` stays green without Docker.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.faiss_store import FaissVectorStore
from src.retrieval.vectorstore import get_vectorstore

IDS = ["a", "b", "c"]
# Orthogonal-ish unit vectors so nearest-neighbor search has an unambiguous answer.
VECTORS = np.array([[1.0, 0.0], [0.0, 1.0], [0.70710678, 0.70710678]], dtype=np.float32)
METADATA = [{"label": "x-axis"}, {"label": "y-axis"}, {"label": "diagonal"}]


def _qdrant_available() -> bool:
    try:
        from qdrant_client import QdrantClient

        QdrantClient(host="localhost", port=6333).get_collections()
        return True
    except Exception:
        return False


QDRANT_UP = _qdrant_available()


def _assert_contract(store) -> None:
    """Shared behavioral contract both backends must satisfy."""
    store.add(IDS, VECTORS, METADATA)

    results = store.search(np.array([0.9, 0.1], dtype=np.float32), k=2)
    assert [r.chunk_id for r in results] == ["a", "c"]
    assert results[0].metadata["label"] == "x-axis"
    assert results[0].score > results[1].score  # ranked best-first

    exact = store.search(np.array([0.0, 1.0], dtype=np.float32), k=1)
    assert exact[0].chunk_id == "b"
    assert exact[0].score == pytest.approx(1.0, abs=1e-4)


class TestFaissVectorStore:
    def test_search_contract(self):
        _assert_contract(FaissVectorStore(dim=2))

    def test_rejects_mismatched_lengths(self):
        store = FaissVectorStore(dim=2)
        with pytest.raises(ValueError):
            store.add(["a", "b"], VECTORS[:1], METADATA[:1])

    def test_persist_and_load_round_trip(self, tmp_path):
        store = FaissVectorStore(dim=2)
        store.add(IDS, VECTORS, METADATA)
        before = store.search(np.array([0.9, 0.1], dtype=np.float32), k=3)

        store.persist(tmp_path / "idx")

        reloaded = FaissVectorStore(dim=2)
        reloaded.load(tmp_path / "idx")
        after = reloaded.search(np.array([0.9, 0.1], dtype=np.float32), k=3)

        assert [r.chunk_id for r in before] == [r.chunk_id for r in after]
        assert [r.score for r in before] == pytest.approx([r.score for r in after])


@pytest.mark.skipif(not QDRANT_UP, reason="Qdrant not reachable on localhost:6333")
class TestQdrantVectorStore:
    def test_search_contract(self):
        store = get_vectorstore("qdrant", dim=2, collection_name="test_contract", recreate=True)
        _assert_contract(store)

    def test_persist_and_load_round_trip(self, tmp_path):
        store = get_vectorstore("qdrant", dim=2, collection_name="test_persist", recreate=True)
        store.add(IDS, VECTORS, METADATA)
        before = store.search(np.array([0.9, 0.1], dtype=np.float32), k=3)

        store.persist(tmp_path / "idx")

        reloaded = get_vectorstore("qdrant", dim=2, collection_name="ignored")
        reloaded.load(tmp_path / "idx")
        after = reloaded.search(np.array([0.9, 0.1], dtype=np.float32), k=3)

        assert [r.chunk_id for r in before] == [r.chunk_id for r in after]
        assert [r.score for r in before] == pytest.approx([r.score for r in after])


@pytest.mark.skipif(not QDRANT_UP, reason="Qdrant not reachable on localhost:6333")
def test_faiss_and_qdrant_agree_on_top_k():
    """The equivalence property Phase 2's acceptance criterion names directly: same
    query, same vectors -> same top-k from both backends."""
    faiss_store = FaissVectorStore(dim=2)
    faiss_store.add(IDS, VECTORS, METADATA)

    qdrant_store = get_vectorstore(
        "qdrant", dim=2, collection_name="test_equivalence", recreate=True
    )
    qdrant_store.add(IDS, VECTORS, METADATA)

    query = np.array([0.9, 0.1], dtype=np.float32)
    faiss_results = faiss_store.search(query, k=3)
    qdrant_results = qdrant_store.search(query, k=3)

    assert [r.chunk_id for r in faiss_results] == [r.chunk_id for r in qdrant_results]
    assert [r.score for r in faiss_results] == pytest.approx(
        [r.score for r in qdrant_results], abs=1e-4
    )
