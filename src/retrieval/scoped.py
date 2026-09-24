"""Dense search over a persisted FAISS index at two scopes: the whole corpus, or only the
chunks of one filing.

The corpus scope is the same exact inner-product search `FaissVectorStore.search` runs.
The filing scope uses FAISS's ID selector, so it is the same computation restricted to one
filing's rows. Scores are therefore comparable across scopes, which the oracle context
(ranking a filing's chunks by retrieval score) relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.retrieval.vectorstore import SearchResult


class ScopedFaissIndex:
    def __init__(self, index_dir: Path):
        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        self.ids: list[str] = meta["ids"]
        self.metadata: list[dict] = meta["metadata"]
        positions: dict[str, list[int]] = {}
        for pos, m in enumerate(self.metadata):
            positions.setdefault(m["doc_id"], []).append(pos)
        self.positions_by_doc = {d: np.asarray(p, dtype=np.int64) for d, p in positions.items()}

    def _results(self, scores: np.ndarray, idx: np.ndarray) -> list[SearchResult]:
        return [
            SearchResult(self.ids[i], float(s), self.metadata[i])
            for s, i in zip(scores[0], idx[0], strict=True)
            if i != -1
        ]

    def search_corpus(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, idx = self.index.search(q, k)
        return self._results(scores, idx)

    def search_filing(
        self, query_vector: np.ndarray, doc_id: str, k: int | None = None
    ) -> list[SearchResult]:
        """Top-k chunks of one filing; `k=None` ranks every chunk of the filing."""
        positions = self.positions_by_doc[doc_id]
        k = len(positions) if k is None else min(k, len(positions))
        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(positions))
        scores, idx = self.index.search(q, k, params=params)
        return self._results(scores, idx)
