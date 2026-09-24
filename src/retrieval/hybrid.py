"""Hybrid retrieval by reciprocal rank fusion (Cormack, Clarke & Buettcher, 2009).

RRF fuses ranks rather than scores, which is the point: cosine similarities and BM25
scores live on incomparable scales, and any score normalisation would be one more tuned
knob. `k_rrf=60` is the constant from the original paper, not tuned here.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.vectorstore import SearchResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]], k_rrf: int = 60
) -> list[SearchResult]:
    """score(d) = sum over rankings containing d of 1 / (k_rrf + rank), rank 1-based.
    Ties are broken by chunk_id so the fused order is deterministic."""
    scores: dict[str, float] = {}
    metadata: dict[str, dict] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k_rrf + rank)
            metadata.setdefault(result.chunk_id, result.metadata)
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [SearchResult(cid, scores[cid], metadata[cid]) for cid in ordered]


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        candidate_depth: int = 100,
        k_rrf: int = 60,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.candidate_depth = candidate_depth
        self.k_rrf = k_rrf

    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        fused = reciprocal_rank_fusion(
            [
                self.dense.retrieve(query, self.candidate_depth),
                self.bm25.retrieve(query, self.candidate_depth),
            ],
            k_rrf=self.k_rrf,
        )
        return fused[:k]
