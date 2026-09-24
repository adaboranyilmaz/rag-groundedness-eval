"""Lexical retrieval with BM25 (Okapi variant, `rank_bm25`) over one chunking strategy's
chunks.

The tokeniser is deliberately simple and financial-text aware rather than borrowed from a
general NLP library: lowercase; thousands separators stripped so "1,577" and "1577" are
the same token; letters and digits split apart so "FY2018" matches a filing's bare
"2018"; decimals kept whole. No stemming and no stopword list — BM25's IDF already
down-weights terms that occur in most chunks, and every extra normalisation step is one
more thing that could silently eat a number.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.vectorstore import SearchResult

_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(_THOUSANDS_SEP.sub("", text.lower()))


def top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the k highest scores, ties broken by lower index so rankings are
    deterministic."""
    k = min(k, len(scores))
    if k == 0:
        return []
    candidates = np.argpartition(-scores, k - 1)[:k]
    # argpartition keeps any k of a tie at the boundary; widen to every index tied with
    # the k-th score before the deterministic sort, so which tied index survives never
    # depends on argpartition's internal order.
    kth = scores[candidates].min()
    candidates = np.union1d(candidates, np.flatnonzero(scores == kth))
    order = sorted(candidates.tolist(), key=lambda i: (-scores[i], i))
    return order[:k]


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks], k1=k1, b=b)

    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        scores = self._bm25.get_scores(tokenize(query))
        return [
            SearchResult(
                chunk_id=self.chunks[i]["chunk_id"],
                score=float(scores[i]),
                metadata=self.chunks[i],
            )
            for i in top_k_indices(scores, k)
        ]
