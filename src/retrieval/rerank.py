"""Cross-encoder reranking (`bge-reranker-base`) of a first-stage candidate list.

A cross-encoder reads query and chunk jointly, so it is far more expensive per candidate
than a bi-encoder lookup; it is applied only to the top `depth` first-stage candidates,
and its latency is measured separately so the grid can say whether it earns it.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from sentence_transformers import CrossEncoder

from src.retrieval.vectorstore import SearchResult

DEFAULT_RERANKER = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, device=self.device, max_length=max_length)

    def rerank(self, query: str, candidates: Sequence[SearchResult], k: int) -> list[SearchResult]:
        if not candidates:
            return []
        pairs = [(query, c.metadata["text"]) for c in candidates]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        # Stable on ties: equal cross-encoder scores keep their first-stage order.
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
        return [
            SearchResult(candidates[i].chunk_id, float(scores[i]), candidates[i].metadata)
            for i in order[:k]
        ]
