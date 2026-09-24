"""Dense retrieval: embed the query with the model's query-side encoding (including the BGE
instruction prefix where the model needs one) and search a VectorStore. Backend-agnostic
by construction — it only sees the `VectorStore` protocol."""

from __future__ import annotations

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vectorstore import SearchResult, VectorStore


class DenseRetriever:
    def __init__(self, store: VectorStore, embed_model: EmbeddingModel):
        self.store = store
        self.embed_model = embed_model

    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        vector = self.embed_model.encode_queries([query])[0]
        return self.store.search(vector, k=k)
