"""Unit tests for src/retrieval/embeddings.py. Runs the real (small) sentence-transformers
models rather than mocking them — the thing worth verifying is that our wrapper applies
the BGE query prefix correctly, and a mock can't catch that being backwards."""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.embeddings import MODEL_REGISTRY, EmbeddingModel

# CPU is enough for a handful of short sentences; keep tests fast regardless of device.
MINILM = "all-MiniLM-L6-v2"  # no query prefix
BGE_SMALL = "bge-small-en-v1.5"  # has a query prefix


class TestEmbeddingModel:
    def test_unknown_model_name_raises(self):
        with pytest.raises(ValueError):
            EmbeddingModel("not-a-real-model")

    def test_corpus_vectors_are_unit_normalized(self):
        model = EmbeddingModel(MINILM, device="cpu")
        vectors = model.encode_corpus(["Revenue increased by 10%.", "Net income was $5 million."])
        assert vectors.shape == (2, MODEL_REGISTRY[MINILM].dim)
        norms = np.linalg.norm(vectors, axis=1)
        assert norms == pytest.approx(1.0, abs=1e-3)

    def test_query_vectors_are_unit_normalized(self):
        model = EmbeddingModel(MINILM, device="cpu")
        vectors = model.encode_queries(["What was net income?"])
        assert vectors.shape == (1, MODEL_REGISTRY[MINILM].dim)
        assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-3)

    def test_bge_query_prefix_changes_query_but_not_corpus_encoding(self):
        model = EmbeddingModel(BGE_SMALL, device="cpu")
        text = "What was net income?"

        corpus_vec = model.encode_corpus([text])[0]
        query_vec = model.encode_queries([text])[0]

        # Same raw string, but the query path prepends an instruction the corpus path
        # does not -- the two encodings must therefore differ.
        assert not np.allclose(corpus_vec, query_vec)

    def test_minilm_query_and_corpus_encoding_match_when_prefix_is_empty(self):
        model = EmbeddingModel(MINILM, device="cpu")
        assert MODEL_REGISTRY[MINILM].query_prefix == ""
        text = "What was net income?"

        corpus_vec = model.encode_corpus([text])[0]
        query_vec = model.encode_queries([text])[0]

        assert np.allclose(corpus_vec, query_vec)
