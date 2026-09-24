"""Unit tests for BM25, reciprocal rank fusion, and reranker ordering in src/retrieval/."""

import numpy as np

from src.retrieval.bm25 import BM25Retriever, tokenize, top_k_indices
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.rerank import CrossEncoderReranker
from src.retrieval.vectorstore import SearchResult


def sr(cid: str, text: str = "") -> SearchResult:
    return SearchResult(cid, 0.0, {"chunk_id": cid, "text": text})


class TestTokenize:
    def test_financial_normalisation(self):
        tokens = tokenize("FY2018 capex was $1,577 million (12.5%)")
        assert tokens == ["fy", "2018", "capex", "was", "1577", "million", "12.5"]

    def test_comma_list_of_numbers_is_not_merged(self):
        # "1,2" is not a thousands separator; only comma + exactly three digits is.
        assert tokenize("items 1,2 and 10,000,000") == ["items", "1", "2", "and", "10000000"]


class TestTopK:
    def test_ties_broken_by_lower_index(self):
        scores = np.array([1.0, 3.0, 3.0, 3.0, 0.5])
        assert top_k_indices(scores, 2) == [1, 2]

    def test_k_larger_than_corpus(self):
        assert top_k_indices(np.array([0.1, 0.2]), 5) == [1, 0]

    def test_empty(self):
        assert top_k_indices(np.array([]), 3) == []


def test_bm25_ranks_lexical_match_first():
    chunks = [
        {"chunk_id": "a", "text": "Dividends paid to shareholders were 3,193"},
        {"chunk_id": "b", "text": "Capital expenditure purchases of property plant"},
        {"chunk_id": "c", "text": "Risk factors and legal proceedings"},
    ]
    results = BM25Retriever(chunks).retrieve("What were capital expenditure purchases?", 2)
    assert results[0].chunk_id == "b"
    assert results[0].metadata is chunks[1]


class TestRRF:
    def test_hand_computed_scores(self):
        fused = reciprocal_rank_fusion([[sr("a"), sr("b")], [sr("b"), sr("c")]], k_rrf=60)
        scores = {r.chunk_id: r.score for r in fused}
        assert scores["b"] == 1 / 62 + 1 / 61
        assert scores["a"] == 1 / 61
        assert scores["c"] == 1 / 62
        assert [r.chunk_id for r in fused] == ["b", "a", "c"]

    def test_ties_broken_by_chunk_id(self):
        fused = reciprocal_rank_fusion([[sr("z")], [sr("y")]], k_rrf=60)
        assert [r.chunk_id for r in fused] == ["y", "z"]

    def test_metadata_preserved(self):
        fused = reciprocal_rank_fusion([[sr("a", "hello")]])
        assert fused[0].metadata["text"] == "hello"


class _StubCrossEncoder:
    """Scores a pair by the length of the chunk text, so the expected order is obvious."""

    def predict(self, pairs, **kwargs):
        return np.array([float(len(text)) for _, text in pairs])


def make_reranker() -> CrossEncoderReranker:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.batch_size = 4
    reranker._model = _StubCrossEncoder()
    return reranker


def test_rerank_orders_by_cross_encoder_score_and_truncates():
    cands = [sr("short", "x"), sr("long", "xxxxx"), sr("mid", "xxx")]
    out = make_reranker().rerank("q", cands, k=2)
    assert [r.chunk_id for r in out] == ["long", "mid"]
    assert out[0].score == 5.0


def test_rerank_ties_keep_first_stage_order():
    cands = [sr("first", "aa"), sr("second", "bb")]
    assert [r.chunk_id for r in make_reranker().rerank("q", cands, k=2)] == ["first", "second"]


def test_rerank_empty():
    assert make_reranker().rerank("q", [], k=5) == []
