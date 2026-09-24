"""Unit tests for src/evaluation/retrieval_metrics.py, against hand-computed values."""

import math

import pytest

from src.evaluation.gold_spans import GoldSpan
from src.evaluation.retrieval_metrics import (
    ChunkSpan,
    factor_decomposition,
    is_relevant,
    paired_bootstrap_diff,
    question_metrics,
    span_recall,
)


class TestIsRelevant:
    gold = GoldSpan("A", 100, 200)

    def test_chunk_mostly_inside_gold(self):
        assert is_relevant(ChunkSpan("c", "A", 120, 220), self.gold)  # 80/100 of chunk

    def test_large_chunk_containing_half_the_gold(self):
        assert is_relevant(ChunkSpan("c", "A", 150, 1000), self.gold)  # 50/100 of gold

    def test_small_overlap_is_not_relevant(self):
        assert not is_relevant(ChunkSpan("c", "A", 190, 1000), self.gold)

    def test_wrong_document(self):
        assert not is_relevant(ChunkSpan("c", "B", 100, 200), self.gold)

    def test_adjacent_is_not_overlap(self):
        assert not is_relevant(ChunkSpan("c", "A", 200, 210), self.gold)


class TestQuestionMetrics:
    golds = [GoldSpan("A", 100, 200), GoldSpan("A", 1000, 1100)]
    ranked = [
        ChunkSpan("c1", "B", 100, 200),  # wrong doc
        ChunkSpan("c2", "A", 150, 450),  # 50 chars = 50% of gold 1 -> relevant
        ChunkSpan("c3", "A", 1000, 1040),  # fully inside gold 2 -> relevant
        ChunkSpan("c4", "A", 500, 600),  # irrelevant
    ]

    def metrics(self):
        return question_metrics(self.ranked, self.golds, 2, [1, 2, 3, 4])

    def test_recall(self):
        m = self.metrics()
        assert (m["recall@1"], m["recall@2"], m["recall@3"]) == (0.0, 0.5, 1.0)

    def test_precision_and_mrr(self):
        m = self.metrics()
        assert m["precision@3"] == pytest.approx(2 / 3)
        assert m["precision@4"] == pytest.approx(0.5)
        assert m["mrr"] == 0.5

    def test_ndcg_uses_corpus_ideal(self):
        m = self.metrics()
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        idcg = 1 + 1 / math.log2(3)
        assert m["ndcg@3"] == pytest.approx(dcg / idcg)
        assert m["ndcg@1"] == 0.0

    def test_span_recall_and_doc_hit(self):
        m = self.metrics()
        assert m["span_recall@3"] == pytest.approx((50 + 40) / 200)
        assert m["doc_hit@1"] == 0.0
        assert m["doc_hit@2"] == 1.0

    def test_no_relevant_retrieved(self):
        m = question_metrics([ChunkSpan("x", "B", 0, 10)], self.golds, 2, [1])
        assert m["mrr"] == 0.0 and m["recall@1"] == 0.0 and m["ndcg@1"] == 0.0

    def test_requires_gold(self):
        with pytest.raises(ValueError):
            question_metrics(self.ranked, [], 0, [1])


def test_span_recall_does_not_double_count_overlapping_chunks():
    golds = [GoldSpan("A", 0, 100)]
    chunks = [ChunkSpan("a", "A", 0, 60), ChunkSpan("b", "A", 40, 80)]
    assert span_recall(chunks, golds) == pytest.approx(0.8)


def test_span_recall_unions_overlapping_gold_spans():
    golds = [GoldSpan("A", 0, 100), GoldSpan("A", 50, 150)]
    assert span_recall([ChunkSpan("a", "A", 0, 150)], golds) == 1.0


class TestPairedBootstrap:
    def test_identical_inputs(self):
        r = paired_bootstrap_diff([1.0, 0.0, 1.0], [1.0, 0.0, 1.0], n_resamples=200)
        assert r["diff"] == 0.0 and r["ci95_low"] == 0.0 and r["ci95_high"] == 0.0
        assert r["p_b_ge_a"] == 1.0

    def test_dominant_a(self):
        r = paired_bootstrap_diff([1.0] * 20, [0.0] * 20, n_resamples=200)
        assert r["diff"] == 1.0 and r["p_b_ge_a"] == 0.0

    def test_deterministic_given_seed(self):
        a, b = [1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1]
        assert paired_bootstrap_diff(a, b, 500, seed=3) == paired_bootstrap_diff(a, b, 500, 3)


class TestFactorDecomposition:
    def test_additive_grid_has_no_interaction(self):
        a_eff = {"x": 0.0, "y": 0.4}
        b_eff = {"p": 0.0, "q": 0.1, "r": 0.2}
        cells = [{"a": a, "b": b} for a in a_eff for b in b_eff]
        values = [a_eff[c["a"]] + b_eff[c["b"]] for c in cells]
        d = factor_decomposition(cells, values, ["a", "b"])
        assert d["interaction_ss_share"] == pytest.approx(0.0, abs=1e-12)
        assert d["factors"]["a"]["range"] == pytest.approx(0.4)
        assert d["factors"]["b"]["range"] == pytest.approx(0.2)
        assert d["factors"]["a"]["ss_share"] > d["factors"]["b"]["ss_share"]
        total = d["factors"]["a"]["ss_share"] + d["factors"]["b"]["ss_share"]
        assert total == pytest.approx(1.0)

    def test_unbalanced_grid_raises(self):
        cells = [{"a": "x"}, {"a": "x"}, {"a": "y"}]
        with pytest.raises(ValueError):
            factor_decomposition(cells, [1.0, 2.0, 3.0], ["a"])
