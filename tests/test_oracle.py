"""Unit tests for the oracle context builder in src/retrieval/oracle.py."""

import pytest

from src.evaluation.gold_spans import GoldSpan
from src.evaluation.retrieval_metrics import ChunkSpan
from src.retrieval.oracle import build_oracle_context


def c(i: int, start: int, doc: str = "D") -> ChunkSpan:
    """A 100-char chunk; its id encodes its retrieval rank."""
    return ChunkSpan(f"r{i}", doc, start, start + 100)


def ids(chunks):
    return [x.chunk_id for x in chunks]


def oracle(ranked, golds, k=5):
    return build_oracle_context(ranked, golds, k, to_span=lambda x: x)


# gold spans: A covers 1000-1300, B covers 5000-5100
A = GoldSpan("D", 1000, 1300)
B = GoldSpan("D", 5000, 5100)


def test_covers_every_gold_span_first():
    # Evidence for B ranks last; evidence for A fills many ranks. With k=2 both spans
    # must still be covered, not two chunks of A.
    ranked = [c(0, 1000), c(1, 1100), c(2, 1200), c(3, 9000), c(4, 5000)]
    assert ids(oracle(ranked, [A, B], k=2)) == ["r0", "r4"]


def test_fills_with_more_evidence_then_distractors_in_rank_order():
    ranked = [c(0, 9000), c(1, 1000), c(2, 7000), c(3, 1100), c(4, 8000), c(5, 6000)]
    # relevant to A: r1, r3. Then best distractors r0, r2, r4. Output keeps rank order.
    assert ids(oracle(ranked, [A], k=5)) == ["r0", "r1", "r2", "r3", "r4"]


def test_evidence_position_follows_retrieval_rank_not_fixed_at_top():
    ranked = [c(0, 9000), c(1, 8000), c(2, 1000)]
    out = oracle(ranked, [A], k=3)
    assert ids(out) == ["r0", "r1", "r2"]  # the evidence stays at rank 3


def test_other_filing_chunks_are_never_relevant():
    ranked = [c(0, 1000, doc="OTHER"), c(1, 1000)]
    assert ids(oracle(ranked, [A], k=1)) == ["r1"]


def test_fewer_candidates_than_k():
    assert ids(oracle([c(0, 1000)], [A], k=5)) == ["r0"]


def test_requires_gold():
    with pytest.raises(ValueError):
        oracle([c(0, 1000)], [])
