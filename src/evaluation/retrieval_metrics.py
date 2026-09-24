"""Retrieval metrics against gold evidence spans, plus the grid-level statistics used to
pick and defend a winning configuration.

Relevance rule (DECISIONS.md, Phase 3): a retrieved chunk is relevant to a gold span iff
it is in the same document and their character overlap is at least `min_overlap_frac` of
the chunk's length OR at least `min_overlap_frac` of the gold span's length. The first
clause credits small chunks lying inside a page-sized gold span; the second credits a
large chunk that contains a small gold span.

Per-question metrics, with ranked chunks truncated at k:
  recall@k       fraction of the question's gold spans hit by >=1 relevant top-k chunk
  precision@k    relevant chunks in top-k / k
  mrr            1 / rank of the first relevant chunk in the full ranked list (0 if none)
  ndcg@k         binary gains; the ideal DCG uses the number of chunks *in the corpus*
                 relevant to the question, not just those retrieved
  span_recall@k  fraction of gold characters covered by the union of top-k chunks — the
                 partial-hit score, which credits chunks below the relevance threshold
  doc_hit@k      whether any top-k chunk comes from a gold document (whole-corpus search
                 can retrieve the right fact from the wrong company or year)
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from src.evaluation.gold_spans import GoldSpan


@dataclass(frozen=True)
class ChunkSpan:
    chunk_id: str
    doc_id: str
    char_start: int
    char_end: int


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def is_relevant(chunk: ChunkSpan, gold: GoldSpan, min_overlap_frac: float = 0.5) -> bool:
    if chunk.doc_id != gold.doc_id:
        return False
    ov = overlap(chunk.char_start, chunk.char_end, gold.char_start, gold.char_end)
    if ov == 0:
        return False
    chunk_len = chunk.char_end - chunk.char_start
    gold_len = gold.char_end - gold.char_start
    return ov >= min_overlap_frac * chunk_len or ov >= min_overlap_frac * gold_len


def count_relevant(
    chunks_by_doc: dict[str, list[ChunkSpan]],
    golds: Sequence[GoldSpan],
    min_overlap_frac: float = 0.5,
) -> int:
    """Chunks relevant to any of `golds` among all chunks of the gold documents: the nDCG
    ideal (`n_relevant_in_corpus` in `question_metrics`)."""
    return sum(
        1
        for d in {g.doc_id for g in golds}
        for c in chunks_by_doc.get(d, [])
        if any(is_relevant(c, g, min_overlap_frac) for g in golds)
    )


def _union_length(intervals: list[tuple[int, int]]) -> int:
    total = 0
    cur_start = cur_end = None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def span_recall(ranked: Sequence[ChunkSpan], golds: Sequence[GoldSpan]) -> float:
    """Fraction of gold characters (union over gold spans) covered by the union of the
    given chunks."""
    gold_by_doc: dict[str, list[tuple[int, int]]] = {}
    for g in golds:
        gold_by_doc.setdefault(g.doc_id, []).append((g.char_start, g.char_end))
    gold_total = sum(_union_length(iv) for iv in gold_by_doc.values())
    if gold_total == 0:
        return 0.0
    covered = 0
    for doc_id, gold_ivs in gold_by_doc.items():
        # Clip every chunk to every gold interval, then take the union, so overlapping
        # chunks (fixed_size uses overlap) are not double-counted.
        clipped = []
        for c in ranked:
            if c.doc_id != doc_id:
                continue
            for gs, ge in gold_ivs:
                s, e = max(c.char_start, gs), min(c.char_end, ge)
                if s < e:
                    clipped.append((s, e))
        covered += _union_length(clipped)
    return covered / gold_total


def question_metrics(
    ranked: Sequence[ChunkSpan],
    golds: Sequence[GoldSpan],
    n_relevant_in_corpus: int,
    k_values: Sequence[int],
    min_overlap_frac: float = 0.5,
) -> dict[str, float]:
    """All metrics for one question. `ranked` is the full ranked list (at least
    max(k_values) long where available); `n_relevant_in_corpus` is how many chunks of the
    same chunking strategy are relevant to any of `golds`, for the nDCG ideal."""
    if not golds:
        raise ValueError("question_metrics needs at least one gold span")
    rel_matrix = [[is_relevant(c, g, min_overlap_frac) for g in golds] for c in ranked]
    rel = [any(row) for row in rel_matrix]
    gold_docs = {g.doc_id for g in golds}

    out: dict[str, float] = {}
    first = next((i for i, r in enumerate(rel) if r), None)
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)

    for k in k_values:
        top = rel_matrix[:k]
        hit_golds = sum(1 for j in range(len(golds)) if any(row[j] for row in top))
        out[f"recall@{k}"] = hit_golds / len(golds)
        out[f"precision@{k}"] = sum(rel[:k]) / k
        dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rel[:k]) if r)
        ideal_n = min(k, n_relevant_in_corpus)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
        out[f"span_recall@{k}"] = span_recall(ranked[:k], golds)
        out[f"doc_hit@{k}"] = float(any(c.doc_id in gold_docs for c in ranked[:k]))
    return out


def mean_metrics(per_question: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = per_question[0].keys()
    return {key: sum(q[key] for q in per_question) / len(per_question) for key in keys}


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap over questions for mean(a) - mean(b). Pairing matters: both
    configurations answer the same questions, so per-question difficulty cancels.
    `p_b_ge_a` is the fraction of resamples in which b does at least as well as a."""
    if len(a) != len(b) or not a:
        raise ValueError("a and b must be equal-length and non-empty")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    rng = random.Random(seed)
    boot = []
    for _ in range(n_resamples):
        boot.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    lo = boot[int(0.025 * n_resamples)]
    hi = boot[min(n_resamples - 1, int(0.975 * n_resamples))]
    return {
        "diff": sum(diffs) / n,
        "ci95_low": lo,
        "ci95_high": hi,
        "p_b_ge_a": sum(1 for d in boot if d <= 0) / n_resamples,
    }


def factor_decomposition(
    cells: Sequence[dict[str, str]], values: Sequence[float], factors: Sequence[str]
) -> dict:
    """Main-effect decomposition of a balanced full-factorial grid.

    For each factor: the mean of `values` at each level, the range between the best and
    worst level, and the main-effect sum of squares as a share of the total sum of
    squares (in a balanced design the main effects are orthogonal, so the shares are
    comparable across factors; what is left over is interactions)."""
    grand = sum(values) / len(values)
    ss_total = sum((v - grand) ** 2 for v in values)
    out: dict = {"grand_mean": grand, "ss_total": ss_total, "factors": {}}
    explained = 0.0
    for factor in factors:
        by_level: dict[str, list[float]] = {}
        for cell, v in zip(cells, values, strict=True):
            by_level.setdefault(cell[factor], []).append(v)
        sizes = {len(vs) for vs in by_level.values()}
        if len(sizes) != 1:
            raise ValueError(f"grid is not balanced over factor {factor!r}")
        level_means = {lvl: sum(vs) / len(vs) for lvl, vs in by_level.items()}
        ss = sum(len(by_level[lvl]) * (m - grand) ** 2 for lvl, m in level_means.items())
        explained += ss
        out["factors"][factor] = {
            "level_means": level_means,
            "range": max(level_means.values()) - min(level_means.values()),
            "ss_share": ss / ss_total if ss_total > 0 else 0.0,
        }
    out["interaction_ss_share"] = 1.0 - explained / ss_total if ss_total > 0 else 0.0
    return out
