"""RAGAS faithfulness against the project's groundedness judge and the author's labels, on
the hand-labelled validation answers (scripts/15_ragas_faithfulness.py).

The comparison is per answer. RAGAS splits an answer into its own statements, so its
verdicts cannot be paired with the author's, which were given on the project judge's claims.
Each rater's per-answer measure:
  score           human, judge, second_judge: supported document claims / document claims,
                  as in src/evaluation/groundedness.py; ragas: faithful statements /
                  statements, over every statement (RAGAS has no claim kinds)
  fully_grounded  score == 1
An answer RAGAS could not score (no statements, or a failed call) is left out of every
comparison that involves RAGAS, and counted. The human labels were made on the judge's own
claim split, which favours the judge in any comparison with RAGAS; the report says so.

No ragas import: this runs, and is tested, in the default environment.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.evaluation.agreement import agreement_summary

CLAIM_RATERS = ("human", "judge", "second_judge")
PAIRS = (
    ("human", "ragas"),
    ("judge", "ragas"),
    ("second_judge", "ragas"),
    ("human", "judge"),
)


def claim_rater_scores(claim_rows: list[dict], rater: str) -> dict[str, dict]:
    """{trace_id: {"score", "fully_grounded", "n_document_claims"}} from the per-claim
    verdicts in judge_agreement.json; None where the rater has no verdict for an answer."""
    by_answer: dict[str, list[str | None]] = {}
    for r in claim_rows:
        if r["kind"] == "document":
            by_answer.setdefault(r["trace_id"], []).append(r[rater])
    out = {}
    for tid, verdicts in by_answer.items():
        if any(v is None for v in verdicts):
            out[tid] = {"score": None, "fully_grounded": None, "n_document_claims": len(verdicts)}
            continue
        s = sum(v == "supported" for v in verdicts) / len(verdicts)
        out[tid] = {"score": s, "fully_grounded": s == 1.0, "n_document_claims": len(verdicts)}
    return out


def average_ranks(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    for v in np.unique(a):  # ties share their mean rank
        tied = a == v
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rho (Pearson on average ranks); None if either side is constant."""
    if len(x) != len(y):
        raise ValueError("paired values only")
    if len(x) < 2:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _percentile_ci(values: list[float]) -> list[float] | None:
    if not values:
        return None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_ci(n: int, stat, n_resamples: int, seed: int) -> list[float] | None:
    """Percentile 95% CI of `stat(indices)` over answers resampled with replacement;
    resamples where the statistic is undefined are dropped."""
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for draw in rng.integers(0, n, size=(n_resamples, n)):
        v = stat(list(draw))
        if v is not None:
            vals.append(v)
    return _percentile_ci(vals)


def continuous_summary(
    a: list[float], b: list[float], n_resamples: int, seed: int
) -> dict[str, object]:
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    return {
        "n": len(a),
        "spearman": spearman(a, b),
        "spearman_ci95": bootstrap_ci(
            len(a), lambda i: spearman([a[j] for j in i], [b[j] for j in i]), n_resamples, seed
        ),
        "mean_a": float(np.mean(a)) if a else None,
        "mean_b": float(np.mean(b)) if b else None,
        "mean_abs_diff": float(np.mean(np.abs(diffs))) if diffs else None,
        "n_a_higher": sum(d > 0 for d in diffs),
        "n_b_higher": sum(d < 0 for d in diffs),
        "n_equal": sum(d == 0 for d in diffs),
    }


def compare_pair(
    per_answer: list[dict], a: str, b: str, n_resamples: int, seed: int
) -> dict[str, object]:
    rows = [r for r in per_answer if r[a]["score"] is not None and r[b]["score"] is not None]
    fa = [r[a]["fully_grounded"] for r in rows]
    fb = [r[b]["fully_grounded"] for r in rows]
    return {
        "fully_grounded": agreement_summary(fa, fb, [True, False], None, n_resamples, seed),
        "score": continuous_summary(
            [r[a]["score"] for r in rows], [r[b]["score"] for r in rows], n_resamples, seed
        ),
    }


def kappa_difference(
    per_answer: list[dict], ref: str, x: str, y: str, n_resamples: int, seed: int
) -> dict[str, object]:
    """kappa(ref, x) - kappa(ref, y) on fully grounded, paired over the answers all three
    rate, with a bootstrap CI over answers: does x agree with ref better than y does?"""
    from src.evaluation.agreement import cohen_kappa

    rows = [r for r in per_answer if all(r[k]["score"] is not None for k in (ref, x, y))]
    f = {k: [r[k]["fully_grounded"] for r in rows] for k in (ref, x, y)}

    def diff(idx: list[int]) -> float | None:
        kx = cohen_kappa([f[ref][i] for i in idx], [f[x][i] for i in idx])
        ky = cohen_kappa([f[ref][i] for i in idx], [f[y][i] for i in idx])
        return None if kx is None or ky is None else kx - ky

    return {
        "n": len(rows),
        "reference": ref,
        "a": x,
        "b": y,
        "kappa_diff": diff(list(range(len(rows)))),
        "kappa_diff_ci95": bootstrap_ci(len(rows), diff, n_resamples, seed),
    }


def build_per_answer(
    items: list[dict], claim_rows: list[dict], ragas: dict[str, dict]
) -> list[dict]:
    """One row per validation answer: each rater's score and fully-grounded flag."""
    claim = {rater: claim_rater_scores(claim_rows, rater) for rater in CLAIM_RATERS}
    out = []
    for it in items:
        tid = it["trace_id"]
        rg = ragas[tid]
        s = rg["score"]
        out.append(
            {
                "trace_id": tid,
                "item_id": it["item_id"],
                "model_key": it["model_key"],
                "condition": it["condition"],
                **{rater: claim[rater][tid] for rater in CLAIM_RATERS},
                "ragas": {
                    "score": s,
                    "fully_grounded": (s == 1.0) if s is not None else None,
                    "n_statements": rg.get("n_statements"),
                    "error": rg.get("error"),
                },
            }
        )
    return out


def compare_all(
    per_answer: list[dict], model_keys: Sequence[str], n_resamples: int, seed: int
) -> dict[str, object]:
    subsets = {"all": per_answer}
    subsets.update({m: [r for r in per_answer if r["model_key"] == m] for m in model_keys})
    return {
        "n_answers": len(per_answer),
        "n_ragas_unscored": sum(r["ragas"]["score"] is None for r in per_answer),
        "ragas_unscored": [
            {"trace_id": r["trace_id"], "error": r["ragas"]["error"]}
            for r in per_answer
            if r["ragas"]["score"] is None
        ],
        "pairs": {
            f"{a}__vs__{b}": {
                name: compare_pair(rows, a, b, n_resamples, seed) for name, rows in subsets.items()
            }
            for a, b in PAIRS
        },
        "human_judge_minus_human_ragas": kappa_difference(
            per_answer, "human", "judge", "ragas", n_resamples, seed
        ),
    }
