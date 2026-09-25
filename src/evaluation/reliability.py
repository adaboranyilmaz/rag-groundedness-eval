"""Phase 6 reliability statistics over the Phase 5 per-trace evaluations. Pure functions;
scripts/07_reliability_analysis.py does the loading and writing. The analysis they serve is
fixed in advance in results/metrics/phase6_preregistration.md.

Every interval resamples questions: a question appears in up to eight traces (four prompts x
two generators), and those are not independent. Agreement between two signals is reported
against a random baseline: the same statistic with one signal permuted within prompt strata,
which keeps each signal's own distribution and each prompt's share of the traces while
breaking any pairing between the two.

Spearman's rho and the Clopper-Pearson interval are written out here (numpy and the
standard library only) rather than imported from scipy, which is not a declared dependency.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Callable, Hashable, Sequence

import numpy as np

from src.evaluation.agreement import cluster_bootstrap_ci

GRADED_LABELS = ("correct", "partially_correct", "incorrect", "unit_error")
QUADRANTS = (
    "correct_grounded",
    "correct_ungrounded",
    "incorrect_grounded",
    "incorrect_ungrounded",
)

PairStat = Callable[[np.ndarray, np.ndarray], float | None]

# --------------------------------------------------------------------------------------
# Definitions


def is_correct(label: str | None, lenient: bool = False) -> bool | None:
    """Strict: `correct` only. Lenient (sensitivity): `partially_correct` counts too. A unit
    error counts as incorrect. None for labels that are not a grade of an answer (declines,
    no answer, unparsed, judge failure)."""
    if label not in GRADED_LABELS:
        return None
    return label == "correct" or (lenient and label == "partially_correct")


def quadrant(correct: bool, grounded: bool) -> str:
    return f"{'correct' if correct else 'incorrect'}_{'grounded' if grounded else 'ungrounded'}"


def dominant_share(values: Sequence[Hashable]) -> float | None:
    """Share of the most common value; the no-variance rule reports a signal whose most
    common value covers >= 90% of a cell instead of an agreement number for it."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][1] / len(values)


# --------------------------------------------------------------------------------------
# Pair statistics: f(x, y) -> value, or None where undefined


def average_ranks(x: Sequence[float]) -> np.ndarray:
    """1-based ranks; tied values share the mean of the ranks they span."""
    x = np.asarray(x, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    new = np.r_[True, sx[1:] != sx[:-1]] if n else np.array([], dtype=bool)
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], n]
    mean_rank = (starts + ends - 1) / 2 + 1
    ranks = np.empty(n)
    ranks[order] = mean_rank[np.cumsum(new) - 1]
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size < 3:
        return None
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else None


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson correlation of the average ranks. None below three items or when either
    variable is constant."""
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")
    if len(x) < 3:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def jaccard(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """|flagged by both| / |flagged by either|; None when neither flags anything."""
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    if x.shape != y.shape:
        raise ValueError("a and b must be the same length")
    either = int((x | y).sum())
    return int((x & y).sum()) / either if either else None


def kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Cohen's kappa for two binary flags; the same value as agreement.cohen_kappa, written
    with array operations because it runs inside bootstrap and permutation loops."""
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    if x.shape != y.shape:
        raise ValueError("a and b must be the same length")
    if x.size == 0:
        return None
    po = float((x == y).mean())
    pa, pb = float(x.mean()), float(y.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def mean_or_none(x: Sequence[float]) -> float | None:
    return float(np.mean(x)) if len(x) else None


# --------------------------------------------------------------------------------------
# Intervals and baselines


def question_ci(
    questions: Sequence[Hashable],
    stat: Callable[[np.ndarray], float | None],
    n_resamples: int,
    seed: int,
) -> list[float] | None:
    """95% percentile interval of `stat(indices)`, resampling questions: every item of a
    drawn question enters, as many times as the question is drawn."""
    order = list(dict.fromkeys(questions))
    index: dict[Hashable, list[int]] = {q: [] for q in order}
    for i, q in enumerate(questions):
        index[q].append(i)
    return cluster_bootstrap_ci(
        order, index, lambda idx: stat(np.asarray(idx, dtype=int)), n_resamples, seed
    )


def mean_with_ci(
    values: Sequence[float], questions: Sequence[Hashable], n_resamples: int, seed: int
) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "n": int(v.size),
        "n_questions": len(set(questions)),
        "mean": float(v.mean()) if v.size else None,
        "ci95": question_ci(questions, lambda idx: float(v[idx].mean()), n_resamples, seed)
        if v.size
        else None,
    }


def permutation_baseline(
    x: Sequence, y: Sequence, strata: Sequence[Hashable], f: PairStat, n_perm: int, seed: int
) -> dict:
    """Distribution of f(x, y') where y' permutes y within each stratum."""
    xa = np.asarray(x)
    ya = np.asarray(y)
    labels = list(dict.fromkeys(strata))
    groups = [np.flatnonzero(np.asarray([s == g for s in strata])) for g in labels]
    rng = np.random.default_rng(seed)
    values = []
    yp = ya.copy()
    for _ in range(n_perm):
        for g in groups:
            yp[g] = ya[rng.permutation(g)]
        v = f(xa, yp)
        if v is not None:
            values.append(v)
    if not values:
        return {"n_perm": n_perm, "n_defined": 0, "mean": None, "p2_5": None, "p97_5": None}
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {
        "n_perm": n_perm,
        "n_defined": len(values),
        "mean": float(np.mean(values)),
        "p2_5": float(lo),
        "p97_5": float(hi),
    }


def pair_agreement(
    x: Sequence,
    y: Sequence,
    questions: Sequence[Hashable],
    strata: Sequence[Hashable],
    f: PairStat,
    n_resamples: int,
    seed: int,
) -> dict:
    """Observed statistic, its question-bootstrap interval, the permutation baseline, and
    the pre-registered call: above chance when the whole interval lies above the baseline
    mean."""
    xa = np.asarray(x)
    ya = np.asarray(y)
    value = f(xa, ya)
    ci = question_ci(questions, lambda idx: f(xa[idx], ya[idx]), n_resamples, seed)
    base = permutation_baseline(xa, ya, strata, f, n_resamples, seed)
    above = None
    if ci is not None and base["mean"] is not None:
        above = ci[0] > base["mean"]
    return {"n": int(xa.size), "value": value, "ci95": ci, "baseline": base, "above_chance": above}


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _bisect(fn: Callable[[float], float], target: float, increasing: bool) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (fn(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float] | None:
    """Exact binomial interval for k successes in n independent trials."""
    if n == 0:
        return None
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    # lower: P(X >= k | p) = alpha/2, increasing in p; upper: P(X <= k | p) = alpha/2, decreasing
    lo = 0.0 if k == 0 else _bisect(lambda p: 1 - _binom_cdf(k - 1, n, p), alpha / 2, True)
    hi = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p), alpha / 2, False)
    return [lo, hi]


# --------------------------------------------------------------------------------------
# Q5 reading and sampling


def q5_reading(d_groundedness_ci: list[float] | None, d_precision_ci: list[float] | None) -> str:
    """The pre-registered reading of a paired prompt contrast, from the intervals of the
    change in groundedness and in citation precision."""
    if d_groundedness_ci is None:
        return "undetermined"
    if d_groundedness_ci[0] > 0:
        return "real_improvement"
    if d_groundedness_ci[1] < 0:
        return "worse"
    if d_precision_ci is not None and d_precision_ci[1] < 0:
        return "apparent_only"
    return "no_effect"


def seeded_sample(items: Sequence[dict], n: int, seed: int, key: str = "trace_id") -> list[dict]:
    """Up to n items drawn at random, independent of the input order."""
    pool = sorted(items, key=lambda r: r[key])
    return random.Random(seed).sample(pool, min(n, len(pool)))


# --------------------------------------------------------------------------------------
# The predictions guard

_PREDICTION_RE = re.compile(r"^- (P\d+\.\d+)\b(.*)$", re.MULTILINE)
_CHECKED_RE = re.compile(r"\[[xX]\]")


def prediction_problems(text: str) -> list[str]:
    """Prediction lines in the pre-registration that do not have exactly one option marked.
    Empty when every prediction is answered, and a problem is also reported when the file
    has no prediction lines at all."""
    found = _PREDICTION_RE.findall(text)
    if not found:
        return ["no prediction lines found"]
    problems = []
    for pid, rest in found:
        n = len(_CHECKED_RE.findall(rest))
        if n != 1:
            problems.append(f"{pid}: {n} options marked")
    return problems
