"""Agreement between raters (human vs judge, judge vs judge, judge vs itself) and bootstrap
confidence intervals.

Cohen's kappa is computed directly, (p_o - p_e) / (1 - p_e) with p_e from the two raters'
marginals, and is undefined (None) when p_e = 1 (both raters used one and the same label
throughout). Kappa depends on prevalence: with one dominant class a high raw agreement can
coexist with a low kappa, so raw agreement, each rater's label distribution and the full
confusion matrix are always reported next to it.

Claims are nested in answers, so claim-level intervals use a cluster bootstrap that resamples
answers (with all their claims), not individual claims.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Sequence

import numpy as np


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def raw_agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else None


def confusion(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> dict:
    """{label_a: {label_b: count}} with rows = rater a."""
    counts = Counter(zip(a, b, strict=True))
    return {x: {y: counts.get((x, y), 0) for y in labels} for x in labels}


def agreement_summary(
    a: Sequence[str],
    b: Sequence[str],
    labels: Sequence[str],
    clusters: Sequence[Hashable] | None = None,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Kappa (with a 95% bootstrap CI), raw agreement, marginals and confusion matrix.
    `clusters` (e.g. the answer each claim belongs to) makes the bootstrap resample whole
    clusters; without it, items are resampled individually."""
    a, b = list(a), list(b)
    groups = clusters if clusters is not None else list(range(len(a)))
    order: list[Hashable] = list(dict.fromkeys(groups))
    index = {g: [i for i, x in enumerate(groups) if x == g] for g in order}

    def stat(idx: list[int]) -> float | None:
        return cohen_kappa([a[i] for i in idx], [b[i] for i in idx])

    ci = cluster_bootstrap_ci(order, index, stat, n_resamples, seed)
    return {
        "n": len(a),
        "n_clusters": len(order),
        "kappa": cohen_kappa(a, b),
        "kappa_ci95": ci,
        "raw_agreement": raw_agreement(a, b),
        "marginals_a": {lab: Counter(a).get(lab, 0) for lab in labels},
        "marginals_b": {lab: Counter(b).get(lab, 0) for lab in labels},
        "confusion_rows_a": confusion(a, b, labels),
    }


def cluster_bootstrap_ci(
    clusters: Sequence[Hashable],
    index: dict[Hashable, list[int]],
    stat: Callable[[list[int]], float | None],
    n_resamples: int,
    seed: int,
) -> list[float] | None:
    """Percentile 95% CI of `stat` over resampled clusters. Resamples where the statistic
    is undefined are dropped and their share reported by the caller if it matters."""
    rng = np.random.default_rng(seed)
    k = len(clusters)
    if k == 0:
        return None
    values = []
    for draw in rng.integers(0, k, size=(n_resamples, k)):
        idx = [i for j in draw for i in index[clusters[j]]]
        v = stat(idx)
        if v is not None:
            values.append(v)
    if not values:
        return None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_mean_ci(
    values: Sequence[float], n_resamples: int = 10_000, seed: int = 0
) -> list[float] | None:
    """Percentile 95% CI of a mean over items (questions), vectorised."""
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if x.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_resamples, x.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]
