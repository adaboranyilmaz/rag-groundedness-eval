"""Phase 6 analyses, one function per pre-registered question, over a flat table of traces,
plus the check of each prediction against its result. The plan is Part A of
results/metrics/phase6_preregistration.md; the prediction rules are its Part B.

Each row is one Phase 4 trace with its Phase 5 evaluation (built by
scripts/07_reliability_analysis.py):
  trace_id, question_id, condition, model, prompt, answerability, status, answered,
  label            correctness label
  groundedness     score (None without document claims), fully_grounded
  citation_precision, n_cited
  numeric_in_cited, numeric_in_context, n_figures    judge-free numeric support
  confidence       stated, 0-100 (None when missing)
  recall5, span_recall5, ndcg5, mrr   None without aligned gold evidence
  top1_score       the best retrieval score in the context
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import combinations

import numpy as np

from src.evaluation import groundedness as groundedness_metric
from src.evaluation.reliability import (
    QUADRANTS,
    dominant_share,
    is_correct,
    jaccard,
    kappa,
    mean_with_ci,
    pair_agreement,
    q5_reading,
    quadrant,
    question_ci,
    seeded_sample,
    spearman,
)

ALIGNED = ("evidence_present", "evidence_absent")
PREDICTORS = ("recall5", "span_recall5", "ndcg5", "mrr", "top1_score")
SIGNALS = ("G", "CP", "NS", "C")
BASE_PROMPT = "v1_zero_shot"
CONTRASTS = (  # (treatment, primary?)
    ("v2_citation_required", True),
    ("v3_chain_of_thought", False),
    ("v4_abstention", False),
)
Q2_VARIANTS = {  # name: (lenient correctness, grounded by)
    "primary": (False, "fully_grounded"),
    "lenient_correct": (True, "fully_grounded"),
    "score_ge_0_5": (False, "score"),
}
EVIDENCE_GROUPS = {
    "evidence_present": "gold_evidence_retrieved",
    "evidence_absent": "gold_evidence_not_retrieved",
    "unanswerable": "filing_not_in_corpus",
    "no_gold": "no_aligned_gold",
}


def scored(r: dict) -> bool:
    return bool(r["answered"]) and r["groundedness"] is not None


def cell_key(condition: str, model: str) -> str:
    return f"{condition}__{model}"


def _cells(rows: list[dict]) -> list[tuple[str, str]]:
    return sorted({(r["condition"], r["model"]) for r in rows})


def _share(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


# --------------------------------------------------------------------------------------
# Q1. Does retrieval quality predict groundedness?


def _question_units(rows: list[dict]) -> list[dict]:
    """One unit per question: its retrieval metrics (the same for every prompt) and its
    groundedness averaged over the scored prompts."""
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["question_id"]].append(r)
    units = []
    for qid, rs in sorted(by_q.items()):
        for p in PREDICTORS:
            if len({r[p] for r in rs}) != 1:
                raise ValueError(f"{qid}: {p} differs between traces of one question")
        units.append(
            {
                "question_id": qid,
                **{p: rs[0][p] for p in PREDICTORS},
                "groundedness": float(np.mean([r["groundedness"] for r in rs])),
                "fully_grounded": float(np.mean([r["fully_grounded"] for r in rs])),
                "n_prompts": len(rs),
            }
        )
    return units


def _unit_corr(units: list[dict], pred: str, outcome: str, nb: int, seed: int) -> dict:
    x = np.asarray([u[pred] for u in units], dtype=float)
    y = np.asarray([u[outcome] for u in units], dtype=float)
    qs = [u["question_id"] for u in units]
    return {
        "n_questions": len(units),
        "rho": spearman(x, y) if len(units) >= 3 else None,
        "ci95": question_ci(qs, lambda idx: spearman(x[idx], y[idx]), nb, seed)
        if len(units) >= 3
        else None,
    }


def q1_retrieval_vs_groundedness(rows: list[dict], nb: int, seed: int) -> dict:
    ret = [r for r in rows if r["condition"] == "retrieved" and r["answerability"] in ALIGNED]
    out: dict = {"by_model": {}, "paired_oracle_vs_retrieved": {}}
    for model in sorted({r["model"] for r in ret}):
        mr = [r for r in ret if r["model"] == model]
        units = _question_units([r for r in mr if scored(r)])
        qs = [u["question_id"] for u in units]
        g = np.asarray([u["groundedness"] for u in units], dtype=float)
        rec = np.asarray([u["recall5"] for u in units], dtype=float)
        top = np.asarray([u["top1_score"] for u in units], dtype=float)
        hit = rec > 0

        def diff(idx: np.ndarray, g=g, hit=hit) -> float | None:
            h = hit[idx]
            if h.all() or not h.any():
                return None
            return float(g[idx][h].mean() - g[idx][~h].mean())

        def top_minus_recall(idx: np.ndarray, g=g, rec=rec, top=top) -> float | None:
            a, b = spearman(top[idx], g[idx]), spearman(rec[idx], g[idx])
            return None if a is None or b is None else a - b

        answer_rate = {}
        for name, flag in (("recall5_gt_0", True), ("recall5_eq_0", False)):
            grp = [r for r in mr if (r["recall5"] > 0) == flag]
            answer_rate[name] = mean_with_ci(
                [float(r["answered"]) for r in grp], [r["question_id"] for r in grp], nb, seed
            )
        out["by_model"][model] = {
            "n_scored_traces": sum(u["n_prompts"] for u in units),
            "spearman_recall5_groundedness": _unit_corr(units, "recall5", "groundedness", nb, seed),
            "groundedness_by_recall": {
                "recall5_gt_0": mean_with_ci(
                    g[hit], [q for q, h in zip(qs, hit, strict=True) if h], nb, seed
                ),
                "recall5_eq_0": mean_with_ci(
                    g[~hit], [q for q, h in zip(qs, hit, strict=True) if not h], nb, seed
                ),
                "difference": diff(np.arange(len(units))),
                "ci95": question_ci(qs, diff, nb, seed),
            },
            "answer_rate": answer_rate,
            "exploratory": {
                p: {
                    o: _unit_corr(units, p, o, nb, seed) for o in ("groundedness", "fully_grounded")
                }
                for p in PREDICTORS
            },
            "top1_minus_recall5_rho": {
                "value": top_minus_recall(np.arange(len(units))) if len(units) >= 3 else None,
                "ci95": question_ci(qs, top_minus_recall, nb, seed) if len(units) >= 3 else None,
            },
            "units": units,
        }

    oracle = {
        (r["question_id"], r["model"], r["prompt"]): r for r in rows if r["condition"] == "oracle"
    }
    for model in sorted({r["model"] for r in ret}):
        pairs = [
            (r, oracle[k])
            for r in ret
            if r["model"] == model
            and r["recall5"] == 0
            and (k := (r["question_id"], r["model"], r["prompt"])) in oracle
        ]
        both = [(a, b) for a, b in pairs if scored(a) and scored(b)]
        bq = [a["question_id"] for a, _ in both]
        out["paired_oracle_vs_retrieved"][model] = {
            "n_pairs": len(pairs),
            "n_pairs_both_scored": len(both),
            "d_groundedness": mean_with_ci(
                [b["groundedness"] - a["groundedness"] for a, b in both], bq, nb, seed
            ),
            "d_fully_grounded": mean_with_ci(
                [float(b["fully_grounded"]) - float(a["fully_grounded"]) for a, b in both],
                bq,
                nb,
                seed,
            ),
            "d_answered": mean_with_ci(
                [float(b["answered"]) - float(a["answered"]) for a, b in pairs],
                [a["question_id"] for a, _ in pairs],
                nb,
                seed,
            ),
        }
    return out


# --------------------------------------------------------------------------------------
# Q2. Where do correctness and groundedness diverge?


def _grounded(r: dict, how: str) -> bool:
    return bool(r["fully_grounded"]) if how == "fully_grounded" else r["groundedness"] >= 0.5


def primary_quadrant(r: dict) -> str:
    return quadrant(bool(is_correct(r["label"])), bool(r["fully_grounded"]))


def q2_quadrants(
    rows: list[dict], nb: int, seed: int, n_review: int, review_seed: int, example_seed: int
) -> dict:
    graded = [r for r in rows if scored(r) and is_correct(r["label"]) is not None]
    prompts = sorted({r["prompt"] for r in graded})
    out: dict = {"cells": {}, "examples": {}, "mechanism_retrieved": {}, "review_sample": {}}
    for cond, model in _cells(graded):
        cr = [r for r in graded if (r["condition"], r["model"]) == (cond, model)]
        entry: dict = {"n": len(cr), "n_unit_error": sum(r["label"] == "unit_error" for r in cr)}
        for name, (lenient, how) in Q2_VARIANTS.items():
            quads = [quadrant(bool(is_correct(r["label"], lenient)), _grounded(r, how)) for r in cr]
            counts = Counter(quads)
            correct = [r for r, q in zip(cr, quads, strict=True) if q.startswith("correct")]
            entry[name] = {
                "counts": {q: counts.get(q, 0) for q in QUADRANTS},
                "ungrounded_given_correct": mean_with_ci(
                    [float(not _grounded(r, how)) for r in correct],
                    [r["question_id"] for r in correct],
                    nb,
                    seed,
                ),
                "per_prompt": {
                    p: {
                        q: sum(
                            1
                            for r, qq in zip(cr, quads, strict=True)
                            if r["prompt"] == p and qq == q
                        )
                        for q in QUADRANTS
                    }
                    for p in prompts
                },
            }
        figs: dict = {}
        for quad in ("correct_ungrounded", "correct_grounded"):
            fr = [r for r in cr if primary_quadrant(r) == quad and r["n_figures"]]
            figs[quad] = {
                "n_with_figures": len(fr),
                "none_in_context": _share([r["numeric_in_context"] == 0 for r in fr]),
                "not_all_in_context": _share([r["numeric_in_context"] < 1 for r in fr]),
            }
        entry["figures_in_context"] = figs
        out["cells"][cell_key(cond, model)] = entry
        out["examples"][cell_key(cond, model)] = {
            q: [
                r["trace_id"]
                for r in seeded_sample([r for r in cr if primary_quadrant(r) == q], 1, example_seed)
            ]
            for q in QUADRANTS
        }

    for model in sorted({r["model"] for r in graded if r["condition"] == "retrieved"}):
        correct = [
            r
            for r in graded
            if r["condition"] == "retrieved" and r["model"] == model and is_correct(r["label"])
        ]
        out["mechanism_retrieved"][model] = {
            group: {
                "fully_grounded": sum(
                    1 for r in correct if r["answerability"] == ans and r["fully_grounded"]
                ),
                "not_fully_grounded": sum(
                    1 for r in correct if r["answerability"] == ans and not r["fully_grounded"]
                ),
            }
            for ans, group in EVIDENCE_GROUPS.items()
        }

    for cond in sorted({r["condition"] for r in graded}):
        pool = [
            r
            for r in graded
            if r["condition"] == cond and primary_quadrant(r) == "correct_ungrounded"
        ]
        out["review_sample"][cond] = {
            "n_in_quadrant": len(pool),
            "trace_ids": [r["trace_id"] for r in seeded_sample(pool, n_review, review_seed)],
        }
    return out


# --------------------------------------------------------------------------------------
# Q3. Do independent reliability signals agree?


def signal(r: dict, name: str, conf_threshold: float) -> tuple[float, bool] | None:
    """(continuous value, flag) for one signal on one trace; None where undefined. The flag
    is True when the signal says the answer is unreliable."""
    if name == "G":
        return float(r["groundedness"]), not r["fully_grounded"]
    if name == "CP":
        v = r["citation_precision"]
        return None if v is None else (float(v), v < 1)
    if name == "NS":
        v = r["numeric_in_cited"]
        return None if v is None else (float(v), v < 1)
    if name == "C":
        v = r["confidence"]
        return None if v is None else (float(v), v < conf_threshold)
    raise ValueError(name)


def pair_name(a: str, b: str) -> str:
    return f"{a}-{b}"


def q3_signal_agreement(
    rows: list[dict],
    nb: int,
    seed: int,
    conf_threshold: float,
    sensitivity_thresholds: list[float],
    no_variance_share: float,
) -> dict:
    out: dict = {}
    for cond, model in _cells([r for r in rows if scored(r)]):
        sc = [r for r in rows if scored(r) and (r["condition"], r["model"]) == (cond, model)]
        variance = {}
        for s in SIGNALS:
            vals = [v[0] for r in sc if (v := signal(r, s, conf_threshold)) is not None]
            share = dominant_share(vals)
            variance[s] = {
                "n": len(vals),
                "dominant_share": share,
                "no_variance": share is not None and share >= no_variance_share,
            }
        pairs: dict = {}
        for a, b in combinations(SIGNALS, 2):
            both = [
                (r, va, vb)
                for r in sc
                if (va := signal(r, a, conf_threshold)) is not None
                and (vb := signal(r, b, conf_threshold)) is not None
            ]
            key = pair_name(a, b)
            flat = [s for s in (a, b) if variance[s]["no_variance"]]
            if flat:
                pairs[key] = {"n": len(both), "status": "no_variance", "signals": flat}
                continue
            qs = [r["question_id"] for r, _, _ in both]
            strata = [r["prompt"] for r, _, _ in both]
            x = np.asarray([va[0] for _, va, _ in both])
            y = np.asarray([vb[0] for _, _, vb in both])
            fa = np.asarray([va[1] for _, va, _ in both])
            fb = np.asarray([vb[1] for _, _, vb in both])
            pairs[key] = {
                "n": len(both),
                "status": "ok",
                "flag_rate": {a: float(fa.mean()), b: float(fb.mean())},
                "spearman": pair_agreement(x, y, qs, strata, spearman, nb, seed),
                "jaccard": pair_agreement(fa, fb, qs, strata, jaccard, nb, seed),
                "kappa": pair_agreement(fa, fb, qs, strata, kappa, nb, seed),
            }
        sens: dict = {}
        if not variance["C"]["no_variance"]:
            for thr in sensitivity_thresholds:
                sens[str(thr)] = {}
                for other in ("G", "CP", "NS"):
                    if variance[other]["no_variance"]:
                        continue
                    both = [
                        (r, vo, vc)
                        for r in sc
                        if (vo := signal(r, other, conf_threshold)) is not None
                        and (vc := signal(r, "C", thr)) is not None
                    ]
                    qs = [r["question_id"] for r, _, _ in both]
                    strata = [r["prompt"] for r, _, _ in both]
                    fo = np.asarray([vo[1] for _, vo, _ in both])
                    fc = np.asarray([vc[1] for _, _, vc in both])
                    sens[str(thr)][pair_name(other, "C")] = {
                        "n": len(both),
                        "jaccard": pair_agreement(fo, fc, qs, strata, jaccard, nb, seed),
                        "kappa": pair_agreement(fo, fc, qs, strata, kappa, nb, seed),
                    }
        out[cell_key(cond, model)] = {
            "n_scored": len(sc),
            "variance": variance,
            "pairs": pairs,
            "confidence_threshold_sensitivity": sens,
        }
    return out


# --------------------------------------------------------------------------------------
# Q5. Citation-required prompting: real or apparent groundedness?


def q5_prompt_effect(rows: list[dict], nb: int, seed: int) -> dict:
    out: dict = {}
    for cond, model in _cells(rows):
        by_prompt: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in rows:
            if (r["condition"], r["model"]) == (cond, model):
                by_prompt[r["prompt"]][r["question_id"]] = r
        base = by_prompt.get(BASE_PROMPT, {})
        entry = {}
        for treat, primary in CONTRASTS:
            tr = by_prompt.get(treat, {})
            qids = sorted(set(tr) & set(base))
            both = [(base[q], tr[q]) for q in qids if scored(base[q]) and scored(tr[q])]
            bq = [b["question_id"] for b, _ in both]

            def paired(field: str, rows_=both) -> dict:
                ok = [(b, t) for b, t in rows_ if b[field] is not None and t[field] is not None]
                return mean_with_ci(
                    [float(t[field]) - float(b[field]) for b, t in ok],
                    [b["question_id"] for b, _ in ok],
                    nb,
                    seed,
                )

            d = {
                "groundedness": mean_with_ci(
                    [t["groundedness"] - b["groundedness"] for b, t in both], bq, nb, seed
                ),
                "fully_grounded": mean_with_ci(
                    [float(t["fully_grounded"]) - float(b["fully_grounded"]) for b, t in both],
                    bq,
                    nb,
                    seed,
                ),
                "citation_precision": paired("citation_precision"),
                "numeric_in_cited": paired("numeric_in_cited"),
                "n_cited": paired("n_cited"),
            }
            entry[f"{treat}_vs_{BASE_PROMPT}"] = {
                "primary": primary,
                "n_questions": len(qids),
                "n_both_scored": len(both),
                "answer_rate": {
                    BASE_PROMPT: _share([base[q]["answered"] for q in qids]),
                    treat: _share([tr[q]["answered"] for q in qids]),
                    "d_answered": mean_with_ci(
                        [float(tr[q]["answered"]) - float(base[q]["answered"]) for q in qids],
                        qids,
                        nb,
                        seed,
                    ),
                },
                "differences": d,
                "reading": q5_reading(d["groundedness"]["ci95"], d["citation_precision"]["ci95"]),
            }
        out[cell_key(cond, model)] = entry
    return out


# --------------------------------------------------------------------------------------
# Human-label check (Q1-Q3 primary statistics on the 50 validation answers)


def human_scores(sample_items: list[dict], human: dict) -> dict[str, dict]:
    """trace_id -> groundedness scored from the author's blind claim labels."""
    out = {}
    for it in sample_items:
        labels = human["items"][it["item_id"]]["claims"]
        if len(labels) != len(it["claims"]):
            raise ValueError(
                f"{it['item_id']}: {len(labels)} labels for {len(it['claims'])} claims"
            )
        out[it["trace_id"]] = groundedness_metric.score(
            it["claims"], [{"verdict": lab["verdict"]} for lab in labels]
        )
    return out


def human_check(rows_by_trace: dict[str, dict], human: dict[str, dict]) -> dict:
    """Each statistic on the same answers twice: the judge's groundedness and the human's.
    No intervals: with n this small it is a check of direction only."""
    items = []
    for tid, h in human.items():
        r = rows_by_trace[tid]
        if h["groundedness"] is None or not scored(r):
            continue
        items.append((r, h))

    def both(select: Callable[[dict], bool], stat: Callable[[list, str], float | None]) -> dict:
        chosen = [(r, h) for r, h in items if select(r)]
        return {
            "n": len(chosen),
            "judge": stat([r for r, _ in chosen], "judge") if chosen else None,
            "human": stat([(r, h) for r, h in chosen], "human") if chosen else None,
        }

    def g(x, who: str) -> tuple[float, bool]:
        if who == "judge":
            return x["groundedness"], bool(x["fully_grounded"])
        _, h = x
        return h["groundedness"], bool(h["fully_grounded"])

    def row(x, who: str) -> dict:
        return x if who == "judge" else x[0]

    def rho(field: str):
        def stat(xs: list, who: str) -> float | None:
            return spearman([row(x, who)[field] for x in xs], [g(x, who)[0] for x in xs])

        return stat

    def ungrounded_share(xs: list, who: str) -> float | None:
        return _share([not g(x, who)[1] for x in xs])

    return {
        "n_items_scored": len(items),
        "q1_spearman_recall5_groundedness": both(
            lambda r: r["condition"] == "retrieved" and r["recall5"] is not None, rho("recall5")
        ),
        "q2_ungrounded_given_correct": both(
            lambda r: is_correct(r["label"]) is True, ungrounded_share
        ),
        "q3_spearman_G_C": both(lambda r: r["confidence"] is not None, rho("confidence")),
        "q3_spearman_G_NS": both(
            lambda r: r["numeric_in_cited"] is not None, rho("numeric_in_cited")
        ),
    }


# --------------------------------------------------------------------------------------
# Predictions: parse Part B and check each against its result

_LINE_RE = re.compile(r"^- (P\d+\.\d+) (.*?):\s*(\[.*)$", re.MULTILINE)
_OPTION_RE = re.compile(r"\[( |x|X)\]\s*([^·]+)")


def parse_predictions(text: str) -> dict[str, dict]:
    out = {}
    for pid, desc, rest in _LINE_RE.findall(text):
        options = [(m.strip().lower() == "x", o.strip()) for m, o in _OPTION_RE.findall(rest)]
        chosen = [o for m, o in options if m]
        out[pid] = {
            "description": desc.strip(),
            "options": [o for _, o in options],
            "predicted": chosen[0] if len(chosen) == 1 else None,
        }
    return out


def parse_range(option: str) -> tuple[float, float]:
    """ ">= 0.5", "0.2 to 0.5", "<= -0.2", "< 10%", "10-25%", "> 75%" -> (lo, hi); percent
    bounds become fractions."""
    s = option.replace("−", "-").strip()
    scale = 0.01 if s.endswith("%") else 1.0
    s = s.rstrip("%").strip()
    if m := re.fullmatch(r"(>=|>)\s*(-?[\d.]+)", s):
        return float(m.group(2)) * scale, math.inf
    if m := re.fullmatch(r"(<=|<)\s*(-?[\d.]+)", s):
        return -math.inf, float(m.group(2)) * scale
    if m := re.fullmatch(r"(-?[\d.]+)\s*(?:to|-)\s*(-?[\d.]+)", s):
        return float(m.group(1)) * scale, float(m.group(2)) * scale
    raise ValueError(f"not a range: {option!r}")


def _range_check(predicted: str, options: list[str], value, ci) -> dict:
    lo, hi = parse_range(predicted)
    observed = (
        next((o for o in options if (b := parse_range(o))[0] <= value <= b[1]), None)
        if value is not None
        else None
    )
    if ci is None:
        return {"observed": observed, "unexpected": None, "note": "no interval"}
    return {"observed": observed, "unexpected": ci[1] < lo or ci[0] > hi}


def _direction_check(predicted: str, ci) -> dict:
    if ci is None:
        return {"observed": None, "unexpected": None, "note": "no interval"}
    observed = "up" if ci[0] > 0 else "down" if ci[1] < 0 else "no change"
    p = predicted.lower()
    return {"observed": observed, "unexpected": observed != p}


def _named_check(predicted: str, observed: str | None) -> dict:
    if observed is None:
        return {"observed": None, "unexpected": None, "note": "not computable"}
    return {"observed": observed, "unexpected": observed != predicted.lower(), "kind": "named"}


def _get(results: dict, *path):
    cur = results
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


SONNET, QWEN = "claude-sonnet-5", "qwen2.5-3b"


def _q3_call(results: dict, pair: str) -> str | None:
    p = _get(results, "q3", cell_key("oracle", SONNET), "pairs", pair)
    if p is None or p.get("status") != "ok":
        return None
    return "above chance" if p["spearman"]["above_chance"] else "not above chance"


def _q3_highest_jaccard(results: dict) -> str | None:
    pairs = _get(results, "q3", cell_key("oracle", SONNET), "pairs") or {}
    vals = {
        k: v["jaccard"]["value"]
        for k, v in pairs.items()
        if v.get("status") == "ok" and v["jaccard"]["value"] is not None
    }
    return max(vals, key=vals.get) if vals else None


def _norm_pair(name: str) -> str:
    return "-".join(sorted(name.split("-")))


def check_predictions(text: str, results: dict) -> list[dict]:
    """Each prediction beside its result and the Part B verdict. `unexpected` is None where
    the result does not exist yet (the adversarial questions before they are run, the human
    review before it is done) or cannot be computed."""
    preds = parse_predictions(text)
    s_ret = _get(results, "q1", "by_model", SONNET)
    q_ret = _get(results, "q1", "by_model", QWEN)

    def rng(pid, stat):
        v = stat or {}
        return _range_check(
            preds[pid]["predicted"], preds[pid]["options"], v.get("value"), v.get("ci95")
        )

    def rho_stat(block):
        c = _get(block or {}, "spearman_recall5_groundedness")
        return None if c is None else {"value": c["rho"], "ci95": c["ci95"]}

    def ugc(cell):
        c = _get(results, "q2", "cells", cell, "primary", "ungrounded_given_correct")
        return None if c is None else {"value": c["mean"], "ci95": c["ci95"]}

    checks: dict[str, Callable[[], dict]] = {
        "P1.1": lambda: rng("P1.1", rho_stat(s_ret)),
        "P1.2": lambda: rng("P1.2", rho_stat(q_ret)),
        "P1.3": lambda: _named_check(
            preds["P1.3"]["predicted"],
            None
            if _get(s_ret or {}, "top1_minus_recall5_rho", "value") is None
            else ("yes" if s_ret["top1_minus_recall5_rho"]["value"] >= 0 else "no"),
        ),
        "P1.4": lambda: _direction_check(
            preds["P1.4"]["predicted"],
            _get(results, "q1", "paired_oracle_vs_retrieved", SONNET, "d_groundedness", "ci95"),
        ),
        "P2.1": lambda: rng("P2.1", ugc(cell_key("oracle", SONNET))),
        "P2.2": lambda: rng("P2.2", ugc(cell_key("retrieved", SONNET))),
        "P2.3": lambda: rng("P2.3", ugc(cell_key("oracle", QWEN))),
        "P2.4": lambda: _named_check(
            preds["P2.4"]["predicted"].replace("+", "_"),
            (
                max(c, key=c.get)
                if (
                    c := _get(
                        results, "q2", "cells", cell_key("retrieved", QWEN), "primary", "counts"
                    )
                )
                else None
            ),
        ),
        "P2.5": lambda: rng("P2.5", _get(results, "q2_review", "judge_right")),
        "P3.1": lambda: _named_check(preds["P3.1"]["predicted"], _q3_call(results, "G-C")),
        "P3.2": lambda: _named_check(preds["P3.2"]["predicted"], _q3_call(results, "G-NS")),
        "P3.3": lambda: _named_check(preds["P3.3"]["predicted"], _q3_call(results, "G-CP")),
        "P3.4": lambda: _named_check(
            _norm_pair(preds["P3.4"]["predicted"]),
            _norm_pair(h).lower() if (h := _q3_highest_jaccard(results)) else None,
        ),
        "P4.1": lambda: rng("P4.1", _get(results, "q4", "premise_rejected", SONNET)),
        "P4.2": lambda: rng("P4.2", _get(results, "q4", "premise_rejected", QWEN)),
        "P4.3": lambda: _named_check(
            preds["P4.3"]["predicted"], _get(results, "q4", "v4_raises_rejection", SONNET)
        ),
        "P4.4": lambda: rng(
            "P4.4", _get(results, "q4", "out_of_corpus_answered_from_memory", SONNET)
        ),
        "P4.5": lambda: _named_check(
            preds["P4.5"]["predicted"],
            {"down": "(b) clearly lower", "no change": "about the same", "up": "(b) higher"}.get(
                _direction_check(
                    "up",
                    _get(results, "q4", "two_filing_minus_one_filing_accuracy", SONNET, "ci95"),
                )["observed"]
            ),
        ),
        "P5.1": lambda: _named_check(
            preds["P5.1"]["predicted"],
            _reading_label(results, cell_key("oracle", SONNET)),
        ),
        "P5.2": lambda: _named_check(
            preds["P5.2"]["predicted"],
            _reading_label(results, cell_key("oracle", QWEN)),
        ),
        "P5.3": lambda: _direction_check(
            preds["P5.3"]["predicted"],
            _get(
                results,
                "q5",
                cell_key("oracle", QWEN),
                f"v2_citation_required_vs_{BASE_PROMPT}",
                "differences",
                "citation_precision",
                "ci95",
            ),
        ),
    }
    out = []
    for pid, p in preds.items():
        if p["predicted"] is None or pid not in checks:
            out.append({"id": pid, **p, "observed": None, "unexpected": None, "note": "no check"})
            continue
        out.append({"id": pid, **p, **checks[pid]()})
    return out


def _reading_label(results: dict, cell: str) -> str | None:
    r = _get(results, "q5", cell, f"v2_citation_required_vs_{BASE_PROMPT}", "reading")
    return None if r in (None, "undetermined") else r.replace("_", " ")


# --------------------------------------------------------------------------------------
# Everything but Q4 (the adversarial set has its own inputs)


def analyse(rows: list[dict], human: dict[str, dict], cfg: dict) -> dict:
    nb, seed = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["seed"]
    q2, q3 = cfg["q2"], cfg["q3"]
    return {
        "q1": q1_retrieval_vs_groundedness(rows, nb, seed),
        "q2": q2_quadrants(
            rows, nb, seed, q2["review_per_condition"], q2["review_seed"], q2["examples_seed"]
        ),
        "q3": q3_signal_agreement(
            rows,
            nb,
            seed,
            q3["confidence_flag_below"],
            q3["sensitivity_thresholds"],
            q3["no_variance_share"],
        ),
        "q5": q5_prompt_effect(rows, nb, seed),
        "human_check": human_check({r["trace_id"]: r for r in rows}, human),
    }
