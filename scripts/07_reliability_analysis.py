"""Phase 6 reliability analysis over the Phase 5 per-trace evaluations.

Runs the plan fixed in results/metrics/phase6_preregistration.md (Part A) and checks each of
the author's predictions (Part B) against its result. It refuses to run while any prediction
is unanswered, so the predictions always exist before the numbers do. No API calls.

  Q1  does retrieval quality predict groundedness?
  Q2  where do correctness and groundedness diverge? (the 2x2, with seeded examples)
  Q3  do independent reliability signals agree, against a random baseline?
  Q5  does citation-required prompting improve groundedness, or only its appearance?
  plus the human-label check of Q1-Q3 on the 50 validation answers, and the author's review
  of a seeded sample of the correct-but-ungrounded quadrant once it exists.
Q4 (the adversarial set) is added when that set has been generated and evaluated.

Writes:
  results/metrics/reliability_analysis.json    every number, and the prediction checks
  results/metrics/reliability_analysis.md      tables generated from the JSON
  results/metrics/quadrant_examples.md         one seeded example per quadrant per cell
  results/labels/quadrant_review_sample.json   the Q2 review sample (fixed once written)
  results/labels/quadrant_review.html          the review page (generated; not committed)
  results/plots/retrieval_vs_groundedness.png, correct_vs_grounded.png,
                signal_agreement.png, prompt_effect.png

Usage:
    uv run python scripts/07_reliability_analysis.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluate import trace_paths
from src.evaluation.labeling_page import render_review_page
from src.evaluation.plots import (
    plot_adversarial,
    plot_prompt_effect,
    plot_quadrants,
    plot_retrieval_vs_groundedness,
    plot_signal_agreement,
)
from src.evaluation.reliability import QUADRANTS, mean_with_ci, prediction_problems
from src.evaluation.reliability_questions import (
    BASE_PROMPT,
    CONTRASTS,
    PREDICTORS,
    Q2_VARIANTS,
    SIGNALS,
    analyse,
    check_predictions,
    human_scores,
    q4_adversarial,
)
from src.generation.trace import read_traces

CONFIG_PATH = Path("configs/reliability.yaml")
FINANCEBENCH_PATH = Path("data/raw/financebench/financebench_merged.jsonl")
RESULTS_DIR = Path("results/metrics")
LABELS_DIR = Path("results/labels")
PLOTS_DIR = Path("results/plots")
CONDITIONS = ["retrieved", "oracle"]  # figure order, as in the Phase 5 reliability diagram
REVIEW_PAGE_PATH = LABELS_DIR / "quadrant_review.html"
K = 5


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def review_id(trace_id: str) -> str:
    """Opaque id for the review page: the trace id names the model and prompt."""
    return hashlib.sha256(("review:" + trace_id).encode("utf-8")).hexdigest()[:12]


def excerpt_header(chunk: dict) -> str:
    """The header src/generation/prompts.py `format_context` gives each excerpt."""
    header = f"[{chunk['label']}] {chunk['doc_id']} | page {chunk['page']}"
    return header + (f" | {chunk['section']}" if chunk.get("section") else "")


# --------------------------------------------------------------------------------------
# Inputs


def flat_row(rec: dict, trace: dict) -> dict:
    """One trace as a row for src/evaluation/reliability_questions.py."""
    g = rec["groundedness"] or {}
    cit = rec["citations"] or {}
    ns = cit.get("numeric_support") or {}
    r = trace["retrieval"]
    if r["k"] != K:
        raise ValueError(f"{rec['trace_id']}: k={r['k']}, expected {K}")
    m = r["context_metrics"] or {}
    return {
        "trace_id": rec["trace_id"],
        "question_id": rec["financebench_id"],
        "condition": rec["condition"],
        "model": rec["model_key"],
        "prompt": rec["prompt_id"],
        "question_type": rec["question_type"],
        "answerability": rec["answerability"],
        "status": rec["abstention"]["status"],
        "answered": rec["abstention"]["status"] in ("answered", "partial"),
        "label": rec["correctness"]["label"],
        "groundedness": g.get("groundedness"),
        "fully_grounded": g.get("fully_grounded"),
        "citation_precision": cit.get("citation_precision"),
        "n_cited": len(cit.get("cited") or []),
        "numeric_in_cited": ns.get("in_cited_rate"),
        "numeric_in_context": ns.get("in_context_rate"),
        "n_figures": ns.get("n_figures"),
        "confidence": rec["confidence"],
        "recall5": m.get(f"recall@{K}"),
        "span_recall5": m.get(f"span_recall@{K}"),
        "ndcg5": m.get(f"ndcg@{K}"),
        "mrr": m.get("mrr"),
        "top1_score": max(c["score"] for c in r["chunks"]),
    }


def load_inputs(cfg: dict) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    inp = cfg["inputs"]
    eval_cfg = yaml.safe_load(Path(inp["eval_config"]).read_text(encoding="utf-8"))
    traces = {t["trace_id"]: t for p in trace_paths(eval_cfg) for t in read_traces(p)}
    lines = Path(inp["eval_per_trace"]).read_text(encoding="utf-8").splitlines()
    recs = {(r := json.loads(x))["trace_id"]: r for x in lines}
    if set(recs) != set(traces):
        raise RuntimeError(
            f"eval records and traces differ: {len(set(recs) ^ set(traces))} trace ids"
        )
    rows = [flat_row(rec, traces[tid]) for tid, rec in sorted(recs.items())]
    return rows, recs, traces


def adversarial_rows(cfg: dict) -> list[dict] | None:
    """Q4's rows from the adversarial evaluation, with the author's premise label where the
    labelling is complete (None otherwise). None when the adversarial set has not been run."""
    inp = cfg["inputs"]
    path = Path(inp["adversarial_eval"])
    if not path.exists():
        return None
    human: dict[str, str] = {}
    pilot_path = Path(inp["premise_pilot"])
    seen = (
        set(json.loads(pilot_path.read_text(encoding="utf-8"))["trace_ids"])
        if pilot_path.exists()
        else set()
    )
    sample_path, labels_path = Path(inp["premise_sample"]), Path(inp["premise_labels"])
    if sample_path.exists() and labels_path.exists():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if labels["sample_sha256"] != sample["sample_sha256"]:
            raise RuntimeError(f"{labels_path} belongs to a different premise sample")
        decided = {
            it["trace_id"]: labels["items"].get(it["label_id"], {}).get("decision")
            for it in sample["items"]
        }
        if all(decided.values()):
            human = decided
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        adv = rec["adversarial"]
        premise = rec.get("premise") or {}
        judge_label = premise.get("handling")
        if adv["category"] == "d" and premise.get("source") == "rule":
            human_label = judge_label  # no answer text: the rule decides for both
        else:
            human_label = human.get(rec["trace_id"])
        g = rec["groundedness"] or {}
        rows.append(
            {
                "trace_id": rec["trace_id"],
                "question_id": rec["financebench_id"],
                "condition": rec["condition"],
                "model": rec["model_key"],
                "prompt": rec["prompt_id"],
                "category": adv["category"],
                "subtype": adv["subtype"],
                "status": rec["abstention"]["status"],
                "answered": rec["abstention"]["status"] in ("answered", "partial"),
                "label": rec["correctness"]["label"],
                "groundedness": g.get("groundedness"),
                "premise_judge": judge_label if adv["category"] == "d" else None,
                "premise_human": (human_label if human else None)
                if adv["category"] == "d"
                else None,
                "label_saw_judge": rec["trace_id"] in seen,
                # "judge" or "rule" (a bare decline token or failed parse, decided for both
                # raters without judgement, so it cannot count as agreement)
                "premise_source": premise.get("source") if adv["category"] == "d" else None,
            }
        )
    return rows


def load_human(cfg: dict) -> dict[str, dict]:
    sample = json.loads(Path(cfg["inputs"]["validation_sample"]).read_text(encoding="utf-8"))
    human = json.loads(Path(cfg["inputs"]["human_labels"]).read_text(encoding="utf-8"))
    if human["sample_sha256"] != sample["sample_sha256"]:
        raise RuntimeError("human labels belong to a different validation sample")
    return human_scores(sample["items"], human)


# --------------------------------------------------------------------------------------
# The Q2 review: a fixed sample, a page, and the author's decisions


def write_review_sample(q2: dict, cfg: dict, rows_by_trace: dict[str, dict]) -> dict:
    """The sample is drawn by the analysis (seeded); here it is given opaque ids and a
    presentation order that interleaves the conditions. Once written it is never redrawn: a
    different draw on a later run means the inputs changed under a review, and stops."""
    items = [
        {"review_id": review_id(tid), "trace_id": tid, "condition": cond}
        for cond, block in sorted(q2["review_sample"].items())
        for tid in block["trace_ids"]
    ]
    random.Random(cfg["q2"]["review_seed"]).shuffle(items)
    path = Path(cfg["inputs"]["review_sample"])
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["items"] != items:
            raise RuntimeError(f"{path} does not match this run's draw; inputs changed?")
        return old
    sample = {
        "meta": {
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "rule": "correct (strict) and not fully grounded; up to "
            f"{cfg['q2']['review_per_condition']} per condition, generators pooled, "
            f"seed {cfg['q2']['review_seed']}",
            "n_in_quadrant": {c: b["n_in_quadrant"] for c, b in q2["review_sample"].items()},
        },
        "sample_sha256": sha256_json(items),
        "items": items,
    }
    path.write_text(
        json.dumps(sample, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return sample


def write_review_page(sample: dict, recs: dict[str, dict], traces: dict[str, dict]) -> None:
    page_items = []
    for it in sample["items"]:
        rec, tr = recs[it["trace_id"]], traces[it["trace_id"]]
        page_items.append(
            {
                "review_id": it["review_id"],
                "question": tr["question"]["question"],
                "answer": rec["answer"],
                "excerpts": [
                    {"header": excerpt_header(c), "text": c["text"]}
                    for c in tr["retrieval"]["chunks"]
                ],
                "claims": [
                    {k: c[k] for k in ("claim", "kind", "verdict", "reason", "supporting_excerpts")}
                    for c in rec["groundedness"]["claims"]
                ],
            }
        )
    REVIEW_PAGE_PATH.write_text(
        render_review_page(page_items, sample["sample_sha256"]), encoding="utf-8", newline="\n"
    )


def review_summary(sample: dict, cfg: dict, rows_by_trace: dict[str, dict]) -> dict:
    path = Path(cfg["inputs"]["review_labels"])
    if not path.exists():
        return {"status": "not_done", "n": len(sample["items"])}
    review = json.loads(path.read_text(encoding="utf-8"))
    if review["sample_sha256"] != sample["sample_sha256"]:
        raise RuntimeError(f"{path} belongs to a different review sample")
    decided = [
        (it, review["items"].get(it["review_id"], {}).get("decision")) for it in sample["items"]
    ]
    if any(d is None for _, d in decided):
        return {
            "status": "incomplete",
            "n": len(decided),
            "n_complete": sum(d is not None for _, d in decided),
        }
    nb, seed = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["seed"]
    right = mean_with_ci(
        [float(d == "judge_right") for _, d in decided],
        [rows_by_trace[it["trace_id"]]["question_id"] for it, _ in decided],
        nb,
        seed,
    )
    counts = {
        cond: {
            k: sum(1 for it, d in decided if it["condition"] == cond and d == k)
            for k in ("judge_right", "judge_wrong", "unclear")
        }
        for cond in sorted({it["condition"] for it in sample["items"]})
    }
    return {
        "status": "complete",
        "n": len(decided),
        "counts_by_condition": counts,
        "judge_right": {"value": right["mean"], "ci95": right["ci95"], "n": right["n"]},
        "notes": {
            it["trace_id"]: review["items"][it["review_id"]].get("note", "")
            for it, _ in decided
            if review["items"][it["review_id"]].get("note")
        },
        **review_categories(cfg, sample),
    }


def review_categories(cfg: dict, sample: dict) -> dict:
    """The author's categories for the answers where the judge was wrong: A, the judge broke
    its own written rules; B, the judge applied them and the author disagrees with the rule."""
    path = Path(cfg["inputs"]["review_categories"])
    if not path.exists():
        return {}
    cats = json.loads(path.read_text(encoding="utf-8"))
    if cats["meta"]["sample_sha256"] != sample["sample_sha256"]:
        raise RuntimeError(f"{path} belongs to a different review sample")
    return {
        "judge_wrong_categories": {
            "definitions": cats["meta"]["categories"],
            "counts": dict(sorted(Counter(i["category"] for i in cats["items"]).items())),
        }
    }


# --------------------------------------------------------------------------------------
# Reports


def _f(x, nd: int = 2) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def _ci(ci, nd: int = 2) -> str:
    return "–" if ci is None else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def _est(d: dict | None, key: str = "mean", nd: int = 2) -> str:
    if not d:
        return "–"
    return f"{_f(d.get(key), nd)} {_ci(d.get('ci95'), nd)}"


def _pair_cell(p: dict) -> str:
    if p.get("status") == "no_variance":
        return f"no variance ({', '.join(p['signals'])} flag)"
    b = p["baseline"]
    mark = " *" if p["above_chance"] else ""
    return f"{_f(p['value'])} {_ci(p['ci95'])} vs {_f(b['mean'])}{mark}"


CATEGORY_NAMES = {
    "a": "(a) one filing",
    "b": "(b) two filings",
    "c": "(c) unanswerable",
    "d": "(d) false premise",
}


def q4_markdown(q4: dict) -> list[str]:
    out = ["", "## Q4. Adversarial questions (40, author-approved)", ""]
    if q4.get("status") == "not_run":
        return out + ["Not run yet."]
    out += [
        "Pooled over the four prompts, intervals resampling questions (10 per category). "
        "Accuracy counts a decline as not correct.",
        "",
        "| Category | Condition | Generator | n | Accuracy | Declined | Groundedness (answered) "
        "| Answered, not disclosed | Answered, filing not in corpus |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cat, conds in q4["by_category"].items():
        for cond, models in conds.items():
            for model, e in models.items():
                acc = _est(e["accuracy_all"]["pooled"]) if "accuracy_all" in e else "–"
                nd = _est(e["answered_not_disclosed"]["pooled"]) if cat == "c" else "–"
                oc = _est(e["answered_filing_not_in_corpus"]["pooled"]) if cat == "c" else "–"
                out.append(
                    f"| {CATEGORY_NAMES[cat]} | {cond} | {model} | {e['n']} | {acc} "
                    f"| {_est(e['declined']['pooled'])} | {_est(e['groundedness_answered'])} "
                    f"| {nd} | {oc} |"
                )
    out += [
        "",
        "False premise (d), retrieved context: share of answers that reject the premise. The "
        "author's blind labels are the measurement; the judge is a second rater.",
        "",
        "| Generator | Rater | Rejects (pooled) | v1 | v2 | v3 | v4 | Counts |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, block in q4["premise"].items():
        for rater in ("human", "judge"):
            b = block[rater]
            if b.get("status") == "not_done":
                out.append(f"| {model} | {rater} | not done | | | | | |")
                continue
            pp = b["rejected"]["per_prompt"]
            cells = [f"{v['k']}/{v['n']}" for p, v in sorted(pp.items())]
            out.append(
                f"| {model} | {rater} | {_est(b['rejected']['pooled'])} | "
                + " | ".join(cells)
                + f" | {b['counts']} |"
            )
    ag = q4["premise_agreement"]
    if ag.get("status") != "not_done":
        out += [
            "",
            f"Author vs premise judge on the {ag['n']} judged answers labelled blind to the "
            f"judge: kappa {_f(ag['kappa_3class'])} (three classes), "
            f"{_f(ag['kappa_rejects'])} (rejects vs not); raw agreement "
            f"{_f(ag['raw_agreement'])}. Including the "
            f"{ag['all_labelled']['n'] - ag['n']} pilot answers, whose verdicts the author saw "
            f"before labelling: kappa {_f(ag['all_labelled']['kappa_3class'])}, raw "
            f"{_f(ag['all_labelled']['raw_agreement'])}.",
        ]
    return out


def report_markdown(res: dict) -> str:
    nb = res["meta"]["config"]["bootstrap"]["n_resamples"]
    out = [
        "# Phase 6 reliability analysis",
        "",
        "Single run. Generated by `scripts/07_reliability_analysis.py` from "
        "`reliability_analysis.json`; the plan and the predictions are fixed in "
        "`phase6_preregistration.md`. Intervals: 95% bootstrap resampling questions "
        f"({nb:,} resamples). Groundedness is measured by the LLM judge (blind human vs judge "
        "kappa 0.52 per answer, `judge_agreement.json`).",
        "",
        "## Predictions against results",
        "",
        "| | Prediction | Predicted | Observed | Unexpected |",
        "|---|---|---|---|---|",
    ]
    for p in res["predictions"]:
        unexpected = {True: "**yes**", False: "no", None: "–"}[p["unexpected"]]
        if p.get("kind") == "named" and p["unexpected"] is not None:
            unexpected += " (named option)"
        out.append(
            f"| {p['id']} | {p['description']} | {p['predicted']} | {p['observed'] or '–'} "
            f"| {unexpected} |"
        )

    q1 = res["q1"]
    out += [
        "",
        "## Q1. Retrieval quality vs groundedness (retrieved condition)",
        "",
        "Unit: question, groundedness averaged over its scored prompts; questions with aligned "
        "gold evidence only.",
        "",
        "| Generator | Questions | Spearman rho(recall@5, groundedness) | Groundedness, "
        "recall@5 > 0 | Groundedness, recall@5 = 0 | Difference | Answer rate, recall@5 > 0 "
        "| Answer rate, recall@5 = 0 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, m in q1["by_model"].items():
        s, d, a = m["spearman_recall5_groundedness"], m["groundedness_by_recall"], m["answer_rate"]
        out.append(
            f"| {model} | {s['n_questions']} | {_f(s['rho'])} {_ci(s['ci95'])} "
            f"| {_est(d['recall5_gt_0'])} (n={d['recall5_gt_0']['n']}) "
            f"| {_est(d['recall5_eq_0'])} (n={d['recall5_eq_0']['n']}) "
            f"| {_f(d['difference'])} {_ci(d['ci95'])} "
            f"| {_est(a['recall5_gt_0'])} | {_est(a['recall5_eq_0'])} |"
        )
    out += [
        "",
        "Exploratory predictors (Spearman rho with the question's mean groundedness score / "
        "share of prompts fully grounded):",
        "",
        "| Generator | " + " | ".join(PREDICTORS) + " |",
        "|---|" + "---|" * len(PREDICTORS),
    ]
    for model, m in q1["by_model"].items():
        cells = [
            f"{_f(m['exploratory'][p]['groundedness']['rho'])} / "
            f"{_f(m['exploratory'][p]['fully_grounded']['rho'])}"
            for p in PREDICTORS
        ]
        out.append(f"| {model} | " + " | ".join(cells) + " |")
    out += [
        "",
        "Paired, exploratory: the same question, generator and prompt under oracle context vs "
        "retrieved context, for questions with recall@5 = 0 under retrieval (oracle minus "
        "retrieved).",
        "",
        "| Generator | Pairs | Both scored | Change in groundedness | Change in fully grounded "
        "| Change in answer rate |",
        "|---|---|---|---|---|---|",
    ]
    for model, p in q1["paired_oracle_vs_retrieved"].items():
        out.append(
            f"| {model} | {p['n_pairs']} | {p['n_pairs_both_scored']} "
            f"| {_est(p['d_groundedness'])} | {_est(p['d_fully_grounded'])} "
            f"| {_est(p['d_answered'])} |"
        )

    q2 = res["q2"]
    out += [
        "",
        "## Q2. Correct vs grounded",
        "",
        "Scored, graded answers. Correct = `correct`; grounded = every document claim supported. "
        "Right answer, wrong reasons = share of correct answers not fully grounded.",
        "",
        "| Cell | n | Correct + grounded | Correct + ungrounded | Incorrect + grounded "
        "| Incorrect + ungrounded | Right answer, wrong reasons | Unit errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell, c in q2["cells"].items():
        p = c["primary"]
        out.append(
            f"| {cell} | {c['n']} | "
            + " | ".join(str(p["counts"][q]) for q in QUADRANTS)
            + f" | {_est(p['ungrounded_given_correct'])} | {c['n_unit_error']} |"
        )
    out += [
        "",
        "Sensitivity (right answer, wrong reasons): "
        + "; ".join(
            f"{name}: "
            + ", ".join(
                f"{cell} {_f(c[name]['ungrounded_given_correct']['mean'])}"
                for cell, c in q2["cells"].items()
            )
            for name in Q2_VARIANTS
            if name != "primary"
        )
        + ".",
        "",
        "Correct answers under retrieval, by whether the gold evidence was in the context "
        "(exploratory):",
        "",
        "| Generator | Group | Fully grounded | Not fully grounded |",
        "|---|---|---|---|",
    ]
    for model, groups in q2["mechanism_retrieved"].items():
        for group, v in groups.items():
            out.append(f"| {model} | {group} | {v['fully_grounded']} | {v['not_fully_grounded']} |")
    out += [
        "",
        "Judge-free: correct answers with checkable figures, share with none of their figures "
        "anywhere in the context:",
        "",
        "| Cell | Correct + ungrounded | Correct + grounded |",
        "|---|---|---|",
    ]
    for cell, c in q2["cells"].items():
        f = c["figures_in_context"]
        out.append(
            f"| {cell} | {_f(f['correct_ungrounded']['none_in_context'])} "
            f"(n={f['correct_ungrounded']['n_with_figures']}) "
            f"| {_f(f['correct_grounded']['none_in_context'])} "
            f"(n={f['correct_grounded']['n_with_figures']}) |"
        )
    rv = res["q2_review"]
    out += ["", f"Human review of the correct + ungrounded quadrant: {rv['status']}."]
    if rv["status"] == "complete":
        out.append(
            f"The judge is right in {_est(rv['judge_right'], 'value')} of {rv['n']} reviewed "
            "answers; by condition: "
            + "; ".join(
                f"{c}: " + ", ".join(f"{k} {v}" for k, v in d.items())
                for c, d in rv["counts_by_condition"].items()
            )
            + "."
        )
        cats = rv.get("judge_wrong_categories")
        if cats:
            n = cats["counts"]
            out.append(
                f"Of the answers where the judge was wrong, {n.get('A', 0)} are the judge "
                f"breaking its own written rules and {n.get('B', 0)} are the author disagreeing "
                "with a rule the judge applied (`quadrant_review_categories.json`)."
            )

    out += [
        "",
        "## Q3. Signal agreement",
        "",
        "Each cell: observed [interval] vs random baseline (permutation mean); * = above chance "
        "(interval entirely above the baseline). G = groundedness, CP = citation precision (same "
        "judge as G), NS = figures found in cited excerpts (judge-free), C = stated confidence.",
    ]
    for cell, c in res["q3"].items():
        flat = [s for s in SIGNALS if c["variance"][s]["no_variance"]]
        out += [
            "",
            f"**{cell}** (n = {c['n_scored']} scored)"
            + (f"; no variance: {', '.join(flat)}" if flat else ""),
            "",
            "| Pair | n | Spearman | Jaccard of flags | Kappa of flags |",
            "|---|---|---|---|---|",
        ]
        for pair, p in c["pairs"].items():
            if p["status"] != "ok":
                out.append(f"| {pair} | {p['n']} | no variance | | |")
                continue
            out.append(
                f"| {pair} | {p['n']} | {_pair_cell(p['spearman'])} | {_pair_cell(p['jaccard'])} "
                f"| {_pair_cell(p['kappa'])} |"
            )

    out += q4_markdown(res["q4"])
    out += [
        "",
        "## Q5. Prompt variants against v1 (paired by question)",
        "",
        "| Cell | Contrast | Both scored | Groundedness | Fully grounded | Citation precision "
        "| NS | Excerpts cited | Answer rate (v1 → variant) | Reading |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell, c in res["q5"].items():
        for treat, primary in CONTRASTS:
            e = c[f"{treat}_vs_{BASE_PROMPT}"]
            d, a = e["differences"], e["answer_rate"]
            label = treat + (" (primary)" if primary else "")
            out.append(
                f"| {cell} | {label} | {e['n_both_scored']} | {_est(d['groundedness'])} "
                f"| {_est(d['fully_grounded'])} | {_est(d['citation_precision'])} "
                f"| {_est(d['numeric_in_cited'])} | {_est(d['n_cited'])} "
                f"| {_f(a[BASE_PROMPT])} → {_f(a[treat])} | {e['reading']} |"
            )

    hc = res["human_check"]
    out += [
        "",
        "## Human-label check (50 validation answers; direction only)",
        "",
        f"{hc['n_items_scored']} of the 50 answers are scored under both.",
        "",
        "| Statistic | Generators | n | Judge | Human |",
        "|---|---|---|---|---|",
    ]
    blocks = [("both", hc)] + [
        (m + " (post hoc)", b) for m, b in hc.get("by_generator", {}).items()
    ]
    for label, block in blocks:
        for k, v in block.items():
            if k in ("n_items_scored", "by_generator"):
                continue
            out.append(f"| {k} | {label} | {v['n']} | {_f(v['judge'])} | {_f(v['human'])} |")

    out += [
        "",
        "## Deviations from the pre-registration",
        "",
        "Made after the first run on the real traces; each is visible in the code history.",
        "",
        "1. **The no-variance rule is applied to the form each statistic uses.** Part A reports "
        "a signal whose most common value covers >= 90% of a cell as no variance. On the first "
        "run it was applied to the continuous values only, and Jaccard and kappa were computed "
        "on flags that were almost constant (Qwen2.5 3B states 100 on nearly everything, so its "
        "confidence flag almost never fires): a kappa of 0.00 [0.00, 0.00] was called above "
        "chance. The rule now also applies to the flags for the flag statistics. No prediction "
        "changed verdict (the cells affected are Qwen2.5 3B's, and retrieved Claude Sonnet 5's "
        "CP-NS).",
        "2. **P1.3 names no generator.** The first run checked it against Claude Sonnet 5 only; "
        "the two generators give opposite answers, so it is reported as mixed, without a verdict.",
        "3. **An empty box typed `[]`** (P4.1) is read as an unmarked option; the prediction "
        "itself was unambiguous.",
        "4. **Added, post hoc:** the human-label check per generator (pooled, it mixes a "
        "generator with near-constant confidence with one whose confidence varies), the G-NS "
        "split by question type, and the right-answer-wrong-reasons split by question type "
        "(both below).",
        '5. **A point estimate on a strict bound** (0.50 against "< 50%") was shown in the '
        "wrong option; strict bounds are now strict. Display only: verdicts use intervals.",
        "6. **Author vs premise judge agreement** is computed on the answers the judge "
        "actually judged, and made blind: the first computation also counted the answers "
        "decided by rule for both raters (bare decline tokens, failed parses), which agree by "
        "construction, and the 8 premise-judge pilot answers, whose verdicts were shown to the "
        "author before labelling (their labels stay in the measurement; they enter only the "
        "all-labelled row).",
        "",
        "## Post hoc, exploratory (added after the first run)",
        "",
        "Groundedness vs NS (figures found in cited excerpts), split by FinanceBench question "
        "type. NS finds only figures printed in the excerpts, so a derived figure (a ratio, a "
        "sum) scores 0 there while the judge may accept it as arithmetic on the excerpts; "
        "metrics-generated questions are the computational ones.",
        "",
        "| Cell | Question type | n | Spearman G-NS |",
        "|---|---|---|---|",
    ]
    for cell, c in res["q3"].items():
        for qtype, v in c.get("post_hoc_G_NS_by_question_type", {}).items():
            rho = _pair_cell({"status": "ok", **v["spearman"]}) if v["spearman"] else "–"
            out.append(f"| {cell} | {qtype} | {v['n']} | {rho} |")
    out += [
        "",
        "Right answer, wrong reasons by FinanceBench question type (added after the author's "
        "review of the quadrant, where several disagreements were about evaluative conclusions "
        "that the groundedness rules count as unsupported unless an excerpt states them). "
        "Domain-relevant questions ask for such judgments.",
        "",
        "| Cell | Question type | Correct answers | Not fully grounded |",
        "|---|---|---|---|",
    ]
    for cell, c in res["q2"]["cells"].items():
        for qtype, v in c.get("post_hoc_by_question_type", {}).items():
            out.append(
                f"| {cell} | {qtype} | {v['n_correct']} | {_est(v['ungrounded_given_correct'])} |"
            )
    return "\n".join(out) + "\n"


def examples_markdown(
    res: dict, recs: dict[str, dict], traces: dict[str, dict], golds: dict[str, dict]
) -> str:
    out = [
        "# Correct vs grounded: one example per quadrant",
        "",
        "Single run. Generated by `scripts/07_reliability_analysis.py`. One answer per quadrant "
        f"per cell, drawn at random (seed {res['meta']['config']['q2']['examples_seed']}), not "
        "chosen by hand. Verdicts and reasons are the groundedness judge's.",
    ]
    for cell, quads in res["q2"]["examples"].items():
        out += ["", f"## {cell}"]
        for q in QUADRANTS:
            out += ["", f"### {q.replace('_', ' + ', 1)}", ""]
            if not quads[q]:
                out.append("_No answer in this quadrant._")
                continue
            tid = quads[q][0]
            rec, tr = recs[tid], traces[tid]
            corr = rec["correctness"]
            reason = (corr.get("judge") or {}).get("reason")
            out += [
                f"`{tid}`",
                "",
                f"- **Question:** {tr['question']['question']}",
                f"- **Gold answer:** {golds[rec['financebench_id']]['answer']}",
                f"- **Answer:** {rec['answer']}",
                f"- **Correctness:** {corr['label']} ({corr['grader']})"
                + (f": {reason}" if reason else ""),
                f"- **Groundedness:** {_f(rec['groundedness']['groundedness'])}; claims:",
            ]
            for c in rec["groundedness"]["claims"]:
                out.append(f"  - [{c['kind']}, {c['verdict']}] {c['claim']} — _{c['reason']}_")
    return "\n".join(out) + "\n"


def write_plots(res: dict, models: list[str], plots_dir: Path) -> None:
    plot_retrieval_vs_groundedness(res["q1"], models, plots_dir / "retrieval_vs_groundedness.png")
    plot_quadrants(res["q2"], CONDITIONS, models, plots_dir / "correct_vs_grounded.png")
    plot_signal_agreement(res["q3"], CONDITIONS, models, plots_dir / "signal_agreement.png")
    plot_prompt_effect(res["q5"], CONDITIONS, models, plots_dir / "prompt_effect.png")
    q4 = res.get("q4")
    if q4 and q4.get("status") != "not_run":
        plot_adversarial(q4, models, plots_dir / "adversarial_breakdown.png")


# --------------------------------------------------------------------------------------


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prereg_path = Path(cfg["inputs"]["preregistration"])
    prereg = prereg_path.read_text(encoding="utf-8")
    problems = prediction_problems(prereg)
    if problems:
        sys.exit(
            f"{prereg_path}: predictions are not complete ({'; '.join(problems)}). "
            "Mark one option per prediction and commit the file before running the analysis."
        )

    rows, recs, traces = load_inputs(cfg)
    rows_by_trace = {r["trace_id"]: r for r in rows}
    human = load_human(cfg)
    golds = {
        (r := json.loads(x))["financebench_id"]: r
        for x in FINANCEBENCH_PATH.read_text(encoding="utf-8").splitlines()
    }
    print(f"{len(rows)} traces; running Q1-Q3, Q5 and the human-label check ...")
    res = analyse(rows, human, cfg)
    adv = adversarial_rows(cfg)
    res["q4"] = (
        q4_adversarial(adv, cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["seed"])
        if adv is not None
        else {"status": "not_run"}
    )

    sample = write_review_sample(res["q2"], cfg, rows_by_trace)
    write_review_page(sample, recs, traces)
    res["q2_review"] = review_summary(sample, cfg, rows_by_trace)

    inputs = {k: sha256_file(Path(v)) for k, v in cfg["inputs"].items() if Path(v).is_file()}
    res["meta"] = {
        "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "single_run": True,
        "n_traces": len(rows),
        "config": cfg,
        "inputs_sha256": inputs,
    }
    res["predictions"] = check_predictions(prereg, res)
    order = ["meta", "predictions", "q1", "q2", "q2_review", "q3", "q4", "q5", "human_check"]
    res = {k: res[k] for k in order}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "reliability_analysis.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    (RESULTS_DIR / "reliability_analysis.md").write_text(
        report_markdown(res), encoding="utf-8", newline="\n"
    )
    (RESULTS_DIR / "quadrant_examples.md").write_text(
        examples_markdown(res, recs, traces, golds), encoding="utf-8", newline="\n"
    )
    write_plots(res, sorted({r["model"] for r in rows}), PLOTS_DIR)
    n_unexpected = sum(1 for p in res["predictions"] if p["unexpected"])
    n_checked = sum(1 for p in res["predictions"] if p["unexpected"] is not None)
    print(f"wrote {RESULTS_DIR / 'reliability_analysis.json'} and .md, quadrant_examples.md")
    print(f"review sample: {len(sample['items'])} answers -> {REVIEW_PAGE_PATH}")
    print(f"predictions: {n_checked} checked so far, {n_unexpected} unexpected")


if __name__ == "__main__":
    main()
