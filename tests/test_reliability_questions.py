"""End-to-end test of the Phase 6 analyses on a synthetic trace table with planted structure:
every question's analysis runs, the planted effects are recovered and the absent ones are
not, the predictions are checked by the Part B rules, and both reports render. Synthetic
data only, so no real result is seen before the predictions are committed."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.reliability import QUADRANTS
from src.evaluation.reliability_questions import (
    analyse,
    check_predictions,
    parse_predictions,
    parse_range,
    primary_quadrant,
)

SONNET, QWEN = "claude-sonnet-5", "qwen2.5-3b"
PROMPTS = ("v1_zero_shot", "v2_citation_required", "v3_chain_of_thought", "v4_abstention")
CFG = {
    "bootstrap": {"n_resamples": 300, "seed": 0},
    "q2": {"examples_seed": 0, "review_per_condition": 15, "review_seed": 0},
    "q3": {
        "confidence_flag_below": 80,
        "sensitivity_thresholds": [70, 90],
        "no_variance_share": 0.9,
    },
}


def make_rows(n_q: int = 60, seed: int = 0) -> list[dict]:
    """Planted: under retrieval, Sonnet's groundedness follows recall@5 and Qwen's does not;
    Sonnet declines half the questions without evidence; under oracle, Sonnet's v2 answers
    are fully grounded (a real improvement) while Qwen's v2 keeps its groundedness but cites
    worse (apparent only); for Sonnet under oracle, NS tracks groundedness and confidence is
    noise; Qwen always states 100 (no variance)."""
    rng = np.random.default_rng(seed)
    rows = []
    for qi in range(n_q):
        qid = f"q{qi:03d}"
        unanswerable = qi >= n_q - 5
        hit = qi % 3 == 0 and not unanswerable
        top1 = 0.6 + 0.2 * hit + 0.05 * float(rng.random())
        for cond in ("retrieved", "oracle"):
            if cond == "oracle" and unanswerable:
                continue
            for model in (SONNET, QWEN):
                sonnet = model == SONNET
                g_base = float(rng.choice([0.0, 0.5, 1.0]))
                cp_base = float(rng.choice([0.5, 1.0]))
                for p in PROMPTS:
                    if cond == "retrieved":
                        recall = None if unanswerable else float(hit)
                        ans = (
                            "unanswerable"
                            if unanswerable
                            else ("evidence_present" if hit else "evidence_absent")
                        )
                    else:
                        recall, ans = 1.0, "answerable"
                    g, cp = g_base, cp_base
                    if cond == "retrieved" and sonnet:
                        g = 1.0 if hit else float(rng.choice([0.0, 0.25, 0.5]))
                    if cond == "oracle" and p == "v2_citation_required":
                        if sonnet:
                            g = 1.0
                        else:
                            cp = cp_base - 0.4
                    answered = not (
                        sonnet and cond == "retrieved" and not hit and rng.random() < 0.5
                    )
                    label = str(
                        rng.choice(["correct", "correct", "incorrect", "partially_correct"])
                    )
                    ns = g if (sonnet and cond == "oracle") else float(rng.choice([0.0, 0.5, 1.0]))
                    rows.append(
                        {
                            "trace_id": f"{cond}__{model}__{p}__{qid}",
                            "question_id": qid,
                            "condition": cond,
                            "model": model,
                            "prompt": p,
                            "answerability": ans,
                            "status": "answered" if answered else "declined",
                            "answered": answered,
                            "label": label if answered else "abstained",
                            "groundedness": g if answered else None,
                            "fully_grounded": (g == 1.0) if answered else None,
                            "citation_precision": cp if answered else None,
                            "n_cited": 2 if p != "v2_citation_required" else 3,
                            "numeric_in_cited": ns if answered else None,
                            "numeric_in_context": float(rng.choice([0.0, 1.0])),
                            "n_figures": 2,
                            "confidence": (100 if not sonnet else int(rng.integers(40, 101)))
                            if answered
                            else None,
                            "recall5": recall,
                            "span_recall5": recall,
                            "ndcg5": recall,
                            "mrr": recall,
                            "top1_score": top1,
                        }
                    )
    return rows


@pytest.fixture(scope="module")
def rows():
    return make_rows()


@pytest.fixture(scope="module")
def human(rows):
    # the human agrees with the judge on twelve scored traces
    scored = [r for r in rows if r["answered"]][:12]
    return {
        r["trace_id"]: {"groundedness": r["groundedness"], "fully_grounded": r["fully_grounded"]}
        for r in scored
    }


@pytest.fixture(scope="module")
def res(rows, human):
    return analyse(rows, human, CFG)


# --------------------------------------------------------------------------------------


class TestQ1:
    def test_planted_correlation_recovered(self, res):
        s = res["q1"]["by_model"][SONNET]["spearman_recall5_groundedness"]
        assert s["rho"] > 0.5 and s["ci95"][0] > 0
        d = res["q1"]["by_model"][SONNET]["groundedness_by_recall"]
        assert d["difference"] > 0.4 and d["ci95"][0] > 0

    def test_absent_correlation_not_invented(self, res):
        s = res["q1"]["by_model"][QWEN]["spearman_recall5_groundedness"]
        assert s["ci95"][0] < 0 < s["ci95"][1]

    def test_only_aligned_questions_enter(self, res):
        # 55 in-corpus questions; the 5 unanswerable ones have no recall@5
        n = res["q1"]["by_model"][QWEN]["spearman_recall5_groundedness"]["n_questions"]
        assert n == 55

    def test_answer_rate_shows_selection(self, res):
        a = res["q1"]["by_model"][SONNET]["answer_rate"]
        assert a["recall5_gt_0"]["mean"] == 1.0
        assert 0.3 < a["recall5_eq_0"]["mean"] < 0.7

    def test_paired_pairs_only_recall_zero(self, res):
        p = res["q1"]["paired_oracle_vs_retrieved"][SONNET]
        # recall@5 = 0 under retrieval: 55 aligned questions minus the 19 hits, x 4 prompts
        assert p["n_pairs"] == (55 - 19) * 4
        assert p["d_answered"]["mean"] > 0  # Sonnet always answers under oracle


class TestQ2:
    def test_counts_cover_every_graded_scored_trace(self, res, rows):
        for cell, c in res["q2"]["cells"].items():
            cond, model = cell.split("__")
            n = sum(
                1
                for r in rows
                if r["condition"] == cond
                and r["model"] == model
                and r["answered"]
                and r["label"] in ("correct", "partially_correct", "incorrect", "unit_error")
            )
            assert c["n"] == n == sum(c["primary"]["counts"].values())

    def test_lenient_moves_partials_to_correct(self, res):
        for c in res["q2"]["cells"].values():
            strict, lenient = c["primary"]["counts"], c["lenient_correct"]["counts"]
            n_correct = strict["correct_grounded"] + strict["correct_ungrounded"]
            assert lenient["correct_grounded"] + lenient["correct_ungrounded"] >= n_correct

    def test_examples_come_from_their_quadrant(self, res, rows):
        by_id = {r["trace_id"]: r for r in rows}
        for quads in res["q2"]["examples"].values():
            for q in QUADRANTS:
                for tid in quads[q]:
                    assert primary_quadrant(by_id[tid]) == q

    def test_review_sample(self, res, rows):
        by_id = {r["trace_id"]: r for r in rows}
        for cond, block in res["q2"]["review_sample"].items():
            assert len(block["trace_ids"]) == min(15, block["n_in_quadrant"])
            for tid in block["trace_ids"]:
                assert by_id[tid]["condition"] == cond
                assert primary_quadrant(by_id[tid]) == "correct_ungrounded"


class TestQ3:
    def test_constant_confidence_is_no_variance(self, res):
        cell = res["q3"]["oracle__qwen2.5-3b"]
        assert cell["variance"]["C"]["no_variance"] is True
        assert cell["pairs"]["G-C"]["status"] == "no_variance"
        assert cell["confidence_threshold_sensitivity"] == {}

    def test_planted_agreement_above_chance(self, res):
        g_ns = res["q3"]["oracle__claude-sonnet-5"]["pairs"]["G-NS"]
        assert g_ns["spearman"]["value"] == pytest.approx(1.0)
        assert g_ns["spearman"]["above_chance"] is True

    def test_noise_not_above_chance(self, res):
        g_c = res["q3"]["oracle__claude-sonnet-5"]["pairs"]["G-C"]
        assert g_c["spearman"]["above_chance"] is False

    def test_baseline_reported_for_every_ok_pair(self, res):
        for cell in res["q3"].values():
            for p in cell["pairs"].values():
                if p["status"] == "ok":
                    for stat in ("spearman", "jaccard", "kappa"):
                        assert p[stat]["baseline"]["n_perm"] == CFG["bootstrap"]["n_resamples"]


class TestQ5:
    def test_real_improvement(self, res):
        e = res["q5"]["oracle__claude-sonnet-5"]["v2_citation_required_vs_v1_zero_shot"]
        assert e["reading"] == "real_improvement"

    def test_apparent_only(self, res):
        e = res["q5"]["oracle__qwen2.5-3b"]["v2_citation_required_vs_v1_zero_shot"]
        assert e["differences"]["groundedness"]["mean"] == pytest.approx(0.0)
        assert e["reading"] == "apparent_only"

    def test_n_cited_difference(self, res):
        e = res["q5"]["oracle__qwen2.5-3b"]["v2_citation_required_vs_v1_zero_shot"]
        assert e["differences"]["n_cited"]["mean"] == pytest.approx(1.0)

    def test_primary_flag(self, res):
        c = res["q5"]["oracle__claude-sonnet-5"]
        assert c["v2_citation_required_vs_v1_zero_shot"]["primary"] is True
        assert c["v4_abstention_vs_v1_zero_shot"]["primary"] is False


class TestHumanCheck:
    def test_identical_labels_give_identical_statistics(self, res):
        hc = res["human_check"]
        assert hc["n_items_scored"] == 12
        for k, v in hc.items():
            if k != "n_items_scored" and v["n"]:
                assert v["judge"] == v["human"] or (
                    v["judge"] is not None and math.isclose(v["judge"], v["human"])
                )


# --------------------------------------------------------------------------------------
# Predictions

FILLED = """
- P1.1 Sonnet rho: [x] >= 0.5 · [ ] 0.2 to 0.5 · [ ] -0.2 to 0.2 · [ ] <= -0.2
- P1.2 Qwen rho: [x] >= 0.5 · [ ] 0.2 to 0.5 · [ ] -0.2 to 0.2 · [ ] <= -0.2
- P1.4 Oracle vs retrieved, Sonnet: [ ] up · [x] no change · [ ] down
- P2.4 Largest quadrant: [x] correct+grounded · [ ] correct+ungrounded · [ ] incorrect+grounded
- P3.1 G-C: [ ] above chance · [x] not above chance
- P3.2 G-NS: [ ] above chance · [x] not above chance
- P4.1 Sonnet rejects (d): [ ] < 25% · [x] 25-50% · [ ] 50-75% · [ ] > 75%
- P5.1 Sonnet v2: [x] real improvement · [ ] apparent only · [ ] no effect · [ ] worse
- P5.2 Qwen v2: [x] real improvement · [ ] apparent only · [ ] no effect · [ ] worse
"""


class TestPredictions:
    def test_parse(self):
        p = parse_predictions(FILLED)
        assert p["P1.1"]["predicted"] == ">= 0.5"
        assert p["P2.4"]["predicted"] == "correct+grounded"
        assert len(p["P1.1"]["options"]) == 4
        assert p["P2.4"]["options"][2] == "incorrect+grounded"

    @pytest.mark.parametrize(
        ("text", "bounds"),
        [
            (">= 0.5", (0.5, math.inf)),
            ("0.2 to 0.5", (0.2, 0.5)),
            ("-0.2 to 0.2", (-0.2, 0.2)),
            ("<= -0.2", (-math.inf, -0.2)),
            ("< 10%", (-math.inf, 0.1)),
            ("10-25%", (0.1, 0.25)),
            ("> 75%", (0.75, math.inf)),
        ],
    )
    def test_parse_range(self, text, bounds):
        lo, hi = parse_range(text)
        assert lo == pytest.approx(bounds[0]) and hi == pytest.approx(bounds[1])

    def test_checks_follow_part_b_rules(self, res):
        by_id = {c["id"]: c for c in check_predictions(FILLED, res)}
        assert by_id["P1.1"]["unexpected"] is False  # planted rho is high
        assert by_id["P1.2"]["unexpected"] is True  # Qwen's interval spans zero, excludes >= 0.5
        assert (
            by_id["P3.1"]["unexpected"] is False and by_id["P3.1"]["observed"] == "not above chance"
        )
        assert by_id["P3.2"]["unexpected"] is True and by_id["P3.2"]["kind"] == "named"
        assert by_id["P5.1"]["unexpected"] is False
        assert by_id["P5.2"]["observed"] == "apparent only" and by_id["P5.2"]["unexpected"] is True
        assert by_id["P4.1"]["unexpected"] is None  # no adversarial results yet

    def test_named_quadrant(self, res):
        by_id = {c["id"]: c for c in check_predictions(FILLED, res)}
        counts = res["q2"]["cells"]["retrieved__qwen2.5-3b"]["primary"]["counts"]
        largest = max(counts, key=counts.get)
        assert by_id["P2.4"]["observed"] == largest
        assert by_id["P2.4"]["unexpected"] == (largest != "correct_grounded")


# --------------------------------------------------------------------------------------
# Reports


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "07_reliability_analysis.py"
    spec = importlib.util.spec_from_file_location("reliability_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reports_render(res, rows):
    script = load_script()
    full = dict(res)
    full["meta"] = {"config": CFG}
    full["q2_review"] = {"status": "not_done", "n": 30}
    full["predictions"] = check_predictions(FILLED, full)
    md = script.report_markdown(full)
    for heading in ("## Predictions", "## Q1.", "## Q2.", "## Q3.", "## Q5.", "## Human-label"):
        assert heading in md
    recs, traces, golds = {}, {}, {}
    for r in rows:
        recs[r["trace_id"]] = {
            "financebench_id": r["question_id"],
            "answer": "An answer.",
            "correctness": {"label": r["label"], "grader": "judge", "judge": {"reason": "why"}},
            "groundedness": {
                "groundedness": r["groundedness"],
                "claims": [
                    {"kind": "document", "verdict": "supported", "claim": "c", "reason": "r"}
                ],
            },
        }
        traces[r["trace_id"]] = {"question": {"question": f"Question {r['question_id']}?"}}
        golds[r["question_id"]] = {"answer": "gold"}
    ex = script.examples_markdown(full, recs, traces, golds)
    assert ex.count("### ") == 16  # four quadrants in each of four cells


def test_plots_render(res, tmp_path):
    script = load_script()
    script.write_plots(res, [SONNET, QWEN], tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "correct_vs_grounded.png",
        "prompt_effect.png",
        "retrieval_vs_groundedness.png",
        "signal_agreement.png",
    ]
    assert all(p.stat().st_size > 10_000 for p in tmp_path.iterdir())
