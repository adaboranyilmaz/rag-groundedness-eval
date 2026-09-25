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
    q3_signal_agreement,
    q4_adversarial,
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
                            "question_type": "metrics-generated" if qi % 2 else "domain-relevant",
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
                        if p[stat]["status"] == "ok":
                            n = p[stat]["baseline"]["n_perm"]
                            assert n == CFG["bootstrap"]["n_resamples"]

    def test_constant_flag_is_no_variance_for_flag_statistics_only(self):
        # confidence varies (80-100) but never falls below the flag threshold of 80: Spearman
        # is computed, Jaccard and kappa are not (both would sit at about zero)
        rows = [
            r | {"confidence": 80 + (i % 21)} if r["model"] == SONNET and r["answered"] else r
            for i, r in enumerate(make_rows())
        ]
        q3 = q3_signal_agreement(rows, 200, 0, 80, [90], 0.9)
        cell = q3["oracle__claude-sonnet-5"]
        assert cell["variance"]["C"]["no_variance"] is False
        assert cell["variance"]["C"]["flag_no_variance"] is True
        g_c = cell["pairs"]["G-C"]
        assert g_c["spearman"]["status"] == "ok"
        assert g_c["jaccard"] == {"status": "no_variance", "signals": ["C"]}
        assert g_c["kappa"]["status"] == "no_variance"
        # at a threshold of 90 the flag varies again
        assert cell["confidence_threshold_sensitivity"]["90"]["G-C"]["jaccard"]["status"] == "ok"

    def test_post_hoc_split_by_question_type(self, res):
        by_type = res["q3"]["oracle__claude-sonnet-5"]["post_hoc_G_NS_by_question_type"]
        assert set(by_type) == {"metrics-generated", "other"}
        assert (
            by_type["metrics-generated"]["n"] + by_type["other"]["n"]
            == (res["q3"]["oracle__claude-sonnet-5"]["pairs"]["G-NS"]["n"])
        )


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
        blocks = [hc, *hc["by_generator"].values()]
        for block in blocks:
            for k, v in block.items():
                if k in ("n_items_scored", "by_generator") or not v["n"]:
                    continue
                assert v["judge"] == v["human"] or (
                    v["judge"] is not None and math.isclose(v["judge"], v["human"])
                )

    def test_by_generator_partitions_the_items(self, res):
        hc = res["human_check"]
        for stat in ("q2_ungrounded_given_correct", "q3_spearman_G_C"):
            assert sum(b[stat]["n"] for b in hc["by_generator"].values()) == hc[stat]["n"]


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

    def test_empty_box_without_space_is_an_option(self):
        p = parse_predictions("- P4.1 Sonnet (d): [] < 25% · [x] 25-50% · [ ] > 75%\n")
        assert p["P4.1"]["options"] == ["< 25%", "25-50%", "> 75%"]
        assert p["P4.1"]["predicted"] == "25-50%"

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
        assert by_id["P2.4"]["observed"] == largest.replace("_", "+")  # the prediction's notation
        assert by_id["P2.4"]["unexpected"] == (largest != "correct_grounded")

    def test_generator_free_prediction_with_split_outcome_is_mixed(self, res):
        text = "- P1.3 Top-1 at least as good: [ ] yes · [x] no\n"
        split = {
            "q1": {
                "by_model": {
                    SONNET: {"top1_minus_recall5_rho": {"value": 0.04, "ci95": [-0.1, 0.2]}},
                    QWEN: {"top1_minus_recall5_rho": {"value": -0.4, "ci95": [-0.6, -0.2]}},
                }
            }
        }
        (c,) = check_predictions(text, split)
        assert c["observed"] == "mixed (Claude Sonnet 5 yes, Qwen2.5 3B no)"
        assert c["unexpected"] is None
        same = {
            "q1": {
                "by_model": {
                    m: {"top1_minus_recall5_rho": {"value": -0.1, "ci95": None}}
                    for m in (SONNET, QWEN)
                }
            }
        }
        (c,) = check_predictions(text, same)
        assert c["observed"] == "no" and c["unexpected"] is False

    def test_pair_names_compare_in_any_order(self):
        text = "- P3.4 Highest Jaccard: [ ] G-C · [x] NS-G\n"
        res = {
            "q3": {
                "oracle__claude-sonnet-5": {
                    "pairs": {
                        "G-NS": {"status": "ok", "jaccard": {"status": "ok", "value": 0.6}},
                        "G-C": {
                            "status": "ok",
                            "jaccard": {"status": "no_variance", "signals": ["C"]},
                        },
                    }
                }
            }
        }
        (c,) = check_predictions(text, res)
        assert c["observed"] == "G-NS" and c["unexpected"] is False


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
    full["q4"] = {"status": "not_run"}
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


# --------------------------------------------------------------------------------------
# Q4


def make_adv_rows(human_done: bool = True) -> list[dict]:
    """Planted: (a) accuracy 0.8, (b) 0.2 under retrieval; Sonnet answers 3 of the 5
    out-of-corpus (c) questions; Sonnet rejects false premises only under v4; Qwen never."""
    rows = []
    cats = {"a": "single_filing", "b": "two_filings"}
    for cat in ("a", "b", "c", "d"):
        for qi in range(10):
            qid = f"adv_{cat}{qi:02d}"
            sub = cats.get(cat) or (
                ("not_disclosed" if qi < 5 else "filing_not_in_corpus")
                if cat == "c"
                else ("premise_contradicted" if qi < 5 else "entity_does_not_exist")
            )
            for model in (SONNET, QWEN):
                for p in PROMPTS:
                    correct = (cat == "a" and qi < 8) or (cat == "b" and qi < 2)
                    answered = cat in ("a", "b", "d") or (
                        cat == "c" and model == SONNET and sub == "filing_not_in_corpus" and qi < 8
                    )
                    handling = None
                    if cat == "d":
                        handling = (
                            "rejects_premise"
                            if model == SONNET and p == "v4_abstention"
                            else "accepts_premise"
                        )
                    rows.append(
                        {
                            "trace_id": f"retrieved__{model}__{p}__{qid}",
                            "question_id": qid,
                            "condition": "retrieved",
                            "model": model,
                            "prompt": p,
                            "category": cat,
                            "subtype": sub,
                            "status": "answered" if answered else "declined",
                            "answered": answered,
                            "label": ("correct" if correct else "incorrect")
                            if cat in ("a", "b")
                            else None,
                            "groundedness": 1.0 if answered else None,
                            "premise_judge": handling,
                            "premise_human": handling if human_done else None,
                        }
                    )
    return rows


class TestQ4:
    def test_planted_structure(self):
        q4 = q4_adversarial(make_adv_rows(), 300, 0)
        a = q4["by_category"]["a"]["retrieved"][SONNET]["accuracy_all"]["pooled"]["mean"]
        b = q4["by_category"]["b"]["retrieved"][SONNET]["accuracy_all"]["pooled"]["mean"]
        assert (a, b) == (0.8, 0.2)
        d = q4["two_filing_minus_one_filing_accuracy"][SONNET]
        assert d["value"] == pytest.approx(-0.6) and d["ci95"][1] < 0
        assert q4["out_of_corpus_answered_from_memory"][SONNET]["value"] == pytest.approx(0.6)
        assert q4["out_of_corpus_answered_from_memory"][QWEN]["value"] == 0.0
        assert q4["premise_rejected"][SONNET]["value"] == pytest.approx(0.25)
        assert q4["v4_raises_rejection"] == {SONNET: "yes", QWEN: "no"}
        per = q4["premise"][SONNET]["human"]["rejected"]["per_prompt"]["v4_abstention"]
        assert (per["k"], per["n"]) == (10, 10) and per["ci95"][0] > 0.6
        assert q4["premise_agreement"]["raw_agreement"] == 1.0

    def test_premise_pending_until_labelled(self):
        q4 = q4_adversarial(make_adv_rows(human_done=False), 200, 0)
        assert q4["premise"][SONNET]["human"] == {"status": "not_done"}
        assert q4["premise_rejected"] == {} and q4["v4_raises_rejection"] == {}
        assert q4["premise_agreement"] == {"status": "not_done"}
        assert "rejected" in q4["premise"][SONNET]["judge"]

    def test_predictions_read_q4(self):
        q4 = q4_adversarial(make_adv_rows(), 300, 0)
        text = (
            "- P4.1 Sonnet (d): [ ] < 25% · [x] 25-50% · [ ] 50-75% · [ ] > 75%\n"
            "- P4.3 v4 raises: [x] yes · [ ] no\n"
            "- P4.5 (b) vs (a): [ ] (b) clearly lower · [ ] about the same · [x] (b) higher\n"
        )
        by_id = {c["id"]: c for c in check_predictions(text, {"q4": q4})}
        assert by_id["P4.1"]["unexpected"] is False  # 0.25 sits on the range's edge
        assert by_id["P4.3"]["unexpected"] is False
        assert by_id["P4.5"]["observed"] == "(b) clearly lower"
        assert by_id["P4.5"]["unexpected"] is True

    def test_report_renders(self):
        script = load_script()
        md = "\n".join(script.q4_markdown(q4_adversarial(make_adv_rows(), 200, 0)))
        assert "## Q4." in md and "not done" not in md
        md = "\n".join(script.q4_markdown(q4_adversarial(make_adv_rows(False), 200, 0)))
        assert "| human | not done |" in md
        assert script.q4_markdown({"status": "not_run"})[-1] == "Not run yet."
