"""Unit tests for the Phase 6 reliability statistics: ranks and Spearman's rho, flag
agreement, exact binomial intervals, question-level bootstrap, the stratified permutation
baseline, quadrant and correctness definitions, the Q5 reading, sampling and the
predictions guard."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.agreement import cohen_kappa
from src.evaluation.reliability import (
    average_ranks,
    clopper_pearson,
    dominant_share,
    is_correct,
    jaccard,
    kappa,
    mean_with_ci,
    pair_agreement,
    permutation_baseline,
    prediction_problems,
    q5_reading,
    quadrant,
    question_ci,
    seeded_sample,
    spearman,
)

# --------------------------------------------------------------------------------------
# Definitions


class TestDefinitions:
    def test_strict_and_lenient_correctness(self):
        assert is_correct("correct") is True
        assert is_correct("partially_correct") is False
        assert is_correct("partially_correct", lenient=True) is True
        assert is_correct("unit_error") is False
        assert is_correct("unit_error", lenient=True) is False
        assert is_correct("incorrect", lenient=True) is False

    @pytest.mark.parametrize("label", ["abstained", "no_answer", "unparsed", None])
    def test_non_grades_are_none(self, label):
        assert is_correct(label) is None
        assert is_correct(label, lenient=True) is None

    def test_quadrant_names(self):
        assert quadrant(True, True) == "correct_grounded"
        assert quadrant(True, False) == "correct_ungrounded"
        assert quadrant(False, True) == "incorrect_grounded"
        assert quadrant(False, False) == "incorrect_ungrounded"

    def test_dominant_share(self):
        assert dominant_share([100, 100, 100, 90]) == 0.75
        assert dominant_share([]) is None


# --------------------------------------------------------------------------------------
# Pair statistics


class TestRanksAndSpearman:
    def test_ties_share_mean_rank(self):
        assert average_ranks([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]
        assert average_ranks([3, 1, 2]).tolist() == [3.0, 1.0, 2.0]
        assert average_ranks([5, 5, 5]).tolist() == [2.0, 2.0, 2.0]

    def test_monotone_and_reversed(self):
        x = [1, 2, 3, 4, 5]
        assert spearman(x, [1, 4, 9, 16, 25]) == pytest.approx(1.0)
        assert spearman(x, [9, 7, 5, 3, 1]) == pytest.approx(-1.0)

    def test_undefined_cases(self):
        assert spearman([1, 2], [2, 1]) is None  # fewer than three items
        assert spearman([1, 2, 3], [4, 4, 4]) is None  # constant variable
        with pytest.raises(ValueError):
            spearman([1, 2, 3], [1, 2])

    def test_matches_scipy_with_ties(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(1)
        for _ in range(20):
            x = rng.integers(0, 5, size=40)  # heavy ties, like recall@5 or confidence
            y = rng.integers(0, 4, size=40) + 0.5 * x
            assert spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic)


class TestFlagAgreement:
    def test_jaccard(self):
        assert jaccard([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(1 / 3)
        assert jaccard([1, 1], [1, 1]) == 1.0
        assert jaccard([0, 0], [0, 0]) is None

    def test_kappa_equals_agreement_module(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            a = rng.random(30) < 0.3
            b = (rng.random(30) < 0.3) | a & (rng.random(30) < 0.5)
            ref = cohen_kappa(a.tolist(), b.tolist())
            got = kappa(a, b)
            assert (got is None and ref is None) or got == pytest.approx(ref)

    def test_kappa_undefined_when_both_constant(self):
        assert kappa([True, True], [True, True]) is None
        assert kappa([], []) is None


# --------------------------------------------------------------------------------------
# Intervals and baselines


class TestClopperPearson:
    # Reference values: the exact (Clopper-Pearson) 95% interval, standard tables.
    @pytest.mark.parametrize(
        ("k", "n", "lo", "hi"),
        [(0, 10, 0.0, 0.30850), (5, 10, 0.18709, 0.81291), (10, 10, 0.69150, 1.0)],
    )
    def test_reference_values(self, k, n, lo, hi):
        got = clopper_pearson(k, n)
        assert got[0] == pytest.approx(lo, abs=1e-5)
        assert got[1] == pytest.approx(hi, abs=1e-5)

    def test_matches_scipy(self):
        stats = pytest.importorskip("scipy.stats")
        for n in (1, 7, 10, 40):
            for k in range(n + 1):
                ref = stats.binomtest(k, n).proportion_ci(method="exact")
                got = clopper_pearson(k, n)
                assert got[0] == pytest.approx(ref.low, abs=1e-7)
                assert got[1] == pytest.approx(ref.high, abs=1e-7)

    def test_edge_cases(self):
        assert clopper_pearson(0, 0) is None
        with pytest.raises(ValueError):
            clopper_pearson(3, 2)


class TestQuestionBootstrap:
    def test_resamples_whole_questions(self):
        # question a has 3 items, b has 1: two drawn questions give 6, 4 or 2 items, never
        # any other count, if questions are resampled whole
        questions = ["a", "a", "a", "b"]
        seen = []

        def stat(idx):
            seen.append(len(idx))
            return float(len(idx))

        question_ci(questions, stat, n_resamples=200, seed=0)
        assert set(seen) <= {2, 4, 6}
        assert len(set(seen)) > 1

    def test_mean_with_ci(self):
        out = mean_with_ci([1.0, 0.0, 1.0, 1.0], ["a", "b", "c", "d"], 2000, 0)
        assert out["mean"] == 0.75
        assert out["n"] == 4 and out["n_questions"] == 4
        assert 0.0 <= out["ci95"][0] <= 0.75 <= out["ci95"][1] <= 1.0

    def test_mean_with_ci_empty(self):
        out = mean_with_ci([], [], 100, 0)
        assert out["mean"] is None and out["ci95"] is None


class TestPermutationBaseline:
    def test_permutes_only_within_strata(self):
        # y is constant within each stratum, so a within-stratum permutation can never change
        # it: every permuted statistic equals the observed one
        x = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=float)
        y = np.array([5, 5, 5, 7, 7, 7, 9, 9, 9], dtype=float)
        strata = ["p1"] * 3 + ["p2"] * 3 + ["p3"] * 3
        base = permutation_baseline(x, y, strata, spearman, n_perm=200, seed=0)
        assert base["mean"] == pytest.approx(1.0)
        assert base["p2_5"] == pytest.approx(1.0) and base["p97_5"] == pytest.approx(1.0)

    def test_null_centres_near_zero_for_spearman(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=200)
        y = x + rng.normal(size=200)
        base = permutation_baseline(x, y, ["s"] * 200, spearman, n_perm=2000, seed=0)
        assert abs(base["mean"]) < 0.02
        assert base["p2_5"] < 0 < base["p97_5"]

    def test_jaccard_baseline_reflects_flag_rates(self):
        # with flag rates p and q, independent flags give Jaccard pq / (p + q - pq)
        rng = np.random.default_rng(4)
        a = rng.random(2000) < 0.5
        b = rng.random(2000) < 0.3
        base = permutation_baseline(a, b, ["s"] * 2000, jaccard, n_perm=500, seed=0)
        p, q = a.mean(), b.mean()
        assert base["mean"] == pytest.approx(p * q / (p + q - p * q), abs=0.01)


class TestPairAgreement:
    def test_related_signals_above_chance(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=120)
        y = x + 0.5 * rng.normal(size=120)
        q = [f"q{i // 2}" for i in range(120)]
        strata = ["v1", "v2"] * 60
        out = pair_agreement(x, y, q, strata, spearman, n_resamples=1000, seed=0)
        assert out["value"] > 0.7
        assert out["above_chance"] is True
        assert out["ci95"][0] <= out["value"] <= out["ci95"][1]

    def test_unrelated_signals_not_above_chance(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=120)
        y = rng.normal(size=120)
        q = [f"q{i // 2}" for i in range(120)]
        strata = ["v1", "v2"] * 60
        out = pair_agreement(x, y, q, strata, spearman, n_resamples=1000, seed=0)
        assert out["above_chance"] is False


# --------------------------------------------------------------------------------------
# Q5 reading and sampling


class TestQ5Reading:
    def test_real_improvement(self):
        assert q5_reading([0.02, 0.10], [-0.2, -0.1]) == "real_improvement"

    def test_worse(self):
        assert q5_reading([-0.10, -0.01], None) == "worse"

    def test_apparent_only(self):
        assert q5_reading([-0.05, 0.04], [-0.15, -0.02]) == "apparent_only"

    def test_no_effect(self):
        assert q5_reading([-0.05, 0.04], [-0.05, 0.03]) == "no_effect"
        assert q5_reading([-0.05, 0.04], None) == "no_effect"

    def test_undetermined(self):
        assert q5_reading(None, [-0.2, -0.1]) == "undetermined"


class TestSeededSample:
    def test_independent_of_input_order(self):
        items = [{"trace_id": f"t{i:02d}"} for i in range(30)]
        a = seeded_sample(items, 5, seed=0)
        b = seeded_sample(list(reversed(items)), 5, seed=0)
        assert a == b and len(a) == 5

    def test_small_pool_returns_everything(self):
        items = [{"trace_id": "b"}, {"trace_id": "a"}]
        assert sorted(r["trace_id"] for r in seeded_sample(items, 15, seed=0)) == ["a", "b"]


class TestPredictionGuard:
    TEMPLATE = (
        "### Q1\n\n"
        "- P1.1 Sonnet, rho: [ ] >= 0.5 · [ ] 0.2 to 0.5 · [ ] below\n"
        "- P1.2 Qwen, rho: [ ] >= 0.5 · [ ] 0.2 to 0.5 · [ ] below\n"
        "- not a prediction line [ ]\n"
    )

    def test_unfilled_template_reports_each_line(self):
        assert prediction_problems(self.TEMPLATE) == [
            "P1.1: 0 options marked",
            "P1.2: 0 options marked",
        ]

    def test_filled(self):
        text = self.TEMPLATE.replace("[ ] >= 0.5", "[x] >= 0.5", 1).replace(
            "[ ] below\n- not", "[X] below\n- not"
        )
        assert prediction_problems(text) == []

    def test_two_marks_is_a_problem(self):
        text = self.TEMPLATE.replace("[ ]", "[x]")
        assert prediction_problems(text) == ["P1.1: 3 options marked", "P1.2: 3 options marked"]

    def test_no_prediction_lines(self):
        assert prediction_problems("# nothing here\n") == ["no prediction lines found"]
