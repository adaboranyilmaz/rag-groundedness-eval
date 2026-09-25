"""Unit tests for the deterministic Phase 5 metrics: figure extraction, numeric correctness
(tolerance, unit errors, answer-figure choice), abstention detection and answerability,
citation checks, agreement statistics, groundedness scoring and calibration."""

from __future__ import annotations

import math

import pytest

from src.evaluation.abstention import (
    answerability,
    final_status,
    has_decline_phrase,
    is_bare_token,
    needs_judge,
    rule_status,
)
from src.evaluation.agreement import (
    agreement_summary,
    bootstrap_mean_ci,
    cohen_kappa,
    confusion,
)
from src.evaluation.citations import (
    evidence_labels,
    judge_citation_metrics,
    numeric_support,
    quote_fidelity,
    quote_status,
)
from src.evaluation.correctness import grade_numeric, numeric_gold, question_hint
from src.evaluation.evaluate import calibration
from src.evaluation.figures import extract_figures
from src.evaluation.gold_spans import GoldSpan
from src.evaluation.groundedness import score

# --------------------------------------------------------------------------------------
# Figures


def numbers(text: str) -> list[float]:
    return [f.value for f in extract_figures(text)]


class TestFigures:
    def test_currency_with_scale_word(self):
        (f,) = extract_figures("Capex was $1,577 million in FY2018.")
        assert (f.number, f.scale, f.currency, f.explicit_scale) == (1577.0, 1e6, True, True)

    def test_scale_letter_only_on_currency(self):
        (f,) = extract_figures("revenue of ~$6,489M")
        assert f.value == 6489e6
        # "3M" without a currency sign is the company, not three million
        assert numbers("3M's capex rose") == []

    def test_percent_and_basis_points(self):
        (p,) = extract_figures("margin of 4.2%")
        assert p.percent and p.number == 4.2
        (b,) = extract_figures("up 50 bps")
        assert b.percent and math.isclose(b.number, 0.5)

    def test_skips_years_dates_labels_and_period_counts(self):
        text = "For the 12 months ended December 31, 2022 (FY2022, Q2, see C3 and the 10-K)"
        assert numbers(text) == []

    def test_keeps_days_as_a_quantity(self):
        assert numbers("DPO is approximately 61.33 days") == [61.33]

    def test_negative_signs(self):
        assert numbers("ROA was −1.53% and -3.7") == [-1.53, -3.7]
        # a hyphen inside a range or token is not a sign
        assert numbers("from 2019-2021") == []

    def test_parentheses_are_not_negative(self):
        assert numbers("FY2021 (15.0%)") == [15.0]

    def test_multiple_suffix(self):
        assert numbers("a ratio of 2.8x") == [2.8]

    def test_precision_as_written(self):
        (f,) = extract_figures("$1577.00")
        assert f.decimals == 2 and f.significant_digits == 6


# --------------------------------------------------------------------------------------
# Numeric correctness

Q_MILLIONS = "What is X's FY2018 capex? Answer in USD millions."
Q_PERCENT = "What is X's margin? Answer in units of percents and round to one decimal place."
Q_RATIO = "What is X's ROA? Round your answer to two decimal places."


class TestQuestionHint:
    @pytest.mark.parametrize(
        "question,scale",
        [
            ("Answer in USD millions.", 1e6),
            ("Answer in USD billions.", 1e9),
            ("in USD million", 1e6),
            ("How much did revenue grow?", None),
        ],
    )
    def test_scale(self, question, scale):
        assert question_hint(question).scale == scale

    def test_percent_and_decimals(self):
        h = question_hint(Q_PERCENT)
        assert h.percent and h.decimals == 1
        assert question_hint(Q_RATIO).decimals == 2
        assert question_hint("round to one decimal place)").decimals == 1


class TestNumericGold:
    def test_single_figure_gold_takes_units_from_question(self):
        g = numeric_gold("$1577.00", Q_MILLIONS)
        assert (g.value, g.unit, g.decimals, g.kind) == (1577.0, 1e6, 2, "currency")

    def test_hint_decimals_override_written(self):
        assert numeric_gold("0.8", Q_RATIO).decimals == 2

    def test_short_trailing_words_allowed(self):
        g = numeric_gold("$400,000,000 increase.", "By how much did it increase?")
        assert g.value == 4e8 and g.unit == 1.0

    def test_prose_gold_is_not_numeric(self):
        assert numeric_gold("Yes. Its net income was $3725 million, the highest.", "Q?") is None
        assert numeric_gold("North America, Latin America, Europe", "Q?") is None


class TestGradeNumeric:
    def grade(self, answer, gold, question, rel_tol=0.01):
        return grade_numeric(answer, numeric_gold(gold, question), rel_tol).label

    @pytest.mark.parametrize(
        "answer",
        [
            "$1,577 million",
            "$1.577 billion",
            "1,577",
            "$1,577,000,000",
            "approximately $1,580 million",
        ],
    )
    def test_equivalent_units_are_correct(self, answer):
        assert self.grade(answer, "$1577.00", Q_MILLIONS) == "correct"

    @pytest.mark.parametrize("answer", ["1,577,000", "$1,577 thousand", "$1,577 billion"])
    def test_wrong_unit_is_a_unit_error(self, answer):
        assert self.grade(answer, "$1577.00", Q_MILLIONS) == "unit_error"

    def test_tolerance(self):
        assert self.grade("$1,600 million", "$1577.00", Q_MILLIONS) == "incorrect"
        assert self.grade("$1,600 million", "$1577.00", Q_MILLIONS, rel_tol=0.05) == "correct"

    def test_percent_gold(self):
        assert self.grade("4.2%", "4.2%", Q_PERCENT) == "correct"
        assert self.grade("0.042", "4.2%", Q_PERCENT) == "correct"  # a fraction, same quantity
        assert self.grade("0.042%", "4.2%", Q_PERCENT) == "unit_error"
        assert self.grade("4.3%", "4.2%", Q_PERCENT) == "incorrect"

    def test_rounding_precision_for_small_golds(self):
        assert self.grade("0.0121", "0.01", Q_RATIO) == "correct"
        assert self.grade("0.016", "0.01", Q_RATIO) == "incorrect"

    def test_ratio_gold_answered_as_percent(self):
        assert self.grade("-1.53% (net income of -$546M / assets)", "-0.02", Q_RATIO) == "correct"
        assert self.grade("-0.02%", "-0.02", Q_RATIO) == "unit_error"

    def test_sign(self):
        assert self.grade("ROA declined by 0.02", "-0.02", Q_RATIO) == "correct"
        res = grade_numeric("0.02", numeric_gold("-0.02", Q_RATIO), 0.01)
        assert res.label == "incorrect" and res.primary.sign_note == "sign_mismatch"

    def test_zero_gold_has_no_unit_error(self):
        assert self.grade("0", "0", "Restructuring costs?") == "correct"
        assert self.grade("$0.3 million", "0", "Restructuring costs?") == "incorrect"

    def test_result_after_calculation_is_the_answer(self):
        assert (
            self.grade("$19,815 million − $13,997 million = $5,818 million", "$5818.00", Q_MILLIONS)
            == "correct"
        )
        assert self.grade("ratio = 5,121.3 / 7,491.5 = 0.68", "0.68", Q_RATIO) == "correct"

    def test_working_in_brackets_is_ignored(self):
        assert self.grade(
            "Approximately 24.26 (revenue of ~$6,489M / $267.5M)", "24.26", Q_RATIO
        ) == ("correct")

    def test_any_figure_sensitivity_and_no_figure(self):
        res = grade_numeric(
            "$12,257 million. Calculation: 9,068 before capex",
            numeric_gold("$9068.00", Q_MILLIONS),
            0.01,
        )
        assert res.label == "incorrect" and res.any_figure_label == "correct"
        none = grade_numeric("Not in the excerpts.", numeric_gold("$9068.00", Q_MILLIONS), 0.01)
        assert none.label is None and none.n_figures == 0


# --------------------------------------------------------------------------------------
# Abstention


def parsed(answer, token=False, status="ok"):
    return {"answer": answer, "abstained_token": token, "status": status}


class TestAbstention:
    def test_bare_tokens(self):
        assert is_bare_token("NONE") and is_bare_token("**NOT_IN_DOCUMENTS**")
        assert not is_bare_token("None of the segments declined")
        assert not needs_judge(parsed("NONE"))
        assert not needs_judge(parsed(None, status="failed"))
        assert needs_judge(parsed("Revenue was $5M."))

    @pytest.mark.parametrize(
        "answer,expected",
        [
            ("The provided excerpts do not contain 3M's net PP&E.", "declined"),
            ("Not stated in the provided excerpts", "declined"),
            ("Cannot be determined from the excerpts.", "declined"),
            (
                "Cannot be calculated from the excerpts; net earnings were $2,707.3 million.",
                "partial",
            ),
            ("Revenue was $5.2 billion in FY2022.", "answered"),
            # a fact about the company, not a statement about the documents
            ("The company did not provide guidance for 2023.", "answered"),
        ],
    )
    def test_rule(self, answer, expected):
        assert rule_status(parsed(answer)) == expected

    def test_decline_phrase_variants(self):
        assert has_decline_phrase("The excerpts do not identify which securities are listed.")
        assert has_decline_phrase("No relevant data on JPM is provided in these excerpts.")
        assert not has_decline_phrase("The excerpts show no acquisitions in fiscal 2023.")

    def test_final_status_precedence(self):
        assert final_status(parsed(None, token=True, status="failed"), "answered") == (
            "declined",
            "token",
        )
        assert final_status(parsed(None, status="failed"), None) == ("unparsed", "parse")
        assert final_status(parsed("NONE"), "answered") == ("declined", "token")
        assert final_status(parsed("Revenue was $5M."), "partial") == ("partial", "judge")
        assert final_status(parsed("Revenue was $5M."), None) == ("answered", "rule")

    def test_answerability(self):
        m = {"recall@5": 0.5}
        assert answerability("oracle", True, m, 5) == "answerable"
        assert answerability("retrieved", False, None, 5) == "unanswerable"
        assert answerability("retrieved", True, None, 5) == "no_gold"
        assert answerability("retrieved", True, m, 5) == "evidence_present"
        assert answerability("retrieved", True, {"recall@5": 0.0}, 5) == "evidence_absent"
        # the audited out-of-corpus questions another filing answers
        assert (
            answerability("retrieved", False, None, 5, answerable_elsewhere=True)
            == "answerable_elsewhere"
        )
        # the flag only matters for out-of-corpus questions
        assert answerability("retrieved", True, m, 5, answerable_elsewhere=True) == (
            "evidence_present"
        )

    def test_audit_applies_only_once_approved(self, tmp_path):
        import json

        from src.evaluation.evaluate import load_answerable_elsewhere

        path = tmp_path / "audit.json"
        assert load_answerable_elsewhere(path) == frozenset()  # no audit
        audit = {
            "meta": {"status": "proposed"},
            "items": [
                {"financebench_id": "q1", "status": "answerable_elsewhere"},
                {"financebench_id": "q2", "status": "unanswerable"},
            ],
        }
        path.write_text(json.dumps(audit), encoding="utf-8")
        assert load_answerable_elsewhere(path) == frozenset()
        audit["meta"]["status"] = "approved"
        path.write_text(json.dumps(audit), encoding="utf-8")
        assert load_answerable_elsewhere(path) == frozenset({"q1"})


# --------------------------------------------------------------------------------------
# Citations


def chunk(label, text, start=0, doc="D"):
    return {
        "label": label,
        "chunk_id": f"{doc}:{label}",
        "doc_id": doc,
        "char_start": start,
        "char_end": start + 100,
        "text": text,
    }


class TestCitations:
    def test_evidence_labels_use_phase3_relevance(self):
        chunks = [chunk("C1", "a", 0), chunk("C2", "b", 1000), chunk("C3", "c", 0, doc="E")]
        golds = [GoldSpan("D", 20, 200)]  # covers 80% of C1, none of C2; C3 is another doc
        assert evidence_labels(chunks, golds, 0.5) == ["C1"]

    def test_judge_citation_metrics(self):
        claims = [
            {"kind": "document"},
            {"kind": "document"},
            {"kind": "document"},
            {"kind": "context"},
        ]
        verdicts = [
            {"verdict": "supported", "supporting_excerpts": ["C1"]},
            {"verdict": "supported", "supporting_excerpts": ["C3"]},
            {"verdict": "unsupported", "supporting_excerpts": []},
            # "the excerpts do not give X" is supported by every excerpt; it must not count
            {"verdict": "supported", "supporting_excerpts": ["C1", "C2", "C3"]},
        ]
        m = judge_citation_metrics(["C1", "C2"], claims, verdicts)
        assert m["citation_precision"] == 0.5  # C2 supports no document claim
        assert m["citation_recall"] == 0.5  # the C3-supported claim is uncited
        assert judge_citation_metrics([], claims, verdicts)["citation_precision"] is None

    def test_quote_status(self):
        chunks = [
            chunk("C1", "Total current assets | | 7,453 | | 7,659 |"),
            chunk("C2", "Net sales increased 5% driven by strong demand in all regions."),
        ]
        assert quote_status("total current assets 7,453  7,659", "C1", chunks) == "exact"
        assert quote_status("Net sales increased 5% driven by strong demand", "C1", chunks) == (
            "other_excerpt"
        )
        # a dropped word is a transcription slip, not a fabrication
        assert quote_status(
            "Net sales increased 5% driven by demand in all regions", "C2", chunks
        ) == ("near")
        # an inserted word is too much to call it the same quote (84% coverage < 90%)
        assert quote_status(
            "Net sales increased 5 percent driven by strong demand", "C2", chunks
        ) == ("not_found")
        assert quote_status("Revenue tripled", "C2", chunks) == "not_found"
        f = quote_fidelity([{"label": "C1", "text": "Total current assets"}], chunks)
        assert f["faithful_rate"] == 1.0

    def test_numeric_support(self):
        chunks = [chunk("C1", "Total revenue | 7,453 | 6,900"), chunk("C2", "Margin was 12.5%")]
        ns = numeric_support(
            "Revenue was $7.45 billion, margin 12.5%, up 8.0%.", chunks[:1], chunks
        )
        assert ns["n_figures"] == 3
        assert ns["n_in_cited"] == 1  # 7.45bn rescales to 7,453
        assert ns["n_in_context"] == 2  # 12.5% is in the uncited C2; 8.0% is derived
        # one significant digit is too ambiguous to check
        assert numeric_support("up 2 points", chunks, chunks)["n_figures"] == 0


# --------------------------------------------------------------------------------------
# Agreement, groundedness, calibration


class TestAgreement:
    def test_kappa_textbook(self):
        # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> p_o 0.7, p_e 0.5, kappa 0.4
        a = ["y"] * 25 + ["n"] * 25
        b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
        assert math.isclose(cohen_kappa(a, b), 0.4)

    def test_kappa_edge_cases(self):
        assert cohen_kappa(["a", "b"], ["a", "b"]) == 1.0
        assert cohen_kappa(["a", "a"], ["a", "a"]) is None  # p_e = 1: undefined
        with pytest.raises(ValueError):
            cohen_kappa(["a"], ["a", "b"])

    def test_confusion_and_summary(self):
        a, b = ["s", "s", "u", "c"], ["s", "u", "u", "c"]
        assert confusion(a, b, ["s", "u", "c"])["s"] == {"s": 1, "u": 1, "c": 0}
        s = agreement_summary(a, b, ["s", "u", "c"], clusters=[1, 1, 2, 2], n_resamples=200)
        assert s["n"] == 4 and s["n_clusters"] == 2 and s["raw_agreement"] == 0.75
        lo, hi = s["kappa_ci95"]
        assert -1.0 <= lo <= hi <= 1.0

    def test_bootstrap_mean_ci_is_seeded(self):
        vals = [0.0, 1.0, 1.0, 0.0, 1.0]
        assert bootstrap_mean_ci(vals, 500, 3) == bootstrap_mean_ci(vals, 500, 3)
        lo, hi = bootstrap_mean_ci(vals, 500, 3)
        assert 0.0 <= lo <= 0.6 <= hi <= 1.0


class TestGroundedness:
    def test_only_document_claims_are_scored(self):
        claims = [
            {"kind": "document"},
            {"kind": "document"},
            {"kind": "general"},
            {"kind": "context"},
        ]
        verdicts = [
            {"verdict": "supported"},
            {"verdict": "contradicted"},
            {"verdict": "unsupported"},
            {"verdict": "contradicted"},
        ]
        s = score(claims, verdicts)
        assert s["groundedness"] == 0.5
        assert s["fully_grounded"] is False and s["any_contradicted"] is True
        assert s["n_general_claims"] == 1 and s["n_context_claims"] == 1
        assert s["n_context_contradicted"] == 1  # a false claim that information is missing

    def test_a_bare_decline_is_not_grounded(self):
        s = score([{"kind": "context"}], [{"verdict": "supported"}])
        assert s["groundedness"] is None and s["n_document_claims"] == 0

    def test_no_document_claims_is_undefined(self):
        s = score([{"kind": "general"}], [{"verdict": "supported"}])
        assert s["groundedness"] is None and s["fully_grounded"] is None


class TestCalibration:
    def test_ece(self):
        # bin 90-100: stated 95 twice, right once -> |0.5 - 0.95| = 0.45, weight 2/3
        # bin 50-60: stated 50, right -> |1.0 - 0.5| = 0.5, weight 1/3
        c = calibration([(95, True), (95, False), (50, True)], 10)
        assert math.isclose(c["ece"], 2 / 3 * 0.45 + 1 / 3 * 0.5)
        assert c["bins"][9]["n"] == 2 and c["bins"][5]["n"] == 1
        assert calibration([(100, True)], 10)["bins"][9]["n"] == 1  # 100 is in the top bin
