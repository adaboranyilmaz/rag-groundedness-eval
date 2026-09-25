"""Unit tests for the adversarial set's validation and review (src/evaluation/adversarial.py),
on synthetic filings."""

from __future__ import annotations

from src.evaluation.adversarial import (
    parse_review,
    quota_problems,
    record_problems,
    render_review,
)
from src.evaluation.gold_spans import NormalisedText

RAW = {
    "ACME_2022_10K": (
        "Revenue was $10 million in 2022.\nTotal revenue | 10 | 8 |\nTotal revenue | 10 | 8 |"
    ),
    "ACME_2018_10K": "Revenue was $6 million in 2018.",
    "OTHER_2022_10K": "Other Co. makes widgets.",
}
DOCS = {d: NormalisedText(t) for d, t in RAW.items()}


def rec(**kw) -> dict:
    base = {
        "id": "adv_a01",
        "category": "a",
        "subtype": "single_filing",
        "question": "What was ACME's 2022 revenue?",
        "expected_behaviour": "answer",
        "answer": "$10 million",
        "gold_evidence": [{"doc_id": "ACME_2022_10K", "text": "Revenue was $10 million in 2022."}],
        "reference_evidence": [],
        "absence_checks": [],
        "note": "",
        "status": "candidate",
    }
    return {**base, **kw}


class TestRecordProblems:
    def test_valid(self):
        assert record_problems(rec(), DOCS, RAW) == []

    def test_evidence_not_in_filing(self):
        r = rec(gold_evidence=[{"doc_id": "ACME_2022_10K", "text": "Revenue was $12 million."}])
        assert any("not found exactly" in p for p in record_problems(r, DOCS, RAW))

    def test_repeated_gold_evidence(self):
        r = rec(gold_evidence=[{"doc_id": "ACME_2022_10K", "text": "Total revenue | 10 | 8 |"}])
        assert any("2 occurrences" in p for p in record_problems(r, DOCS, RAW))

    def test_repeated_reference_evidence_is_allowed(self):
        r = rec(
            id="adv_d01",
            category="d",
            subtype="premise_contradicted",
            expected_behaviour="reject_premise",
            gold_evidence=[],
            reference_evidence=[{"doc_id": "ACME_2022_10K", "text": "Total revenue | 10 | 8 |"}],
        )
        assert record_problems(r, DOCS, RAW) == []

    def test_two_filings_need_two_filings(self):
        r = rec(category="b", subtype="two_filings")
        assert any("fewer than two filings" in p for p in record_problems(r, DOCS, RAW))
        r["gold_evidence"].append(
            {"doc_id": "ACME_2018_10K", "text": "Revenue was $6 million in 2018."}
        )
        assert record_problems(r, DOCS, RAW) == []

    def test_absence_check_hit_and_scope(self):
        base = dict(
            id="adv_c01",
            category="c",
            subtype="not_disclosed",
            expected_behaviour="decline",
            gold_evidence=[],
        )
        ok = rec(**base, absence_checks=[{"pattern": "gadgets", "scope": "corpus", "flags": "i"}])
        assert record_problems(ok, DOCS, RAW) == []
        hit = rec(**base, absence_checks=[{"pattern": "WIDGETS", "scope": "OTHER", "flags": "i"}])
        assert any("hits ['OTHER_2022_10K']" in p for p in record_problems(hit, DOCS, RAW))
        # the same pattern is absent from a scope that excludes the filing holding it
        scoped = rec(**base, absence_checks=[{"pattern": "widgets", "scope": "ACME", "flags": "i"}])
        assert record_problems(scoped, DOCS, RAW) == []
        bad_scope = rec(**base, absence_checks=[{"pattern": "x", "scope": "NOPE", "flags": "i"}])
        assert any("matches no filing" in p for p in record_problems(bad_scope, DOCS, RAW))

    def test_unanswerable_needs_an_absence_check_and_no_gold(self):
        r = rec(category="c", subtype="not_disclosed", expected_behaviour="decline")
        problems = record_problems(r, DOCS, RAW)
        assert any("gold evidence on an unanswerable" in p for p in problems)
        assert any("without an absence check" in p for p in problems)

    def test_category_consistency(self):
        r = rec(subtype="two_filings", expected_behaviour="decline")
        problems = record_problems(r, DOCS, RAW)
        assert any("subtype" in p for p in problems)
        assert any("expected_behaviour" in p for p in problems)


def test_quota():
    records = []
    for cat, subs in (
        ("a", ["single_filing"] * 10),
        ("b", ["two_filings"] * 10),
        ("c", ["not_disclosed"] * 5 + ["filing_not_in_corpus"] * 5),
        ("d", ["premise_contradicted"] * 6 + ["entity_does_not_exist"] * 4),
    ):
        records += [{"category": cat, "subtype": s, "status": "approved"} for s in subs]
    records.append({"category": "a", "subtype": "single_filing", "status": "rejected"})
    assert quota_problems(records) == [
        "(d) premise_contradicted: 6 approved, need 5",
        "(d) entity_does_not_exist: 4 approved, need 5",
    ]


class TestReview:
    RECORDS = [
        rec(),
        rec(
            id="adv_c01",
            category="c",
            subtype="not_disclosed",
            expected_behaviour="decline",
            gold_evidence=[],
            absence_checks=[{"pattern": "gadgets", "scope": "corpus", "flags": "i"}],
        ),
    ]

    def test_round_trip(self):
        text = render_review(self.RECORDS)
        assert "Decision adv_a01: [ ] approve · [ ] reject" in text
        filled = text.replace(
            "Decision adv_a01: [ ] approve", "Decision adv_a01: [x] approve"
        ).replace("· [ ] reject\n- Comment adv_c01:", "· [X] reject\n- Comment adv_c01: too easy")
        decisions, comments, problems = parse_review(filled)
        assert problems == []
        assert decisions == {"adv_a01": "approved", "adv_c01": "rejected"}
        assert comments == {"adv_c01": "too easy"}

    def test_unmarked_and_double_marked(self):
        text = render_review(self.RECORDS).replace(
            "Decision adv_c01: [ ] approve · [ ] reject",
            "Decision adv_c01: [x] approve · [x] reject",
        )
        _, _, problems = parse_review(text)
        assert problems == ["adv_a01: 0 options marked", "adv_c01: 2 options marked"]

    def test_display_cleans_filing_whitespace(self):
        r = rec(gold_evidence=[{"doc_id": "ACME_2022_10K", "text": "a\xa0b\n\n​\nc"}])
        assert "```\na b\nc\n```" in render_review([r])


# --------------------------------------------------------------------------------------
# scripts/07b_adversarial_run.py: the parts that decide without the judge


def load_run_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "07b_adversarial_run.py"
    spec = importlib.util.spec_from_file_location("adversarial_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRunScript:
    def test_premise_rule(self):
        run = load_run_script()
        assert run.premise_rule({"answer": None}) == "unparsed"
        assert run.premise_rule({"answer": "NOT_IN_DOCUMENTS"}) == "declines_without_addressing"
        assert run.premise_rule({"answer": "Sales fell 3.2%."}) is None  # the judge decides

    def test_trace_question(self):
        run = load_run_script()
        c = rec(
            id="adv_c07",
            category="c",
            subtype="filing_not_in_corpus",
            expected_behaviour="decline",
            gold_evidence=[],
        )
        q = run.trace_question(c)
        assert q["financebench_id"] == "adv_c07" and q["in_corpus"] is False
        assert q["question_type"] == "adversarial_c" and q["doc_name"] is None
        assert q["adversarial"]["expected_behaviour"] == "decline"
        assert run.trace_question(rec())["doc_name"] == "ACME_2022_10K"


def test_premise_page_hides_what_it_should():
    from src.evaluation.labeling_page import render_premise_page

    page = render_premise_page(
        [{"label_id": "abc", "question": "Q?", "premise_note": "False: x.", "answer": "A </b>"}],
        "f" * 64,
    )
    assert '"label_id": "abc"' in page and "trace_id" not in page
    assert "<\/b>" in page  # filing text cannot close the script element
