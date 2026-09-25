"""The adversarial question set (Phase 6, Q4): validation, and the author's review.

`data/adversarial/questions.jsonl` holds every candidate with its status (`candidate`,
`approved`, `rejected`). Each record carries what makes it checkable without trusting the
person who drafted it:
  gold_evidence       (a, b) passages that answer the question; each must align exactly and
                      once in its filing, as FinanceBench evidence does (src/evaluation/
                      gold_spans.py), so retrieval metrics and the oracle context apply
  reference_evidence  (c, d) passages that show why a question is unanswerable or its
                      premise false; shown to the reviewer and the labeller, never used as
                      gold
  absence_checks      regexes that must have no hit in their scope (the corpus, or filings
                      whose id starts with the scope): the unanswerable fact, the invented
                      segment
The approved set is 10 questions per category, and for (c) and (d) 5 per subtype
(results/metrics/phase6_preregistration.md, Part A, Q4).
"""

from __future__ import annotations

import re
from collections import Counter

from src.evaluation.gold_spans import NormalisedText, align_evidence

CATEGORIES = {
    "a": ("single_filing",),
    "b": ("two_filings",),
    "c": ("not_disclosed", "filing_not_in_corpus"),
    "d": ("premise_contradicted", "entity_does_not_exist"),
}
EXPECTED = {"a": "answer", "b": "answer", "c": "decline", "d": "reject_premise"}
PER_CATEGORY = 10
STATUSES = ("candidate", "approved", "rejected")


def record_problems(rec: dict, docs: dict[str, NormalisedText], raw: dict[str, str]) -> list[str]:
    """Everything wrong with one record; empty when it is valid. `docs` maps doc id to the
    normalised full text (for alignment), `raw` to the full text (for absence checks)."""
    out = []
    cat = rec.get("category")
    if cat not in CATEGORIES:
        return [f"unknown category {cat!r}"]
    if rec.get("subtype") not in CATEGORIES[cat]:
        out.append(f"subtype {rec.get('subtype')!r} not valid for ({cat})")
    if rec.get("expected_behaviour") != EXPECTED[cat]:
        out.append(f"expected_behaviour should be {EXPECTED[cat]!r}")
    if rec.get("status") not in STATUSES:
        out.append(f"unknown status {rec.get('status')!r}")
    gold = rec.get("gold_evidence") or []
    if cat in ("a", "b") and not gold:
        out.append("no gold evidence")
    if cat in ("c", "d") and gold:
        out.append("gold evidence on an unanswerable or false-premise question")
    if cat == "b" and len({e["doc_id"] for e in gold}) < 2:
        out.append("two-filing question with evidence from fewer than two filings")
    for kind, unique in (("gold_evidence", True), ("reference_evidence", False)):
        for ev in rec.get(kind) or []:
            doc = docs.get(ev["doc_id"])
            if doc is None:
                out.append(f"{kind}: {ev['doc_id']} not in the corpus")
                continue
            al = align_evidence(ev["text"], ev["doc_id"], doc)
            if al.status != "exact":
                out.append(f"{kind}: not found exactly in {ev['doc_id']} ({al.status})")
            elif unique and al.n_occurrences != 1:
                out.append(f"{kind}: {al.n_occurrences} occurrences in {ev['doc_id']}")
    if cat == "c" and not rec.get("absence_checks"):
        out.append("unanswerable question without an absence check")
    for chk in rec.get("absence_checks") or []:
        rx = re.compile(chk["pattern"], re.IGNORECASE if "i" in chk.get("flags", "") else 0)
        scope = chk["scope"]
        hits = [
            d for d, t in raw.items() if (scope == "corpus" or d.startswith(scope)) and rx.search(t)
        ]
        if not any(scope == "corpus" or d.startswith(scope) for d in raw):
            out.append(f"absence check scope {scope!r} matches no filing")
        if hits:
            out.append(f"absence check {chk['pattern']!r} hits {hits}")
    return out


def quota_problems(records: list[dict]) -> list[str]:
    """The approved set's composition against the pre-registered design."""
    approved = [r for r in records if r["status"] == "approved"]
    out = []
    by_cat = Counter(r["category"] for r in approved)
    for cat, subtypes in CATEGORIES.items():
        if by_cat[cat] != PER_CATEGORY:
            out.append(f"({cat}): {by_cat[cat]} approved, need {PER_CATEGORY}")
        if len(subtypes) == 2:
            by_sub = Counter(r["subtype"] for r in approved if r["category"] == cat)
            for s in subtypes:
                if by_sub[s] != PER_CATEGORY // 2:
                    out.append(f"({cat}) {s}: {by_sub[s]} approved, need {PER_CATEGORY // 2}")
    return out


# --------------------------------------------------------------------------------------
# The review file: one block per candidate, the decision marked with [x]

_DECISION_RE = re.compile(r"^- Decision (adv_\w+):(.*)$", re.MULTILINE)
_COMMENT_RE = re.compile(r"^- Comment (adv_\w+):[ \t]*(.*)$", re.MULTILINE)
_MARK_RE = re.compile(r"\[([ xX])\]\s*(approve|reject)")


def _display(text: str) -> str:
    """Evidence as a reader sees it: NBSP and zero-width spaces as spaces, no blank lines."""
    text = text.replace("\xa0", " ").replace("​", "")
    return "\n".join(ln.rstrip() for ln in text.splitlines() if ln.strip())


def render_review(records: list[dict]) -> str:
    out = [
        "# Adversarial set: candidate review",
        "",
        "Approve **10 per category**; for (c) and (d), **5 of each subtype**. Mark one option "
        "per decision line with `[x]`. To change a question's wording or answer, approve or "
        "reject it and say what to change on its comment line; changes are applied and "
        "re-validated before anything is generated.",
        "",
        "Every candidate below has passed validation: gold evidence occurs exactly once in its "
        "filing, and every absence check has no hit in its scope.",
    ]
    names = {
        "a": "(a) answerable from one filing",
        "b": "(b) answerable only by combining two filings",
        "c": "(c) plausible but unanswerable from the corpus",
        "d": "(d) false premise",
    }
    for cat in CATEGORIES:
        out += ["", f"## {names[cat]}"]
        for r in (r for r in records if r["category"] == cat):
            out += [
                "",
                f"### {r['id']} · {r['subtype'].replace('_', ' ')}",
                "",
                f"**Question:** {r['question']}",
                "",
                f"**Expected:** {r['expected_behaviour'].replace('_', ' ')} — {r['answer']}",
            ]
            for kind, label in (("gold_evidence", "Evidence"), ("reference_evidence", "Reference")):
                for ev in r.get(kind) or []:
                    out += [
                        "",
                        f"**{label}** ({ev['doc_id']}):",
                        "",
                        "```",
                        _display(ev["text"]),
                        "```",
                    ]
            for chk in r.get("absence_checks") or []:
                out += ["", f"**Absent:** `{chk['pattern']}` in {chk['scope']} (0 hits)"]
            if r.get("note"):
                out += ["", f"**Note:** {r['note']}"]
            out += [
                "",
                f"- Decision {r['id']}: [ ] approve · [ ] reject",
                f"- Comment {r['id']}:",
            ]
    return "\n".join(out) + "\n"


def parse_review(text: str) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """(decisions, comments, problems). A decision line must have exactly one mark."""
    decisions, problems = {}, []
    for qid, rest in _DECISION_RE.findall(text):
        marked = [d for m, d in _MARK_RE.findall(rest) if m.lower() == "x"]
        if len(marked) != 1:
            problems.append(f"{qid}: {len(marked)} options marked")
        else:
            decisions[qid] = "approved" if marked[0] == "approve" else "rejected"
    comments = {qid: c.strip() for qid, c in _COMMENT_RE.findall(text) if c.strip()}
    return decisions, comments, problems
