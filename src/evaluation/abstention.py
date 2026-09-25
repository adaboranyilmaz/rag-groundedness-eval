"""Abstention: did the model decline, and was declining right?

Detection. Three signals are recorded per trace, and the final status is resolved from them:
  token  the parser found the fixed NOT_IN_DOCUMENTS token, or the whole ANSWER is a bare
         token such as NONE or N/A (the 3B writes NONE, borrowed from the CITATIONS line)
  rule   a phrase detector for free-text declines ("the excerpts do not contain...",
         "cannot be calculated..."); a decline phrase alongside a figure or a yes/no
         conclusion reads as partial. A deliberately simple, judge-free cross-check.
  judge  the claim-decomposition judge's response type (answered / partial / declined)
Final status: `declined` when the ANSWER is a bare token or the parse failed with the token
present; `unparsed` when the parse failed without it (no ANSWER field at all); otherwise the
judge's type when available, else the rule's. Rule-vs-judge agreement is reported, so the
choice of the judge as the primary signal is checkable.

Answerability, per trace (see DECISIONS.md, Phase 5):
  unanswerable      the question's filing is not in the corpus (retrieved condition)
  answerable        oracle condition: the gold evidence is guaranteed to be in the context
  evidence_present  retrieved, in corpus, >=1 gold span hit in the top k (recall@k > 0)
  evidence_absent   retrieved, in corpus, no gold span hit. Not scored as right or wrong:
                    missing the aligned gold span does not prove the context lacks the answer
  no_gold           retrieved, in corpus, no aligned gold evidence to check against
"""

from __future__ import annotations

import re

from src.evaluation.figures import extract_figures

STATUSES = ("answered", "partial", "declined", "unparsed")

_BARE_TOKEN_RE = re.compile(r"^\W*(?:NOT[_ ]IN[_ ]DOCUMENTS|NONE|N/?A|UNKNOWN)\W*$", re.IGNORECASE)
_DOC = (
    r"(?:excerpts?|documents?|context|filings?|passages?|sources?|materials?|text|"
    r"data provided|information provided|provided (?:information|data))"
)
_DECLINE_RES = [
    re.compile(
        rf"\b{_DOC}\b[^.;\n]{{0,80}}?\b(?:do|does|did)\s*(?:not|n't)\s+(?:\w+\s+)?"
        r"(?:contain|provide|include|mention|specify|state|disclose|report|show|give|present|"
        r"have|list|break\s+out|offer|cover|supply|identify|describe|discuss|indicate|detail|"
        r"address|reference|explain|reveal)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bnot\s+(?:\w+\s+)?(?:provided|included|available|mentioned|stated|disclosed|specified|"
        r"found|contained|reported|given|shown|listed|present|detailed|broken\s+out)\s+"
        rf"(?:in|within|among)\s+(?:the\s+|these\s+|any\s+(?:of\s+the\s+)?)?(?:provided\s+|given\s+|"
        rf"available\s+)?{_DOC}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:cannot|can\s*not|can't|unable\s+to|not\s+possible\s+to|impossible\s+to)\s+"
        r"(?:be\s+)?(?:fully\s+|reliably\s+|accurately\s+|precisely\s+|definitively\s+)?"
        r"(?:determine|determined|calculate|calculated|compute|computed|answer|answered|derive|"
        r"derived|identify|identified|assess|assessed|confirm|confirmed|verify|verified|"
        r"quantify|quantified)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:insufficient|not\s+enough)\s+(?:\w+\s+){0,2}(?:information|data|details?)\b", re.I
    ),
    re.compile(
        r"\bno\s+(?:\w+\s+){0,2}(?:information|data|details?|figures?)\b[^.;\n]{0,60}?"
        rf"\b(?:provided|available|given|included|in\s+(?:the|these)\s+{_DOC})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:information|data)\s+(?:is|are)\s+(?:not\s+available|unavailable|missing)\b", re.I
    ),
]
_YES_NO_RE = re.compile(r"^\W*(?:yes|no)\b", re.IGNORECASE)


def is_bare_token(answer: str | None) -> bool:
    return answer is not None and bool(_BARE_TOKEN_RE.match(answer))


def has_decline_phrase(answer: str) -> bool:
    return any(r.search(answer) for r in _DECLINE_RES)


def rule_status(parsed: dict) -> str:
    answer = parsed.get("answer")
    if answer is None:
        return "declined" if parsed.get("abstained_token") else "unparsed"
    if is_bare_token(answer):
        return "declined"
    decline = parsed.get("abstained_token") or has_decline_phrase(answer)
    if not decline:
        return "answered"
    substantive = bool(extract_figures(answer)) or bool(_YES_NO_RE.match(answer))
    return "partial" if substantive else "declined"


def final_status(parsed: dict, judge_type: str | None) -> tuple[str, str]:
    """(status, source). Hard facts about the output first, then the judge, then the rule."""
    answer = parsed.get("answer")
    if answer is None:
        return ("declined", "token") if parsed.get("abstained_token") else ("unparsed", "parse")
    if is_bare_token(answer):
        return "declined", "token"
    if judge_type is not None:
        return judge_type, "judge"
    return rule_status(parsed), "rule"


def needs_judge(parsed: dict) -> bool:
    """Whether the output has answer text for the judge to decompose."""
    answer = parsed.get("answer")
    return answer is not None and not is_bare_token(answer)


def answerability(condition: str, in_corpus: bool, context_metrics: dict | None, k: int) -> str:
    if condition == "oracle":
        return "answerable"
    if not in_corpus:
        return "unanswerable"
    if context_metrics is None:
        return "no_gold"
    return "evidence_present" if context_metrics[f"recall@{k}"] > 0 else "evidence_absent"
