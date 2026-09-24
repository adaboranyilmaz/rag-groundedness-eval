"""Parse a model's raw output into the fields the prompt asked for.

The parser records what the output says; it does not judge it. Deciding whether a
free-text reply ("the excerpts do not state...") is an abstention, whether a quote really
appears in the chunk it cites, or whether the answer is right are Phase 5 metrics. Here
`abstained_token` only records whether the fixed `NOT_IN_DOCUMENTS` token was used.

Every deviation from the requested format is listed in `issues`, and `status` is:
  ok       every field the prompt asked for is present and well-formed
  partial  ANSWER was found but something else is missing or malformed
  failed   no ANSWER field; `answer` is None and the raw output stays in the trace
Nothing is silently repaired: a malformed confidence is kept verbatim in
`confidence_raw` and parsed as None, never coerced.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

FIELDS = ("REASONING", "ANSWER", "CITATIONS", "QUOTES", "CONFIDENCE")
ABSTAIN_TOKEN = "NOT_IN_DOCUMENTS"

# A field label at the start of a line, tolerating markdown emphasis or headings around
# it: "ANSWER:", "**ANSWER:**", "**ANSWER**:", "## Answer:". Case-insensitive. A label
# mid-line ("my final answer: ...") is deliberately not a field boundary.
_LABEL_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?[*_]{0,2}[ \t]*(" + "|".join(FIELDS) + r")[ \t]*[*_]{0,2}[ \t]*:"
    r"[ \t]*[*_]{0,2}",
    re.IGNORECASE | re.MULTILINE,
)
# A label written straight after another label on the same line ("ANSWER: CONFIDENCE: 80")
# means the first field was left empty. Matched only at the very start of a field's content.
_CHAINED_LABEL_RE = re.compile(
    r"[ \t]*[*_]{0,2}[ \t]*(" + "|".join(FIELDS) + r")[ \t]*[*_]{0,2}[ \t]*:[ \t]*[*_]{0,2}",
    re.IGNORECASE,
)
_CITE_RE = re.compile(r"\bC(\d+)\b", re.IGNORECASE)
_QUOTE_LINE_RE = re.compile(r"^[ \t]*[-*]?[ \t]*\[?C(\d+)\]?[ \t]*:[ \t]*(.+?)[ \t]*$", re.I)
_INT_RE = re.compile(r"^\s*(\d{1,3})\s*%?\s*(?:/\s*100)?\s*$")
_ABSTAIN_RE = re.compile(r"\bNOT[_ ]IN[_ ]DOCUMENTS\b", re.IGNORECASE)
_NONE_RE = re.compile(r"^\s*[*_]*\s*(none|n/?a)\s*[*_.]*\s*$", re.IGNORECASE)
_QUOTE_CHARS = "\"'“”‘’"


@dataclass
class ParsedOutput:
    status: str  # ok | partial | failed
    answer: str | None
    citations: list[str] = field(default_factory=list)  # valid labels, first-seen order
    invalid_citations: list[str] = field(default_factory=list)  # labels outside C1..Ck
    quotes: list[dict[str, str]] = field(default_factory=list)  # [{"label", "text"}]
    confidence: int | None = None
    confidence_raw: str | None = None
    reasoning: str | None = None
    abstained_token: bool = False
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _sections(text: str) -> tuple[dict[str, str], list[str]]:
    """Split output into labelled sections. A repeated label keeps its last occurrence
    (the final output block follows any reasoning) and is reported as an issue."""
    matches = list(_LABEL_RE.finditer(text))
    sections: dict[str, str] = {}
    issues: list[str] = []
    for i, m in enumerate(matches):
        label = m.group(1).upper()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[m.end() : end]
        pieces = []
        while chained := _CHAINED_LABEL_RE.match(content):
            pieces.append((label, ""))
            issues.append(f"empty_{label.lower()}_chained")
            label, content = chained.group(1).upper(), content[chained.end() :]
        pieces.append((label, content.strip()))
        for lab, body in pieces:
            if lab in sections:
                issues.append(f"duplicate_{lab.lower()}")
            sections[lab] = body
    if matches and text[: matches[0].start()].strip():
        issues.append("text_before_first_field")
    return sections, issues


def _strip_emphasis(s: str) -> str:
    return s.strip().strip("*_").strip()


def parse_output(text: str, output_fields: tuple[str, ...], n_chunks: int) -> ParsedOutput:
    sections, issues = _sections(text)
    requested = set(output_fields)
    for extra in sorted(set(sections) - requested):
        issues.append(f"unrequested_{extra.lower()}")

    if "ANSWER" not in sections or not _strip_emphasis(sections["ANSWER"]):
        # No answer field, but the fixed token may still have been used (e.g. written
        # without its ANSWER label): record that, since the token's presence is a fact
        # about the output, not a repair of it.
        return ParsedOutput(
            status="failed",
            answer=None,
            abstained_token=bool(_ABSTAIN_RE.search(text)),
            issues=issues + ["missing_answer"],
        )

    out = ParsedOutput(status="ok", answer=_strip_emphasis(sections["ANSWER"]))
    out.abstained_token = bool(_ABSTAIN_RE.search(out.answer))

    # CITATIONS: labels C<n>; "NONE" means an explicit empty list.
    if "CITATIONS" in sections:
        body = sections["CITATIONS"]
        seen: list[str] = []
        for n in _CITE_RE.findall(body):
            label = f"C{int(n)}"
            if label not in seen:
                seen.append(label)
        out.citations = [c for c in seen if 1 <= int(c[1:]) <= n_chunks]
        out.invalid_citations = [c for c in seen if c not in out.citations]
        if out.invalid_citations:
            issues.append("citation_out_of_range")
        if not body.strip():
            if "empty_citations_chained" not in issues:
                issues.append("empty_citations")
        elif not seen and not _NONE_RE.match(body):
            issues.append("citations_unparseable")
    elif "CITATIONS" in requested:
        issues.append("missing_citations")

    # QUOTES: one "C<n>: "text"" per line; continuation lines join the previous quote.
    if "QUOTES" in sections:
        for line in sections["QUOTES"].splitlines():
            m = _QUOTE_LINE_RE.match(line)
            if m:
                out.quotes.append({"label": f"C{int(m.group(1))}", "text": m.group(2)})
            elif line.strip() and out.quotes:
                out.quotes[-1]["text"] += "\n" + line.strip()
        for q in out.quotes:
            q["text"] = q["text"].strip().strip(_QUOTE_CHARS).strip()
        if not out.quotes and not _NONE_RE.match(sections["QUOTES"] or "none"):
            issues.append("quotes_unparseable")
        uncited = sorted({q["label"] for q in out.quotes} - set(out.citations))
        if uncited:
            issues.append("quote_for_uncited_label")
        if out.citations and not out.quotes:
            issues.append("citations_without_quotes")
    elif "QUOTES" in requested:
        issues.append("missing_quotes")

    # CONFIDENCE: an integer 0-100, optionally with "%" or "/100". Anything else is kept
    # raw and parsed as None rather than guessed at.
    if "CONFIDENCE" in sections:
        raw = _strip_emphasis(
            sections["CONFIDENCE"].splitlines()[0] if sections["CONFIDENCE"] else ""
        )
        out.confidence_raw = raw
        m = _INT_RE.match(raw)
        if m and 0 <= int(m.group(1)) <= 100:
            out.confidence = int(m.group(1))
        else:
            issues.append("confidence_unparseable")
        if len(sections["CONFIDENCE"].splitlines()) > 1:
            issues.append("text_after_confidence")
    elif "CONFIDENCE" in requested:
        issues.append("missing_confidence")

    if "REASONING" in sections:
        out.reasoning = sections["REASONING"]
        if not out.reasoning:
            issues.append("empty_reasoning")
    elif "REASONING" in requested:
        issues.append("missing_reasoning")

    out.issues = issues
    if any(
        i.startswith(
            (
                "missing_",
                "citations_unparseable",
                "confidence_unparseable",
                "quotes_unparseable",
                "empty_reasoning",
                "empty_citations",
            )
        )
        for i in issues
    ):
        out.status = "partial"
    return out
