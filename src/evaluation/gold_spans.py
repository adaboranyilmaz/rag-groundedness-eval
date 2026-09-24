"""Align FinanceBench gold evidence to character spans in this project's parsed filings.

FinanceBench's `evidence_page_num` indexes the PDF rendering of each filing, but this
corpus is parsed from EDGAR's HTML, whose page numbers come from CSS page breaks — the
two page numberings do not line up, so page-based matching would silently mis-score
retrieval. Alignment is done on text instead, in a normalised space (lowercase ASCII
letters and digits only) that is insensitive to the whitespace, line-break, and
punctuation differences between PDF-extracted and HTML-extracted text.

Two alignment tiers:
  * `exact`   — the whole normalised evidence text occurs in the normalised document.
  * `partial` — fixed-size windows of the evidence are located individually; only windows
                that occur exactly once in the document are used as anchors (boilerplate
                like "table of contents" repeats and would anchor anywhere), and anchors
                that imply an evidence start position inconsistent with the median are
                discarded as outliers. Accepted only if the surviving anchors cover at
                least `min_window_coverage` of the evidence's windows.
Everything else is `unaligned` and is reported, never silently dropped.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class GoldSpan:
    doc_id: str
    char_start: int  # offsets into the parsed document's `full_text`
    char_end: int


@dataclass(frozen=True)
class Alignment:
    status: str  # "exact" | "partial" | "unaligned"
    span: GoldSpan | None
    window_coverage: float  # fraction of evidence windows supporting the span (1.0 if exact)
    n_occurrences: int  # exact tier only: how many times the evidence occurs in the doc


class NormalisedText:
    """A document's normalised text plus a map from each normalised character back to its
    offset in the original text, so matches found in normalised space can be reported as
    real `full_text` offsets."""

    def __init__(self, text: str):
        chars: list[str] = []
        offsets: list[int] = []
        for i, ch in enumerate(text):
            low = ch.lower()
            # ASCII-only on purpose: `str.isalnum` accepts e.g. superscript digits, and
            # `lower()` can change string length for some non-ASCII characters.
            if len(low) == 1 and ("a" <= low <= "z" or "0" <= low <= "9"):
                chars.append(low)
                offsets.append(i)
        self.text = "".join(chars)
        self.offsets = offsets

    def to_original(self, norm_start: int, norm_end: int) -> tuple[int, int]:
        """Map the half-open normalised range [norm_start, norm_end) to a half-open
        original range spanning the first through last matched original characters."""
        return self.offsets[norm_start], self.offsets[norm_end - 1] + 1


def normalise(text: str) -> str:
    return NormalisedText(text).text


def _find_all(haystack: str, needle: str) -> list[int]:
    positions = []
    start = haystack.find(needle)
    while start != -1:
        positions.append(start)
        start = haystack.find(needle, start + 1)
    return positions


def align_evidence(
    evidence_text: str,
    doc_id: str,
    document: NormalisedText,
    window_chars: int = 30,
    min_window_coverage: float = 0.5,
) -> Alignment:
    needle = normalise(evidence_text)
    if not needle:
        return Alignment("unaligned", None, 0.0, 0)

    occurrences = _find_all(document.text, needle)
    if occurrences:
        # First occurrence is used; `n_occurrences` is recorded so ambiguous short
        # evidence strings are visible in the alignment report rather than hidden.
        pos = occurrences[0]
        start, end = document.to_original(pos, pos + len(needle))
        return Alignment("exact", GoldSpan(doc_id, start, end), 1.0, len(occurrences))

    # Non-overlapping windows; the final window is flush with the end of the evidence so
    # its tail is never ignored.
    if len(needle) <= window_chars:
        return Alignment("unaligned", None, 0.0, 0)
    window_offsets = list(range(0, len(needle) - window_chars + 1, window_chars))
    if window_offsets[-1] != len(needle) - window_chars:
        window_offsets.append(len(needle) - window_chars)

    anchors: list[tuple[int, int]] = []  # (offset in evidence, position in document)
    for off in window_offsets:
        hits = _find_all(document.text, needle[off : off + window_chars])
        if len(hits) == 1:
            anchors.append((off, hits[0]))
    if not anchors:
        return Alignment("unaligned", None, 0.0, 0)

    # Each anchor implies where the evidence would start in the document. Genuine anchors
    # agree up to drift from text present in one rendering but not the other (PDF running
    # headers, page numbers); tolerate drift up to the evidence's own length.
    implied_starts = [pos - off for off, pos in anchors]
    centre = statistics.median(implied_starts)
    kept = [
        (off, pos)
        for (off, pos), s in zip(anchors, implied_starts, strict=True)
        if abs(s - centre) <= len(needle)
    ]
    coverage = len(kept) / len(window_offsets)
    if coverage < min_window_coverage:
        return Alignment("unaligned", None, coverage, 0)

    norm_start = min(pos for _, pos in kept)
    norm_end = max(pos for _, pos in kept) + window_chars
    start, end = document.to_original(norm_start, norm_end)
    return Alignment("partial", GoldSpan(doc_id, start, end), coverage, 0)
