"""Unit tests for gold-evidence alignment in src/evaluation/gold_spans.py."""

import random

from src.evaluation.gold_spans import NormalisedText, align_evidence, normalise


def random_words(n: int, seed: int) -> str:
    """Deterministic pseudo-text whose 30-char normalised windows are unique."""
    rng = random.Random(seed)
    return " ".join(
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 9)))
        for _ in range(n)
    )


class TestNormalisedText:
    def test_offsets_map_back_to_original(self):
        text = "Net  Income,\n$1,577 (Millions)"
        nt = NormalisedText(text)
        assert nt.text == "netincome1577millions"
        start, end = nt.to_original(0, len("netincome"))
        assert text[start:end] == "Net  Income"

    def test_non_ascii_is_dropped_not_miscounted(self):
        # "İ".lower() is two code points and superscript two is a digit to
        # str.isdigit; neither may shift the offset map.
        text = "İx²y"
        nt = NormalisedText(text)
        assert nt.text == "xy"
        assert [text[i] for i in nt.offsets] == ["x", "y"]


class TestAlignEvidence:
    def test_exact_match_despite_whitespace_and_punctuation(self):
        doc = "Intro text.\n\nPurchases of property, plant\nand equipment (PP&E)  (1,577)\n\nEnd."
        evidence = "Purchases of property plant and equipment PP E 1577"
        al = align_evidence(evidence, "D", NormalisedText(doc))
        assert al.status == "exact"
        assert doc[al.span.char_start : al.span.char_end].startswith("Purchases")
        assert doc[al.span.char_start : al.span.char_end].endswith("1,577")
        assert al.n_occurrences == 1

    def test_multiple_occurrences_are_counted(self):
        doc = "Revenue 100. Other stuff. Revenue 100."
        al = align_evidence("revenue 100", "D", NormalisedText(doc))
        assert al.status == "exact"
        assert al.n_occurrences == 2
        assert al.span.char_start == 0

    def test_partial_alignment_when_pdf_adds_text_missing_from_html(self):
        body = random_words(120, seed=1)
        doc = random_words(50, seed=2) + "\n" + body + "\n" + random_words(50, seed=3)
        # PDF-extracted evidence carries a running header and a page number the HTML lacks.
        evidence = "Table of Contents 3M COMPANY " + body[:300] + " 60 " + body[300:]
        al = align_evidence(evidence, "D", NormalisedText(doc))
        assert al.status == "partial"
        assert al.window_coverage >= 0.5
        start = doc.index(body)
        # Span lies within the true body, covering nearly all of it.
        assert al.span.char_start >= start
        assert al.span.char_end <= start + len(body)
        assert (al.span.char_end - al.span.char_start) > 0.8 * len(body)

    def test_repeated_boilerplate_windows_are_not_anchors(self):
        # A long block that recurs in the filing (a repeated statement header, say) makes
        # up most of the evidence. Its windows match twice each, so they must not anchor
        # anything -- otherwise it alone would clear the coverage threshold and "align"
        # evidence whose distinctive part is absent from the document.
        boiler = random_words(60, seed=4)
        doc = boiler + " " + random_words(200, seed=5) + " " + boiler
        tail = random_words(10, seed=10)  # not in the document at all
        al = align_evidence(boiler + " " + tail, "D", NormalisedText(doc))
        assert al.status == "unaligned"

    def test_outlier_anchor_is_discarded(self):
        body = random_words(80, seed=6)
        far_away = random_words(400, seed=7)
        stray = body[:40]  # the evidence's opening also appears, far from the rest
        doc = body[40:] + " " + far_away + " " + stray
        al = align_evidence(body, "D", NormalisedText(doc))
        assert al.status == "partial"
        # The span must not stretch across `far_away` to reach the stray copy.
        assert al.span.char_end - al.span.char_start < len(body) * 1.5

    def test_below_min_coverage_is_unaligned(self):
        body = random_words(60, seed=8)
        doc = body[: len(body) // 5] + " " + random_words(60, seed=9)
        al = align_evidence(body, "D", NormalisedText(doc), min_window_coverage=0.5)
        assert al.status == "unaligned"
        assert 0 < al.window_coverage < 0.5

    def test_short_non_matching_evidence_is_unaligned(self):
        al = align_evidence("not here", "D", NormalisedText("something else entirely"))
        assert al.status == "unaligned"

    def test_empty_evidence_is_unaligned(self):
        assert align_evidence("  --  ", "D", NormalisedText("abc")).status == "unaligned"

    def test_normalise(self):
        assert normalise("A-1,000.5\n") == "a10005"
