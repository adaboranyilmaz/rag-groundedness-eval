"""Numeric figures in free text: extraction for the numeric correctness matcher and for the
judge-free check of whether an answer's figures appear in the chunks it cites.

Extraction records what the text says and nothing more: the written number, its sign, its
written precision, an explicit scale (thousand/million/billion/trillion, or K/M/B/T glued
to a currency figure), a percent marker (basis points are converted to percent units) and
a currency marker. Parentheses are never read as a negative sign: in prose answers they
are parentheticals ("FY2021 (15.0%)"), and the judge-free citation check compares
magnitudes anyway.

Numbers that are not quantities are skipped:
  - years: a bare integer 1900-2100 with no currency, scale or percent marker
  - day of month after a month name ("December 31, 2022")
  - numbers glued to letters ("FY2022", "Q2", "C3", "3M", "2nd") or to "-letter" ("10-K")
  - counts of periods ("12 months ended")
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SCALE_WORDS = {
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
    "mn": 1e6,
    "mm": 1e6,
    "bn": 1e9,
    "tn": 1e12,
}
# Glued to a number ("$6,489M"): only read as a scale when the figure is a currency figure,
# so a company name like "3M" is not three million.
SCALE_LETTERS = {"k": 1e3, "m": 1e6, "mm": 1e6, "mn": 1e6, "b": 1e9, "bn": 1e9, "t": 1e12}

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_FIGURE_RE = re.compile(
    r"""
    (?<![\w.,$])
    (?P<sign>[-−–]\s?)?
    (?P<cur>US\$|\$|USD\s?)?
    (?P<sign2>[-−–]\s?)?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?P<glued>[A-Za-z]+)?
    (?:\s?(?P<scale>trillion|billion|million|thousand|tn|bn|mn|mm)s?\b)?
    (?:\s?(?P<cur2>USD|dollars?)\b)?
    (?:\s?(?P<pct>%|percent\b|per\s?cent\b|pct\b|percentage\s+points?\b|bps\b|basis\s+points?\b))?
    """,
    re.IGNORECASE | re.VERBOSE,
)
_MONTH_BEFORE_RE = re.compile(rf"(?:{_MONTHS})\.?\s+$", re.IGNORECASE)
# Days are deliberately not here: "61.33 days" is a quantity (days payable outstanding).
_PERIOD_AFTER_RE = re.compile(r"\s*-?\s*(?:months?|weeks?|quarters?)\b", re.IGNORECASE)
_HYPHEN_LETTER_RE = re.compile(r"-[A-Za-z]")


@dataclass(frozen=True)
class Figure:
    number: float  # the written number with its sign, before any scale
    decimals: int  # digits after the decimal point, as written
    scale: float  # 1.0 unless an explicit scale was written
    explicit_scale: bool
    percent: bool  # in percent units (basis points already divided by 100)
    currency: bool
    start: int
    end: int
    text: str

    @property
    def value(self) -> float:
        """The quantity in base units (or percent units for a percent figure)."""
        return self.number * self.scale

    @property
    def significant_digits(self) -> int:
        digits = self.text_digits.lstrip("0")
        return len(digits)

    @property
    def text_digits(self) -> str:
        return re.sub(r"\D", "", f"{abs(self.number):.{self.decimals}f}")


def _is_negative(m: re.Match) -> bool:
    return bool(m.group("sign") or m.group("sign2"))


def extract_figures(text: str) -> list[Figure]:
    out: list[Figure] = []
    for m in _FIGURE_RE.finditer(text):
        raw_num = m.group("num").replace(",", "")
        number = float(raw_num)
        decimals = len(raw_num.split(".")[1]) if "." in raw_num else 0
        currency = bool(m.group("cur") or m.group("cur2"))
        scale, explicit = 1.0, False
        glued = (m.group("glued") or "").lower()
        if glued:
            if glued in SCALE_LETTERS and (currency or len(glued) > 1):
                scale, explicit = SCALE_LETTERS[glued], True
            elif glued == "x":  # "2.8x": a multiple, the number stands as written
                pass
            else:  # "3M", "2nd", "10K" without a currency sign, "4Q": a token, not a figure
                continue
        if m.group("scale"):
            scale, explicit = SCALE_WORDS[m.group("scale").lower()], True
        pct_word = (m.group("pct") or "").lower()
        percent = bool(pct_word)
        if pct_word.startswith(("bps", "basis")):
            number /= 100.0
            decimals += 2

        after = text[m.end() : m.end() + 12]
        before = text[max(0, m.start() - 12) : m.start()]
        if not glued and not m.group("scale") and _HYPHEN_LETTER_RE.match(after):
            continue  # "10-K", "8-K"
        if _MONTH_BEFORE_RE.search(before):
            continue  # "December 31"
        if (
            decimals == 0
            and not (currency or explicit or percent)
            and _PERIOD_AFTER_RE.match(after)
        ):
            continue  # "12 months ended"
        is_year = (
            decimals == 0
            and "," not in m.group("num")
            and not (currency or explicit or percent)
            and 1900 <= number <= 2100
        )
        if is_year:
            continue
        if _is_negative(m):
            number = -number
        out.append(
            Figure(
                number=number,
                decimals=decimals,
                scale=scale,
                explicit_scale=explicit,
                percent=percent,
                currency=currency,
                start=m.start(),
                end=m.end(),
                text=m.group(0).strip(),
            )
        )
    return out
