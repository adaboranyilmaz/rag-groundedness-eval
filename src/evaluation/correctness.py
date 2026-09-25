"""Answer correctness against FinanceBench gold answers.

Two graders, chosen per question and recorded per trace:

numeric  For the questions whose gold answer is a single figure (all 50 metrics-generated
         questions plus 3 others), a deterministic matcher. The question's own unit and
         rounding instructions ("Answer in USD millions", "round to one decimal place") fix
         the gold's units and precision. A figure in the answer is read in every legitimate
         way (a bare number as the requested unit or as literal dollars; a bare number
         against a percent gold as percent units or as a fraction) and labelled:
           correct     some reading equals the gold within tolerance
           unit_error  no reading does, but one does after a unit factor (10^+-3, 10^+-6,
                       10^+-9: thousands/millions/billions; 10^+-2: percent vs fraction).
                       Spec: a figure in the wrong unit is a unit bug, not a hallucination.
           incorrect   otherwise
         Tolerance: within `rel_tol` of the gold (default 1%), or equal to the gold at its
         rounding precision (|pred - gold| <= half a unit in the last place). A figure whose
         magnitude matches but whose sign is opposite counts as correct only when the gold is
         negative and the answer expresses the direction in words ("declined 3.7%").
         The answer's figure is chosen by `primary_figure`: one that can state the gold's
         kind (a percent for a percent gold; a currency or scaled figure for a currency gold;
         a plain number or a percent for a plain ratio gold), in the lead sentence, outside
         brackets, after the last '=' when the answer shows its calculation. A sensitivity
         label over *any* figure in the answer is recorded alongside. An answer with
         no figure at all is left to the LLM judge (e.g. "no restructuring costs" for a gold
         of 0).
judge    Every other question, and numeric questions whose answer has no figure: the LLM
         correctness judge (src/evaluation/judge.py, prompts/judge/judge_correctness_*.md).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from src.evaluation.figures import Figure, extract_figures

UNIT_FACTORS = (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9, 100.0, 0.01)
_SCALE_NAMES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_UNIT_HINT_RE = re.compile(
    r"\b(?:in\s+(?:USD|US\s*dollars?|dollars?|\$)?\s*|USD\s+)(thousand|million|billion|trillion)s?\b",
    re.IGNORECASE,
)
_PERCENT_HINT_RE = re.compile(
    r"units\s+of\s+percents?|as\s+a\s+percent(?:age)?|in\s+percent(?:age)?s?\b", re.IGNORECASE
)
_WORD_DIGITS = {"one": 1, "two": 2, "three": 3, "four": 4}
_DECIMALS_HINT_RE = re.compile(
    r"round\w*\s+(?:\w+\s+){0,3}?to\s+(one|two|three|four|\d)\s+decimal", re.IGNORECASE
)
_NEGATIVE_WORDS_RE = re.compile(
    r"\b(?:decreas\w*|declin\w*|fell|fall\w*|drop\w*|lower\w*|negative|loss\w*|down|"
    r"reduc\w*|contract\w*|shr[au]nk|deficit|deteriorat\w*)\b",
    re.IGNORECASE,
)
LABEL_RANK = {"correct": 2, "unit_error": 1, "incorrect": 0}


@dataclass(frozen=True)
class UnitHint:
    scale: float | None  # "in USD millions" -> 1e6
    percent: bool  # "answer in units of percents"
    decimals: int | None  # "round to two decimal places" -> 2


def question_hint(question: str) -> UnitHint:
    m = _UNIT_HINT_RE.search(question)
    d = _DECIMALS_HINT_RE.search(question)
    decimals = None
    if d:
        tok = d.group(1).lower()
        decimals = _WORD_DIGITS.get(tok, int(tok) if tok.isdigit() else None)
    return UnitHint(
        scale=_SCALE_NAMES[m.group(1).lower()] if m else None,
        percent=bool(_PERCENT_HINT_RE.search(question)),
        decimals=decimals,
    )


@dataclass(frozen=True)
class NumericGold:
    value: float  # in the gold's own units (as written)
    unit: float  # base-unit factor of those units: "$1577.00" in USD millions -> 1e6
    percent: bool
    currency: bool
    decimals: int  # rounding precision used for tolerance
    text: str

    @property
    def kind(self) -> str:
        if self.percent:
            return "percent"
        if self.currency or self.unit != 1.0:
            return "currency"
        return "plain"


def numeric_gold(gold_answer: str, question: str) -> NumericGold | None:
    """The gold as a single figure, or None when the gold is not a single-figure answer."""
    figures = extract_figures(gold_answer)
    if len(figures) != 1:
        return None
    # Words beyond the figure are allowed only if they are few ("$400,000,000 increase.").
    rest = (gold_answer[: figures[0].start] + gold_answer[figures[0].end :]).split()
    if len(rest) > 2:
        return None
    f = figures[0]
    hint = question_hint(question)
    if f.explicit_scale:
        unit = f.scale
    elif hint.scale is not None:
        unit = hint.scale
    else:
        unit = 1.0
    return NumericGold(
        value=f.number,
        unit=unit,
        percent=f.percent or hint.percent,
        currency=f.currency,
        decimals=hint.decimals if hint.decimals is not None else f.decimals,
        text=gold_answer.strip(),
    )


def readings(fig: Figure, gold: NumericGold) -> list[float]:
    """Every legitimate reading of an answer figure, in the gold's units."""
    if gold.percent:
        if fig.percent:
            return [fig.number]
        if fig.explicit_scale:
            return [fig.value]
        return [fig.number, fig.number * 100.0]  # percent units, or a fraction
    if fig.percent:
        return [fig.number / 100.0 / gold.unit]  # 68% of a ratio gold 0.68
    if fig.explicit_scale:
        return [fig.value / gold.unit]
    if gold.unit == 1.0:
        return [fig.number]
    return [fig.number, fig.number / gold.unit]  # the requested unit, or literal dollars


def within(pred: float, gold: NumericGold, rel_tol: float) -> bool:
    tol = max(rel_tol * abs(gold.value), 0.5 * 10.0**-gold.decimals)
    return abs(pred - gold.value) <= tol * (1 + 1e-9) + 1e-12


def _figure_kind(fig: Figure) -> str:
    if fig.percent:
        return "percent"
    if fig.currency or fig.explicit_scale:
        return "currency"
    return "plain"


# Figure kinds that can state an answer of each gold kind: a plain ratio gold (ROA -0.02)
# can be answered as a percent (-1.53%), but not as a dollar amount.
_ANSWER_KINDS = {
    "percent": ("percent",),
    "currency": ("currency",),
    "plain": ("plain", "percent"),
}


_SENTENCE_END_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(])|\n")


def _bracket_depths(text: str) -> list[int]:
    depths, d = [], 0
    for ch in text:
        if ch in ")]}":
            d = max(0, d - 1)
        depths.append(d)
        if ch in "([{":
            d += 1
    return depths


def primary_figure(answer: str, figures: list[Figure], gold: NumericGold) -> Figure | None:
    """The figure the answer gives as its answer. Answers state the result first and put
    working in brackets ("24.26 (revenue of $6,489M / ...)"), or end a calculation with
    it ("$19,815M - $13,997M = $5,818M"). So: within the first sentence, outside brackets,
    the first fitting figure after the last '=' if there is one, else the first fitting
    figure; failing that, the same over the whole answer; then any fitting figure; then
    the first figure."""
    if not figures:
        return None

    def fits(f: Figure) -> bool:
        return _figure_kind(f) in _ANSWER_KINDS[gold.kind]

    depths = _bracket_depths(answer)
    m = _SENTENCE_END_RE.search(answer)
    for scope_end in (m.start() if m else len(answer), len(answer)):
        top = [f for f in figures if f.start < scope_end and depths[f.start] == 0 and fits(f)]
        eqs = [i for i in range(scope_end) if answer[i] in "=≈" and depths[i] == 0]
        if eqs:
            after = [f for f in top if f.start > eqs[-1]]
            if after:
                return after[0]
        if top:
            return top[0]
    fitting = [f for f in figures if fits(f)]
    return fitting[0] if fitting else figures[0]


@dataclass
class FigureMatch:
    label: str  # correct | unit_error | incorrect
    figure: str
    reading: float | None  # the reading that matched (gold units), if any
    unit_factor: float | None = None  # for unit_error
    sign_note: str | None = None  # "sign_in_words" | "sign_mismatch"


def match_figure(fig: Figure, gold: NumericGold, answer: str, rel_tol: float) -> FigureMatch:
    reads = readings(fig, gold)
    for r in reads:
        if within(r, gold, rel_tol):
            return FigureMatch("correct", fig.text, r)
    sign_note = None
    for r in reads:
        if r != 0 and within(-r, gold, rel_tol):
            if gold.value < 0 < r and _NEGATIVE_WORDS_RE.search(answer):
                return FigureMatch("correct", fig.text, r, sign_note="sign_in_words")
            sign_note = "sign_mismatch"
    # A gold of 0 has no unit to get wrong: any tiny reading times 10^-9 would "match".
    for factor in UNIT_FACTORS if gold.value != 0 else ():
        for r in reads:
            if within(r * factor, gold, rel_tol) or (
                gold.value < 0 < r and within(-r * factor, gold, rel_tol)
            ):
                return FigureMatch("unit_error", fig.text, r, unit_factor=factor)
    return FigureMatch("incorrect", fig.text, None, sign_note=sign_note)


@dataclass
class NumericResult:
    label: str | None  # correct | unit_error | incorrect; None = no figure in the answer
    primary: FigureMatch | None
    any_figure_label: str | None  # best label over every figure in the answer
    n_figures: int
    sensitivity: dict[str, str | None] = field(default_factory=dict)  # rel_tol -> label

    def to_dict(self) -> dict:
        return asdict(self)


def grade_numeric(
    answer: str, gold: NumericGold, rel_tol: float, sensitivity_tols: tuple[float, ...] = ()
) -> NumericResult:
    figures = extract_figures(answer)
    fig = primary_figure(answer, figures, gold)
    if fig is None:
        return NumericResult(None, None, None, 0, {str(t): None for t in sensitivity_tols})
    primary = match_figure(fig, gold, answer, rel_tol)
    best = max((match_figure(f, gold, answer, rel_tol).label for f in figures), key=LABEL_RANK.get)
    sens = {str(t): match_figure(fig, gold, answer, t).label for t in sensitivity_tols}
    return NumericResult(primary.label, primary, best, len(figures), sens)
