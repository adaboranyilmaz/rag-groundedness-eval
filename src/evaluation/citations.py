"""Citation metrics: does what the model cites actually back what it says?

From the verification judge (src/evaluation/groundedness.py), which lists, for each claim it
finds supported, the excerpts holding the supporting text. The judge never sees the model's
own citations, so these compare two independent pointings at the context. Only `document`
claims count: a supported claim about the excerpts themselves ("the excerpts do not give X")
is "supported" by every excerpt, and would make any citation look precise.
  citation_precision  cited excerpts that support >=1 document claim / cited excerpts
  citation_recall     supported document claims with >=1 supporting excerpt among those
                      cited / supported document claims
These share the groundedness judge, so their agreement with groundedness in Phase 6 is not
fully independent. Three judge-free checks exist for that reason:
  gold_cited       cited >=1 excerpt holding gold evidence (Phase 3 relevance rule), among
                   traces whose context contains gold evidence
  quote fidelity   (v2 only) each quote is `exact` (occurs in the excerpt it is attributed
                   to, in the normalised space of src/evaluation/gold_spans.py), `near`
                   (>=90% of it found there in runs of >=8 characters), `other_excerpt`
                   (found exactly in a different excerpt: misattributed), or `not_found`
  numeric support  the answer's figures found in the cited excerpts, and in any excerpt.
                   Only figures with >=2 significant digits are checked; a figure matches an
                   excerpt figure at the answer's written precision, allowing a thousands /
                   millions / billions rescale for figures with >=3 significant digits
                   ("$7.45 billion" vs a table's "7,453"). Derived figures (ratios, growth
                   rates) rarely appear verbatim, so this is a lower bound on grounding.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from src.evaluation.figures import Figure, extract_figures
from src.evaluation.gold_spans import GoldSpan, normalise
from src.evaluation.retrieval_metrics import ChunkSpan, is_relevant

NEAR_MATCH_COVERAGE = 0.9
NEAR_MATCH_MIN_RUN = 8
RESCALE_EXPONENTS = (3, 6, 9, -3, -6, -9)


def evidence_labels(
    chunks: list[dict], golds: list[GoldSpan], min_overlap_frac: float
) -> list[str]:
    """Labels of the context chunks relevant to any gold span."""
    out = []
    for c in chunks:
        span = ChunkSpan(c["chunk_id"], c["doc_id"], c["char_start"], c["char_end"])
        if any(is_relevant(span, g, min_overlap_frac) for g in golds):
            out.append(c["label"])
    return out


def judge_citation_metrics(cited: list[str], claims: list[dict], verdicts: list[dict]) -> dict:
    supported = [
        v
        for c, v in zip(claims, verdicts, strict=True)
        if c["kind"] == "document" and v["verdict"] == "supported"
    ]
    supporting = {lab for v in supported for lab in v["supporting_excerpts"]}
    n_cited_supporting = sum(1 for c in cited if c in supporting)
    return {
        "n_cited": len(cited),
        "n_cited_supporting": n_cited_supporting,
        "citation_precision": n_cited_supporting / len(cited) if cited else None,
        "citation_recall": (
            sum(1 for v in supported if set(v["supporting_excerpts"]) & set(cited)) / len(supported)
            if supported
            else None
        ),
    }


def _near_coverage(quote: str, text: str) -> float:
    blocks = SequenceMatcher(None, quote, text, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= NEAR_MATCH_MIN_RUN) / len(quote)


def quote_status(quote: str, label: str, chunks: list[dict]) -> str:
    q = normalise(quote)
    if not q:
        return "empty"
    by_label = {c["label"]: normalise(c["text"]) for c in chunks}
    own = by_label.get(label)
    if own is not None:
        if q in own:
            return "exact"
        if _near_coverage(q, own) >= NEAR_MATCH_COVERAGE:
            return "near"
    if any(q in t for lab, t in by_label.items() if lab != label):
        return "other_excerpt"
    return "not_found"


def quote_fidelity(quotes: list[dict], chunks: list[dict]) -> dict:
    statuses = [quote_status(q["text"], q["label"], chunks) for q in quotes]
    n = len(statuses)
    return {
        "n_quotes": n,
        "statuses": statuses,
        "faithful_rate": sum(s in ("exact", "near") for s in statuses) / n if n else None,
        "exact_rate": sum(s == "exact" for s in statuses) / n if n else None,
    }


def _figure_matches(a: Figure, c: Figure) -> bool:
    tol = 0.5 * 10.0**-a.decimals * (1 + 1e-9) + 1e-12
    target = abs(a.number)
    if abs(abs(c.number) - target) <= tol:
        return True
    if a.significant_digits < 3:
        return False
    return any(abs(abs(c.number) * 10.0**-k - target) <= tol for k in RESCALE_EXPONENTS)


def numeric_support(answer: str, cited_chunks: list[dict], all_chunks: list[dict]) -> dict:
    figures = [f for f in extract_figures(answer) if f.significant_digits >= 2]
    cited_figs = [f for c in cited_chunks for f in extract_figures(c["text"])]
    all_figs = [f for c in all_chunks for f in extract_figures(c["text"])]
    in_cited = sum(any(_figure_matches(a, c) for c in cited_figs) for a in figures)
    in_context = sum(any(_figure_matches(a, c) for c in all_figs) for a in figures)
    n = len(figures)
    return {
        "n_figures": n,
        "n_in_cited": in_cited,
        "n_in_context": in_context,
        "in_cited_rate": in_cited / n if n and cited_chunks else None,
        "in_context_rate": in_context / n if n else None,
    }
