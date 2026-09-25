"""Groundedness: the share of an answer's claims that its context supports.

Written from scratch, before (and without) RAGAS, as the spec requires. An answer is split
into atomic claims by one judge call (src/evaluation/judge.py, `decompose`), and each claim
is checked against the context the generator was given by a second call (`verify`):

  groundedness     supported document claims / document claims
  fully_grounded   every document claim supported
  any_contradicted at least one document claim contradicted by the context

Only `document` claims (about the company, its figures, its filings) enter the score. The
other two kinds are verified and counted but kept out of it:
  general  definitions, formulas, what a ratio indicates. The context is a set of filing
           excerpts; whether it restates textbook finance says nothing about whether the
           answer is grounded in it.
  context  statements about what the excerpts contain or lack ("the excerpts do not give
           the cost of sales"). They describe the context, not the company, and would
           otherwise let a bare decline score as perfectly grounded. A contradicted one is a
           false claim of absence: the model said the information was missing when an
           excerpt holds it (`n_context_contradicted`), reported separately.
The score is undefined (None) for an answer with no document claims. Declined responses can
still state facts they found; they are scored like any other, and reported separately.
"""

from __future__ import annotations


def score(claims: list[dict], verdicts: list[dict]) -> dict:
    if len(claims) != len(verdicts):
        raise ValueError("one verdict per claim")
    by_kind: dict[str, list[str]] = {"document": [], "context": [], "general": []}
    for c, v in zip(claims, verdicts, strict=True):
        by_kind[c["kind"]].append(v["verdict"])
    doc = by_kind["document"]
    n = len(doc)
    supported = doc.count("supported")
    contradicted = doc.count("contradicted")
    return {
        "n_claims": len(claims),
        "n_document_claims": n,
        "n_context_claims": len(by_kind["context"]),
        "n_general_claims": len(by_kind["general"]),
        "n_supported": supported,
        "n_unsupported": doc.count("unsupported"),
        "n_contradicted": contradicted,
        "n_context_contradicted": by_kind["context"].count("contradicted"),
        "n_general_supported": by_kind["general"].count("supported"),
        "groundedness": supported / n if n else None,
        "fully_grounded": (supported == n) if n else None,
        "any_contradicted": (contradicted > 0) if n else None,
    }
