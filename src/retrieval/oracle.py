"""Oracle context: the k chunks a perfect retriever would return, for the generation
condition that holds generation fixed while guaranteeing the evidence is present.

Given every chunk of the question's filing ranked by the real retriever's score, the
oracle picks, in order:
  1. for each gold span, its best-scoring relevant chunk (so every span is covered),
  2. further relevant chunks by score (more of each span's text),
  3. the best-scoring non-relevant chunks of the same filing, until k are chosen.
The chosen chunks are then ordered by retrieval score, not by relevance, so the
evidence's position in the prompt varies the way it would under real retrieval instead of
always sitting at C1. Relevance is the Phase 3 rule (`is_relevant`), unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from src.evaluation.gold_spans import GoldSpan
from src.evaluation.retrieval_metrics import ChunkSpan, is_relevant


def build_oracle_context(
    ranked: Sequence[Any],
    golds: Sequence[GoldSpan],
    k: int,
    to_span: Callable[[Any], ChunkSpan],
    min_overlap_frac: float = 0.5,
) -> list[Any]:
    """`ranked`: every candidate chunk, best retrieval score first. `to_span` maps a
    candidate to its ChunkSpan. Returns up to k candidates in their `ranked` order."""
    if not golds:
        raise ValueError("an oracle context needs at least one gold span")
    spans = [to_span(c) for c in ranked]
    relevant_to = [
        [j for j, g in enumerate(golds) if is_relevant(s, g, min_overlap_frac)] for s in spans
    ]
    chosen: list[int] = []

    def take(i: int) -> None:
        if i not in chosen and len(chosen) < k:
            chosen.append(i)

    for j in range(len(golds)):
        best = next((i for i, rel in enumerate(relevant_to) if j in rel), None)
        if best is not None:
            take(best)
    for i, rel in enumerate(relevant_to):
        if rel:
            take(i)
    for i, rel in enumerate(relevant_to):
        if not rel:
            take(i)
    return [ranked[i] for i in sorted(chosen)]
