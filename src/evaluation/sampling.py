"""Fixed, seeded samples of traces: the judge pilot and the hand-labelled validation sample.

The two never share a question. The pilot's judge verdicts are read (by the author and in
review) before the judge prompts are frozen; the validation sample must be labelled blind to
every judge verdict, so its questions are drawn from outside the pilot.

Seeds are strings combining the configured seed with the stratum, so each stratum's draw is
independent of the others and of Python's hash randomisation.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from src.evaluation.abstention import needs_judge


def _cell(t: dict) -> tuple[str, str]:
    return t["retrieval"]["condition"], t["generation"]["model_key"]


def _fid(t: dict) -> str:
    return t["question"]["financebench_id"]


def _round_robin_by_prompt(traces: list[dict], rng: random.Random) -> list[dict]:
    """Shuffle each prompt's traces, then interleave prompts in a fixed order."""
    by_prompt: dict[str, list[dict]] = {}
    for t in sorted(traces, key=lambda t: t["trace_id"]):
        by_prompt.setdefault(t["prompt"]["id"], []).append(t)
    for ts in by_prompt.values():
        rng.shuffle(ts)
    out, prompts = [], sorted(by_prompt)
    while any(by_prompt[p] for p in prompts):
        for p in prompts:
            if by_prompt[p]:
                out.append(by_prompt[p].pop(0))
    return out


def pilot_questions(traces: list[dict], n_in: int, n_out: int, seed: int) -> list[str]:
    """In-corpus questions with aligned gold (they appear in the oracle condition) and
    out-of-corpus questions (the naturally unanswerable ones)."""
    oracle = sorted({_fid(t) for t in traces if t["retrieval"]["condition"] == "oracle"})
    outside = sorted({_fid(t) for t in traces if not t["question"]["in_corpus"]})
    rng = random.Random(f"{seed}:pilot_questions")
    return sorted(rng.sample(oracle, n_in) + rng.sample(outside, n_out))


def pilot_sample(traces: list[dict], fids: list[str], per_cell: int, seed: int) -> list[dict]:
    wanted = set(fids)
    out = []
    for cell in sorted({_cell(t) for t in traces}):
        pool = [
            t for t in traces if _cell(t) == cell and _fid(t) in wanted and needs_judge(t["parsed"])
        ]
        rng = random.Random(f"{seed}:pilot:{cell[0]}:{cell[1]}")
        out.extend(_round_robin_by_prompt(pool, rng)[:per_cell])
    return sorted(out, key=lambda t: t["trace_id"])


def validation_quotas(n_per_generator: dict[str, int], conditions: list[str]) -> dict:
    """Split each generator's quota over the conditions as evenly as possible, alternating
    which condition gets the odd item so both conditions total the same across models."""
    quotas = {}
    for i, (model, n) in enumerate(sorted(n_per_generator.items())):
        base, extra = divmod(n, len(conditions))
        for j, cond in enumerate(conditions):
            bump = 1 if (j - i) % len(conditions) < extra else 0
            quotas[(cond, model)] = base + bump
    return quotas


def validation_candidates(
    traces: list[dict], exclude_fids: set[str], quotas: dict, seed: int
) -> Iterator[tuple[tuple[str, str], dict]]:
    """Candidates in the order they should be tried: cells take turns, each cell's list
    is prompt-balanced. The caller accepts or rejects each (after decomposing it) and
    stops a cell once its quota is met; questions must be unique across the sample."""
    lists = {}
    for cell in sorted(quotas):
        pool = [
            t
            for t in traces
            if _cell(t) == cell and _fid(t) not in exclude_fids and needs_judge(t["parsed"])
        ]
        rng = random.Random(f"{seed}:validation:{cell[0]}:{cell[1]}")
        lists[cell] = _round_robin_by_prompt(pool, rng)
    while any(lists.values()):
        for cell in sorted(lists):
            if lists[cell]:
                yield cell, lists[cell].pop(0)
