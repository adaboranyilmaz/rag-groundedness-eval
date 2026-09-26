"""Repeated runs of an evaluated arm: the requests, the traces, and the spread of the Phase 5
metrics over runs (scripts/16_replicates.py).

A replicate re-sends each original generation request unchanged, with only a `_replicate`
tag added to its params: metadata that is part of the cache key but never sent
(src/generation/llm.py). The original request is rebuilt from its trace, prompt file and
retrieved chunks, and must hash to the trace's recorded cache key, so the replicate provably
asks the model exactly what run 0 asked, over exactly the same retrieved excerpts. Sonnet 5
takes no sampling parameters, so each run is a fresh sample of the same model.

Run 0 is the Phase 4/5 run. A run's metrics are those Phase 5 and the pipeline selection
report (src/evaluation/evaluate.py `aggregate`, src/tracking/selection.py), computed on that
run's records alone; the spread is the sample standard deviation over the runs.
"""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Sequence
from typing import Any

from src.generation.llm import GenerationRequest, GenerationResponse
from src.generation.parsing import parse_output
from src.generation.prompts import PromptTemplate
from src.generation.trace import build_trace, request_from_trace
from src.tracking.selection import (
    arm_key,
    arm_rates,
    correct_and_grounded,
    paired_difference,
    rank,
)


def replicate_request(trace: dict, template: PromptTemplate, tag: str) -> GenerationRequest:
    """The trace's own request with `_replicate` added; refuses if the rebuilt original does
    not hash to the key the trace recorded."""
    p = trace["prompt"]
    if template.file_sha256 != p["file_sha256"] or template.version != p["version"]:
        raise ValueError(f"{trace['trace_id']}: prompt file changed since run 0")
    original = request_from_trace(trace, template)
    if original.cache_key != trace["generation"]["cache_key"]:
        raise ValueError(f"{trace['trace_id']}: rebuilt request differs from run 0's")
    return dataclasses.replace(original, params={**original.params, "_replicate": tag})


def replicate_trace(
    trace: dict,
    template: PromptTemplate,
    request: GenerationRequest,
    response: GenerationResponse,
    run: int,
    cost_usd: float | None,
) -> dict:
    """A trace in the Phase 4 schema for the replicate's answer, plus its run number. The
    trace id stays the question's, so runs pair by it."""
    parsed = parse_output(response.text, template.output_fields, len(trace["retrieval"]["chunks"]))
    out = build_trace(
        question=trace["question"],
        retrieval=trace["retrieval"],
        template=template,
        request=request,
        model_key=trace["generation"]["model_key"],
        response=response,
        parsed=parsed,
        rendered_sha256=trace["prompt"]["rendered_sha256"],
        cost_usd=cost_usd,
    )
    out["replicate"] = run
    return out


# --------------------------------------------------------------------------------------
# Metrics per run


def selection_rows(records: list[dict]) -> list[dict]:
    """Records as the selection functions read them. A record with no groundedness (a judge
    failure) cannot be correct-and-grounded; it is kept, counted as not, and reported."""
    return [
        r if r["groundedness"] is not None else {**r, "groundedness": {"fully_grounded": None}}
        for r in records
    ]


def run_metrics(records: list[dict], aggregate_row: dict) -> dict[str, Any]:
    """The reported metrics of one arm in one run; `aggregate_row` is `aggregate()` of the
    same records."""
    n = len(records)
    rows = selection_rows(records)
    a = aggregate_row
    status = a["abstention"]["status_counts"]
    return {
        "n": n,
        "n_correct_and_grounded": sum(correct_and_grounded(r) for r in rows),
        "correct_and_grounded": sum(correct_and_grounded(r) for r in rows) / n,
        "n_correct": sum(r["correctness"]["label"] == "correct" for r in records),
        "accuracy_all": sum(r["correctness"]["label"] == "correct" for r in records) / n,
        "accuracy_answered": a["correctness"]["accuracy_answered"]["rate"],
        "n_declined": status["declined"],
        "declined_rate": status["declined"] / n,
        "n_answered_or_partial": status["answered"] + status["partial"],
        "n_groundedness_scored": a["groundedness"]["n_scored"],
        "mean_groundedness": a["groundedness"]["mean_groundedness"]["mean"],
        "fully_grounded_rate": a["groundedness"]["fully_grounded"]["rate"],
        "citation_precision": a["citations"]["citation_precision"]["mean"],
        "citation_recall": a["citations"]["citation_recall"]["mean"],
        "gold_cited_rate": a["citations"]["gold_cited"]["rate"],
        "n_judge_errors": sum(a["judge_errors"].values()),
    }


def spread(values: Sequence[float | None]) -> dict[str, Any]:
    """Mean, sample std (n-1), min and max over runs; None where any run is undefined."""
    if any(v is None for v in values):
        return {"n_runs": len(values), "values": list(values), "mean": None, "std": None}
    vals = [float(v) for v in values]
    return {
        "n_runs": len(vals),
        "values": vals,
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else None,
        "min": min(vals),
        "max": max(vals),
    }


def stability(records_by_run: dict[int, list[dict]], arm: str) -> dict[str, Any]:
    """How many questions change outcome between runs (any change over all runs)."""
    per_q: dict[str, list[dict]] = {}
    for run in sorted(records_by_run):
        for r in selection_rows(records_by_run[run]):
            if arm_key(r) == arm:
                per_q.setdefault(r["financebench_id"], []).append(r)
    complete = {q: rs for q, rs in per_q.items() if len(rs) == len(records_by_run)}

    def changes(f) -> int:
        return sum(len({f(r) for r in rs}) > 1 for rs in complete.values())

    return {
        "n_questions": len(complete),
        "correct_and_grounded_changes": changes(correct_and_grounded),
        "correct_changes": changes(lambda r: r["correctness"]["label"] == "correct"),
        "correctness_label_changes": changes(lambda r: r["correctness"]["label"]),
        "declined_changes": changes(lambda r: r["abstention"]["status"] == "declined"),
        "n_correct_and_grounded_in_every_run": sum(
            all(correct_and_grounded(r) for r in rs) for rs in complete.values()
        ),
        "n_correct_and_grounded_in_some_run": sum(
            any(correct_and_grounded(r) for r in rs) for rs in complete.values()
        ),
    }


def selection_per_run(records: list[dict], n_resamples: int, seed: int) -> dict[str, Any]:
    """The Phase 7 selection rule applied to one run's records of the replicated arms."""
    rows = selection_rows(records)
    arms = arm_rates(rows, n_resamples, seed)
    ranking = rank(arms)
    return {
        "ranking": ranking,
        "winner": ranking[0],
        "rates": {
            k: {
                "correct_and_grounded": a["correct_and_grounded"]["mean"],
                "correct_and_grounded_ci95": a["correct_and_grounded"]["ci95"],
                "accuracy_all": a["accuracy_all"],
            }
            for k, a in arms.items()
        },
        "first_minus_second": paired_difference(rows, ranking[0], ranking[1], n_resamples, seed),
    }


def pooled_difference(
    records_by_run: dict[int, list[dict]], a: str, b: str, n_resamples: int, seed: int
) -> dict[str, Any]:
    """Arm a minus arm b in correct-and-grounded, averaged over runs per question first, with
    a paired question-bootstrap interval: the three-run version of the selection margin."""
    import numpy as np

    from src.evaluation.reliability import question_ci

    def per_question(arm: str) -> dict[str, float]:
        vals: dict[str, list[float]] = {}
        for run in records_by_run:
            for r in selection_rows(records_by_run[run]):
                if arm_key(r) == arm:
                    vals.setdefault(r["financebench_id"], []).append(float(correct_and_grounded(r)))
        return {q: statistics.fmean(v) for q, v in vals.items() if len(v) == len(records_by_run)}

    va, vb = per_question(a), per_question(b)
    qids = sorted(set(va) & set(vb))
    d = np.asarray([va[q] - vb[q] for q in qids])
    return {
        "a": a,
        "b": b,
        "n_questions": len(qids),
        "n_runs": len(records_by_run),
        "diff": float(d.mean()) if qids else None,
        "ci95": question_ci(qids, lambda idx: float(d[idx].mean()), n_resamples, seed)
        if qids
        else None,
    }
