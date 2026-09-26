"""Replicated runs (src/replication/replicates.py): the request a replicate sends, its trace,
and the per-run metrics and spread. Uses the committed run-0 traces and records; no API."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from src.evaluation.evaluate import aggregate
from src.generation.llm import GenerationResponse
from src.generation.prompts import load_registry
from src.generation.trace import read_traces
from src.replication.replicates import (
    pooled_difference,
    replicate_request,
    replicate_trace,
    run_metrics,
    selection_per_run,
    spread,
    stability,
)

TRACES = Path("results/traces/retrieved/claude-sonnet-5__v1_zero_shot.jsonl")
RECORDS = Path("results/metrics/eval_per_trace.jsonl")
ARMS = (
    "retrieved__claude-sonnet-5__v1_zero_shot",
    "retrieved__claude-sonnet-5__v2_citation_required",
)


@pytest.fixture(scope="module")
def trace():
    if not TRACES.exists():
        pytest.skip("run-0 traces not present")
    return read_traces(TRACES)[0]


@pytest.fixture(scope="module")
def template():
    return load_registry()["v1_zero_shot"]


@pytest.fixture(scope="module")
def run0():
    if not RECORDS.exists():
        pytest.skip("run-0 records not present")
    rows = [json.loads(x) for x in RECORDS.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if f"{r['condition']}__{r['model_key']}__{r['prompt_id']}" in ARMS]


class TestReplicateRequest:
    def test_only_the_tag_differs(self, trace, template):
        req = replicate_request(trace, template, "replicates_run1")
        assert req.params == {**trace["generation"]["params"], "_replicate": "replicates_run1"}
        assert req.cache_key != trace["generation"]["cache_key"]
        assert req.model == trace["generation"]["model_requested"]
        assert req.max_tokens == trace["generation"]["max_tokens"]

    def test_tag_is_never_sent(self, trace, template):
        from src.generation.llm import AnthropicBackend

        sent = AnthropicBackend.call_params(replicate_request(trace, template, "t"))
        assert "_replicate" not in sent

    def test_refuses_if_the_context_differs_from_run_0(self, trace, template):
        t = copy.deepcopy(trace)
        t["retrieval"]["chunks"][0]["text"] = "X" + t["retrieval"]["chunks"][0]["text"]
        with pytest.raises(ValueError, match="differs from run 0"):
            replicate_request(t, template, "t")

    def test_refuses_if_the_prompt_file_changed(self, trace, template):
        t = copy.deepcopy(trace)
        t["prompt"]["file_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="prompt file changed"):
            replicate_request(t, template, "t")


def test_replicate_trace_keeps_the_question_id_and_records_the_run(trace, template):
    req = replicate_request(trace, template, "t")
    resp = GenerationResponse(
        text="ANSWER: 42\nCITATIONS: C1\nCONFIDENCE: 80",
        model_reported="claude-sonnet-5",
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
        latency_ms=0.0,
        created_utc="2026-09-26T00:00:00+00:00",
    )
    out = replicate_trace(trace, template, req, resp, 2, 0.001)
    assert out["trace_id"] == trace["trace_id"] and out["replicate"] == 2
    assert out["generation"]["cache_key"] == req.cache_key
    assert out["parsed"]["answer"] == "42" and out["parsed"]["citations"] == ["C1"]


class TestSpread:
    def test_mean_and_sample_std(self):
        s = spread([0.08, 0.10, 0.12])
        assert s["mean"] == pytest.approx(0.10) and s["std"] == pytest.approx(0.02)
        assert (s["min"], s["max"]) == (0.08, 0.12)

    def test_undefined_run_makes_the_spread_undefined(self):
        assert spread([0.1, None])["mean"] is None


class TestOnRun0:
    def test_run_metrics_reproduce_the_selection(self, run0):
        sel = json.loads(Path("results/metrics/pipeline_selection.json").read_text())
        cfg = yaml.safe_load(Path("configs/evaluation.yaml").read_text(encoding="utf-8"))
        for arm in ARMS:
            rs = [r for r in run0 if r["trace_id"].startswith(arm)]
            m = run_metrics(rs, aggregate(rs, cfg))
            assert m["n_correct_and_grounded"] == sel["arms"][arm]["n_correct_and_grounded"]
            assert m["accuracy_all"] == pytest.approx(sel["arms"][arm]["accuracy_all"])

    def test_selection_per_run_reproduces_the_tie_break(self, run0):
        s = selection_per_run(run0, 200, 0)
        assert s["winner"] == ARMS[0]  # tied on correct-and-grounded, ahead on accuracy
        assert s["first_minus_second"]["diff"] == pytest.approx(0.0)

    def test_identical_runs_are_perfectly_stable(self, run0):
        twice = {0: run0, 1: [dict(r, replicate=1) for r in run0]}
        st = stability(twice, ARMS[0])
        assert st["n_questions"] == 150
        assert st["correct_and_grounded_changes"] == 0 and st["declined_changes"] == 0
        assert st["n_correct_and_grounded_in_every_run"] == 12
        d = pooled_difference(twice, ARMS[0], ARMS[1], 200, 0)
        assert d["diff"] == pytest.approx(0.0) and d["n_runs"] == 2
