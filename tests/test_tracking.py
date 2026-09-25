"""Unit tests for Phase 7: the pipeline-selection rule, the flattening of results blocks into
MLflow metrics, the served pipeline (against a fake index, embedder and backend), its pyfunc
wrapper, and the API skeleton."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.generation.llm import CacheMiss, GenerationResponse, ResponseCache
from src.generation.prompts import load_registry
from src.generation.requests import build_request
from src.tracking.selection import arm_rates, correct_and_grounded, rank, select

REPO = Path(__file__).resolve().parent.parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# Selection


def row(qid, model="m1", prompt="v1", version=1, label="correct", grounded=True, cond="retrieved"):
    return {
        "financebench_id": qid,
        "condition": cond,
        "model_key": model,
        "prompt_id": prompt,
        "prompt_version": version,
        "correctness": {"label": label},
        "groundedness": {"fully_grounded": grounded},
    }


def arm(model, prompt, version, outcomes, cond="retrieved"):
    """outcomes: one (label, fully_grounded) per question q0, q1, ..."""
    return [
        row(f"q{i}", model, prompt, version, label, g, cond)
        for i, (label, g) in enumerate(outcomes)
    ]


class TestSelection:
    def test_correct_and_grounded_needs_both(self):
        assert correct_and_grounded(row("q", label="correct", grounded=True))
        assert not correct_and_grounded(row("q", label="correct", grounded=False))
        assert not correct_and_grounded(row("q", label="incorrect", grounded=True))
        assert not correct_and_grounded(row("q", label="partially_correct", grounded=True))
        assert not correct_and_grounded(row("q", label="abstained", grounded=None))

    def test_declining_stays_in_the_denominator(self):
        rows = arm("m", "v1", 1, [("correct", True), ("abstained", None), ("abstained", None)])
        a = arm_rates(rows, 200, 0)["retrieved__m__v1"]
        assert a["n"] == 3 and a["n_correct_and_grounded"] == 1
        assert a["correct_and_grounded"]["mean"] == pytest.approx(1 / 3)

    def test_ranks_by_rate_then_accuracy_then_prompt_version(self):
        rows = (
            arm("m", "v2", 2, [("correct", True), ("correct", False), ("incorrect", True)])
            + arm("m", "v1", 1, [("correct", True), ("incorrect", False), ("incorrect", True)])
            + arm("m", "v3", 3, [("correct", True), ("correct", False), ("incorrect", False)])
            + arm("m", "v4", 4, [("correct", True), ("correct", True), ("correct", True)], "oracle")
        )
        arms = arm_rates(rows, 200, 0)
        # all three retrieved arms have rate 1/3; v2 and v3 have accuracy 2/3, v1 has 1/3;
        # v2 beats v3 on prompt version. The oracle arm is never a candidate.
        assert rank(arms) == ["retrieved__m__v2", "retrieved__m__v3", "retrieved__m__v1"]

    def test_winner_and_paired_difference(self):
        rows = arm("a", "v1", 1, [("correct", True)] * 4) + arm(
            "b", "v1", 1, [("correct", True), ("incorrect", True)] * 2
        )
        sel = select(rows, 500, 0)
        assert sel["winner"] == "retrieved__a__v1"
        d = sel["winner_minus_runner_up"]
        assert d["n_questions"] == 4 and d["diff"] == pytest.approx(0.5)
        assert d["ci95"][0] <= 0.5 <= d["ci95"][1]

    def test_duplicate_question_in_an_arm_is_refused(self):
        rows = arm("m", "v1", 1, [("correct", True)]) * 2
        with pytest.raises(ValueError, match="more than one trace"):
            arm_rates(rows, 100, 0)


# --------------------------------------------------------------------------------------
# Results -> MLflow metrics


class TestFlatten:
    def test_names_intervals_and_skips(self):
        track = load_script("09_track_experiments")
        out = track.flatten(
            {
                "metrics": {"recall@5": 0.2, "ok": True, "missing": None},
                "rate": {"mean": 0.5, "ci95": [0.25, 0.75]},
                "bad": math.nan,
                "labels": ["a", "b"],
            }
        )
        assert out == {
            "metrics.recall_at_5": 0.2,
            "rate.mean": 0.5,
            "rate.ci95_lo": 0.25,
            "rate.ci95_hi": 0.75,
        }


# --------------------------------------------------------------------------------------
# Served pipeline


class FakeEmbed:
    def encode_queries(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


class FakeIndex:
    def __init__(self, n=8):
        self.results = [
            SimpleNamespace(
                chunk_id=f"D:fixed_size:{i}",
                score=1.0 - i / 10,
                metadata={
                    "chunk_id": f"D:fixed_size:{i}",
                    "doc_id": "D",
                    "page": i,
                    "section": None,
                    "char_start": 0,
                    "char_end": 5,
                    "is_table": False,
                    "text": f"text {i}",
                },
            )
            for i in range(n)
        ]
        self.depths = []

    def search_corpus(self, qv, k):
        self.depths.append(k)
        return self.results[:k]


def config(prompt_id="v2_citation_required", **retrieval):
    template = load_registry(REPO / "prompts")[prompt_id]
    return {
        "retrieval": {
            "chunking": "fixed_size",
            "embedding": "bge-small-en-v1.5",
            "method": "dense",
            "backend": "faiss",
            "k": 5,
            "search_depth": 100,
            **retrieval,
        },
        "generator": {
            "key": "claude-sonnet-5",
            "model": {"backend": "anthropic", "model": "claude-sonnet-5", "thinking": "disabled"},
        },
        "prompt": {
            "id": prompt_id,
            "version": template.version,
            "file_sha256": template.file_sha256,
        },
        "max_tokens": 2048,
    }


def response(text):
    return GenerationResponse(text, "claude-sonnet-5", "end_turn", 10, 5, 1.0, "2026-01-01")


def pipeline(tmp_path, cfg=None, backend=None):
    from src.serving.pipeline import RAGPipeline

    return RAGPipeline(
        cfg or config(),
        root=REPO,
        backend=backend or SimpleNamespace(name="anthropic", generate=lambda r: pytest.fail()),
        cache=ResponseCache(tmp_path / "cache"),
        embed_model=FakeEmbed(),
        index=FakeIndex(),
    )


class TestPipeline:
    def test_request_is_the_generation_request_and_is_served_from_cache(self, tmp_path):
        p = pipeline(tmp_path)
        chunks = p.retrieve("What was revenue?")
        assert [c["label"] for c in chunks] == ["C1", "C2", "C3", "C4", "C5"]
        assert p.index.depths == [100]  # searched at search_depth, then cut to k
        req, _ = build_request(
            p.model_cfg, 2048, p.template, {"question": "What was revenue?"}, chunks
        )
        p.cache.put(
            req, response('ANSWER: $5\nCITATIONS: C2\nQUOTES:\n[C2] "text 1"\nCONFIDENCE: 80')
        )
        out = p.answer("What was revenue?")
        assert out["cache_key"] == req.cache_key and out["from_cache"]
        assert out["answer"] == "$5" and out["citations"] == ["C2"] and out["confidence"] == 80
        assert set(out["timings_ms"]) == {"retrieve", "generate"}

    def test_uncached_api_call_without_ledger_is_refused(self, tmp_path):
        from src.generation.llm import BudgetExceeded

        with pytest.raises(BudgetExceeded):
            pipeline(tmp_path).answer("Anything?")

    def test_replay_only_miss_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RAG_REPLAY_ONLY", "1")
        with pytest.raises(CacheMiss):
            pipeline(tmp_path).answer("Anything?")

    def test_changed_prompt_file_is_refused(self, tmp_path):
        cfg = config()
        cfg["prompt"]["file_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="has changed"):
            pipeline(tmp_path, cfg)

    def test_depth_below_k_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="search_depth"):
            pipeline(tmp_path, config(search_depth=3))

    def test_pyfunc_predict_accepts_frame_or_list(self):
        pd = pytest.importorskip("pandas")
        from src.tracking.pipeline_model import RAGPipelineModel

        m = RAGPipelineModel()
        m.pipeline = SimpleNamespace(answer=lambda q: {"question": q})
        frame = pd.DataFrame({"question": ["a", "b"]})
        assert m.predict(None, frame) == [{"question": "a"}, {"question": "b"}]
        assert m.predict(None, ["c"]) == [{"question": "c"}]


# --------------------------------------------------------------------------------------
# API skeleton


class TestApp:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        from src.serving import app as app_module

        sel = {
            "meta": {"rule": "r"},
            "arms": {"retrieved__m__v1": {"correct_and_grounded": {"mean": 0.5}}},
            "winner": "retrieved__m__v1",
            "winner_minus_runner_up": {"diff": 0.1},
            "pipeline_config": {"prompt": {"id": "v1"}},
        }
        path = tmp_path / "sel.json"
        path.write_text(json.dumps(sel), encoding="utf-8")
        monkeypatch.setattr(app_module, "SELECTION_PATH", path)
        monkeypatch.setattr(
            app_module, "_probe", lambda url, timeout=2.0: {"status": "unreachable"}
        )
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        return TestClient(app_module.app), app_module, path

    def test_health_reports_unreachable_dependencies_as_degraded(self, client):
        c, _, _ = client
        body = c.get("/health").json()
        assert body["status"] == "degraded"
        assert set(body["dependencies"]) == {"qdrant", "mlflow"}

    def test_health_ok_when_dependencies_answer(self, client, monkeypatch):
        c, app_module, _ = client
        monkeypatch.setattr(app_module, "_probe", lambda url, timeout=2.0: {"status": "ok"})
        assert c.get("/health").json()["status"] == "ok"

    def test_config_serves_the_selection(self, client):
        c, _, _ = client
        body = c.get("/config").json()
        assert body["selected_arm"] == "retrieved__m__v1"
        assert body["pipeline"] == {"prompt": {"id": "v1"}}

    def test_config_missing_selection_is_503(self, client, monkeypatch, tmp_path):
        c, app_module, _ = client
        monkeypatch.setattr(app_module, "SELECTION_PATH", tmp_path / "absent.json")
        assert c.get("/config").status_code == 503
