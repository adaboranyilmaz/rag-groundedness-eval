"""Unit tests for Phase 8 serving: the groundedness judge applied to one answer, the query
service, the API (with a fake service), metrics and the request log, and the check script's
comparison. Judge requests are compared with the Phase 5 harness's, so a served score is
the evaluated measurement and not a lookalike."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.evaluation.evaluate import Judge, decompose_values, verify_values
from src.evaluation.judge import DECOMPOSE_SCHEMA, verify_schema
from src.generation.llm import (
    BudgetExceeded,
    CacheMiss,
    GenerationRequest,
    GenerationResponse,
    ResponseCache,
)
from src.serving.grounding import GroundednessJudge, load_validation
from src.serving.observability import Metrics, RequestLog
from src.serving.service import ApiKeyMissing, LayeredCache, NoKeyBackend, QueryService

REPO = Path(__file__).resolve().parent.parent
EVAL_CFG = yaml.safe_load((REPO / "configs/evaluation.yaml").read_text(encoding="utf-8"))
AGREEMENT = REPO / "results/metrics/judge_agreement.json"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resp(text: str, stop="end_turn") -> GenerationResponse:
    return GenerationResponse(text, "claude-sonnet-5", stop, 10, 5, 12.0, "2026-01-01")


def chunks(n=3):
    return [
        {
            "label": f"C{i}",
            "score": 1.0 - i / 10,
            "chunk_id": f"D:fixed_size:{i}",
            "doc_id": "D",
            "page": i,
            "section": "Item 7" if i == 1 else None,
            "char_start": 0,
            "char_end": 5,
            "is_table": False,
            "text": f"Revenue was {i} million.  Second   sentence.",
        }
        for i in range(1, n + 1)
    ]


def parsed(answer="Revenue was $1 million.", token=False):
    return {"answer": answer, "abstained_token": token, "citations": ["C1"], "status": "ok"}


DECOMP = {
    "response_type": "answered",
    "claims": [
        {"claim": "Revenue was $1 million", "kind": "document"},
        {"claim": "Revenue measures sales", "kind": "general"},
        {"claim": "Revenue fell", "kind": "document"},
    ],
}
VERDICTS = {
    "verdicts": [
        {"claim_id": 1, "reason": "r", "verdict": "supported", "supporting_excerpts": ["C1"]},
        {"claim_id": 2, "reason": "r", "verdict": "supported", "supporting_excerpts": []},
        {"claim_id": 3, "reason": "r", "verdict": "contradicted", "supporting_excerpts": []},
    ]
}


class ScriptedBackend:
    """Answers decompose and verify requests (told apart by their schema) with fixed text,
    or raises."""

    name = "fake"

    def __init__(self, decompose=DECOMP, verify=VERDICTS, raise_on=None, stop="end_turn"):
        self.outputs = {"decompose": decompose, "verify": verify}
        self.raise_on, self.stop = raise_on, stop
        self.calls: list[str] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        schema = request.params["output_config"]["format"]["schema"]
        stage = "decompose" if "response_type" in schema["properties"] else "verify"
        self.calls.append(stage)
        if self.raise_on == stage:
            raise CacheMiss("not cached")
        out = self.outputs[stage]
        return resp(out if isinstance(out, str) else json.dumps(out), self.stop)


def judge(tmp_path, backend=None, validation=None, recoverable=(CacheMiss,)):
    return GroundednessJudge(
        EVAL_CFG,
        backend=backend or ScriptedBackend(),
        cache=ResponseCache(tmp_path / "cache"),
        ledger=None,
        prompts_dir=REPO / "prompts" / "judge",
        validation=validation,
        recoverable=recoverable,
    )


# --------------------------------------------------------------------------------------
# Groundedness judge


class TestGroundednessJudge:
    def test_scores_document_claims_only(self, tmp_path):
        out = judge(tmp_path).assess("What was revenue?", parsed(), chunks())
        g = out["groundedness"]
        assert g["score"] == pytest.approx(0.5) and g["unavailable_reason"] is None
        assert (g["n_document_claims"], g["n_supported"], g["n_contradicted"]) == (2, 1, 1)
        assert g["any_contradicted"] and g["fully_grounded"] is False
        assert [c["verdict"] for c in g["claims"]] == ["supported", "supported", "contradicted"]
        assert (out["status"], out["status_source"]) == ("answered", "judge")
        assert [c["purpose"] for c in out["llm_calls"]] == ["judge_decompose", "judge_verify"]
        assert set(out["timings_ms"]) == {"judge_decompose", "judge_verify"}

    def test_requests_are_the_evaluation_harness_requests(self, tmp_path):
        """Same prompts, model, schema and ceilings as src/evaluation/evaluate.py, so a
        benchmark answer replays from the evaluated cache entries."""
        backend = ScriptedBackend()
        j = judge(tmp_path, backend)
        j.assess("What was revenue?", parsed(), chunks())
        harness = Judge(EVAL_CFG, EVAL_CFG["judge"]["model"], "offline", j.cache, None)
        trace = {
            "question": {"question": "What was revenue?"},
            "parsed": parsed(),
            "retrieval": {"chunks": chunks()},
        }
        labels = [c["label"] for c in chunks()]
        expected = [
            harness.request("decompose", decompose_values(trace), DECOMPOSE_SCHEMA).cache_key,
            harness.request(
                "verify", verify_values(trace, DECOMP["claims"]), verify_schema(labels)
            ).cache_key,
        ]
        assert [j.cache.has(k) for k in expected] == [True, True]

    def test_bare_decline_is_not_judged(self, tmp_path):
        backend = ScriptedBackend()
        out = judge(tmp_path, backend).assess("Q?", parsed(None, token=True), chunks())
        assert out["status"] == "declined" and backend.calls == []
        assert out["groundedness"] == {"score": None, "unavailable_reason": "no_answer_text"}

    def test_no_claims_skips_verify(self, tmp_path):
        backend = ScriptedBackend(decompose={"response_type": "declined", "claims": []})
        out = judge(tmp_path, backend).assess("Q?", parsed("I cannot tell."), chunks())
        assert backend.calls == ["decompose"] and out["status"] == "declined"
        g = out["groundedness"]
        assert g["score"] is None and g["unavailable_reason"] == "no_document_claims"

    def test_only_context_claims_leave_the_score_undefined(self, tmp_path):
        backend = ScriptedBackend(
            decompose={
                "response_type": "declined",
                "claims": [{"claim": "The excerpts lack it", "kind": "context"}],
            },
            verify={
                "verdicts": [
                    {
                        "claim_id": 1,
                        "reason": "r",
                        "verdict": "supported",
                        "supporting_excerpts": [],
                    }
                ]
            },
        )
        g = judge(tmp_path, backend).assess("Q?", parsed("Not given."), chunks())["groundedness"]
        assert g["score"] is None and g["unavailable_reason"] == "no_document_claims"
        assert g["n_context_claims"] == 1 and len(g["claims"]) == 1

    def test_malformed_judge_output_is_reported_not_guessed(self, tmp_path):
        backend = ScriptedBackend(decompose="not json")
        out = judge(tmp_path, backend).assess("Q?", parsed(), chunks())
        reason = out["groundedness"]["unavailable_reason"]
        assert reason.startswith("judge_error: decompose: invalid_output")
        assert out["groundedness"]["score"] is None and out["status_source"] == "rule"

    def test_truncated_judge_output_is_an_error(self, tmp_path):
        out = judge(tmp_path, ScriptedBackend(stop="max_tokens")).assess("Q?", parsed(), chunks())
        assert out["groundedness"]["unavailable_reason"] == (
            "judge_error: decompose: stop_reason_max_tokens"
        )

    def test_recoverable_failure_keeps_the_answer_and_the_claims(self, tmp_path):
        backend = ScriptedBackend(raise_on="verify")
        out = judge(tmp_path, backend).assess("Q?", parsed(), chunks())
        g = out["groundedness"]
        assert g["unavailable_reason"] == "judge_error: verify: CacheMiss"
        assert len(g["claims"]) == 3 and g["score"] is None
        assert "judge_verify" in out["timings_ms"]

    def test_unrecoverable_failure_raises(self, tmp_path):
        with pytest.raises(CacheMiss):
            judge(tmp_path, ScriptedBackend(raise_on="decompose"), recoverable=()).assess(
                "Q?", parsed(), chunks()
            )

    def test_validation_is_read_from_the_results_file(self):
        v = load_validation(AGREEMENT)
        report = json.loads(AGREEMENT.read_text(encoding="utf-8"))
        block = report["agreement"]["human__vs__judge"]["document_claims"]
        assert v["claim_supported_kappa"]["kappa"] == block["claim_supported"]["kappa"]
        assert v["answer_fully_grounded_kappa"]["n"] == block["answer_fully_grounded"]["n"]

    def test_prompts_on_disk_are_the_validated_ones(self, tmp_path):
        judge(tmp_path, validation=load_validation(AGREEMENT))  # raises if they differ

    def test_changed_judge_prompt_is_refused(self, tmp_path):
        v = load_validation(AGREEMENT)
        v["judge_prompt_sha256"]["verify"] = "0" * 64
        with pytest.raises(ValueError, match="changed since the judge was validated"):
            judge(tmp_path, validation=v)


# --------------------------------------------------------------------------------------
# Service


class TestService:
    def test_layered_cache_reads_bundle_first_and_writes_state_only(self, tmp_path):
        bundle, writable = ResponseCache(tmp_path / "b"), ResponseCache(tmp_path / "w")
        req = GenerationRequest("anthropic", "m", "s", "u", 10)
        bundle.put(req, resp("bundle"))
        layered = LayeredCache(bundle, writable)
        assert layered.get(req.cache_key).text == "bundle"
        other = GenerationRequest("anthropic", "m", "s", "other", 10)
        layered.put(other, resp("new"))
        assert writable.has(other.cache_key) and not bundle.has(other.cache_key)
        assert layered.get(other.cache_key).text == "new"

    def test_no_key_backend_refuses(self):
        with pytest.raises(ApiKeyMissing):
            NoKeyBackend().generate(GenerationRequest("anthropic", "m", "s", "u", 10))

    def test_query_assembles_answer_citations_latency_and_cost(self):
        answer = {
            "answer": "Revenue was $1 million.",
            "confidence": 90,
            "citations": ["C2", "C9"],  # C9 is not a retrieved chunk: dropped
            "parsed": parsed(),
            "chunks": chunks(),
            "cache_key": "k",
            "from_cache": False,
            "cost_usd": 0.01,
            "model_latency_ms": 900.0,
            "model": "m",
            "input_tokens": 100,
            "output_tokens": 10,
            "timings_ms": {"embed": 5.0, "search": 1.0, "generate": 900.0},
        }
        assessed = {
            "status": "answered",
            "status_source": "judge",
            "groundedness": {"score": 1.0, "unavailable_reason": None, "claims": []},
            "llm_calls": [
                {"purpose": "judge_decompose", "cached": True, "cost_usd": 0.0},
                {"purpose": "judge_verify", "cached": False, "cost_usd": 0.02},
            ],
            "timings_ms": {"judge_decompose": 1.0, "judge_verify": 800.0},
            "model": "claude-sonnet-5",
            "decompose_prompt": "judge_decompose",
            "verify_prompt": "judge_verify",
        }
        selection = {
            "winner": "retrieved__m__v1",
            "pipeline_config": {
                "retrieval": {"chunking": "c", "embedding": "e", "method": "dense", "k": 3},
                "generator": {"model": {"model": "m"}},
                "prompt": {"id": "v1"},
            },
        }
        svc = QueryService(
            SimpleNamespace(answer=lambda q: answer),
            SimpleNamespace(assess=lambda q, p, c: assessed, validation={"v": 1}),
            selection,
        )
        out = svc.query("What was revenue?")
        assert [c["label"] for c in out["citations"]] == ["C2"]
        assert out["citations"][0]["snippet"] == "Revenue was 2 million. Second sentence."
        assert out["context"] is None
        assert out["cost_usd"] == pytest.approx(0.03)
        assert set(out["latency_ms"]) == {
            "embed",
            "search",
            "generate",
            "judge_decompose",
            "judge_verify",
            "total",
        }
        assert out["groundedness"]["validation"] == {"v": 1}
        assert out["pipeline"]["selected_arm"] == "retrieved__m__v1"
        with_ctx = svc.query("Q?", include_context=True)
        assert [c["text"] for c in with_ctx["context"]] == [c["text"] for c in chunks()]


# --------------------------------------------------------------------------------------
# API


def served(score=0.5, reason=None):
    v = load_validation(AGREEMENT)
    return {
        "request_id": "abc",
        "question": "What was revenue?",
        "answer": "Revenue was $1 million.",
        "status": "answered",
        "confidence": 90,
        "citations": [
            {
                "label": "C1",
                "chunk_id": "D:1",
                "doc_id": "D",
                "page": 1,
                "section": None,
                "score": 0.9,
                "snippet": "Revenue",
            }
        ],
        "context": None,
        "chunk_ids": ["D:1"],
        "groundedness": {
            "score": score,
            "unavailable_reason": reason,
            "n_document_claims": 2,
            "n_supported": 1,
            "claims": [{"claim": "c", "kind": "document", "verdict": "supported"}],
            "judge_model": "claude-sonnet-5",
            "judge_prompts": {"decompose": "judge_decompose", "verify": "judge_verify"},
            "validation": v,
        },
        "latency_ms": {"embed": 5.0, "generate": 900.0, "total": 1000.0},
        "llm_calls": [
            {
                "purpose": "generate",
                "cached": True,
                "cost_usd": 0.0,
                "cache_key": "k",
                "model_latency_ms": 900.0,
            }
        ],
        "cost_usd": 0.0,
        "pipeline": {"selected_arm": "a", "k": 5},
    }


@pytest.fixture
def api(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from src.serving import app as app_module

    monkeypatch.setenv("RAG_SKIP_LOAD", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setattr(app_module.state, "metrics", Metrics())
    monkeypatch.setattr(app_module.state, "request_log", RequestLog(tmp_path / "logs"))
    monkeypatch.setattr(app_module, "_probe", lambda url, timeout=2.0: {"status": "ok"})

    def install(query=None, status=None):
        svc = SimpleNamespace(
            query=query or (lambda q, include_context=False: served()),
            status=lambda: status or {"replay_only": False},
        )
        monkeypatch.setattr(app_module.state, "service", svc)

    monkeypatch.setattr(app_module.state, "service", None)
    monkeypatch.setattr(app_module.state, "load_error", "FileNotFoundError: no bundle")
    return TestClient(app_module.app), install, tmp_path / "logs"


def log_lines(log_dir: Path) -> list[dict]:
    return [json.loads(x) for p in log_dir.glob("*.jsonl") for x in p.read_text().splitlines()]


class TestAPI:
    def test_not_ready_before_the_bundle_loads(self, api):
        client, _, _ = api
        assert client.get("/ready").status_code == 503
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "not_ready"
        assert "no bundle" in health.json()["load_error"]
        r = client.post("/query", json={"question": "Q?"})
        assert r.status_code == 503 and r.json()["detail"]["error"] == "not_ready"

    def test_query_returns_the_answer_and_logs_and_counts_it(self, api):
        client, install, logs = api
        install()
        r = client.post("/query", json={"question": "What was revenue?"})
        assert r.status_code == 200
        body = r.json()
        assert body["groundedness"]["score"] == 0.5 and "chunk_ids" not in body
        assert body["groundedness"]["validation"]["claim_supported_kappa"]["n"] > 0
        (line,) = log_lines(logs)
        assert line["outcome"] == "ok" and line["chunk_ids"] == ["D:1"]
        assert line["groundedness"]["score"] == 0.5 and "ts_utc" in line
        metrics = client.get("/metrics").text
        assert 'rag_requests_total{outcome="ok"} 1.0' in metrics
        assert "rag_groundedness_score_count 1.0" in metrics
        assert 'rag_llm_calls_total{cache="hit",purpose="generate"} 1.0' in metrics
        assert 'rag_stage_latency_seconds_count{stage="total"} 1.0' in metrics

    def test_undefined_score_is_counted_by_reason(self, api):
        client, install, _ = api
        install(query=lambda q, include_context=False: served(None, "judge_error: verify: X"))
        assert client.post("/query", json={"question": "Q?"}).status_code == 200
        metrics = client.get("/metrics").text
        assert 'rag_groundedness_unavailable_total{reason="judge_error"} 1.0' in metrics
        assert "rag_groundedness_score_count 0.0" in metrics

    @pytest.mark.parametrize(
        "exc,code,name",
        [
            (CacheMiss("miss"), 503, "not_cached"),
            (ApiKeyMissing("no key"), 503, "no_api_key"),
            (BudgetExceeded("cap"), 503, "budget_exhausted"),
        ],
    )
    def test_upstream_failures_map_to_status_codes(self, api, exc, code, name):
        client, install, logs = api

        def fail(q, include_context=False):
            raise exc

        install(query=fail)
        r = client.post("/query", json={"question": "Q?"})
        assert r.status_code == code and r.json()["error"] == name
        assert log_lines(logs)[0]["outcome"] == name
        assert f'rag_requests_total{{outcome="{name}"}} 1.0' in client.get("/metrics").text

    def test_model_api_error_is_502(self, api):
        import anthropic
        import httpx

        client, install, _ = api

        def fail(q, include_context=False):
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))

        install(query=fail)
        r = client.post("/query", json={"question": "Q?"})
        assert r.status_code == 502 and r.json()["error"] == "upstream_error"

    def test_empty_question_is_rejected(self, api):
        client, install, _ = api
        install()
        assert client.post("/query", json={"question": ""}).status_code == 422

    def test_ready_and_health_once_loaded(self, api):
        client, install, _ = api
        install(status={"replay_only": True})
        assert client.get("/ready").status_code == 200
        body = client.get("/health").json()
        assert body["status"] == "ok" and body["service"] == {"replay_only": True}


# --------------------------------------------------------------------------------------
# Check script


class TestCheckCompare:
    def setup_method(self):
        self.mod = load_script("12_serving")
        cs = chunks()
        self.trace = {
            "retrieval": {"chunks": cs},
            "generation": {"cache_key": "g"},
            "parsed": {"answer": "A", "citations": ["C1"]},
        }
        claims = [{"claim": "c", "kind": "document", "verdict": "supported",
                   "supporting_excerpts": ["C1"], "reason": "r"}]  # fmt: skip
        g = {k: 0 for k in self.mod.GROUNDEDNESS_FIELDS} | {"groundedness": 1.0, "claims": claims}
        self.ev = {
            "abstention": {"status": "answered"},
            "judge": {"decompose": "d", "verify": "v"},
            "groundedness": g,
        }
        self.served = {
            "answer": "A",
            "status": "answered",
            "citations": [{"label": "C1"}],
            "context": [dict(c) for c in cs],
            "groundedness": {**g, "score": 1.0, "claims": [dict(c) for c in claims]},
            "llm_calls": [
                {"purpose": "generate", "cache_key": "g", "cached": True},
                {"purpose": "judge_decompose", "cache_key": "d", "cached": True},
                {"purpose": "judge_verify", "cache_key": "v", "cached": True},
            ],
        }

    def test_identical_answer_passes(self):
        ok, diff = self.mod.compare(self.trace, self.ev, self.served)
        assert all(ok.values()) and diff == 0.0

    def test_changed_verdict_fails_groundedness(self):
        self.served["groundedness"]["claims"][0]["verdict"] = "unsupported"
        ok, _ = self.mod.compare(self.trace, self.ev, self.served)
        assert [k for k, v in ok.items() if not v] == ["groundedness"]

    def test_reordered_chunks_and_score_drift_fail(self):
        ctx = self.served["context"]
        ctx[0], ctx[1] = ctx[1], ctx[0]
        ok, diff = self.mod.compare(self.trace, self.ev, self.served)
        assert not ok["chunks"] and not ok["scores_within_tol"] and diff > 1e-5

    def test_uncached_call_fails(self):
        self.served["llm_calls"][2]["cached"] = False
        ok, _ = self.mod.compare(self.trace, self.ev, self.served)
        assert [k for k, v in ok.items() if not v] == ["all_cached"]


# --------------------------------------------------------------------------------------
# Deployment files


class TestDeploymentFiles:
    def test_k8s_and_compose_run_the_same_image(self):
        compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
        docs = list(yaml.safe_load_all((REPO / "k8s/api.yaml").read_text(encoding="utf-8")))
        deploy = next(d for d in docs if d["kind"] == "Deployment")
        container = deploy["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == compose["services"]["api"]["image"]

    def test_probes_point_at_served_routes(self):
        from src.serving import app as app_module

        routes = {r.path for r in app_module.app.routes}
        docs = list(yaml.safe_load_all((REPO / "k8s/api.yaml").read_text(encoding="utf-8")))
        deploy = next(d for d in docs if d["kind"] == "Deployment")
        container = deploy["spec"]["template"]["spec"]["containers"][0]
        probes = [container[k]["httpGet"]["path"] for k in container if k.endswith("Probe")]
        assert len(probes) == 3 and set(probes) <= routes
        dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
        assert "http://localhost:8000/ready" in dockerfile and "/ready" in routes
