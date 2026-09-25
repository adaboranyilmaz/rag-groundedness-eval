"""Unit tests for the Phase 5 judge plumbing: judge prompts and output validation, the
ledger's batch pricing, the Message Batches runner (against a fake client), sampling, and
the two-stage judge pipeline feeding one trace's record."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.correctness import numeric_gold
from src.evaluation.evaluate import Inputs, evaluate_trace, judge_traces
from src.evaluation.judge import (
    DECOMPOSE_SCHEMA,
    JudgeOutputError,
    build_request,
    format_claims,
    load_judge_registry,
    parse_correctness,
    parse_decomposition,
    parse_verification,
    render_judge,
    verify_schema,
)
from src.evaluation.sampling import (
    pilot_questions,
    pilot_sample,
    validation_candidates,
    validation_quotas,
)
from src.generation.batch import pending_records, run_batch_cached
from src.generation.llm import (
    AnthropicBackend,
    BudgetExceeded,
    GenerationRequest,
    GenerationResponse,
    ResponseCache,
    SpendLedger,
)

JUDGE_PROMPTS = Path(__file__).resolve().parent.parent / "prompts" / "judge"
SONNET = {"backend": "anthropic", "model": "claude-sonnet-5", "thinking": "disabled"}
HAIKU = {"backend": "anthropic", "model": "claude-haiku-4-5", "temperature": 0}
PRICES = {"claude-sonnet-5": {"input": 2.0, "output": 10.0}}


# --------------------------------------------------------------------------------------
# Judge prompts, requests, output validation


class TestJudgePrompts:
    def test_repo_judge_prompts_load(self):
        reg = load_judge_registry(JUDGE_PROMPTS)
        purposes = {p.purpose for p in reg.values()}
        assert purposes == {"decompose", "verify", "correctness", "premise"}
        assert reg["judge_verify"].placeholders == ("context", "claims")
        assert reg["judge_premise"].placeholders == ("question", "premise_note", "answer")

    def test_render_is_single_pass(self):
        prompt = load_judge_registry(JUDGE_PROMPTS)["judge_decompose"]
        _, user = render_judge(prompt, {"question": "Q {{answer}}", "answer": "A"})
        assert "Q {{answer}}" in user  # text in a value is never substituted again

    def test_missing_value_is_an_error(self):
        prompt = load_judge_registry(JUDGE_PROMPTS)["judge_decompose"]
        with pytest.raises(ValueError):
            render_judge(prompt, {"question": "Q"})


class TestJudgeRequests:
    def prompt(self):
        return load_judge_registry(JUDGE_PROMPTS)["judge_decompose"]

    def test_schema_and_model_settings_are_in_the_request(self):
        r = build_request(
            SONNET, self.prompt(), {"question": "Q", "answer": "A"}, DECOMPOSE_SCHEMA, 99
        )
        assert r.params["output_config"]["format"]["schema"] == DECOMPOSE_SCHEMA
        assert r.params["thinking"] == {"type": "disabled"} and "temperature" not in r.params
        h = build_request(
            HAIKU, self.prompt(), {"question": "Q", "answer": "A"}, DECOMPOSE_SCHEMA, 99
        )
        assert h.params["temperature"] == 0 and "thinking" not in h.params

    def test_replicate_changes_the_key_but_not_the_call(self):
        vals = {"question": "Q", "answer": "A"}
        a = build_request(SONNET, self.prompt(), vals, DECOMPOSE_SCHEMA, 99)
        b = build_request(SONNET, self.prompt(), vals, DECOMPOSE_SCHEMA, 99, replicate=1)
        assert a.cache_key != b.cache_key
        assert AnthropicBackend.call_params(a) == AnthropicBackend.call_params(b)
        assert "_replicate" not in AnthropicBackend.call_params(b)

    def test_sampling_params_go_in_extra_body_for_direct_calls(self):
        # SDK 1.x rejects `temperature=` in messages.create(); the API still honours it
        r = build_request(
            HAIKU, self.prompt(), {"question": "Q", "answer": "A"}, DECOMPOSE_SCHEMA, 9
        )
        direct = AnthropicBackend.call_params(r, direct=True)
        assert "temperature" not in direct and direct["extra_body"] == {"temperature": 0}
        batch = AnthropicBackend.call_params(r)
        assert batch["temperature"] == 0 and "extra_body" not in batch

    def test_direct_call_accepts_the_translated_params(self):
        sent = {}

        def create(**kw):
            sent.update(kw)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="{}")],
                model="claude-haiku-4-5",
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

        backend = AnthropicBackend(client=SimpleNamespace(messages=SimpleNamespace(create=create)))
        r = build_request(
            HAIKU, self.prompt(), {"question": "Q", "answer": "A"}, DECOMPOSE_SCHEMA, 9
        )
        backend.generate(r)
        assert sent["extra_body"] == {"temperature": 0} and "temperature" not in sent

    def test_format_claims(self):
        assert format_claims([{"claim": "a"}, {"claim": "b"}]) == "1. a\n2. b"


class TestJudgeOutputs:
    def test_decomposition(self):
        d = parse_decomposition(
            '{"claims": [{"claim": " X grew. ", "kind": "document"},'
            ' {"claim": "The excerpts do not give Y.", "kind": "context"}],'
            ' "response_type": "partial"}'
        )
        assert d == {
            "response_type": "partial",
            "claims": [
                {"claim": "X grew.", "kind": "document"},
                {"claim": "The excerpts do not give Y.", "kind": "context"},
            ],
        }
        for bad in (
            "not json",
            '{"claims": [], "response_type": "maybe"}',
            '{"claims": [{"claim": "", "kind": "document"}], "response_type": "answered"}',
            '{"claims": [{"claim": "x", "kind": "fact"}], "response_type": "answered"}',
        ):
            with pytest.raises(JudgeOutputError):
                parse_decomposition(bad)

    def verdict(self, cid, verdict="supported", support=("C1",)):
        return {
            "claim_id": cid,
            "reason": "r",
            "verdict": verdict,
            "supporting_excerpts": list(support),
        }

    def test_verification_orders_and_cleans(self):
        text = json.dumps(
            {
                "verdicts": [
                    self.verdict(2, "unsupported", ["C2"]),
                    self.verdict(1, "supported", ["C3", "C1"]),
                ]
            }
        )
        v = parse_verification(text, 2, ["C1", "C2", "C3"])
        assert v[0]["supporting_excerpts"] == ["C1", "C3"]
        assert v[1]["supporting_excerpts"] == [] and v[1]["dropped_support"] is True

    @pytest.mark.parametrize(
        "verdicts",
        [
            [{"claim_id": 1, "reason": "r", "verdict": "supported", "supporting_excerpts": ["C1"]}],
            [
                {"claim_id": 1, "reason": "r", "verdict": "supported", "supporting_excerpts": []},
                {"claim_id": 1, "reason": "r", "verdict": "supported", "supporting_excerpts": []},
            ],
            [
                {
                    "claim_id": 1,
                    "reason": "r",
                    "verdict": "supported",
                    "supporting_excerpts": ["C9"],
                },
                {"claim_id": 2, "reason": "r", "verdict": "supported", "supporting_excerpts": []},
            ],
        ],
    )
    def test_verification_rejects_incomplete_or_invalid(self, verdicts):
        with pytest.raises(JudgeOutputError):
            parse_verification(json.dumps({"verdicts": verdicts}), 2, ["C1", "C2"])

    def test_correctness(self):
        c = parse_correctness('{"reason": "ok", "grade": "correct", "unit_error": false}')
        assert c["grade"] == "correct" and c["unit_error"] is False
        with pytest.raises(JudgeOutputError):
            parse_correctness('{"reason": "ok", "grade": "correct", "unit_error": "no"}')

    def test_verify_schema_restricts_labels(self):
        items = verify_schema(["C1", "C2"])["properties"]["verdicts"]["items"]
        assert items["properties"]["supporting_excerpts"]["items"]["enum"] == ["C1", "C2"]


# --------------------------------------------------------------------------------------
# Ledger and batches


def ledger(tmp_path, phase_cap=10.0, project_cap=30.0):
    return SpendLedger(tmp_path / "spend.json", PRICES, project_cap, "p5", phase_cap)


def req(i: int, max_tokens: int = 100) -> GenerationRequest:
    return GenerationRequest("anthropic", "claude-sonnet-5", "sys", f"user {i}", max_tokens, {})


class TestLedgerBatch:
    def test_batch_is_half_price(self, tmp_path):
        led = ledger(tmp_path)
        assert led.cost("claude-sonnet-5", 1_000_000, 0, batch=True) == 1.0
        assert led.settle(0.0, "claude-sonnet-5", 1_000_000, 0, batch=True) == 1.0
        assert led.state["by_phase"]["p5"]["n_batch_calls"] == 1

    def test_reserve_amount_respects_caps_unless_forced(self, tmp_path):
        led = ledger(tmp_path, phase_cap=1.0)
        with pytest.raises(BudgetExceeded):
            led.reserve_amount(1.5)
        led.reserve_amount(1.5, force=True)
        assert led.headroom() < 0

    def test_earlier_phase_caps_are_kept(self, tmp_path):
        path = tmp_path / "spend.json"
        path.write_text(
            json.dumps(
                {
                    "total_usd": 0,
                    "n_calls": 0,
                    "by_phase": {},
                    "by_model": {},
                    "caps": {"project_usd": 30.0, "p4_usd": 10.0},
                }
            )
        )
        led = SpendLedger(path, PRICES, 30.0, "p5", 12.0)
        led.settle(0.0, "claude-sonnet-5", 10, 10)
        caps = json.loads(path.read_text())["caps"]
        assert caps == {"project_usd": 30.0, "p4_usd": 10.0, "p5_usd": 12.0}


class FakeBatches:
    def __init__(self, fail=(), usage=(1000, 100)):
        self.created: dict[str, list[dict]] = {}
        self.fail = set(fail)
        self.usage = usage

    def create(self, requests):
        bid = f"batch_{len(self.created)}"
        self.created[bid] = list(requests)
        return SimpleNamespace(id=bid)

    def retrieve(self, bid):
        return SimpleNamespace(processing_status="ended", request_counts=None)

    def results(self, bid):
        for r in self.created[bid]:
            user = r["params"]["messages"][0]["content"]
            if user in self.fail:
                yield SimpleNamespace(
                    custom_id=r["custom_id"],
                    result=SimpleNamespace(type="errored", error=SimpleNamespace(type="api_error")),
                )
                continue
            msg = SimpleNamespace(
                content=[SimpleNamespace(type="text", text=f"echo {user}")],
                model=r["params"]["model"],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=self.usage[0], output_tokens=self.usage[1]),
            )
            yield SimpleNamespace(
                custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg)
            )


def fake_client(**kw):
    return SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches(**kw)))


def run(requests, cache, led, client, batch_dir, **kw):
    return run_batch_cached(
        requests,
        cache,
        led,
        client,
        poll_seconds=0,
        batch_dir=batch_dir,
        log=lambda s: None,
        sleep=lambda s: None,
        **kw,
    )


class TestBatchRunner:
    def test_submits_only_uncached_and_caches_results(self, tmp_path):
        cache, led, client = ResponseCache(tmp_path / "c"), ledger(tmp_path), fake_client()
        cached = req(0)
        cache.put(cached, GenerationResponse("old", "m", "end_turn", 1, 1, 1.0, "t"))
        out = run([cached, req(1), req(2), req(1)], cache, led, client, tmp_path / "b")
        assert (out.n_requested, out.n_cached_before, out.n_submitted, out.n_succeeded) == (
            3,
            1,
            2,
            2,
        )
        sent = [
            r["params"]["messages"][0]["content"]
            for rs in client.messages.batches.created.values()
            for r in rs
        ]
        assert sorted(sent) == ["user 1", "user 2"]
        resp = cache.get(req(1).cache_key)
        assert resp.text == "echo user 1" and resp.extra["service"] == "batch"
        # 2 calls x (1000 in x $2 + 100 out x $10) / 1e6, at half price
        assert abs(out.new_cost_usd - 2 * 0.003 * 0.5) < 1e-12
        assert abs(led.state["total_usd"] - out.new_cost_usd) < 1e-12
        assert led._reserved == pytest.approx(0.0)
        assert pending_records(tmp_path / "b") == []

    def test_failures_are_reported_and_left_uncached(self, tmp_path):
        cache, led = ResponseCache(tmp_path / "c"), ledger(tmp_path)
        out = run([req(1), req(2)], cache, led, fake_client(fail={"user 2"}), tmp_path / "b")
        assert out.n_succeeded == 1 and len(out.failures) == 1
        assert out.failures[0]["type"] == "errored" and not cache.has(req(2).cache_key)

    def test_reattaches_to_a_pending_batch_instead_of_resubmitting(self, tmp_path):
        cache, led, client = ResponseCache(tmp_path / "c"), ledger(tmp_path), fake_client()
        batches = client.messages.batches
        pending = [req(1), req(2)]
        batches.created["batch_old"] = [
            {"custom_id": r.cache_key, "params": AnthropicBackend.call_params(r)} for r in pending
        ]
        rec = {
            "batch_id": "batch_old",
            "status": "submitted",
            "reserved_usd": 0.01,
            "requests": {r.cache_key: r.__dict__ for r in pending},
        }
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "batch_old.json").write_text(json.dumps(rec))
        out = run(pending, cache, led, client, tmp_path / "b")
        assert out.n_submitted == 0 and list(batches.created) == ["batch_old"]
        assert all(cache.has(r.cache_key) for r in pending)
        assert led._reserved == pytest.approx(0.0)

    def test_packs_batches_under_the_cap(self, tmp_path):
        # Worst case per request: (7 est. input tokens x $2 + 100 x $10) / 1e6 x 0.5 = $0.000507;
        # real usage (5 in, 50 out) costs $0.000255. Cap $0.002: 3 fit in the first batch, the
        # other 2 only after it settles.
        cache, led, client = (
            ResponseCache(tmp_path / "c"),
            ledger(tmp_path, phase_cap=0.002),
            fake_client(usage=(5, 50)),
        )
        out = run([req(i) for i in range(5)], cache, led, client, tmp_path / "b")
        assert out.n_succeeded == 5
        assert [len(v) for v in client.messages.batches.created.values()] == [3, 2]
        assert led.phase_spent() <= 0.002

    def test_waits_instead_of_sending_slivers(self, tmp_path):
        # Worst case $0.000507 per request, real cost $0.000255; cap $0.0017, at most 2 per
        # batch. After the first batch of 2, headroom fits 1 more: with in-flight batches
        # and 3 still queued, that sliver waits; the tail of 1 is sent when it is all left.
        cache, led, client = (
            ResponseCache(tmp_path / "c"),
            ledger(tmp_path, phase_cap=0.0017),
            fake_client(usage=(5, 50)),
        )
        out = run(
            [req(i) for i in range(5)],
            cache,
            led,
            client,
            tmp_path / "b",
            max_requests_per_batch=2,
            min_requests_per_batch=2,
        )
        assert out.n_succeeded == 5
        assert [len(v) for v in client.messages.batches.created.values()] == [2, 2, 1]

    def test_refuses_when_nothing_fits(self, tmp_path):
        cache, led = ResponseCache(tmp_path / "c"), ledger(tmp_path, phase_cap=0.0)
        with pytest.raises(BudgetExceeded):
            run([req(1)], cache, led, fake_client(), tmp_path / "b")


# --------------------------------------------------------------------------------------
# Sampling


def trace(
    fid, cond="retrieved", model="m1", prompt="v1", answer="Revenue was $5M.", in_corpus=True
):
    return {
        "trace_id": f"{cond}__{model}__{prompt}__{fid}",
        "question": {
            "financebench_id": fid,
            "question": "What was revenue?",
            "question_type": "novel-generated",
            "in_corpus": in_corpus,
        },
        "retrieval": {
            "condition": cond,
            "k": 5,
            "context_metrics": {"recall@5": 1.0} if in_corpus else None,
            "chunks": [
                {
                    "label": "C1",
                    "chunk_id": "D:1",
                    "doc_id": "D",
                    "page": 1,
                    "section": None,
                    "char_start": 0,
                    "char_end": 100,
                    "text": "Revenue was $5 million.",
                },
                {
                    "label": "C2",
                    "chunk_id": "D:2",
                    "doc_id": "D",
                    "page": 2,
                    "section": None,
                    "char_start": 100,
                    "char_end": 200,
                    "text": "Other text.",
                },
            ],
        },
        "generation": {"model_key": model},
        "prompt": {"id": prompt, "version": 2},
        "parsed": {
            "status": "ok",
            "answer": answer,
            "abstained_token": False,
            "citations": ["C1"],
            "quotes": [],
            "confidence": 90,
        },
    }


class TestSampling:
    def traces(self):
        out = []
        for i in range(12):
            fid = f"q{i:02d}"
            for model in ("m1", "m2"):
                for prompt in ("v1", "v2"):
                    out.append(trace(fid, "retrieved", model, prompt, in_corpus=i < 9))
                    if i < 9:
                        out.append(trace(fid, "oracle", model, prompt))
        return out

    def test_pilot_questions_are_seeded_and_stratified(self):
        ts = self.traces()
        a = pilot_questions(ts, 2, 1, seed=0)
        assert a == pilot_questions(ts, 2, 1, seed=0) and len(a) == 3
        assert sum(fid >= "q09" for fid in a) == 1  # one out-of-corpus question

    def test_pilot_sample_spreads_prompts(self):
        ts = self.traces()
        s = pilot_sample(ts, ["q00", "q01"], per_cell=2, seed=0)
        assert len(s) == 2 * 4  # 2 per (condition, model) cell
        for cell in {(t["retrieval"]["condition"], t["generation"]["model_key"]) for t in s}:
            prompts = [
                t["prompt"]["id"]
                for t in s
                if (t["retrieval"]["condition"], t["generation"]["model_key"]) == cell
            ]
            assert sorted(prompts) == ["v1", "v2"]

    def test_validation_quotas_balance_models_and_conditions(self):
        q = validation_quotas({"a": 25, "b": 25}, ["retrieved", "oracle"])
        assert q == {
            ("retrieved", "a"): 13,
            ("oracle", "a"): 12,
            ("retrieved", "b"): 12,
            ("oracle", "b"): 13,
        }

    def test_validation_candidates_exclude_pilot_and_rotate_cells(self):
        ts = self.traces()
        quotas = validation_quotas({"m1": 2, "m2": 2}, ["retrieved", "oracle"])
        cands = list(validation_candidates(ts, {"q00"}, quotas, seed=0))
        assert all(t["question"]["financebench_id"] != "q00" for _, t in cands)
        assert [c for c, _ in cands[:4]] == sorted(quotas)


# --------------------------------------------------------------------------------------
# Pipeline: two judge stages into one record


class StubJudge:
    """Duck-types Judge: canned outputs keyed by purpose."""

    def __init__(self, outputs: dict[str, str]):
        self.outputs = outputs
        self.requested: list[str] = []

    def log(self, _):
        pass

    def request(self, purpose, values, schema):
        self.requested.append(purpose)
        return GenerationRequest(
            "anthropic", "judge", purpose, json.dumps(values, sort_keys=True), 1, {}
        )

    def run(self, requests):
        return {
            r.cache_key: GenerationResponse(
                self.outputs[r.system], "judge", "end_turn", 1, 1, 0.0, "t"
            )
            for r in requests
        }


DECOMP = json.dumps(
    {
        "claims": [{"claim": "Acme's revenue was $5 million.", "kind": "document"}],
        "response_type": "answered",
    }
)
VERIFY = json.dumps(
    {
        "verdicts": [
            {
                "claim_id": 1,
                "reason": "C1 says so.",
                "verdict": "supported",
                "supporting_excerpts": ["C1"],
            }
        ]
    }
)
GRADE = json.dumps({"reason": "Matches.", "grade": "correct", "unit_error": False})


def inputs_for(traces, gold="Revenue was $5 million.", question="What was revenue?"):
    rows = {
        t["question"]["financebench_id"]: {
            "answer": gold,
            "justification": "",
            "question": question,
        }
        for t in traces
    }
    numeric = {fid: g for fid, r in rows.items() if (g := numeric_gold(r["answer"], question))}
    inp = Inputs(traces, rows, numeric, {}, 0.5)
    inp.evidence = {t["trace_id"]: ["C1"] for t in traces}
    return inp


class TestPipeline:
    def test_answered_trace(self):
        t = trace("q1")
        inp = inputs_for([t])
        judged = judge_traces(
            [t], inp, StubJudge({"decompose": DECOMP, "verify": VERIFY, "correctness": GRADE})
        )
        rec = evaluate_trace(t, inp, judged[t["trace_id"]], 0.01, ())
        assert rec["abstention"]["status"] == "answered" and rec["abstention"]["source"] == "judge"
        assert rec["correctness"] == {
            "label": "correct",
            "grader": "judge",
            "numeric": None,
            "judge": {"grade": "correct", "unit_error": False, "reason": "Matches."},
        }
        assert rec["groundedness"]["groundedness"] == 1.0
        assert rec["groundedness"]["claims"][0]["verdict"] == "supported"
        assert (
            rec["citations"]["citation_precision"] == 1.0 and rec["citations"]["gold_cited"] is True
        )
        assert rec["judge_errors"] == {}

    def test_numeric_grader_takes_precedence(self):
        t = trace("q1", answer="$4 million")
        inp = inputs_for([t], gold="$5.00", question="Revenue? Answer in USD millions.")
        judged = judge_traces(
            [t], inp, StubJudge({"decompose": DECOMP, "verify": VERIFY, "correctness": GRADE})
        )
        rec = evaluate_trace(t, inp, judged[t["trace_id"]], 0.01, ())
        assert (
            rec["correctness"]["label"] == "incorrect" and rec["correctness"]["grader"] == "numeric"
        )
        assert rec["correctness"]["judge"]["grade"] == "correct"  # kept for the cross-check

    def test_declined_is_not_graded(self):
        t = trace("q1", answer="The excerpts do not contain revenue.")
        stub = StubJudge({"decompose": json.dumps({"claims": [], "response_type": "declined"})})
        inp = inputs_for([t])
        judged = judge_traces([t], inp, stub)
        assert stub.requested == ["decompose"]
        rec = evaluate_trace(t, inp, judged[t["trace_id"]], 0.01, ())
        assert rec["correctness"]["label"] == "abstained"
        assert (
            rec["groundedness"]["groundedness"] is None and rec["citations"]["gold_cited"] is None
        )

    def test_bare_token_skips_the_judge(self):
        t = trace("q1", answer="NOT_IN_DOCUMENTS")
        t["parsed"]["abstained_token"] = True
        stub = StubJudge({})
        judged = judge_traces([t], inputs_for([t]), stub)
        assert stub.requested == [] and judged[t["trace_id"]] == {}

    def test_judge_failure_is_recorded_not_guessed(self):
        t = trace("q1")
        inp = inputs_for([t])
        judged = judge_traces(
            [t], inp, StubJudge({"decompose": DECOMP, "verify": "{oops", "correctness": GRADE})
        )
        rec = evaluate_trace(t, inp, judged[t["trace_id"]], 0.01, ())
        assert rec["groundedness"] is None
        assert rec["judge_errors"]["verify"].startswith("invalid_output")


# --------------------------------------------------------------------------------------
# Labelling page


def test_labeling_page_embeds_data_safely():
    from src.evaluation.labeling_page import render_labeling_page

    items = [
        {"item_id": "t", "question": "q", "answer": "a </script><b>", "excerpts": [], "claims": []}
    ]
    html = render_labeling_page(items, "abc123")
    blob = html.split('id="data">')[1].split("</script>")[0]
    data = json.loads(blob)
    assert data["sample_sha256"] == "abc123" and data["items"][0]["answer"] == "a </script><b>"
    # nothing the page hides is present in the embedded data
    assert not {"trace_id", "model_key", "condition", "gold", "citations"} & set(data["items"][0])


class TestPremiseParsing:
    def test_valid(self):
        from src.evaluation.judge import parse_premise

        out = parse_premise('{"reason": "It says sales fell.", "handling": "rejects_premise"}')
        assert out == {"handling": "rejects_premise", "reason": "It says sales fell."}

    @pytest.mark.parametrize(
        "text",
        ['{"reason": "x", "handling": "partly"}', '{"reason": "x"}', "not json", "[]"],
    )
    def test_invalid(self, text):
        from src.evaluation.judge import JudgeOutputError, parse_premise

        with pytest.raises(JudgeOutputError):
            parse_premise(text)
