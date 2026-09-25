"""Unit tests for src/generation/: prompt registry, rendering, output parsing, response
cache, spend ledger, backends (against fake clients), and trace replay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.generation.llm import (
    AnthropicBackend,
    BudgetExceeded,
    CacheMiss,
    ContextOverflow,
    GenerationRequest,
    GenerationResponse,
    OllamaBackend,
    ResponseCache,
    SpendLedger,
    generate_cached,
)
from src.generation.parsing import parse_output
from src.generation.prompts import (
    format_context,
    load_registry,
    parse_prompt_file,
    render,
)
from src.generation.trace import build_trace, replay_trace, trace_chunks

REPO_PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
V1 = ("ANSWER", "CITATIONS", "CONFIDENCE")
V2 = ("ANSWER", "CITATIONS", "QUOTES", "CONFIDENCE")
V3 = ("REASONING", "ANSWER", "CITATIONS", "CONFIDENCE")

PROMPT_FILE = """---
id: t1
version: 1
variant: test
output_fields: [ANSWER, CITATIONS, CONFIDENCE]
description: test prompt
---
<!-- system -->
System text.
<!-- user -->
Excerpts:

{{context}}

Question: {{question}}
"""


def chunk(i: int, text: str = "Revenue was $1,577 million.", section=None) -> dict:
    return {
        "chunk_id": f"D:fixed_size:{i}",
        "doc_id": "3M_2018_10K",
        "page": 10 + i,
        "section": section,
        "char_start": 0,
        "char_end": len(text),
        "is_table": False,
        "text": text,
    }


# --------------------------------------------------------------------------------------
# Registry and rendering


class TestRegistry:
    def test_repo_prompts_load(self):
        reg = load_registry(REPO_PROMPTS)
        assert set(reg) == {
            "v1_zero_shot",
            "v2_citation_required",
            "v3_chain_of_thought",
            "v4_abstention",
        }
        assert reg["v2_citation_required"].output_fields == V2
        assert reg["v3_chain_of_thought"].output_fields == V3

    def test_variants_share_the_base_text(self):
        # Design rule: every variant = the same base + one added instruction. The first two
        # system paragraphs are the base and must be identical across all four prompts.
        reg = load_registry(REPO_PROMPTS)
        bases = {pid: "\n\n".join(t.system.split("\n\n")[:2]) for pid, t in reg.items()}
        assert len(set(bases.values())) == 1
        users = {t.user for t in reg.values()}
        assert len(users) == 1

    def test_confidence_line_identical_and_neutral(self):
        # The confidence question must mean the same thing in every variant, and must not
        # hint that declining is allowed (that permission is v4's manipulation alone).
        reg = load_registry(REPO_PROMPTS)
        lines = {
            next(ln for ln in t.system.splitlines() if ln.startswith("CONFIDENCE:"))
            for t in reg.values()
        }
        assert len(lines) == 1
        line = lines.pop().lower()
        assert not any(w in line for w in ("not contain", "abstain", "decline"))

    def test_only_v4_mentions_the_abstention_token(self):
        reg = load_registry(REPO_PROMPTS)
        mentions = {pid for pid, t in reg.items() if "NOT_IN_DOCUMENTS" in t.system}
        assert mentions == {"v4_abstention"}

    def test_missing_placeholder_rejected(self):
        with pytest.raises(ValueError, match="question"):
            parse_prompt_file(PROMPT_FILE.replace("{{question}}", "?"))

    def test_duplicate_placeholder_rejected(self):
        with pytest.raises(ValueError, match="context"):
            parse_prompt_file(PROMPT_FILE.replace("Excerpts:", "{{context}}"))

    def test_unknown_output_field_rejected(self):
        with pytest.raises(ValueError, match="unknown output_fields"):
            parse_prompt_file(PROMPT_FILE.replace("CONFIDENCE]", "SCORE]"))

    def test_file_hash_covers_front_matter(self):
        a = parse_prompt_file(PROMPT_FILE)
        b = parse_prompt_file(PROMPT_FILE.replace("version: 1", "version: 2"))
        assert a.file_sha256 != b.file_sha256


class TestRender:
    def test_labels_and_headers(self):
        ctx = format_context([chunk(1), chunk(2, section="Item 7")])
        assert ctx.startswith("[C1] 3M_2018_10K | page 11\nRevenue")
        assert "[C2] 3M_2018_10K | page 12 | Item 7\n" in ctx

    def test_single_pass_substitution(self):
        # Filing text containing a literal placeholder must not be substituted again.
        t = parse_prompt_file(PROMPT_FILE)
        r = render(t, "What was revenue?", [chunk(1, "odd text {{question}} here")])
        assert "odd text {{question}} here" in r.user
        assert r.user.endswith("Question: What was revenue?")

    def test_hash_is_stable_and_sensitive(self):
        t = parse_prompt_file(PROMPT_FILE)
        a = render(t, "Q?", [chunk(1)])
        assert a.sha256 == render(t, "Q?", [chunk(1)]).sha256
        assert a.sha256 != render(t, "Q?", [chunk(1, "Revenue was $1,578 million.")]).sha256


# --------------------------------------------------------------------------------------
# Parsing


class TestParse:
    def test_clean_v1(self):
        p = parse_output("ANSWER: $1,577 million\nCITATIONS: C2, C4\nCONFIDENCE: 85", V1, 5)
        assert (p.status, p.answer, p.citations, p.confidence) == (
            "ok",
            "$1,577 million",
            ["C2", "C4"],
            85,
        )
        assert p.issues == []

    def test_markdown_bold_labels(self):
        text = "**ANSWER:** 12.5%\n**CITATIONS:** [C1]\n**CONFIDENCE:** 70%"
        p = parse_output(text, V1, 5)
        assert (p.status, p.answer, p.citations, p.confidence) == ("ok", "12.5%", ["C1"], 70)

    def test_none_citations_is_explicit_empty(self):
        p = parse_output("ANSWER: NOT_IN_DOCUMENTS\nCITATIONS: NONE\nCONFIDENCE: 90", V1, 5)
        assert p.status == "ok" and p.citations == [] and p.abstained_token

    def test_unlabelled_token_is_recorded_on_failed_parse(self):
        # Seen from Sonnet 5 (oracle, v4): the token written without its ANSWER label.
        p = parse_output("NOT_IN_DOCUMENTS\n\nCITATIONS: NONE\nCONFIDENCE: 90", V1, 5)
        assert p.status == "failed" and p.answer is None and p.abstained_token

    def test_failed_parse_without_token(self):
        p = parse_output("The revenue was $1,577 million.", V1, 5)
        assert p.status == "failed" and not p.abstained_token

    def test_free_text_decline_is_not_the_token(self):
        # Semantic abstention detection is a Phase 5 metric; the parser only records the token.
        p = parse_output(
            "ANSWER: The excerpts do not state this.\nCITATIONS: NONE\nCONFIDENCE: 20", V1, 5
        )
        assert not p.abstained_token

    def test_out_of_range_citation(self):
        p = parse_output("ANSWER: x\nCITATIONS: C1, C7\nCONFIDENCE: 50", V1, 5)
        assert p.citations == ["C1"] and p.invalid_citations == ["C7"]
        assert "citation_out_of_range" in p.issues

    def test_missing_answer_fails(self):
        p = parse_output("The revenue was $1,577 million.", V1, 5)
        assert p.status == "failed" and p.answer is None

    def test_missing_confidence_is_partial(self):
        p = parse_output("ANSWER: x\nCITATIONS: C1", V1, 5)
        assert p.status == "partial" and "missing_confidence" in p.issues

    @pytest.mark.parametrize("raw", ["0.85", "high", "150", "about 80"])
    def test_malformed_confidence_kept_raw_not_coerced(self, raw):
        p = parse_output(f"ANSWER: x\nCITATIONS: C1\nCONFIDENCE: {raw}", V1, 5)
        assert p.confidence is None and p.confidence_raw == raw and p.status == "partial"

    def test_v2_quotes_with_curly_quotes_and_continuation(self):
        text = (
            "ANSWER: $1,577 million\nCITATIONS: C1, C3\nQUOTES:\n"
            "C1: “Purchases of property, plant and equipment (1,577)”\n"
            '- C3: "Capital spending was\n$1.6 billion"\nCONFIDENCE: 90'
        )
        p = parse_output(text, V2, 5)
        assert p.status == "ok"
        assert p.quotes == [
            {"label": "C1", "text": "Purchases of property, plant and equipment (1,577)"},
            {"label": "C3", "text": "Capital spending was\n$1.6 billion"},
        ]

    def test_v2_citations_without_quotes_flagged(self):
        p = parse_output("ANSWER: x\nCITATIONS: C1\nQUOTES:\nCONFIDENCE: 60", V2, 5)
        assert "citations_without_quotes" in p.issues

    def test_v3_reasoning_with_midline_answer_word(self):
        text = (
            "REASONING: From C2 the figure is 1,577; my answer: that value.\n"
            "Step 2: no calculation needed.\nANSWER: $1,577 million\nCITATIONS: C2\n"
            "CONFIDENCE: 80"
        )
        p = parse_output(text, V3, 5)
        assert p.status == "ok" and p.answer == "$1,577 million"
        assert p.reasoning.startswith("From C2") and "Step 2" in p.reasoning

    def test_v3_missing_reasoning_is_partial(self):
        p = parse_output("ANSWER: x\nCITATIONS: C1\nCONFIDENCE: 50", V3, 5)
        assert p.status == "partial" and "missing_reasoning" in p.issues

    def test_label_chained_after_empty_answer(self):
        # Seen from the 3B: the answer slot left empty, the next field on the same line.
        # The answer must not become the text "CONFIDENCE: 80".
        p = parse_output("ANSWER: CONFIDENCE: 80\nCITATIONS: C1, C5", V1, 5)
        assert p.status == "failed" and p.answer is None
        assert "empty_answer_chained" in p.issues

    def test_chained_label_citations_recovered(self):
        text = "ANSWER: 1.21\nCITATIONS: CONFIDENCE: 70"
        p = parse_output(text, V1, 5)
        assert p.answer == "1.21" and p.citations == [] and p.confidence == 70
        assert "empty_citations_chained" in p.issues and "citations_unparseable" not in p.issues
        assert p.status == "partial"  # an empty CITATIONS field is a format deviation

    def test_blank_citations_line_is_empty_not_unparseable(self):
        p = parse_output("ANSWER: NONE\nCITATIONS: \nCONFIDENCE: 40", V1, 5)
        assert "empty_citations" in p.issues and "citations_unparseable" not in p.issues
        assert p.status == "partial"

    def test_answer_starting_with_ordinary_word_is_not_split(self):
        p = parse_output("ANSWER: Answers vary: 12%\nCITATIONS: C1\nCONFIDENCE: 50", V1, 5)
        assert p.answer == "Answers vary: 12%"

    def test_repeated_label_keeps_last_and_reports(self):
        text = "ANSWER: draft\nANSWER: final\nCITATIONS: C1\nCONFIDENCE: 50"
        p = parse_output(text, V1, 5)
        assert p.answer == "final" and "duplicate_answer" in p.issues


# --------------------------------------------------------------------------------------
# Cache and ledger


def req(**kw) -> GenerationRequest:
    base = dict(
        backend="anthropic",
        model="m",
        system="s",
        user="u",
        max_tokens=100,
        params={"thinking": {"type": "disabled"}},
    )
    base.update(kw)
    return GenerationRequest(**base)


def resp(text="ANSWER: x\nCITATIONS: C1\nCONFIDENCE: 50", **kw) -> GenerationResponse:
    base = dict(
        text=text,
        model_reported="m",
        stop_reason="end_turn",
        input_tokens=1000,
        output_tokens=100,
        latency_ms=12.0,
        created_utc="2026-09-24T00:00:00+00:00",
    )
    base.update(kw)
    return GenerationResponse(**base)


class TestCacheKey:
    def test_param_order_does_not_matter(self):
        a = req(params={"options": {"seed": 0, "temperature": 0}})
        b = req(params={"options": {"temperature": 0, "seed": 0}})
        assert a.cache_key == b.cache_key

    @pytest.mark.parametrize(
        "change",
        [{"model": "m2"}, {"system": "s2"}, {"user": "u2"}, {"max_tokens": 101}, {"params": {}}],
    )
    def test_any_change_is_a_miss(self, change):
        assert req().cache_key != req(**change).cache_key

    def test_roundtrip(self, tmp_path):
        cache = ResponseCache(tmp_path)
        r = req()
        assert cache.get(r.cache_key) is None
        cache.put(r, resp())
        assert cache.get(r.cache_key) == resp()


def ledger(tmp_path, project=1.0, phase=0.5) -> SpendLedger:
    return SpendLedger(
        tmp_path / "spend.json", {"m": {"input": 2.0, "output": 10.0}}, project, "p4", phase
    )


class TestLedger:
    def test_settle_records_actual_cost_and_persists(self, tmp_path):
        led = ledger(tmp_path)
        reserved = led.reserve(req())
        cost = led.settle(reserved, "m", 1000, 100)
        assert cost == pytest.approx((1000 * 2 + 100 * 10) / 1e6)
        again = ledger(tmp_path)
        assert again.state["total_usd"] == pytest.approx(cost)
        assert again.state["by_phase"]["p4"]["n_calls"] == 1

    def test_refuses_call_that_could_exceed_phase_cap(self, tmp_path):
        led = ledger(tmp_path, phase=0.0005)
        with pytest.raises(BudgetExceeded, match="p4 cap"):
            led.reserve(req(max_tokens=100))  # worst case 100 output tokens = $0.001

    def test_in_flight_reservations_count(self, tmp_path):
        led = ledger(tmp_path, phase=0.0015)
        led.reserve(req(max_tokens=100))
        with pytest.raises(BudgetExceeded):
            led.reserve(req(max_tokens=100))

    def test_unpriced_model_refused(self, tmp_path):
        with pytest.raises(BudgetExceeded, match="no price"):
            ledger(tmp_path).reserve(req(model="unknown"))

    def test_api_call_without_ledger_refused(self, tmp_path):
        backend = SimpleNamespace(name="anthropic", generate=lambda r: resp())
        with pytest.raises(BudgetExceeded):
            generate_cached(backend, req(), ResponseCache(tmp_path))

    def test_cached_call_costs_nothing(self, tmp_path):
        calls = []
        backend = SimpleNamespace(name="anthropic", generate=lambda r: calls.append(r) or resp())
        cache, led = ResponseCache(tmp_path / "c"), ledger(tmp_path)
        _, cached1, cost1 = generate_cached(backend, req(), cache, led)
        _, cached2, cost2 = generate_cached(backend, req(), cache, led)
        assert (cached1, cached2, len(calls)) == (False, True, 1)
        assert cost1 > 0 and cost2 == 0.0
        assert led.state["n_calls"] == 1

    def test_failed_call_releases_reservation(self, tmp_path):
        def boom(r):
            raise ConnectionError("network")

        led = ledger(tmp_path, phase=0.0015)
        backend = SimpleNamespace(name="anthropic", generate=boom)
        with pytest.raises(ConnectionError):
            generate_cached(backend, req(max_tokens=100), ResponseCache(tmp_path), led)
        led.reserve(req(max_tokens=100))  # would raise if the failed reservation leaked


class TestReplayOnly:
    def test_miss_raises_without_calling_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RAG_REPLAY_ONLY", "1")
        calls = []
        backend = SimpleNamespace(name="ollama", generate=lambda r: calls.append(r) or resp())
        with pytest.raises(CacheMiss):
            generate_cached(backend, req(), ResponseCache(tmp_path))
        assert calls == []

    def test_hit_is_served(self, tmp_path, monkeypatch):
        cache = ResponseCache(tmp_path)
        cache.put(req(), resp())
        monkeypatch.setenv("RAG_REPLAY_ONLY", "1")
        backend = SimpleNamespace(name="ollama", generate=lambda r: pytest.fail("called"))
        _, was_cached, cost = generate_cached(backend, req(), cache)
        assert (was_cached, cost) == (True, 0.0)

    def test_batch_miss_raises_before_submitting(self, tmp_path, monkeypatch):
        from src.generation.batch import run_batch_cached

        monkeypatch.setenv("RAG_REPLAY_ONLY", "1")
        client = SimpleNamespace()  # any attribute access would raise
        with pytest.raises(CacheMiss, match="1 batch requests"):
            run_batch_cached(
                [req()],
                ResponseCache(tmp_path / "c"),
                ledger(tmp_path),
                client,
                batch_dir=tmp_path / "b",
            )


# --------------------------------------------------------------------------------------
# Backends against fake clients


class FakeAnthropicMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ANSWER: x")],
            model="claude-sonnet-5",
            stop_reason="end_turn",
            stop_details=None,
            usage=SimpleNamespace(input_tokens=900, output_tokens=40),
            _request_id="req_1",
        )


class TestAnthropicBackend:
    def test_request_shape(self):
        fake = FakeAnthropicMessages()
        backend = AnthropicBackend(client=SimpleNamespace(messages=fake))
        r = req(model="claude-sonnet-5", params=AnthropicBackend.request_params("disabled"))
        out = backend.generate(r)
        assert fake.kwargs["thinking"] == {"type": "disabled"}
        assert "temperature" not in fake.kwargs  # rejected by current models
        assert fake.kwargs["system"] == "s"
        assert fake.kwargs["messages"] == [{"role": "user", "content": "u"}]
        assert (out.text, out.model_reported, out.input_tokens, out.request_id) == (
            "ANSWER: x",
            "claude-sonnet-5",
            900,
            "req_1",
        )


class FakeOllama:
    def __init__(self):
        self.kwargs = None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            message=SimpleNamespace(content="ANSWER: y"),
            model="qwen2.5:3b-instruct",
            done_reason="stop",
            prompt_eval_count=800,
            eval_count=30,
            load_duration=0,
            prompt_eval_duration=1e6,
            eval_duration=2e6,
        )

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model="qwen2.5:3b-instruct", digest="abc")])


class TestOllamaBackend:
    def test_options_and_digest(self):
        fake = FakeOllama()
        r = req(
            backend="ollama",
            model="qwen2.5:3b-instruct",
            params=OllamaBackend.request_params(0, 0, 8192),
        )
        out = OllamaBackend(client=fake).generate(r)
        assert fake.kwargs["options"] == {
            "temperature": 0,
            "seed": 0,
            "num_ctx": 8192,
            "num_predict": 100,
        }
        assert out.extra["digest"] == "abc" and out.text == "ANSWER: y"

    def test_prompt_that_would_be_truncated_is_refused(self):
        fake = FakeOllama()
        r = req(
            backend="ollama",
            model="qwen2.5:3b-instruct",
            user="x" * 20_000,
            max_tokens=2048,
            params=OllamaBackend.request_params(0, 0, 8192),
        )
        with pytest.raises(ContextOverflow):
            OllamaBackend(client=fake).generate(r)
        assert fake.kwargs is None  # never sent


# --------------------------------------------------------------------------------------
# Trace and replay


def make_trace(tmp_path, template, raw):
    results = [SimpleNamespace(score=0.9 - i / 10, metadata=chunk(i)) for i in range(1, 4)]
    chunks = trace_chunks(results)
    question = {"financebench_id": "fb_1", "question": "What was revenue?", "in_corpus": True}
    rendered = render(template, question["question"], chunks)
    r = GenerationRequest(
        "anthropic", "m", rendered.system, rendered.user, 100, {"thinking": {"type": "disabled"}}
    )
    cache = ResponseCache(tmp_path / "cache")
    response = resp(text=raw)
    cache.put(r, response)
    trace = build_trace(
        question=question,
        retrieval={"condition": "retrieved", "config": "c", "k": 3, "chunks": chunks},
        template=template,
        request=r,
        model_key="mk",
        response=response,
        parsed=parse_output(raw, template.output_fields, len(chunks)),
        rendered_sha256=rendered.sha256,
        cost_usd=0.001,
    )
    return json.loads(json.dumps(trace)), cache  # through JSON, as when read from disk


class TestReplay:
    RAW = "ANSWER: $1,577 million\nCITATIONS: C1\nCONFIDENCE: 80"

    def test_roundtrip_ok(self, tmp_path):
        t = parse_prompt_file(PROMPT_FILE)
        trace, cache = make_trace(tmp_path, t, self.RAW)
        r = replay_trace(trace, {"t1": t}, cache)
        assert r.ok and r.output_source == "cache" and r.problems == []

    def test_without_cache_uses_trace_output(self, tmp_path):
        t = parse_prompt_file(PROMPT_FILE)
        trace, _ = make_trace(tmp_path, t, self.RAW)
        r = replay_trace(trace, {"t1": t}, ResponseCache(tmp_path / "empty"))
        assert r.ok and r.output_source == "trace_only" and r.output_ok is None

    def test_edited_prompt_file_detected(self, tmp_path):
        t = parse_prompt_file(PROMPT_FILE)
        trace, cache = make_trace(tmp_path, t, self.RAW)
        edited = parse_prompt_file(PROMPT_FILE.replace("System text.", "System text!"))
        r = replay_trace(trace, {"t1": edited}, cache)
        assert not r.ok and not r.prompt_file_ok and not r.prompt_render_ok

    def test_tampered_output_detected(self, tmp_path):
        t = parse_prompt_file(PROMPT_FILE)
        trace, cache = make_trace(tmp_path, t, self.RAW)
        trace["raw_output"] = trace["raw_output"].replace("1,577", "1,578")
        r = replay_trace(trace, {"t1": t}, cache)
        assert not r.ok and r.output_ok is False and not r.parse_ok

    def test_tampered_chunk_text_detected(self, tmp_path):
        t = parse_prompt_file(PROMPT_FILE)
        trace, cache = make_trace(tmp_path, t, self.RAW)
        trace["retrieval"]["chunks"][0]["text"] = "Revenue was $9 million."
        r = replay_trace(trace, {"t1": t}, cache)
        assert not r.prompt_render_ok and not r.cache_key_ok
