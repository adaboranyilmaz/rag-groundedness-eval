"""RAGAS faithfulness comparison (src/external/faithfulness_comparison.py) and the RAGAS
transport adapter (src/external/ragas_llm.py).

The comparison tests run everywhere. The adapter tests need ragas, which lives in its own
environment (pyproject.toml group `ragas`); they are skipped elsewhere and run with:
    UV_PROJECT_ENVIRONMENT=.venv-ragas uv run --frozen --no-default-groups --group ragas \
        --group dev pytest tests/test_ragas.py
"""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from src.external.faithfulness_comparison import (
    average_ranks,
    build_per_answer,
    claim_rater_scores,
    compare_all,
    compare_pair,
    kappa_difference,
    spearman,
)
from src.generation.llm import (
    CacheMiss,
    GenerationResponse,
    ResponseCache,
    SpendLedger,
)

# --------------------------------------------------------------------------------------
# Comparison


def claim(tid, kind, human, judge, second="supported"):
    return {"trace_id": tid, "kind": kind, "human": human, "judge": judge, "second_judge": second}


class TestClaimRaterScores:
    def test_document_claims_only(self):
        rows = [
            claim("a", "document", "supported", "supported"),
            claim("a", "document", "unsupported", "supported"),
            claim("a", "context", "unsupported", "unsupported"),  # not in the score
        ]
        h = claim_rater_scores(rows, "human")["a"]
        j = claim_rater_scores(rows, "judge")["a"]
        assert h == {"score": 0.5, "fully_grounded": False, "n_document_claims": 2}
        assert j == {"score": 1.0, "fully_grounded": True, "n_document_claims": 2}

    def test_missing_verdict_makes_answer_unscored(self):
        rows = [claim("a", "document", "supported", None)]
        assert claim_rater_scores(rows, "judge")["a"]["score"] is None


class TestSpearman:
    def test_average_ranks_share_ties(self):
        assert list(average_ranks([0.5, 1.0, 0.5, 0.0])) == [2.5, 4.0, 2.5, 1.0]

    def test_monotone_is_one_and_reversed_is_minus_one(self):
        assert spearman([0.1, 0.4, 0.9], [1, 2, 3]) == pytest.approx(1.0)
        assert spearman([0.1, 0.4, 0.9], [3, 2, 1]) == pytest.approx(-1.0)

    def test_matches_pearson_on_ranks_with_ties(self):
        x, y = [1.0, 1.0, 0.5, 0.0, 1.0], [1.0, 0.5, 0.5, 0.0, 0.8]
        rx, ry = average_ranks(x), average_ranks(y)
        mx, my = rx.mean(), ry.mean()
        expected = ((rx - mx) * (ry - my)).sum() / math.sqrt(
            ((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum()
        )
        assert spearman(x, y) == pytest.approx(expected)

    def test_constant_side_is_undefined(self):
        assert spearman([1.0, 1.0, 1.0], [0.1, 0.5, 0.9]) is None


def per_answer_rows():
    items = [
        {"trace_id": t, "item_id": t, "model_key": m, "condition": "retrieved"}
        for t, m in [("a", "m1"), ("b", "m1"), ("c", "m2"), ("d", "m2")]
    ]
    rows = [
        claim("a", "document", "supported", "supported"),
        claim("b", "document", "unsupported", "unsupported"),
        claim("c", "document", "supported", "unsupported"),
        claim("d", "document", "unsupported", "supported"),
    ]
    ragas = {
        "a": {"score": 1.0, "n_statements": 2},
        "b": {"score": 0.5, "n_statements": 2},
        "c": {"score": None, "n_statements": 0, "error": "no_statements"},
        "d": {"score": 1.0, "n_statements": 1},
    }
    return build_per_answer(items, rows, ragas)


class TestCompare:
    def test_unscored_ragas_answer_left_out_of_ragas_pairs_only(self):
        pa = per_answer_rows()
        hr = compare_pair(pa, "human", "ragas", 200, 0)
        hj = compare_pair(pa, "human", "judge", 200, 0)
        assert hr["fully_grounded"]["n"] == 3  # c has no RAGAS score
        assert hj["fully_grounded"]["n"] == 4

    def test_fully_grounded_is_score_one(self):
        pa = {r["trace_id"]: r for r in per_answer_rows()}
        assert pa["a"]["ragas"]["fully_grounded"] is True
        assert pa["b"]["ragas"]["fully_grounded"] is False
        assert pa["c"]["ragas"]["fully_grounded"] is None

    def test_compare_all_counts_and_subsets(self):
        out = compare_all(per_answer_rows(), ["m1", "m2"], 200, 0)
        assert out["n_ragas_unscored"] == 1
        assert out["ragas_unscored"] == [{"trace_id": "c", "error": "no_statements"}]
        assert set(out["pairs"]["human__vs__ragas"]) == {"all", "m1", "m2"}
        # a, b, d: human T,F,F vs ragas T,F,T -> 2 of 3 agree
        fg = out["pairs"]["human__vs__ragas"]["all"]["fully_grounded"]
        assert fg["raw_agreement"] == pytest.approx(2 / 3)

    def test_kappa_difference_is_zero_for_identical_raters(self):
        pa = per_answer_rows()
        for r in pa:
            r["copy"] = dict(r["judge"])
        d = kappa_difference(pa, "human", "judge", "copy", 200, 0)
        assert d["kappa_diff"] == pytest.approx(0.0)
        assert d["kappa_diff_ci95"] == [pytest.approx(0.0), pytest.approx(0.0)]


# --------------------------------------------------------------------------------------
# Adapter (needs ragas)


class FakeBackend:
    """Answers RAGAS's two steps with fixed JSON, keyed on the response model's schema."""

    name = "anthropic"

    def __init__(self, stop_reason="end_turn"):
        self.requests = []
        self.stop_reason = stop_reason

    def generate(self, request):
        self.requests.append(request)
        props = request.params["output_config"]["format"]["schema"]["properties"]
        items = props["statements"]["items"]
        if items.get("type") == "string":  # statement generation
            text = json.dumps({"statements": ["X earned 5.", "X is in Y."]})
        else:  # NLI verdicts
            text = json.dumps(
                {
                    "statements": [
                        {"statement": "X earned 5.", "reason": "stated", "verdict": 1},
                        {"statement": "X is in Y.", "reason": "not stated", "verdict": 0},
                    ]
                }
            )
        return GenerationResponse(
            text=text,
            model_reported="claude-sonnet-5",
            stop_reason=self.stop_reason,
            input_tokens=1000,
            output_tokens=100,
            latency_ms=1.0,
            created_utc="2026-09-26T00:00:00+00:00",
        )


MODEL_CFG = {"backend": "anthropic", "model": "claude-sonnet-5", "thinking": "disabled"}


def make_ledger(tmp_path, cap=1.0):
    return SpendLedger(
        tmp_path / "spend.json",
        {"claude-sonnet-5": {"input": 2.0, "output": 10.0}},
        30.0,
        "ragas_faithfulness",
        cap,
    )


def score(llm):
    from ragas.metrics.collections import Faithfulness

    return asyncio.run(
        Faithfulness(llm=llm).ascore(
            user_input="What did X earn?", response="X earned 5 in Y.", retrieved_contexts=["c"]
        )
    )


class TestAdapter:
    @pytest.fixture(autouse=True)
    def _ragas(self):
        pytest.importorskip("ragas")

    def test_ragas_scores_through_the_adapter(self, tmp_path):
        from src.external.ragas_llm import ProjectRagasLLM

        backend = FakeBackend()
        llm = ProjectRagasLLM(
            MODEL_CFG, 2048, ResponseCache(tmp_path / "c"), make_ledger(tmp_path), backend
        )
        assert score(llm).value == pytest.approx(0.5)  # RAGAS's own arithmetic: 1 of 2
        assert [c["step"] for c in llm.calls] == ["StatementGeneratorOutput", "NLIStatementOutput"]
        req = backend.requests[0]
        assert req.system == "" and req.user.startswith("Given a question and an answer")
        assert not {"temperature", "top_p", "top_k"} & set(req.params)
        assert req.params["thinking"] == {"type": "disabled"}

    def test_second_run_is_served_from_the_cache(self, tmp_path):
        from src.external.ragas_llm import ProjectRagasLLM

        cache, ledger = ResponseCache(tmp_path / "c"), make_ledger(tmp_path)
        score(ProjectRagasLLM(MODEL_CFG, 2048, cache, ledger, FakeBackend()))
        again = FakeBackend()
        llm = ProjectRagasLLM(MODEL_CFG, 2048, cache, ledger, again)
        assert score(llm).value == pytest.approx(0.5)
        assert again.requests == [] and all(c["cached"] for c in llm.calls)
        spent = json.loads((tmp_path / "spend.json").read_text())
        assert spent["by_phase"]["ragas_faithfulness"]["n_calls"] == 2  # paid once

    def test_replay_only_miss_raises(self, tmp_path, monkeypatch):
        from src.external.ragas_llm import ProjectRagasLLM

        monkeypatch.setenv("RAG_REPLAY_ONLY", "1")
        llm = ProjectRagasLLM(MODEL_CFG, 2048, ResponseCache(tmp_path / "c"), None, FakeBackend())
        with pytest.raises(CacheMiss):
            score(llm)

    def test_cut_off_output_is_an_error_not_a_score(self, tmp_path):
        from src.external.ragas_llm import ProjectRagasLLM, RagasCallError

        llm = ProjectRagasLLM(
            MODEL_CFG,
            2048,
            ResponseCache(tmp_path / "c"),
            make_ledger(tmp_path),
            FakeBackend(stop_reason="max_tokens"),
        )
        with pytest.raises(RagasCallError, match="stop_reason_max_tokens"):
            score(llm)

    def test_strict_schema_inlines_refs_and_closes_objects(self):
        from ragas.metrics.collections.faithfulness.util import NLIStatementOutput

        from src.external.ragas_llm import strict_schema

        s = strict_schema(NLIStatementOutput)
        text = json.dumps(s)
        assert "$ref" not in text and "$defs" not in text and '"title"' not in text
        item = s["properties"]["statements"]["items"]
        assert s["additionalProperties"] is False and item["additionalProperties"] is False
        assert item["required"] == ["statement", "reason", "verdict"]
