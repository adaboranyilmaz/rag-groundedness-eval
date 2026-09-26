"""Groundedness for a served answer: the Phase 5 measurement, applied to one request.

The same two judge calls as the evaluation harness (src/evaluation/evaluate.py), built by the
same functions from the same prompts, model and output ceilings (configs/evaluation.yaml):
  1. decompose  question + the answer text -> atomic claims and the response type
  2. verify     the context the generator saw + the numbered claims -> a verdict per claim
then `groundedness.score`. For a benchmark question of the selected arm, both requests are
therefore the evaluated ones byte for byte and replay from the same cache entries
(`scripts/12_serving.py check` verifies this for all 150).

A judge failure (no response, cut off at max_tokens, refused, malformed output) gives no
score and says why; it is never guessed, as in the harness. The score is also undefined
when the answer has no text to decompose (a bare decline or an unparsed output) or no claim
about the documents.

`validation` travels with every score: the judge's agreement with the author's hand labels
(results/metrics/judge_agreement.json), so a consumer sees how far the number can be trusted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from src.evaluation import groundedness
from src.evaluation.abstention import final_status, needs_judge
from src.evaluation.judge import (
    DECOMPOSE_SCHEMA,
    JudgeOutputError,
    build_request,
    format_claims,
    load_judge_registry,
    parse_decomposition,
    parse_verification,
    verify_schema,
)
from src.generation.llm import GenerationRequest, ResponseCache, SpendLedger, generate_cached
from src.generation.prompts import format_context

AGREEMENT_KEYS = {
    # the author's labels vs the primary judge, document claims (the claims the score uses)
    "claim_supported_kappa": ("document_claims", "claim_supported"),
    "claim_3class_kappa": ("document_claims", "claim_3class"),
    "answer_fully_grounded_kappa": ("document_claims", "answer_fully_grounded"),
}


def load_validation(path: Path) -> dict[str, Any]:
    """The judge-vs-author agreement figures served with every score, read from the results
    file (never restated in code), plus the judge prompt hashes they were measured with."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    block = report["agreement"]["human__vs__judge"]
    out: dict[str, Any] = {}
    for name, (scope, measure) in AGREEMENT_KEYS.items():
        m = block[scope][measure]
        out[name] = {"kappa": m["kappa"], "ci95": m["kappa_ci95"], "n": m["n"]}
    out["source"] = Path(path).as_posix()
    out["note"] = (
        "Cohen's kappa between the author's blind hand labels and this judge on a sample of "
        f"{report['meta']['n_answers']} answers; single run"
    )
    out["judge_prompt_sha256"] = {
        purpose: p["file_sha256"] for purpose, p in report["meta"]["judge_prompts"].items()
    }
    return out


class JudgeCallError(RuntimeError):
    def __init__(self, stage: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason = stage, reason


class GroundednessJudge:
    def __init__(
        self,
        eval_cfg: dict,
        backend: Any,
        cache: ResponseCache,
        ledger: SpendLedger | None,
        prompts_dir: Path = Path("prompts/judge"),
        validation: dict | None = None,
        recoverable: tuple[type[BaseException], ...] = (),
    ):
        """`recoverable`: exceptions from a judge call (cache miss, budget, API error) that
        leave the score undefined, with the reason, instead of failing the request."""
        self.recoverable = recoverable
        jc = eval_cfg["judge"]
        self.model_cfg = jc["models"][jc["model"]]
        registry = load_judge_registry(prompts_dir)
        self.prompts = {p: registry[jc["prompts"][p]] for p in ("decompose", "verify")}
        if validation is not None:  # the agreement figures describe these exact prompts
            for purpose, prompt in self.prompts.items():
                if prompt.file_sha256 != validation["judge_prompt_sha256"][purpose]:
                    raise ValueError(
                        f"judge prompt {prompt.id} has changed since the judge was validated"
                    )
        self.max_tokens = jc["max_tokens"]
        self.backend, self.cache, self.ledger = backend, cache, ledger
        self.validation = validation

    @classmethod
    def from_config(cls, eval_config_path: Path, **kw) -> GroundednessJudge:
        cfg = yaml.safe_load(Path(eval_config_path).read_text(encoding="utf-8"))
        return cls(cfg, **kw)

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_cfg["model"],
            "decompose_prompt": self.prompts["decompose"].id,
            "verify_prompt": self.prompts["verify"].id,
        }

    def request(self, purpose: str, values: dict[str, str], schema: dict) -> GenerationRequest:
        return build_request(
            self.model_cfg, self.prompts[purpose], values, schema, self.max_tokens[purpose]
        )

    def _call(self, stage: str, request: GenerationRequest, parse, calls: list, timings: dict):
        t0 = time.perf_counter()
        try:
            response, cached, cost = generate_cached(self.backend, request, self.cache, self.ledger)
        except self.recoverable as e:
            raise JudgeCallError(stage, type(e).__name__) from e
        finally:
            timings[f"judge_{stage}"] = (time.perf_counter() - t0) * 1000
        calls.append(
            {
                "purpose": f"judge_{stage}",
                "cache_key": request.cache_key,
                "cached": cached,
                "cost_usd": cost,
                "model_latency_ms": response.latency_ms,
                "model": response.model_reported,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        )
        if response.stop_reason in ("max_tokens", "refusal"):
            raise JudgeCallError(stage, f"stop_reason_{response.stop_reason}")
        try:
            return parse(response.text)
        except JudgeOutputError as e:
            raise JudgeCallError(stage, f"invalid_output: {e}") from e

    def assess(self, question: str, parsed: dict, chunks: list[dict]) -> dict[str, Any]:
        """-> {"status", "status_source", "groundedness": {...}, "llm_calls", "timings_ms"}.
        `parsed` is the generator output's parsed fields (ParsedOutput.to_dict())."""
        calls: list[dict] = []
        timings: dict[str, float] = {}
        result: dict[str, Any] = {"llm_calls": calls, "timings_ms": timings, **self.describe()}
        g: dict[str, Any] = {"score": None, "unavailable_reason": None}

        if not needs_judge(parsed):
            status, source = final_status(parsed, None)
            g["unavailable_reason"] = "no_answer_text"
            return {"status": status, "status_source": source, "groundedness": g, **result}

        values = {"question": question, "answer": parsed["answer"]}
        try:
            dec = self._call(
                "decompose",
                self.request("decompose", values, DECOMPOSE_SCHEMA),
                parse_decomposition,
                calls,
                timings,
            )
        except JudgeCallError as e:
            status, source = final_status(parsed, None)
            g["unavailable_reason"] = f"judge_error: {e}"
            return {"status": status, "status_source": source, "groundedness": g, **result}
        status, source = final_status(parsed, dec["response_type"])

        claims = dec["claims"]
        if claims:
            labels = [c["label"] for c in chunks]
            verify_values = {"context": format_context(chunks), "claims": format_claims(claims)}
            try:
                verdicts = self._call(
                    "verify",
                    self.request("verify", verify_values, verify_schema(labels)),
                    lambda text: parse_verification(text, len(claims), labels),
                    calls,
                    timings,
                )
            except JudgeCallError as e:
                g["unavailable_reason"] = f"judge_error: {e}"
                g["claims"] = claims
                return {"status": status, "status_source": source, "groundedness": g, **result}
            scored = groundedness.score(claims, verdicts)
            scored["claims"] = [
                {**c, **{k: v[k] for k in ("verdict", "reason", "supporting_excerpts")}}
                for c, v in zip(claims, verdicts, strict=True)
            ]
        else:
            scored = groundedness.score([], [])
            scored["claims"] = []
        g.update(scored)
        g["score"] = scored["groundedness"]
        if g["score"] is None:
            g["unavailable_reason"] = "no_document_claims"
        return {"status": status, "status_source": source, "groundedness": g, **result}
