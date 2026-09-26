"""RAGAS's LLM interface over the project's own Anthropic transport, so RAGAS's calls are
cached on disk by request hash and capped by the spend ledger like every other call (spec
§3.8), and replay from the cache at no cost.

Only the transport is replaced. RAGAS renders its own prompts (`BasePrompt.to_string`: the
instruction, the output JSON schema, its worked example and the input), splits the answer
into statements, and computes the score; this module sends the rendered prompt unchanged as
the user message and hands RAGAS back its own response model. RAGAS's stock Anthropic path
(`llm_factory`, through `instructor`) sends `temperature=0.01, top_p=0.1`, which Sonnet 5
rejects, so it cannot run this model as shipped.

Structured output is requested the way the project's judge requests it
(`output_config.format`, a JSON schema, src/evaluation/judge.py): the schema is RAGAS's own
response model's, made strict (references inlined, no extra properties, every field
required). The response is validated with that model; a cut-off, refused or invalid output
raises `RagasCallError`, and the caller records the answer as unscored rather than guessing.

Imports ragas, so it is only importable in the `ragas` environment (pyproject.toml).
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

from pydantic import BaseModel, ValidationError
from ragas.llms.base import InstructorBaseRagasLLM

from src.generation.llm import (
    AnthropicBackend,
    GenerationRequest,
    GenerationResponse,
    ResponseCache,
    SpendLedger,
    generate_cached,
)


class RagasCallError(RuntimeError):
    pass


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema with `$ref`s inlined, titles dropped, and every object closed
    (`additionalProperties: false`, all properties required), as structured outputs need."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            out = {k: walk(v) for k, v in node.items() if k != "title"}
            if out.get("type") == "object":
                out["additionalProperties"] = False
                out["required"] = list(out.get("properties", {}))
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


class _LazyAnthropic:
    """Creates the API client on the first uncached call only, so replay needs no key."""

    name = "anthropic"

    def __init__(self):
        self._backend: AnthropicBackend | None = None

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._backend is None:
            self._backend = AnthropicBackend()
        return self._backend.generate(request)


class ProjectRagasLLM(InstructorBaseRagasLLM):
    """One instance per scored answer, so `calls` holds that answer's calls in order."""

    def __init__(
        self,
        model_cfg: dict,
        max_tokens: int,
        cache: ResponseCache,
        ledger: SpendLedger | None,
        backend: Any = None,
    ):
        if model_cfg["backend"] != "anthropic":
            raise ValueError("RAGAS runs on the Anthropic API only here")
        self.model_cfg = model_cfg
        self.max_tokens = max_tokens
        self.cache = cache
        self.ledger = ledger
        self.backend = backend if backend is not None else _LazyAnthropic()
        self.calls: list[dict[str, Any]] = []

    def request(self, prompt: str, response_model: type[BaseModel]) -> GenerationRequest:
        return GenerationRequest(
            backend="anthropic",
            model=self.model_cfg["model"],
            system="",
            user=prompt,
            max_tokens=self.max_tokens,
            params={
                "output_config": {
                    "format": {"type": "json_schema", "schema": strict_schema(response_model)}
                },
                "thinking": {"type": self.model_cfg["thinking"]},
            },
        )

    def generate(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        req = self.request(prompt, response_model)
        resp, was_cached, cost = generate_cached(self.backend, req, self.cache, self.ledger)
        call = {
            "step": response_model.__name__,
            "cache_key": req.cache_key,
            "cached": was_cached,
            "new_cost_usd": cost,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "stop_reason": resp.stop_reason,
        }
        self.calls.append(call)
        if resp.stop_reason in ("max_tokens", "refusal"):
            call["error"] = f"stop_reason_{resp.stop_reason}"
            raise RagasCallError(call["error"])
        try:
            parsed = response_model.model_validate_json(resp.text)
        except ValidationError as e:
            call["error"] = "invalid_output"
            raise RagasCallError(f"invalid output: {e}") from e
        call["output"] = parsed.model_dump()
        return parsed

    async def agenerate(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        return await asyncio.to_thread(self.generate, prompt, response_model)
