"""The generation request for a (model, prompt, question, context): one definition shared by
the Phase 4 generation script and the served pipeline (src/serving/pipeline.py), so a served
answer is built exactly as the evaluated traces were and replays from the same cache entry."""

from __future__ import annotations

from src.generation.llm import AnthropicBackend, GenerationRequest, OllamaBackend
from src.generation.prompts import PromptTemplate, render


def request_params(model_cfg: dict) -> dict:
    if model_cfg["backend"] == "anthropic":
        return AnthropicBackend.request_params(thinking=model_cfg["thinking"])
    return OllamaBackend.request_params(
        temperature=model_cfg["temperature"], seed=model_cfg["seed"], num_ctx=model_cfg["num_ctx"]
    )


def build_request(
    model_cfg: dict, max_tokens: int, template: PromptTemplate, question: dict, chunks: list[dict]
) -> tuple[GenerationRequest, str]:
    rendered = render(template, question["question"], chunks)
    req = GenerationRequest(
        backend=model_cfg["backend"],
        model=model_cfg["model"],
        system=rendered.system,
        user=rendered.user,
        max_tokens=max_tokens,
        params=request_params(model_cfg),
    )
    return req, rendered.sha256


def make_backend(model_cfg: dict):
    if model_cfg["backend"] == "anthropic":
        return AnthropicBackend()
    if model_cfg["backend"] == "ollama":
        return OllamaBackend()
    raise ValueError(f"unknown backend {model_cfg['backend']!r}")
