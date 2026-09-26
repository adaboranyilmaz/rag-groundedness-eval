"""Request and response bodies of `POST /query`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    include_context: bool = Field(
        False, description="return the full text of every retrieved chunk, not only citations"
    )


class Citation(BaseModel):
    label: str  # C1..Ck, as the model saw the chunks
    chunk_id: str
    doc_id: str
    page: int | None
    section: str | None
    score: float
    snippet: str


class ContextChunk(Citation):
    text: str


class Claim(BaseModel):
    claim: str
    kind: Literal["document", "context", "general"]
    verdict: Literal["supported", "unsupported", "contradicted"] | None = None
    reason: str | None = None
    supporting_excerpts: list[str] = []


class KappaFigure(BaseModel):
    kappa: float
    ci95: list[float]
    n: int


class JudgeValidation(BaseModel):
    claim_supported_kappa: KappaFigure
    claim_3class_kappa: KappaFigure
    answer_fully_grounded_kappa: KappaFigure
    source: str
    note: str


class Groundedness(BaseModel):
    score: float | None = Field(
        description="supported document claims / document claims; null when undefined"
    )
    unavailable_reason: str | None = None
    fully_grounded: bool | None = None
    any_contradicted: bool | None = None
    n_document_claims: int | None = None
    n_supported: int | None = None
    n_unsupported: int | None = None
    n_contradicted: int | None = None
    n_context_claims: int | None = None
    n_general_claims: int | None = None
    claims: list[Claim] = []
    judge_model: str
    judge_prompts: dict[str, str]
    validation: JudgeValidation


class LLMCall(BaseModel):
    purpose: str
    cached: bool
    cost_usd: float
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_latency_ms: float = Field(
        description="the model call's own latency, recorded when it was made (for a cached "
        "response, the original call)"
    )


class QueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str | None
    status: Literal["answered", "partial", "declined", "unparsed"]
    confidence: int | None
    citations: list[Citation]
    context: list[ContextChunk] | None = None
    groundedness: Groundedness
    latency_ms: dict[str, float] = Field(
        description="this request's time per stage: embed, search, generate, judge_decompose, "
        "judge_verify, total"
    )
    llm_calls: list[LLMCall]
    cost_usd: float
    pipeline: dict[str, str | int]
