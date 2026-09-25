"""Generation traces: one JSON line per (model, prompt, question), and their replay.

A trace stores everything needed to reconstruct the call without the index or the
corpus: the question, the retrieved chunks with their full text, the prompt id, version
and file hash, the exact request parameters, the raw output and the parsed fields.

Traces are deterministic given the cache: timestamps and latencies come from the cached
response (when the call was actually made), not from the run that wrote the file, so
re-running generation over a warm cache rewrites byte-identical trace files.

Replay (`replay_trace`) checks, for one trace:
  1. prompt   the prompt file on disk, rendered with the trace's question and chunks,
              hashes to the trace's `rendered_sha256` (and the file hash matches)
  2. request  the rebuilt request hashes to the trace's `cache_key`
  3. output   the cached response for that key has exactly the trace's raw output
              (skipped, and reported as such, when the response cache is absent, as in a
              fresh clone, where the raw output in the trace is the record)
  4. parse    re-parsing the raw output reproduces the trace's parsed fields
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.generation.llm import GenerationRequest, GenerationResponse, ResponseCache
from src.generation.parsing import ParsedOutput, parse_output
from src.generation.prompts import PromptTemplate, render

TRACES_DIR = Path("results/traces")
SCHEMA_VERSION = 1

CHUNK_FIELDS = (
    "chunk_id",
    "doc_id",
    "page",
    "section",
    "char_start",
    "char_end",
    "is_table",
    "text",
)


def trace_chunks(results: list[Any]) -> list[dict[str, Any]]:
    """Retrieval results -> the chunk records stored in a trace (and rendered into the
    prompt), labelled C1..Ck in rank order."""
    out = []
    for i, r in enumerate(results, start=1):
        rec = {"label": f"C{i}", "score": round(float(r.score), 6)}
        rec.update({k: r.metadata.get(k) for k in CHUNK_FIELDS})
        out.append(rec)
    return out


def build_trace(
    *,
    question: dict[str, Any],
    retrieval: dict[str, Any],
    template: PromptTemplate,
    request: GenerationRequest,
    model_key: str,
    response: GenerationResponse,
    parsed: ParsedOutput,
    rendered_sha256: str,
    cost_usd: float | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": (
            f"{retrieval['condition']}__{model_key}__{template.id}__{question['financebench_id']}"
        ),
        "question": question,
        "retrieval": retrieval,
        "prompt": {
            "id": template.id,
            "version": template.version,
            "variant": template.variant,
            "file_sha256": template.file_sha256,
            "rendered_sha256": rendered_sha256,
        },
        "generation": {
            "model_key": model_key,
            "backend": request.backend,
            "model_requested": request.model,
            "model_reported": response.model_reported,
            "max_tokens": request.max_tokens,
            "params": request.params,
            "cache_key": request.cache_key,
            "request_id": response.request_id,
            "stop_reason": response.stop_reason,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": round(response.latency_ms, 1),
            "created_utc": response.created_utc,
            "cost_usd": cost_usd,
            "extra": response.extra,
        },
        "raw_output": response.text,
        "parsed": parsed.to_dict(),
    }


def write_traces(path: Path, traces: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(traces, key=lambda t: t["question"]["financebench_id"])
    lines = [json.dumps(t, ensure_ascii=False, sort_keys=True) for t in ordered]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def read_traces(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def request_from_trace(trace: dict[str, Any], template: PromptTemplate) -> GenerationRequest:
    rendered = render(template, trace["question"]["question"], trace["retrieval"]["chunks"])
    g = trace["generation"]
    return GenerationRequest(
        backend=g["backend"],
        model=g["model_requested"],
        system=rendered.system,
        user=rendered.user,
        max_tokens=g["max_tokens"],
        params=g["params"],
    )


@dataclass
class ReplayResult:
    trace_id: str
    prompt_file_ok: bool
    prompt_render_ok: bool
    cache_key_ok: bool
    output_source: str  # "cache" | "trace_only"
    output_ok: bool | None  # None when there is no cache entry to compare against
    parse_ok: bool
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.prompt_file_ok
            and self.prompt_render_ok
            and self.cache_key_ok
            and self.output_ok is not False
            and self.parse_ok
        )


def replay_trace(
    trace: dict[str, Any], registry: dict[str, PromptTemplate], cache: ResponseCache | None
) -> ReplayResult:
    problems: list[str] = []
    p = trace["prompt"]
    template = registry.get(p["id"])
    if template is None:
        return ReplayResult(
            trace["trace_id"],
            False,
            False,
            False,
            "trace_only",
            None,
            False,
            [f"prompt {p['id']!r} not in registry"],
        )

    file_ok = template.file_sha256 == p["file_sha256"] and template.version == p["version"]
    if not file_ok:
        problems.append("prompt file changed since the trace was written")
    rendered = render(template, trace["question"]["question"], trace["retrieval"]["chunks"])
    render_ok = rendered.sha256 == p["rendered_sha256"]
    if not render_ok:
        problems.append("re-rendered prompt hash differs")
    request = request_from_trace(trace, template)
    key_ok = request.cache_key == trace["generation"]["cache_key"]
    if not key_ok:
        problems.append("rebuilt request cache key differs")

    source, output_ok = "trace_only", None
    if cache is not None:
        hit = cache.get(trace["generation"]["cache_key"])
        if hit is not None:
            source = "cache"
            output_ok = hit.text == trace["raw_output"]
            if not output_ok:
                problems.append("cached output differs from trace raw_output")

    reparsed = parse_output(
        trace["raw_output"], template.output_fields, len(trace["retrieval"]["chunks"])
    ).to_dict()
    parse_ok = reparsed == trace["parsed"]
    if not parse_ok:
        problems.append("re-parsed output differs from trace")
    return ReplayResult(
        trace["trace_id"], file_ok, render_ok, key_ok, source, output_ok, parse_ok, problems
    )
