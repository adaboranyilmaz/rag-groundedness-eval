"""LLM backends behind one interface, with an on-disk response cache and a spend ledger.

- `GenerationRequest` is everything sent to a model. Its `cache_key` is a SHA-256 of that
  request in canonical JSON, so any change to the prompt text, the model or a parameter
  is a cache miss, and an identical request is never paid for twice (spec §3.8).
- `SpendLedger` enforces the API cost caps. Before an uncached API call it reserves the
  call's worst case (estimated input plus the full `max_tokens` of output); after the call
  it settles the reservation at the real cost from the response's token usage. A call that
  would take either the phase cap or the project cap past its limit is refused before it
  is sent. Reservations are thread-safe, so concurrent API calls cannot overshoot.
- The Anthropic backend sends no sampling parameters (current models reject them) and
  disables thinking, so a prompt variant is the only source of explicit reasoning.
- The Ollama backend runs at `temperature=0` with a fixed seed. Ollama silently drops the
  start of a prompt that exceeds `num_ctx`, so the request is refused beforehand if a
  deliberately pessimistic token estimate would not fit alongside `max_tokens` of output.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

CACHE_DIR = Path("data/cache/llm")

# Pessimistic characters-per-token, used only for pre-flight checks (context fit, budget
# reservation), never for reported numbers. Filing excerpts are token-dense (figures,
# table pipes): measured on this corpus, ~2.4 chars/token for Claude (mean) and down to
# 2.5 for qwen2.5 (worst prompt of the pilot), so 1.5 over-estimates with a wide margin.
PESSIMISTIC_CHARS_PER_TOKEN = 1.5


def estimate_tokens_upper(text: str) -> int:
    return int(len(text) / PESSIMISTIC_CHARS_PER_TOKEN) + 1


@dataclass(frozen=True)
class GenerationRequest:
    backend: str  # "anthropic" | "ollama"
    model: str
    system: str
    user: str
    max_tokens: int
    params: dict[str, Any] = field(default_factory=dict)  # backend-specific, all recorded

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @property
    def cache_key(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()


@dataclass
class GenerationResponse:
    text: str
    model_reported: str  # the model string the server says it ran
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    created_utc: str
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


# --------------------------------------------------------------------------------------
# Cache


class ResponseCache:
    """One JSON file per request, `<dir>/<key[:2]>/<key>.json`, holding the request and the
    response. Writes go through a temp file and rename, so an interrupted run never
    leaves a half-written entry that later reads as a valid response."""

    def __init__(self, root: Path = CACHE_DIR):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> GenerationResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        return GenerationResponse(**entry["response"])

    def put(self, request: GenerationRequest, response: GenerationResponse) -> None:
        path = self._path(request.cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"request": asdict(request), "response": asdict(response)}
        tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# Spend ledger


class BudgetExceeded(RuntimeError):
    pass


class SpendLedger:
    """Cumulative API spend across runs and phases, persisted as JSON.

    `prices` maps model -> {"input": usd_per_mtok, "output": usd_per_mtok}. A model with no
    price entry cannot be called through the ledger: an unpriced call cannot be capped.
    """

    def __init__(
        self,
        path: Path,
        prices: dict[str, dict[str, float]],
        project_cap_usd: float,
        phase: str,
        phase_cap_usd: float,
    ):
        self.path = Path(path)
        self.prices = prices
        self.project_cap_usd = project_cap_usd
        self.phase = phase
        self.phase_cap_usd = phase_cap_usd
        self._lock = threading.Lock()
        self._reserved = 0.0
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = {"total_usd": 0.0, "n_calls": 0, "by_phase": {}, "by_model": {}}

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if model not in self.prices:
            raise BudgetExceeded(f"no price configured for {model!r}; refusing to call it")
        p = self.prices[model]
        return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

    def phase_spent(self) -> float:
        return self.state["by_phase"].get(self.phase, {}).get("usd", 0.0)

    def reserve(self, request: GenerationRequest) -> float:
        worst = self.cost(
            request.model, estimate_tokens_upper(request.system + request.user), request.max_tokens
        )
        with self._lock:
            project_after = self.state["total_usd"] + self._reserved + worst
            phase_after = self.phase_spent() + self._reserved + worst
            if project_after > self.project_cap_usd:
                raise BudgetExceeded(
                    f"project cap ${self.project_cap_usd:.2f} would be exceeded "
                    f"(spent ${self.state['total_usd']:.4f}, in flight ${self._reserved:.4f}, "
                    f"this call up to ${worst:.4f})"
                )
            if phase_after > self.phase_cap_usd:
                raise BudgetExceeded(
                    f"{self.phase} cap ${self.phase_cap_usd:.2f} would be exceeded "
                    f"(spent ${self.phase_spent():.4f}, in flight ${self._reserved:.4f}, "
                    f"this call up to ${worst:.4f})"
                )
            self._reserved += worst
        return worst

    def settle(self, reserved: float, model: str, input_tokens: int, output_tokens: int) -> float:
        actual = self.cost(model, input_tokens, output_tokens)
        with self._lock:
            self._reserved -= reserved
            self.state["total_usd"] += actual
            self.state["n_calls"] += 1
            for bucket, name in (("by_phase", self.phase), ("by_model", model)):
                b = self.state[bucket].setdefault(
                    name, {"usd": 0.0, "n_calls": 0, "input_tokens": 0, "output_tokens": 0}
                )
                b["usd"] += actual
                b["n_calls"] += 1
                b["input_tokens"] += input_tokens
                b["output_tokens"] += output_tokens
            self.state["updated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
            self.state["caps"] = {
                "project_usd": self.project_cap_usd,
                f"{self.phase}_usd": self.phase_cap_usd,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        return actual

    def release(self, reserved: float) -> None:
        """Drop a reservation for a call that failed before any tokens were billed."""
        with self._lock:
            self._reserved -= reserved


# --------------------------------------------------------------------------------------
# Backends


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, client: Any = None, max_retries: int = 5):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=max_retries)
        self.client = client

    @staticmethod
    def request_params(thinking: str = "disabled") -> dict[str, Any]:
        return {"thinking": {"type": thinking}}

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        t0 = time.perf_counter()
        msg = self.client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
            **request.params,
        )
        latency = (time.perf_counter() - t0) * 1000
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        extra: dict[str, Any] = {}
        if msg.stop_reason == "refusal" and getattr(msg, "stop_details", None) is not None:
            extra["stop_details"] = {
                "category": getattr(msg.stop_details, "category", None),
                "explanation": getattr(msg.stop_details, "explanation", None),
            }
        return GenerationResponse(
            text=text,
            model_reported=msg.model,
            stop_reason=msg.stop_reason,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            latency_ms=latency,
            created_utc=_now(),
            request_id=getattr(msg, "_request_id", None),
            extra=extra,
        )


class ContextOverflow(ValueError):
    pass


class OllamaBackend:
    name = "ollama"

    def __init__(self, client: Any = None, host: str | None = None):
        if client is None:
            import ollama

            client = ollama.Client(host=host or os.environ.get("OLLAMA_HOST") or None)
        self.client = client
        self._digests: dict[str, str | None] = {}

    @staticmethod
    def request_params(temperature: float, seed: int, num_ctx: int) -> dict[str, Any]:
        return {"options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}}

    def digest(self, model: str) -> str | None:
        """Content digest of the local model weights: the pinned version string for a
        local model, since a tag like `qwen2.5:3b-instruct` can be re-pointed upstream."""
        if model not in self._digests:
            listed = {m.model: m.digest for m in self.client.list().models}
            self._digests[model] = listed.get(model)
        return self._digests[model]

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        options = dict(request.params["options"])
        num_ctx = options["num_ctx"]
        needed = estimate_tokens_upper(request.system + request.user) + request.max_tokens
        if needed > num_ctx:
            raise ContextOverflow(
                f"prompt (pessimistic estimate) + max_tokens = {needed} tokens exceeds "
                f"num_ctx={num_ctx}; Ollama would silently truncate the prompt"
            )
        options["num_predict"] = request.max_tokens
        t0 = time.perf_counter()
        resp = self.client.chat(
            model=request.model,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            options=options,
            stream=False,
        )
        latency = (time.perf_counter() - t0) * 1000
        return GenerationResponse(
            text=resp.message.content or "",
            model_reported=resp.model,
            stop_reason=resp.done_reason,
            input_tokens=resp.prompt_eval_count,
            output_tokens=resp.eval_count,
            latency_ms=latency,
            created_utc=_now(),
            extra={
                "digest": self.digest(request.model),
                "load_ms": (resp.load_duration or 0) / 1e6,
                "prompt_eval_ms": (resp.prompt_eval_duration or 0) / 1e6,
                "eval_ms": (resp.eval_duration or 0) / 1e6,
            },
        )


# --------------------------------------------------------------------------------------


def generate_cached(
    backend: Backend,
    request: GenerationRequest,
    cache: ResponseCache,
    ledger: SpendLedger | None = None,
) -> tuple[GenerationResponse, bool, float]:
    """Return (response, was_cached, new_cost_usd). API backends must pass a ledger."""
    cached = cache.get(request.cache_key)
    if cached is not None:
        return cached, True, 0.0
    if backend.name == "anthropic" and ledger is None:
        raise BudgetExceeded("API calls require a SpendLedger")
    reserved = ledger.reserve(request) if ledger is not None else 0.0
    try:
        response = backend.generate(request)
    except BaseException:
        if ledger is not None:
            ledger.release(reserved)
        raise
    cost = 0.0
    if ledger is not None:
        cost = ledger.settle(
            reserved, request.model, response.input_tokens or 0, response.output_tokens or 0
        )
    cache.put(request, response)
    return response, False, cost
