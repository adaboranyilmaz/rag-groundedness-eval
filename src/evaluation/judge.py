"""LLM judge: versioned judge prompts, request building, and strict output validation.

Judge prompts live in `prompts/judge/` in the same registry style as the generation prompts
(YAML front-matter, then `<!-- system -->` and `<!-- user -->` sections); each file names the
placeholders its user section must contain exactly once. Every judge request asks for JSON
constrained by a schema (`output_config.format`), and the schema is part of the request, so
it is part of the response-cache key: changing a schema can never reuse an old answer.

Three judge calls:
  decompose    question + the ANSWER field -> atomic claims (kind: document | context | general) and
               the response type (answered | partial | declined). Sees no context, so it
               cannot shape claims to fit it, and no gold answer.
  verify       the context excerpts, labelled as the generator saw them, + numbered claims
               -> per claim a reason, a verdict (supported | unsupported | contradicted) and
               the excerpts that support it. Sees neither the model's citations (so it
               cannot be anchored by them) nor the gold answer.
  correctness  question + gold answer + gold justification + the ANSWER field -> reason,
               grade (correct | partially_correct | incorrect | no_answer), unit_error flag.
               Sees no context.
Outputs are validated field by field; anything malformed raises `JudgeOutputError`, which
the harness records as a judge failure for that trace rather than guessing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.generation.llm import GenerationRequest
from src.generation.prompts import SYSTEM_MARKER, USER_MARKER, sha256_text

JUDGE_PROMPTS_DIR = Path("prompts/judge")
PURPOSES = ("decompose", "verify", "correctness")
RESPONSE_TYPES = ("answered", "partial", "declined")
CLAIM_KINDS = ("document", "context", "general")
VERDICTS = ("supported", "unsupported", "contradicted")
GRADES = ("correct", "partially_correct", "incorrect", "no_answer")
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class JudgeOutputError(ValueError):
    pass


@dataclass(frozen=True)
class JudgePrompt:
    id: str
    version: int
    purpose: str
    placeholders: tuple[str, ...]
    description: str
    system: str
    user: str
    file_sha256: str


def parse_judge_prompt(text: str) -> JudgePrompt:
    if not text.startswith("---\n"):
        raise ValueError("judge prompt must start with YAML front-matter")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("unterminated front-matter")
    meta = yaml.safe_load(text[4:end])
    for key in ("id", "version", "purpose", "placeholders", "description"):
        if key not in meta:
            raise ValueError(f"front-matter missing {key!r}")
    if meta["purpose"] not in PURPOSES:
        raise ValueError(f"unknown purpose {meta['purpose']!r}")
    body = text[end + len("\n---\n") :]
    if body.count(SYSTEM_MARKER) != 1 or body.count(USER_MARKER) != 1:
        raise ValueError("body needs exactly one system and one user marker")
    s_at, u_at = body.index(SYSTEM_MARKER), body.index(USER_MARKER)
    if s_at > u_at:
        raise ValueError("system section must come before user section")
    system = body[s_at + len(SYSTEM_MARKER) : u_at].strip()
    user = body[u_at + len(USER_MARKER) :].strip()
    placeholders = tuple(meta["placeholders"])
    found = _PLACEHOLDER_RE.findall(user)
    if sorted(found) != sorted(placeholders):
        raise ValueError(f"user section placeholders {found} != declared {list(placeholders)}")
    if _PLACEHOLDER_RE.search(system):
        raise ValueError("placeholders belong in the user section only")
    return JudgePrompt(
        id=meta["id"],
        version=int(meta["version"]),
        purpose=meta["purpose"],
        placeholders=placeholders,
        description=" ".join(str(meta["description"]).split()),
        system=system,
        user=user,
        file_sha256=sha256_text(text),
    )


def load_judge_prompt(path: Path) -> JudgePrompt:
    prompt = parse_judge_prompt(Path(path).read_text(encoding="utf-8"))
    if prompt.id != Path(path).stem:
        raise ValueError(f"judge prompt id {prompt.id!r} does not match {Path(path).name}")
    return prompt


def load_judge_registry(prompts_dir: Path = JUDGE_PROMPTS_DIR) -> dict[str, JudgePrompt]:
    reg = {p.stem: load_judge_prompt(p) for p in sorted(Path(prompts_dir).glob("*.md"))}
    if not reg:
        raise FileNotFoundError(f"no judge prompts in {prompts_dir}")
    return reg


def render_judge(prompt: JudgePrompt, values: dict[str, str]) -> tuple[str, str]:
    """Single-pass substitution, so text that contains a literal `{{name}}` (filing text,
    an answer) is never substituted a second time."""
    missing = set(prompt.placeholders) - set(values)
    if missing:
        raise ValueError(f"missing values for {sorted(missing)}")
    user = _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], prompt.user)
    return prompt.system, user


# --------------------------------------------------------------------------------------
# Schemas


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


DECOMPOSE_SCHEMA = _obj(
    {
        "claims": {
            "type": "array",
            "items": _obj(
                {"claim": {"type": "string"}, "kind": {"type": "string", "enum": list(CLAIM_KINDS)}}
            ),
        },
        "response_type": {"type": "string", "enum": list(RESPONSE_TYPES)},
    }
)
CORRECTNESS_SCHEMA = _obj(
    {
        "reason": {"type": "string"},
        "grade": {"type": "string", "enum": list(GRADES)},
        "unit_error": {"type": "boolean"},
    }
)


def verify_schema(labels: list[str]) -> dict[str, Any]:
    return _obj(
        {
            "verdicts": {
                "type": "array",
                "items": _obj(
                    {
                        "claim_id": {"type": "integer"},
                        "reason": {"type": "string"},
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                        "supporting_excerpts": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(labels)},
                        },
                    }
                ),
            }
        }
    )


# --------------------------------------------------------------------------------------
# Requests


def request_params(judge_cfg: dict, schema: dict, replicate: int | None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "output_config": {"format": {"type": "json_schema", "schema": schema}}
    }
    if "thinking" in judge_cfg:
        params["thinking"] = {"type": judge_cfg["thinking"]}
    if "temperature" in judge_cfg:
        params["temperature"] = judge_cfg["temperature"]
    if replicate:
        params["_replicate"] = replicate
    return params


def build_request(
    judge_cfg: dict,
    prompt: JudgePrompt,
    values: dict[str, str],
    schema: dict,
    max_tokens: int,
    replicate: int | None = None,
) -> GenerationRequest:
    if judge_cfg["backend"] != "anthropic":
        raise ValueError("the judge runs on the Anthropic API only")
    system, user = render_judge(prompt, values)
    return GenerationRequest(
        backend="anthropic",
        model=judge_cfg["model"],
        system=system,
        user=user,
        max_tokens=max_tokens,
        params=request_params(judge_cfg, schema, replicate),
    )


def format_claims(claims: list[dict]) -> str:
    return "\n".join(f"{i}. {c['claim']}" for i, c in enumerate(claims, start=1))


# --------------------------------------------------------------------------------------
# Output validation


def _load_json(text: str) -> dict:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise JudgeOutputError(f"not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise JudgeOutputError("top level is not an object")
    return obj


def _enum(value: Any, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise JudgeOutputError(f"{name} {value!r} not in {allowed}")
    return value


def parse_decomposition(text: str) -> dict:
    obj = _load_json(text)
    rtype = _enum(obj.get("response_type"), RESPONSE_TYPES, "response_type")
    claims = obj.get("claims")
    if not isinstance(claims, list):
        raise JudgeOutputError("claims is not a list")
    out = []
    for c in claims:
        if not isinstance(c, dict) or not isinstance(c.get("claim"), str):
            raise JudgeOutputError(f"malformed claim {c!r}")
        text_ = c["claim"].strip()
        if not text_:
            raise JudgeOutputError("empty claim")
        out.append({"claim": text_, "kind": _enum(c.get("kind"), CLAIM_KINDS, "kind")})
    return {"response_type": rtype, "claims": out}


def parse_verification(text: str, n_claims: int, labels: list[str]) -> list[dict]:
    """Verdicts in claim order. Every claim id 1..n must appear exactly once. Supporting
    excerpts listed for a verdict other than `supported` are dropped (and counted), since
    by instruction only supported claims have supporting excerpts."""
    obj = _load_json(text)
    verdicts = obj.get("verdicts")
    if not isinstance(verdicts, list):
        raise JudgeOutputError("verdicts is not a list")
    by_id: dict[int, dict] = {}
    for v in verdicts:
        if not isinstance(v, dict) or not isinstance(v.get("claim_id"), int):
            raise JudgeOutputError(f"malformed verdict {v!r}")
        cid = v["claim_id"]
        if cid in by_id:
            raise JudgeOutputError(f"claim {cid} judged twice")
        verdict = _enum(v.get("verdict"), VERDICTS, "verdict")
        support = v.get("supporting_excerpts")
        if not isinstance(support, list) or any(s not in labels for s in support):
            raise JudgeOutputError(f"bad supporting_excerpts {support!r}")
        by_id[cid] = {
            "verdict": verdict,
            "reason": str(v.get("reason", "")).strip(),
            "supporting_excerpts": sorted(set(support), key=labels.index)
            if verdict == "supported"
            else [],
            "dropped_support": bool(support) and verdict != "supported",
        }
    if sorted(by_id) != list(range(1, n_claims + 1)):
        raise JudgeOutputError(f"claim ids {sorted(by_id)} != 1..{n_claims}")
    return [by_id[i] for i in range(1, n_claims + 1)]


def parse_correctness(text: str) -> dict:
    obj = _load_json(text)
    if not isinstance(obj.get("unit_error"), bool):
        raise JudgeOutputError("unit_error is not a boolean")
    return {
        "grade": _enum(obj.get("grade"), GRADES, "grade"),
        "unit_error": obj["unit_error"],
        "reason": str(obj.get("reason", "")).strip(),
    }
