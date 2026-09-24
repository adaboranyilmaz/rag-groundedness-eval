"""Prompt registry: prompts are versioned files under `prompts/`, never inline strings.

File format: YAML front-matter (`id`, `version`, `variant`, `output_fields`,
`description`), then a `<!-- system -->` section and a `<!-- user -->` section. The user
section holds exactly one `{{context}}` and one `{{question}}` placeholder.

Rendering substitutes both placeholders in a single regex pass, so filing text that happens
to contain a literal `{{question}}` is never substituted a second time. Every rendered
prompt carries a SHA-256 of exactly what the model is sent; replay recomputes it from the
prompt file plus the chunks stored in a trace and checks it matches.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROMPTS_DIR = Path("prompts")
SYSTEM_MARKER = "<!-- system -->"
USER_MARKER = "<!-- user -->"
PLACEHOLDERS = ("context", "question")
_PLACEHOLDER_RE = re.compile(r"\{\{(context|question)\}\}")
REQUIRED_KEYS = ("id", "version", "variant", "output_fields", "description")
KNOWN_FIELDS = {"REASONING", "ANSWER", "CITATIONS", "QUOTES", "CONFIDENCE"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    version: int
    variant: str
    output_fields: tuple[str, ...]
    description: str
    system: str
    user: str
    file_sha256: str  # hash of the whole file, front-matter included


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str

    @property
    def sha256(self) -> str:
        return sha256_text(json.dumps({"system": self.system, "user": self.user}))


def parse_prompt_file(text: str) -> PromptTemplate:
    if not text.startswith("---\n"):
        raise ValueError("prompt file must start with YAML front-matter ('---')")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("unterminated front-matter")
    meta = yaml.safe_load(text[4:end])
    missing = [k for k in REQUIRED_KEYS if k not in meta]
    if missing:
        raise ValueError(f"front-matter missing keys: {missing}")
    unknown = set(meta["output_fields"]) - KNOWN_FIELDS
    if unknown:
        raise ValueError(f"unknown output_fields: {sorted(unknown)}")
    if "ANSWER" not in meta["output_fields"]:
        raise ValueError("output_fields must include ANSWER")

    body = text[end + len("\n---\n") :]
    if body.count(SYSTEM_MARKER) != 1 or body.count(USER_MARKER) != 1:
        raise ValueError("body needs exactly one system and one user marker")
    s_at, u_at = body.index(SYSTEM_MARKER), body.index(USER_MARKER)
    if s_at > u_at:
        raise ValueError("system section must come before user section")
    system = body[s_at + len(SYSTEM_MARKER) : u_at].strip()
    user = body[u_at + len(USER_MARKER) :].strip()
    for name in PLACEHOLDERS:
        n = user.count("{{" + name + "}}")
        if n != 1:
            raise ValueError(f"user section must contain {{{{{name}}}}} exactly once (found {n})")
    if _PLACEHOLDER_RE.search(system):
        raise ValueError("placeholders belong in the user section only")

    return PromptTemplate(
        id=meta["id"],
        version=int(meta["version"]),
        variant=meta["variant"],
        output_fields=tuple(meta["output_fields"]),
        description=" ".join(str(meta["description"]).split()),
        system=system,
        user=user,
        file_sha256=sha256_text(text),
    )


def load_prompt(path: Path) -> PromptTemplate:
    template = parse_prompt_file(Path(path).read_text(encoding="utf-8"))
    if template.id != Path(path).stem:
        raise ValueError(f"prompt id {template.id!r} does not match file name {Path(path).name}")
    return template


def load_registry(prompts_dir: Path = PROMPTS_DIR) -> dict[str, PromptTemplate]:
    registry = {p.stem: load_prompt(p) for p in sorted(Path(prompts_dir).glob("*.md"))}
    if not registry:
        raise FileNotFoundError(f"no prompt files in {prompts_dir}")
    return registry


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Label chunks C1..Ck in retrieval order. The header names the filing, page and
    section so the model can tell which company and period an excerpt comes from."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[C{i}] {c['doc_id']} | page {c['page']}"
        if c.get("section"):
            header += f" | {c['section']}"
        blocks.append(f"{header}\n{c['text'].strip()}")
    return "\n\n".join(blocks)


def render(template: PromptTemplate, question: str, chunks: list[dict[str, Any]]) -> RenderedPrompt:
    values = {"context": format_context(chunks), "question": question.strip()}
    user = _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template.user)
    return RenderedPrompt(system=template.system, user=user)
