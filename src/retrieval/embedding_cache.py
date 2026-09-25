"""Content-addressed cache of chunk embeddings.

Each `(chunking_strategy, embedding_model)` cache is a `.npy` matrix plus a parallel
`.keys.json` list holding the SHA-256 of the exact chunk text each row embeds. Rows are
reused by text hash, never by position, so re-chunking a corpus in which most chunk texts
are unchanged re-embeds only the new texts — and a changed or reordered chunk can never
silently pick up another chunk's vector (which a row-count check alone would allow).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def keys_path(npy_path: Path) -> Path:
    return npy_path.with_suffix(".keys.json")


def save(npy_path: Path, vectors: np.ndarray, texts: Sequence[str]) -> None:
    if vectors.shape[0] != len(texts):
        raise ValueError("one vector per text required")
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, vectors)
    keys_path(npy_path).write_text(json.dumps([text_key(t) for t in texts]), encoding="utf-8")


def load(npy_path: Path) -> dict[str, np.ndarray]:
    """Map text hash -> vector for an existing cache; empty if absent or keyless."""
    kp = keys_path(npy_path)
    if not (npy_path.exists() and kp.exists()):
        return {}
    vectors = np.load(npy_path)
    keys = json.loads(kp.read_text(encoding="utf-8"))
    if len(keys) != vectors.shape[0]:
        raise ValueError(f"{npy_path} and its keys file disagree on row count")
    return {k: vectors[i] for i, k in enumerate(keys)}


def get_or_embed(
    npy_path: Path,
    texts: Sequence[str],
    embed: Callable[[list[str]], np.ndarray],
) -> tuple[np.ndarray, int]:
    """Vectors for `texts` in order, embedding only texts not already cached; rewrites
    the cache to exactly `texts` when that changes it (and only then, so a caller that only
    reads, such as the index build, leaves the files byte-identical). Returns (vectors,
    number of texts newly embedded)."""
    cached = load(npy_path)
    kp = keys_path(npy_path)
    stored_keys = json.loads(kp.read_text(encoding="utf-8")) if cached else None
    keys = [text_key(t) for t in texts]
    missing = sorted({i for i, k in enumerate(keys) if k not in cached})
    # Embed each distinct missing text once, even if it recurs (boilerplate chunks do).
    first_of: dict[str, int] = {}
    for i in missing:
        first_of.setdefault(keys[i], i)
    if first_of:
        new_vectors = embed([texts[i] for i in first_of.values()])
        cached.update(zip(first_of.keys(), new_vectors, strict=True))
    vectors = np.stack([cached[k] for k in keys]) if keys else np.zeros((0, 0), np.float32)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if first_of or stored_keys != keys:
        save(npy_path, vectors, texts)
    return vectors, len(first_of)
