"""Check that a pipeline rebuild (`dvc repro`) reproduced the committed results.

Compares a results tree with a baseline copy of it taken before the rebuild, file by file:
JSON and JSONL semantically, every other file byte for byte. A difference is allowed only in
a field matched by an EXEMPT rule (file, field path, reason). The rules are as narrow as the
evidence allowed; each records something about the run rather than a result (when it ran,
how long it took, on what machine, what the whole project had spent by then), or a value
that is not reproducible by construction, with the reason. Every other difference is a
reproduction failure.

Writes the report to --out (default results/metrics/reproduction_check.json) and exits 1 if
anything differs outside the rules.

Usage:
    cp -r results /tmp/baseline && dvc repro --force && \\
    uv run python scripts/11_verify_reproduction.py /tmp/baseline
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
from pathlib import Path

# (file glob, regex on the dotted field path, reason): a difference between two runs of the
# same pipeline that is allowed. Paths look like `.configs.x.qdrant.index_size_bytes`, with
# `[i]` for list items (JSONL files are lists of records).
EXEMPT: list[tuple[str, str, str]] = [
    ("*", r"\.(updated_utc|run_utc)$", "when the run happened"),
    (
        "metrics/generation_runs*.json",
        r"\.meta\.api_spend_to_date_usd$",
        "a snapshot of the project's cumulative API spend ledger at run time, including "
        "spend by later phases; not a property of the run",
    ),
    (
        "metrics/generation_runs*.json",
        r"\.this_run\.(cache_hits|new_calls|new_cost_usd)$",
        "what that invocation took from the response cache versus paid for; a rebuild "
        "takes every response from the cache",
    ),
    (
        "metrics/generation_runs*.json",
        r"\.meta\.environment\.ollama_server$",
        "the Ollama server version asked at run time; a replay does not query the server "
        "(the weights digest per arm comes from the cached responses and is compared)",
    ),
    (
        "metrics/index_stats.json",
        r"\.(build_time_sec|query_latency\.p\d+_ms)$",
        "measured wall-clock time",
    ),
    (
        "metrics/index_stats.json",
        r"\.qdrant\.index_size_bytes$",
        "Qdrant's on-disk size (du of its storage) depends on segment and write-ahead-log "
        "state when measured; the FAISS index sizes are compared",
    ),
    (
        "metrics/index_stats.json",
        r"\.equivalence_check\.[^.]+\.(exact_top_k_matches|match_rate)$",
        "FAISS-vs-Qdrant exact-order agreement on 30 queries: Qdrant orders exactly tied "
        "scores arbitrarily between builds, so a tied pair flips it between 29 and 30 "
        "(DECISIONS.md, Phases 2 and 7)",
    ),
    (
        "metrics/retrieval_grid.json",
        r"\.(latency\.(mean|p50|p95)_ms|bm25_build_sec)$",
        "measured wall-clock time",
    ),
]

SKIP_FILES = {
    # This report, and the smoke evaluation's (written by CI / scripts/10_smoke_eval.py).
    "metrics/reproduction_check.json",
    "metrics/smoke_eval.json",
    # Generated labelling pages, not results (gitignored).
    "labels/groundedness_labeling.html",
    "labels/premise_labeling.html",
    "labels/quadrant_review.html",
}


def differences(a, b, path: str = "") -> list[tuple[str, str]]:
    """Every (field path, description) at which two JSON values differ."""
    numbers = isinstance(a, int | float) and isinstance(b, int | float)
    if type(a) is not type(b) and not numbers:
        return [(path, f"type {type(a).__name__} vs {type(b).__name__}")]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", "present in only one"))
            else:
                out += differences(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [(path, f"length {len(a)} vs {len(b)}")]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += differences(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [(path, f"{a!r} vs {b!r}"[:200])]


def exemption(rel: str, path: str) -> str | None:
    for glob, pattern, reason in EXEMPT:
        if fnmatch.fnmatch(rel, glob) and re.search(pattern, path):
            return reason
    return None


def load(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    return json.loads(text)


def compare_file(rel: str, base: Path, cur: Path) -> dict:
    if base.read_bytes() == cur.read_bytes():
        return {"status": "identical"}
    if base.suffix not in (".json", ".jsonl"):
        return {"status": "different", "detail": ["bytes differ"]}
    diffs = differences(load(base), load(cur))
    if not diffs:  # same content, different formatting
        return {"status": "identical"}
    unexplained = [f"{p}: {d}" for p, d in diffs if exemption(rel, p) is None]
    if unexplained:
        return {"status": "different", "n": len(unexplained), "detail": unexplained[:20]}
    reasons = sorted({exemption(rel, p) for p, _ in diffs})
    return {"status": "exempt_only", "n_fields": len(diffs), "reasons": reasons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", help="copy of the results directory taken before the rebuild")
    parser.add_argument("--current", default="results")
    parser.add_argument("--out", default="results/metrics/reproduction_check.json")
    args = parser.parse_args()
    base_root, cur_root = Path(args.baseline), Path(args.current)

    def files(root: Path) -> set[str]:
        return {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and p.relative_to(root).as_posix() not in SKIP_FILES
        }

    base_files, cur_files = files(base_root), files(cur_root)
    report: dict[str, dict] = {}
    for rel in sorted(base_files | cur_files):
        if rel not in cur_files:
            report[rel] = {"status": "missing_after_rebuild"}
        elif rel not in base_files:
            report[rel] = {"status": "new_after_rebuild"}
        else:
            report[rel] = compare_file(rel, base_root / rel, cur_root / rel)
    counts: dict[str, int] = {}
    for r in report.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    failures = {k: v for k, v in report.items() if v["status"] not in ("identical", "exempt_only")}
    out = {
        "meta": {
            "baseline": "results/ as committed, copied before the rebuild",
            "exempt": [{"files": g, "field": f, "reason": r} for g, f, r in EXEMPT],
            "baseline_sha256_of_file_list": hashlib.sha256(
                "\n".join(sorted(base_files)).encode()
            ).hexdigest(),
        },
        "counts": counts,
        "passed": not failures,
        "failures": failures,
        "exempt_only": {k: v for k, v in report.items() if v["status"] == "exempt_only"},
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{len(report)} files: {counts}")
    for k, v in failures.items():
        print(f"  DIFFERENT {k} ({v.get('n', '')}):")
        for d in v.get("detail", [v["status"]])[:8]:
            print(f"      {d}")
    print("PASSED" if not failures else "FAILED", f"-> {args.out}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
