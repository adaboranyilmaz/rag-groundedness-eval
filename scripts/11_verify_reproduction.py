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

# A difference between two runs of the same pipeline that is allowed: files (glob), field
# (regex on the dotted field path; `<bytes>` for a non-JSON file), reason, and abs_tol: None
# allows any value, a number allows a numeric difference up to it and nothing more. Paths look
# like `.configs.x.qdrant.index_size_bytes`, with `[i]` for list items (a JSONL file is a list
# of records).
EXEMPT: list[tuple[str, str, str, float | None]] = [
    ("*", r"\.(updated_utc|run_utc)$", "when the run happened", None),
    (
        "metrics/generation_runs*.json",
        r"\.meta\.api_spend_to_date_usd$",
        "a snapshot of the project's cumulative API spend ledger at run time, including "
        "spend by later phases; not a property of the run",
        None,
    ),
    (
        "metrics/generation_runs*.json",
        r"\.this_run\.(cache_hits|new_calls|new_cost_usd)$",
        "what that invocation took from the response cache versus paid for; a rebuild "
        "takes every response from the cache",
        None,
    ),
    (
        "metrics/generation_runs*.json",
        r"\.meta\.environment\.ollama_server$",
        "the Ollama server version asked at run time; a replay does not query the server "
        "(the weights digest per arm comes from the cached responses and is compared)",
        None,
    ),
    (
        "metrics/index_stats.json",
        r"\.(build_time_sec|query_latency\.p\d+_ms)$",
        "measured wall-clock time",
        None,
    ),
    (
        "metrics/index_stats.json",
        r"\.qdrant\.index_size_bytes$",
        "Qdrant's on-disk size (du of its storage) depends on segment and write-ahead-log "
        "state when measured; the FAISS index sizes are compared",
        None,
    ),
    (
        "metrics/index_stats.json",
        r"\.equivalence_check\.[^.]+\.(exact_top_k_matches|match_rate)$",
        "FAISS-vs-Qdrant exact-order agreement on 30 queries: Qdrant orders exactly tied "
        "scores arbitrarily between builds, so a tied pair flips it between 29 and 30 "
        "(DECISIONS.md, Phases 2 and 7)",
        None,
    ),
    (
        "metrics/retrieval_grid.json",
        r"\.(latency\.(mean|p50|p95)_ms|bm25_build_sec)$",
        "measured wall-clock time",
        None,
    ),
    # Embeddings computed from scratch are not bit-identical to the committed ones: GPU
    # arithmetic depends on how texts are batched, and the committed vectors were embedded
    # in several increments (DECISIONS.md, Phase 3). Rows agree to <= 1.5e-6 (cosine
    # >= 0.99999976), which moves a score stored to 6 decimals by 1e-6.
    (
        "traces/*",
        r"\.retrieval\.chunks\[\d+\]\.score$",
        "retrieval score of a context chunk, stored to 6 decimals; the chunks and their "
        "order are compared exactly",
        1e-5,
    ),
    (
        "metrics/reliability_analysis.json",
        r"\.units\[\d+\]\.top1_score$",
        "a question's top-1 retrieval score (6 decimals); the Q1 statistics are compared",
        1e-5,
    ),
    (
        "metrics/retrieval_per_question.jsonl",
        r"^\[\d+\]\.top_ids\[[5-9]\]$",
        "stored ranking beyond the top 5: a near-tied pair (score gap < 1e-7) can swap "
        "places; every metric of the row is compared and must match",
        None,
    ),
    (
        "plots/retrieval_vs_groundedness.png",
        r"^<bytes>$",
        "draws the top-1 scores above; one marker's anti-aliasing moves (the plotted values "
        "come from reliability_analysis.json, which is compared)",
        None,
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


def differences(a, b, path: str = "") -> list[tuple[str, object, object, str]]:
    """Every (field path, value a, value b, description) at which two JSON values differ."""
    numbers = isinstance(a, int | float) and isinstance(b, int | float)
    if type(a) is not type(b) and not numbers:
        return [(path, a, b, f"type {type(a).__name__} vs {type(b).__name__}")]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", a.get(k), b.get(k), "present in only one"))
            else:
                out += differences(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [(path, a, b, f"length {len(a)} vs {len(b)}")]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += differences(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [(path, a, b, f"{a!r} vs {b!r}"[:200])]


def exemption(rel: str, path: str, a=None, b=None) -> str | None:
    for glob, pattern, reason, tol in EXEMPT:
        if not (fnmatch.fnmatch(rel, glob) and re.search(pattern, path)):
            continue
        if tol is None:
            return reason
        numeric = isinstance(a, int | float) and isinstance(b, int | float)
        if numeric and abs(a - b) <= tol:
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
        reason = exemption(rel, "<bytes>")
        if reason:
            return {"status": "exempt_only", "n_fields": 1, "reasons": [reason]}
        return {"status": "different", "detail": ["bytes differ"]}
    diffs = differences(load(base), load(cur))
    if not diffs:  # same content, different formatting
        return {"status": "identical"}
    unexplained = [f"{p}: {d}" for p, a, b, d in diffs if exemption(rel, p, a, b) is None]
    if unexplained:
        return {"status": "different", "n": len(unexplained), "detail": unexplained[:20]}
    reasons = sorted({exemption(rel, p, a, b) for p, a, b, _ in diffs})
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
            "exempt": [
                {"files": g, "field": f, "reason": r, "abs_tol": t} for g, f, r, t in EXEMPT
            ],
            "baseline_sha256_of_file_list": hashlib.sha256(
                "\n".join(sorted(base_files)).encode()
            ).hexdigest(),
        },
        "counts": counts,
        "passed": not failures,
        "failures": failures,
        "exempt_only": {k: v for k, v in report.items() if v["status"] == "exempt_only"},
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(report)} files: {counts}")
    for k, v in failures.items():
        print(f"  DIFFERENT {k} ({v.get('n', '')}):")
        for d in v.get("detail", [v["status"]])[:8]:
            print(f"      {d}")
    print("PASSED" if not failures else "FAILED", f"-> {args.out}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
