"""Measure how long each pipeline stage takes, for the README's reproduction section (a
runtime there is a number, so it must come from a results file).

Two parts, both in replay mode (every command carries configs/replay.env: no model API call
and no EDGAR request is possible):
  from scratch  ingest, chunk, the nine embedding configurations and the index build, timed
                once with the embeddings set aside so they are computed, not read back. Needs
                Qdrant running (`docker compose up -d qdrant`). Embeddings computed on a GPU
                are not bit-identical to the committed ones (batch composition, ~1e-6), so
                afterwards the original embeddings are put back, the index build is re-run
                from them (untimed: it rewrites the FAISS files and the Qdrant collections as
                committed), and `dvc checkout` restores anything else DVC tracks.
  replayed      every later stage, timed `--runs` times (default 2) in pipeline order; the
                slower run is reported.
The stages rewrite their outputs, so the results tree and the serving bundle (which copies
results files) are copied first and restored afterwards byte for byte: the only file this
leaves changed is results/metrics/runtimes.json.

Usage:
    uv run python scripts/18_stage_runtimes.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EMBEDDINGS = ROOT / "data/processed/embeddings"
# restored with the results: the serving bundle bakes in results files that stages rewrite
RESTORED = [RESULTS, ROOT / "build/serving"]
OUT = RESULTS / "metrics/runtimes.json"
# children flush their progress to the log as they go
UNBUFFERED = {**os.environ, "PYTHONUNBUFFERED": "1"}
FROM_SCRATCH = {  # stage -> the script it runs (the README table's row)
    "ingest": "01_ingest.py",
    "chunk": "02_chunk.py",
    "embed": "03_build_index.py --embed-only (9 configurations)",
    "index": "03_build_index.py --all",
}
REPLAYED = {
    "retrieve": "04_eval_retrieval.py",
    "retrieve_scoped": "04b_scoped_retrieval.py",
    "generate": "05_generate.py",
    "replay_check": "05_generate.py --verify-replay",
    "evaluate": "06_eval_generation.py",
    "judge_validation": "06b_judge_validation.py",
    "adversarial_generate": "07b_adversarial_run.py (generate)",
    "adversarial_evaluate": "07b_adversarial_run.py (evaluate)",
    "analyse": "07_reliability_analysis.py",
    "select": "09_track_experiments.py select",
    "track": "09_track_experiments.py track",
    "serving_bundle": "12_serving.py bundle",
    "serving_check": "12_serving.py check",
    "ragas_faithfulness": "15_ragas_faithfulness.py",
    "replicates": "16_replicates.py",
}


def commands(stages: dict, name: str) -> list[str]:
    stage = stages[name]
    cmd = stage["cmd"]
    cmds = cmd if isinstance(cmd, list) else [cmd]
    if "matrix" not in stage:
        return cmds
    out = []  # expand a matrix stage (embed) over its items, in dvc.yaml order
    m = stage["matrix"]
    keys = list(m)
    combos = [{}]
    for k in keys:
        combos = [{**c, k: v} for c in combos for v in m[k]]
    for c in combos:
        for cmd in cmds:
            for k, v in c.items():
                cmd = cmd.replace("${item." + k + "}", str(v))
            out.append(cmd)
    return out


def run_timed(stages: dict, name: str) -> float:
    t0 = time.perf_counter()
    for cmd in commands(stages, name):
        print(f"[{name}] {cmd}", flush=True)
        rc = subprocess.run(cmd, shell=True, cwd=ROOT, env=UNBUFFERED).returncode
        if rc != 0:
            raise RuntimeError(f"stage {name} failed ({rc}): {cmd}")
    seconds = round(time.perf_counter() - t0, 1)
    print(f"[{name}] {seconds} s", flush=True)
    return seconds


def run(cmd: str) -> None:
    print(f"[restore] {cmd}", flush=True)
    if subprocess.run(cmd, shell=True, cwd=ROOT, env=UNBUFFERED).returncode != 0:
        raise RuntimeError(f"restore step failed: {cmd}")


def from_scratch(stages: dict) -> dict:
    aside = EMBEDDINGS.with_name("embeddings.timing_aside")
    if aside.exists():
        sys.exit(f"{aside} exists: a previous run was interrupted; restore it first")
    EMBEDDINGS.rename(aside)
    timings = {}
    try:
        for name, script in FROM_SCRATCH.items():
            timings[name] = {"script": script, "seconds": run_timed(stages, name)}
    finally:
        if EMBEDDINGS.exists():
            shutil.rmtree(EMBEDDINGS)
        aside.rename(EMBEDDINGS)
        for cmd in commands(stages, "index"):  # FAISS files and Qdrant collections as committed
            run(cmd)
        run("uv run dvc checkout")
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2, help="replays per later stage")
    args = parser.parse_args()
    stages = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    backup = Path(tempfile.mkdtemp(prefix="results_backup_"))
    for d in RESTORED:
        shutil.copytree(d, backup / d.relative_to(ROOT))
    try:
        scratch = from_scratch(stages)
        replayed = {}
        for _ in range(args.runs):
            for name, script in REPLAYED.items():
                t = run_timed(stages, name)
                replayed.setdefault(name, {"script": script, "runs_s": []})["runs_s"].append(t)
        for r in replayed.values():
            r["seconds"] = max(r["runs_s"])
    finally:
        for d in RESTORED:
            shutil.rmtree(d)
            shutil.copytree(backup / d.relative_to(ROOT), d)
        shutil.rmtree(backup, ignore_errors=True)
    out = {
        "meta": {
            "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "what": "wall-clock time of each stage's command on the development machine, in "
            "pipeline order; `from_scratch` computes the embeddings rather than reading them "
            f"back (one run); `replayed` replays from the response caches ({args.runs} runs, "
            "`seconds` is the slower)",
            "host": {"platform": platform.platform(), "python": platform.python_version()},
        },
        "from_scratch": scratch,
        "replayed": replayed,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
