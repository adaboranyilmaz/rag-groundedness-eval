"""The 10-question smoke evaluation: the pipeline scripts, end to end, on a slice of the
benchmark (configs/smoke.yaml), compared with the committed results.

  fixture  draw the questions (seeded, once) and write the cached model outputs their traces
           and evaluations need -> tests/fixtures/smoke/{questions.json,responses.jsonl}.
           Responses only: a cached request holds the filing excerpts, and the fixture is
           committed, so requests are not stored (the response cache is keyed by their hash).
  run      in a separate workspace, with no API access (RAG_REPLAY_ONLY=1, no key):
             01 ingest (the slice's filings only)   02 chunk
             03 embed + FAISS index (the winning configuration only)
             04 align gold evidence                 05 generate (oracle, replayed)
             06 evaluate (judge replayed)
           then compare every trace and evaluation with the committed one: the context's
           chunks and order, the generation request (cache key), the output and its parse
           must be identical, and so must each evaluation record; retrieval scores must agree
           within `score_abs_tol`. Exit status 1 on any difference.
           -> <workspace>/smoke_eval.json (and results/metrics/smoke_eval.json with --record)

EDGAR: with `--edgar-cache DIR`, filings are read from that cache without network access
(EDGAR_OFFLINE=1); without it they are fetched from EDGAR into the workspace (CI), which
needs EDGAR_USER_AGENT.

Usage:
    uv run python scripts/10_smoke_eval.py fixture
    uv run python scripts/10_smoke_eval.py run --edgar-cache data/raw/edgar_cache
    uv run python scripts/10_smoke_eval.py run --workspace /tmp/smoke   # CI: fetches filings
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.reliability import seeded_sample  # noqa: E402

CONFIG_PATH = REPO / "configs" / "smoke.yaml"
FINANCEBENCH = Path("data/raw/financebench/financebench_merged.jsonl")
TRACES = Path("results/traces")
EVAL_PER_TRACE = Path("results/metrics/eval_per_trace.jsonl")
TRACE_FIELDS = ("question", "prompt", "raw_output", "parsed", "schema_version", "trace_id")
GENERATION_FIELDS = ("backend", "cache_key", "model_requested", "model_reported", "params")


def load_cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def arms(cfg: dict) -> list[str]:
    return [f"{m}__{p}" for m in cfg["models"] for p in cfg["prompts"]]


def committed_traces(cfg: dict, qids: set[str]) -> dict[str, dict]:
    out = {}
    for a in arms(cfg):
        for t in load_jsonl(REPO / TRACES / cfg["condition"] / f"{a}.jsonl"):
            if t["question"]["financebench_id"] in qids:
                out[t["trace_id"]] = t
    return out


# --------------------------------------------------------------------------------------
# fixture


def cmd_fixture() -> None:
    from src.generation.llm import ResponseCache

    cfg = load_cfg()
    fdir = REPO / cfg["fixture_dir"]
    first = load_jsonl(REPO / TRACES / cfg["condition"] / f"{arms(cfg)[0]}.jsonl")
    candidates = sorted(
        {t["question"]["financebench_id"] for t in first}, key=lambda q: q
    )  # oracle traces exist only for questions with aligned gold evidence
    picked = seeded_sample(
        [{"financebench_id": q} for q in candidates],
        cfg["n_questions"],
        cfg["seed"],
        key="financebench_id",
    )
    qids = sorted(p["financebench_id"] for p in picked)
    traces = committed_traces(cfg, set(qids))
    rows = [r for r in load_jsonl(REPO / EVAL_PER_TRACE) if r["trace_id"] in traces]
    if len(rows) != len(traces):
        raise SystemExit(f"{len(traces)} traces but {len(rows)} evaluation records")
    keys = {t["generation"]["cache_key"] for t in traces.values()}
    keys |= {k for r in rows for k in r["judge"].values() if k}
    cache = ResponseCache(REPO / "data/cache/llm")
    missing = [k for k in keys if not cache.has(k)]
    if missing:
        raise SystemExit(f"{len(missing)} fixture responses are not in the response cache")
    fdir.mkdir(parents=True, exist_ok=True)
    lines = []
    for k in sorted(keys):
        entry = json.loads(cache._path(k).read_text(encoding="utf-8"))
        lines.append(json.dumps({"cache_key": k, "response": entry["response"]}, sort_keys=True))
    (fdir / "responses.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "questions": qids,
        "condition": cfg["condition"],
        "arms": arms(cfg),
        "n_traces": len(traces),
        "n_responses": len(keys),
        "doc_names": sorted({t["question"]["doc_name"] for t in traces.values()}),
    }
    (fdir / "questions.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{len(qids)} questions, {len(manifest['doc_names'])} filings, {len(traces)} traces")
    print(f"wrote {len(keys)} responses to {fdir / 'responses.jsonl'}")


# --------------------------------------------------------------------------------------
# run


def financebench_rows() -> list[dict]:
    local = REPO / FINANCEBENCH
    if local.exists():
        return load_jsonl(local)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="PatronusAI/financebench",
        repo_type="dataset",
        filename="financebench_merged.jsonl",
    )
    return load_jsonl(Path(path))


def build_workspace(ws: Path, cfg: dict, manifest: dict) -> None:
    for d in ("configs", "prompts"):
        shutil.copytree(REPO / d, ws / d, dirs_exist_ok=True)
    labels = ws / "results" / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "results/labels/out_of_corpus_audit.json", labels)

    # Evaluate exactly the slice's arms.
    ev_path = ws / "configs" / "evaluation.yaml"
    ev = yaml.safe_load(ev_path.read_text(encoding="utf-8"))
    ev["traces"]["conditions"] = [cfg["condition"]]
    ev["traces"]["models"] = cfg["models"]
    ev["traces"]["prompts"] = cfg["prompts"]
    ev_path.write_text(yaml.safe_dump(ev, sort_keys=False), encoding="utf-8", newline="\n")

    qids = set(manifest["questions"])
    rows = [r for r in financebench_rows() if r["financebench_id"] in qids]
    if len(rows) != len(qids):
        raise SystemExit(f"FinanceBench has {len(rows)} of the {len(qids)} smoke questions")
    fb = ws / FINANCEBENCH
    fb.parent.mkdir(parents=True, exist_ok=True)
    fb.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")

    # The response cache, from the fixture (entries without their request).
    n = 0
    for line in (REPO / cfg["fixture_dir"] / "responses.jsonl").read_text("utf-8").splitlines():
        rec = json.loads(line)
        k = rec["cache_key"]
        path = ws / "data/cache/llm" / k[:2] / f"{k}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"request": None, "response": rec["response"]}), "utf-8", newline="\n"
        )
        n += 1
    print(f"workspace {ws}: {len(rows)} questions, {n} cached responses")


def run_stages(ws: Path, cfg: dict, edgar_cache: Path | None) -> dict[str, float]:
    env = {
        **os.environ,
        "RAG_REPLAY_ONLY": "1",
        # A placeholder, not "": Windows drops empty variables, and .env would then supply the
        # real key. With it, a call that got past replay-only mode would fail to authenticate.
        "ANTHROPIC_API_KEY": "replay-only-no-key",
        "PYTHONUTF8": "1",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }
    if edgar_cache is not None:
        env["EDGAR_CACHE_DIR"] = str(edgar_cache.resolve())
        env["EDGAR_OFFLINE"] = "1"
    rc = cfg["retrieval"]
    one = ["--chunking-strategy", rc["chunking"], "--embedding-model", rc["embedding"]]
    stages = {
        "ingest": ["01_ingest.py"],
        "chunk": ["02_chunk.py"],
        "embed": ["03_build_index.py", "--embed-only", *one],
        "index": ["03_build_index.py", *one, "--backend", "faiss"],
        "align": ["04_eval_retrieval.py", "--align-only"],
        "generate": [
            "05_generate.py",
            "--conditions",
            cfg["condition"],
            "--models",
            *cfg["models"],
            "--prompts",
            *cfg["prompts"],
            "--no-mlflow",
        ],
        "evaluate": ["06_eval_generation.py", "--offline", "--no-mlflow"],
    }
    timings = {}
    for name, (script, *args) in stages.items():
        print(f"\n=== {name}: {script} {' '.join(args)}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / script), *args], cwd=ws, env=env
        )
        timings[name] = round(time.perf_counter() - t0, 1)
        if proc.returncode != 0:
            raise SystemExit(f"stage {name} failed (exit {proc.returncode})")
    return timings


def compare(ws: Path, cfg: dict, manifest: dict) -> dict:
    qids = set(manifest["questions"])
    want = committed_traces(cfg, qids)
    got = {}
    for a in arms(cfg):
        for t in load_jsonl(ws / TRACES / cfg["condition"] / f"{a}.jsonl"):
            got[t["trace_id"]] = t
    problems: list[str] = []
    if set(got) != set(want):
        problems.append(f"trace ids differ: {sorted(set(got) ^ set(want))[:5]}")
    max_score_diff = 0.0
    for tid in sorted(set(got) & set(want)):
        g, w = got[tid], want[tid]
        for f in TRACE_FIELDS:
            if g[f] != w[f]:
                problems.append(f"{tid}: {f} differs")
        for f in GENERATION_FIELDS:
            if g["generation"][f] != w["generation"][f]:
                problems.append(f"{tid}: generation.{f} differs")
        gc, wc = g["retrieval"]["chunks"], w["retrieval"]["chunks"]
        if [c["chunk_id"] for c in gc] != [c["chunk_id"] for c in wc]:
            problems.append(f"{tid}: context chunks or their order differ")
        else:
            for a, b in zip(gc, wc, strict=True):
                max_score_diff = max(max_score_diff, abs(a["score"] - b["score"]))
                if {k: v for k, v in a.items() if k != "score"} != {
                    k: v for k, v in b.items() if k != "score"
                }:
                    problems.append(f"{tid}: chunk {a['chunk_id']} differs")
        if g["retrieval"]["context_metrics"] != w["retrieval"]["context_metrics"]:
            problems.append(f"{tid}: context metrics differ")
    if max_score_diff > cfg["score_abs_tol"]:
        problems.append(f"retrieval scores differ by up to {max_score_diff:.2e}")

    want_eval = {
        r["trace_id"]: r for r in load_jsonl(REPO / EVAL_PER_TRACE) if r["trace_id"] in want
    }
    got_eval = {r["trace_id"]: r for r in load_jsonl(ws / EVAL_PER_TRACE)}
    if set(got_eval) != set(want_eval):
        problems.append("evaluated trace ids differ")
    n_eval_same = 0
    for tid in sorted(set(got_eval) & set(want_eval)):
        g, w = got_eval[tid], want_eval[tid]
        diff = sorted(k for k in set(g) | set(w) if g.get(k) != w.get(k))
        if diff:
            problems.append(f"{tid}: evaluation differs in {diff}")
        else:
            n_eval_same += 1
    return {
        "n_traces": len(got),
        "n_traces_expected": len(want),
        "n_evaluations_identical": n_eval_same,
        "max_retrieval_score_abs_diff": max_score_diff,
        "problems": problems,
        "passed": not problems,
    }


def cmd_run(args) -> None:
    cfg = load_cfg()
    manifest = json.loads((REPO / cfg["fixture_dir"] / "questions.json").read_text("utf-8"))
    if manifest["arms"] != arms(cfg) or manifest["condition"] != cfg["condition"]:
        raise SystemExit("the fixture was made for other arms: run `10_smoke_eval.py fixture`")
    ws = Path(args.workspace) if args.workspace else Path(tempfile.mkdtemp(prefix="smoke_"))
    if ws.exists() and any(ws.iterdir()) and not args.reuse:
        raise SystemExit(f"workspace {ws} is not empty (pass --reuse to keep its downloads)")
    ws.mkdir(parents=True, exist_ok=True)
    build_workspace(ws, cfg, manifest)
    edgar = Path(args.edgar_cache) if args.edgar_cache else None
    timings = run_stages(ws, cfg, edgar)
    result = {
        "meta": {
            "config": cfg,
            "questions": manifest["questions"],
            "doc_names": manifest["doc_names"],
            "edgar": "offline, local cache" if edgar else "fetched from EDGAR",
            "stage_seconds": timings,
        },
        **compare(ws, cfg, manifest),
    }
    text = json.dumps(result, indent=2) + "\n"
    (ws / "smoke_eval.json").write_text(text, encoding="utf-8", newline="\n")
    if args.record:
        (REPO / "results/metrics/smoke_eval.json").write_text(text, encoding="utf-8", newline="\n")
    print(
        f"\nsmoke eval: {result['n_traces']}/{result['n_traces_expected']} traces, "
        f"{result['n_evaluations_identical']} evaluations identical, "
        f"max score diff {result['max_retrieval_score_abs_diff']:.2e}"
    )
    for p in result["problems"][:20]:
        print(f"  PROBLEM {p}")
    print("PASSED" if result["passed"] else "FAILED")
    sys.exit(0 if result["passed"] else 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fixture")
    run = sub.add_parser("run")
    run.add_argument("--workspace", default=None)
    run.add_argument("--reuse", action="store_true", help="allow a non-empty workspace")
    run.add_argument("--edgar-cache", default=None, help="read filings from this EDGAR cache")
    run.add_argument("--record", action="store_true", help="also write results/metrics/")
    args = parser.parse_args()
    cmd_fixture() if args.command == "fixture" else cmd_run(args)


if __name__ == "__main__":
    main()
