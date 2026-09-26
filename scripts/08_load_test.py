"""Load test of the served API, and a live sample of its model-call latency.

  replay   drive the running container (replay-only mode, so every model response is the
           baked-in cached one) with locust: closed-loop users at each concurrency level in
           configs/serving.yaml, each sending the next question as soon as the last answer
           returns, drawn from the 150 benchmark questions in a seeded random order. Per
           level: throughput, client-side latency p50/p95/p99, errors, and the server's
           per-stage latency. This measures the service (query embedding on CPU, FAISS
           search, cache reads, parsing), not the model API.
  live     the same service in-process (CPU embeddings, as in the container) answering a
           seeded sample of the benchmark questions one at a time with real API calls,
           through a cache of its own: the latency of each model call (generation and the
           two judge calls), the cost per question, and how often the fresh answer and its
           groundedness differ from the evaluated ones. The stage latencies are the model
           calls' own, recorded when they were made, so re-running replays the same numbers
           from the cache at no cost. `--dry-run` prints the cost estimate and stops.

Both write their section of results/metrics/load_test.json and keep the other's. Single run
each, on one machine; the load generator shares the host with the container.

Usage:
    RAG_REPLAY_ONLY=1 docker compose up -d        # the container, replay-only
    uv run --group loadtest python scripts/08_load_test.py replay
    uv run python scripts/08_load_test.py live --dry-run
    uv run python scripts/08_load_test.py live
"""

from __future__ import annotations

import sys

if __name__ == "__main__" and sys.argv[1:2] == ["replay"]:
    from gevent import monkey  # locust runs on gevent: patch before anything opens a socket

    monkey.patch_all()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SERVING_CONFIG = Path("configs/serving.yaml")
EVAL_CONFIG = Path("configs/evaluation.yaml")
OUT_PATH = Path("results/metrics/load_test.json")
STAGES = ("embed", "search", "generate", "judge_decompose", "judge_verify", "total")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]


def dist(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    q = np.percentile(a, [50, 95, 99])
    return {
        "n": len(values),
        "mean": round(float(a.mean()), 3),
        "p50": round(float(q[0]), 3),
        "p95": round(float(q[1]), 3),
        "p99": round(float(q[2]), 3),
        "max": round(float(a.max()), 3),
    }


def write_section(name: str, section: dict) -> None:
    report = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    report[name] = section
    report = {k: report[k] for k in sorted(report)}
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT_PATH} [{name}]")


def host_info() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        out = subprocess.run(
            ["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}} {{.ServerVersion}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=True,
        ).stdout.split()
        info["docker"] = {"cpus": int(out[0]), "memory_gb": round(int(out[1]) / 2**30, 1)}
        info["docker"]["server_version"] = out[2]
    except Exception as e:  # noqa: BLE001 - recorded, not needed
        info["docker"] = f"unavailable: {type(e).__name__}"
    return info


# --------------------------------------------------------------------------------------
# replay


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def cmd_replay() -> None:
    import gevent
    from locust import HttpUser, constant, task
    from locust.env import Environment

    cfg = load_yaml(SERVING_CONFIG)
    rc = cfg["load_test"]["replay"]
    url = rc["url"].rstrip("/")
    health = get_json(f"{url}/health")
    if not health.get("ready"):
        sys.exit(f"the API at {url} is not ready: {health.get('load_error')}")
    service = health["service"]
    if not service["replay_only"]:
        sys.exit("the API is not in replay-only mode: start it with RAG_REPLAY_ONLY=1")

    questions = [t["question"]["question"] for t in read_jsonl(Path(cfg["arm_traces"]))]
    rng = random.Random(rc["seed"])

    def question_stream():
        while True:
            order = list(questions)
            rng.shuffle(order)
            yield from order

    stream = question_stream()
    for _ in range(rc["warmup_requests"]):  # load the model and index pages; not recorded
        req = urllib.request.Request(
            f"{url}/query",
            json.dumps({"question": next(stream)}).encode(),
            {"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=120).read()

    records: list[dict] = []
    ctl = {"stop": False, "in_flight": 0}

    class QueryUser(HttpUser):
        host = url
        wait_time = constant(0)

        @task
        def ask(self):
            if ctl["stop"]:  # draining: no new requests, let in-flight ones finish
                gevent.sleep(0.05)
                return
            ctl["in_flight"] += 1
            t0 = time.perf_counter()
            try:
                with self.client.post(
                    "/query", json={"question": next(stream)}, catch_response=True
                ) as r:
                    t1 = time.perf_counter()
                    rec = {"start": t0, "ms": (t1 - t0) * 1000, "status": r.status_code}
                    if r.status_code == 200:
                        body = r.json()
                        rec["server"] = body["latency_ms"]
                        rec["uncached"] = sum(not c["cached"] for c in body["llm_calls"])
                        r.success()
                    else:
                        r.failure(f"HTTP {r.status_code}")
                    records.append(rec)
            finally:
                ctl["in_flight"] -= 1

    levels = []
    for users in rc["users"]:
        records.clear()
        ctl["stop"] = False
        env = Environment(user_classes=[QueryUser])
        runner = env.create_local_runner()
        runner.start(users, spawn_rate=users)
        gevent.sleep(rc["settle_s"])
        m0 = time.perf_counter()
        gevent.sleep(rc["duration_s"])
        m1 = time.perf_counter()
        ctl["stop"] = True
        deadline = time.perf_counter() + 300
        while ctl["in_flight"] and time.perf_counter() < deadline:
            gevent.sleep(0.05)
        runner.quit()
        window = [r for r in records if m0 <= r["start"] < m1]
        ok = [r for r in window if r["status"] == 200]
        errors = [r for r in window if r["status"] != 200]
        level = {
            "users": users,
            "window_s": round(m1 - m0, 3),
            "n_requests": len(window),
            "n_errors": len(errors),
            "error_rate": round(len(errors) / len(window), 6) if window else None,
            "errors_by_status": {
                str(s): sum(r["status"] == s for r in errors) for s in {r["status"] for r in errors}
            },
            "throughput_rps": round(len(ok) / (m1 - m0), 3),
            "latency_ms": dist([r["ms"] for r in ok]),
            "server_stage_ms": {
                s: dist([r["server"][s] for r in ok if s in r["server"]]) for s in STAGES
            },
            "n_uncached_model_calls": sum(r.get("uncached", 0) for r in ok),
        }
        levels.append(level)
        lat = level["latency_ms"] or {}
        print(
            f"  {users:>2} users: {level['throughput_rps']:.2f} req/s, "
            f"p50 {lat.get('p50')} ms, p95 {lat.get('p95')} ms, p99 {lat.get('p99')} ms, "
            f"{level['n_errors']} errors"
        )
    write_section(
        "replay",
        {
            "meta": {
                "single_run": True,
                "what": "the served API in replay-only mode: every model response is the "
                "cached one, so latency is the service's own (CPU query embedding, FAISS "
                "search over the full index, cache reads, parsing, JSON), not the model API's",
                "load": "closed loop: each user sends its next request when the last returns; "
                "questions: the selected arm's 150 benchmark questions in a seeded random order",
                "caveat": "the load generator (locust) runs on the same machine as the "
                "container and competes with it for CPU",
                "settings": rc,
                "bundle_sha256": service["bundle_sha256"],
                "pipeline": service["pipeline"],
                "host": host_info(),
            },
            "levels": levels,
        },
    )


# --------------------------------------------------------------------------------------
# live


def cached_entry(cache_dir: Path, key: str) -> dict:
    return json.loads((cache_dir / key[:2] / f"{key}.json").read_text(encoding="utf-8"))


def cmd_live(dry_run: bool) -> None:
    from dotenv import load_dotenv

    from src.generation.llm import GenerationRequest, SpendLedger

    cfg = load_yaml(SERVING_CONFIG)
    lc = cfg["load_test"]["live"]
    ecfg = load_yaml(EVAL_CONFIG)
    b = ecfg["budget"]
    ledger = SpendLedger(
        Path(b["ledger"]),
        b["prices_usd_per_mtok"],
        b["project_cap_usd"],
        lc["phase"],
        lc["phase_cap_usd"],
        b["batch_discount"],
    )
    traces = read_jsonl(Path(cfg["arm_traces"]))
    sample = random.Random(lc["seed"]).sample(traces, lc["n_questions"])
    ids = {t["trace_id"] for t in sample}
    evals = {
        r["trace_id"]: r for r in read_jsonl(Path(cfg["arm_evaluations"])) if r["trace_id"] in ids
    }

    # estimate from the evaluated calls' own token counts, at the standard (not batch) price
    bundle_cache = Path(cfg["bundle_dir"]) / "cache"
    expected = worst = 0.0
    for t in sample:
        keys = [t["generation"]["cache_key"]]
        keys += [evals[t["trace_id"]]["judge"][s] for s in ("decompose", "verify")
                 if s in evals[t["trace_id"]]["judge"]]  # fmt: skip
        for k in keys:
            entry = cached_entry(bundle_cache, k)
            resp = entry["response"]
            expected += ledger.cost(
                entry["request"]["model"], resp["input_tokens"], resp["output_tokens"]
            )
            worst += ledger.worst_case(GenerationRequest(**entry["request"]))
    print(
        f"live sample: {len(sample)} questions; expected ${expected:.4f} (the evaluated calls' "
        f"tokens at the standard price), worst case ${worst:.4f}; phase cap "
        f"${lc['phase_cap_usd']:.2f}, headroom ${ledger.headroom():.4f}"
    )
    if dry_run:
        return

    load_dotenv()
    os.environ.pop("RAG_REPLAY_ONLY", None)
    from src.serving.service import NoKeyBackend, QueryService

    service = QueryService.from_bundle(
        Path(cfg["bundle_dir"]),
        Path(lc["state_dir"]),
        device=cfg["check_device"],
        use_bundle_cache=False,
        ledger=ledger,
    )
    if isinstance(service.pipeline.backend, NoKeyBackend):
        print("no ANTHROPIC_API_KEY: only responses already in the live cache can be used")
    spent_before = ledger.state["total_usd"]
    rows = []
    for i, t in enumerate(sample, 1):
        ev = evals[t["trace_id"]]
        r = service.query(t["question"]["question"])
        calls = {c["purpose"]: c for c in r["llm_calls"]}
        stages = {"embed": r["latency_ms"]["embed"], "search": r["latency_ms"]["search"]}
        stages.update({p: c["model_latency_ms"] for p, c in calls.items()})
        stages["total"] = sum(stages.values())
        by_call = {
            c["purpose"]: ledger.cost(c["model"], c["input_tokens"] or 0, c["output_tokens"] or 0)
            for c in r["llm_calls"]
            if c["model"] is not None
        }
        cost = sum(by_call.values())
        eg = ev["groundedness"]
        rows.append(
            {
                "trace_id": t["trace_id"],
                "stage_ms": {k: round(v, 3) for k, v in stages.items()},
                "cost_usd": round(cost, 6),
                "cost_by_call_usd": {k: round(v, 6) for k, v in by_call.items()},
                "made_this_run": sum(not c["cached"] for c in r["llm_calls"]),
                "same_answer_text": r["answer"] == t["parsed"]["answer"],
                "status": {"evaluated": ev["abstention"]["status"], "live": r["status"]},
                "groundedness": {
                    "evaluated": eg["groundedness"] if eg else None,
                    "live": r["groundedness"]["score"],
                    "live_unavailable_reason": r["groundedness"]["unavailable_reason"],
                },
            }
        )
        print(
            f"  {i}/{len(sample)} {t['trace_id'].rsplit('__', 1)[1]}: "
            f"{stages['total'] / 1000:.1f} s, ${cost:.4f}, status {r['status']}"
        )
    spent = ledger.state["total_usd"] - spent_before

    def same_score(row):
        g = row["groundedness"]
        if g["evaluated"] is None or g["live"] is None:
            return g["evaluated"] is None and g["live"] is None
        return abs(g["evaluated"] - g["live"]) < 1e-9

    both = [
        r for r in rows if None not in (r["groundedness"]["evaluated"], r["groundedness"]["live"])
    ]
    write_section(
        "live",
        {
            "meta": {
                "single_run": True,
                "what": "the served pipeline in-process (CPU embeddings) answering a seeded "
                "sample of the benchmark questions one at a time with real API calls, each "
                "through a cache of its own; stage latencies of model calls are the calls' "
                "own, recorded when made (so a re-run replays them)",
                "settings": lc,
                "pipeline": service.pipeline_info,
                "judge": service.judge.describe(),
                "cost_basis": "tokens of each response at the standard price "
                "(configs/evaluation.yaml budget.prices_usd_per_mtok)",
                "host": host_info(),
            },
            "n_questions": len(rows),
            "stage_ms": {
                s: dist([r["stage_ms"][s] for r in rows if s in r["stage_ms"]]) for s in STAGES
            },
            "cost_usd": {
                "total": round(sum(r["cost_usd"] for r in rows), 6),
                "per_question": dist([r["cost_usd"] for r in rows]),
                "by_call": {
                    p: round(sum(r["cost_by_call_usd"].get(p, 0.0) for r in rows), 6)
                    for p in ("generate", "judge_decompose", "judge_verify")
                },
            },
            # the groundedness judge's part of the summed cost and of the summed end-to-end time
            "judge_share": {
                "cost": round(
                    sum(
                        v
                        for r in rows
                        for k, v in r["cost_by_call_usd"].items()
                        if k.startswith("judge_")
                    )
                    / sum(r["cost_usd"] for r in rows),
                    4,
                ),
                "latency": round(
                    sum(v for r in rows for k, v in r["stage_ms"].items() if k.startswith("judge_"))
                    / sum(r["stage_ms"]["total"] for r in rows),
                    4,
                ),
            },  # fmt: skip
            "spent_this_run_usd": round(spent, 6),
            "vs_evaluated": {
                "same_answer_text": sum(r["same_answer_text"] for r in rows),
                "same_status": sum(r["status"]["evaluated"] == r["status"]["live"] for r in rows),
                "same_groundedness": sum(same_score(r) for r in rows),
                "both_scored": len(both),
                "mean_abs_groundedness_diff_both_scored": round(
                    float(
                        np.mean(
                            [
                                abs(r["groundedness"]["evaluated"] - r["groundedness"]["live"])
                                for r in both
                            ]
                        )
                    ),
                    6,  # fmt: skip
                )
                if both
                else None,
            },
            "questions": rows,
        },
    )
    print(f"spent this run: ${spent:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("replay")
    live = sub.add_parser("live")
    live.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "replay":
        cmd_replay()
    else:
        cmd_live(args.dry_run)


if __name__ == "__main__":
    main()
