"""The deployment criterion, timed: from a fresh clone, `docker compose up` to a real answer.

Run from this repository against a fresh clone elsewhere (the clone is made by hand, and its
duration passed in, so the script never touches git):

  1. preconditions  the clone has no local images of the stack (so every image is pulled, as
                    on a new machine) and a `.env` with an API key (the one manual step)
  2. up             `docker compose up -d` in the clone, timed until `/ready`
  3. answers        a benchmark question (answered from the image's baked-in responses) and
                    a question outside the benchmark (three real API calls, about $0.02,
                    capped by the container's own spend cap), each timed; the benchmark
                    answer must come from the cache with its groundedness, the other must
                    have made its model calls (declining it is a valid answer)
  -> results/metrics/deploy_check.json: every duration measured, the images pulled and
  their sizes, and whether clone-to-answer took under five minutes. Single run, on this
  machine and this network; the pull dominates, so another connection gives another time.

Port 8000 must be free: stop any other copy of the stack first (`docker compose down`).

Usage (PowerShell, in a new directory, e.g. C:\\tmp):
    Measure-Command { git clone -b <branch> https://github.com/<owner>/<repo>.git rag-deploy-check }
    Copy-Item rag-deploy-check\\.env.example rag-deploy-check\\.env   # then set the API key
    uv run python scripts/14_deploy_check.py --clone C:\\tmp\\rag-deploy-check --clone-seconds <s>
    docker compose -p rag-deploy-check down -v       # afterwards, in the clone
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

OUT_PATH = Path("results/metrics/deploy_check.json")
SERVING_CONFIG = Path("configs/serving.yaml")
LIMIT_S = 300
URL = "http://localhost:8000"
# A benchmark question with a grounded answer (the demo's), served from the baked-in cache
BENCH_QUESTION = "How much has the effective tax rate of Corning changed between FY2021 and FY2022?"
# Not a FinanceBench question, so not in the baked-in cache: it exercises the API key path.
LIVE_QUESTION = "What were Nike's total revenues in fiscal 2023?"


def run(cmd: list[str], cwd: Path | None = None, timeout: float = 1800) -> str:
    r = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {r.stderr.strip()[-2000:]}")
    return r.stdout


def compose_images(clone: Path) -> list[str]:
    cfg = yaml.safe_load((clone / "docker-compose.yml").read_text(encoding="utf-8"))
    return [s["image"] for s in cfg["services"].values() if "image" in s]


def image_present(image: str) -> bool:
    r = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    return r.returncode == 0


def ask(question: str) -> tuple[float, int, dict]:
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{URL}/query",
        json.dumps({"question": question}).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            status, body = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        status, body = e.code, json.loads(e.read() or b"{}")
    return time.perf_counter() - t0, status, body


def answer_summary(elapsed: float, status: int, body: dict) -> dict:
    g = body.get("groundedness") or {}
    return {
        "seconds": round(elapsed, 2),
        "http_status": status,
        "answer_status": body.get("status"),
        "has_answer_text": bool(body.get("answer")),
        "groundedness_score": g.get("score"),
        "groundedness_unavailable_reason": g.get("unavailable_reason"),
        "model_calls": [(c["purpose"], c["cached"]) for c in body.get("llm_calls", [])],
        "cost_usd": body.get("cost_usd"),
        "error": body.get("error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone", type=Path, required=True)
    parser.add_argument("--clone-seconds", type=float, required=True)
    args = parser.parse_args()
    clone = args.clone.resolve()
    project = clone.name

    images = compose_images(clone)
    present = [i for i in images if image_present(i)]
    if present:
        sys.exit(f"remove the local images first, so they are pulled as on a new machine: "
                 f"docker image rm {' '.join(present)}")  # fmt: skip
    env = (clone / ".env").read_text(encoding="utf-8") if (clone / ".env").exists() else ""
    if not any(ln.startswith("ANTHROPIC_API_KEY=") and len(ln.strip()) > 18
               for ln in env.splitlines()):  # fmt: skip
        sys.exit(f"{clone / '.env'} has no ANTHROPIC_API_KEY: that is the one manual step")

    cfg = yaml.safe_load(SERVING_CONFIG.read_text(encoding="utf-8"))
    arm = Path(cfg["arm_traces"]).read_text(encoding="utf-8").splitlines()
    if BENCH_QUESTION not in {json.loads(x)["question"]["question"] for x in arm}:
        sys.exit("the benchmark question is not in the selected arm's traces")
    bench_q = BENCH_QUESTION

    print(f"docker compose up -d in {clone} (project {project}); pulling {len(images)} images")
    t0 = time.perf_counter()
    run(["docker", "compose", "-p", project, "up", "-d"], cwd=clone)
    t_up = time.perf_counter() - t0
    ready_at = None
    while time.perf_counter() - t0 < 900:
        try:
            with urllib.request.urlopen(f"{URL}/ready", timeout=5) as r:
                if r.status == 200:
                    ready_at = time.perf_counter() - t0
                    break
        except OSError:
            pass
        time.sleep(0.5)
    if ready_at is None:
        sys.exit("the API did not become ready within 15 minutes")
    bench = answer_summary(*ask(bench_q))
    first_answer_at = time.perf_counter() - t0
    live = answer_summary(*ask(LIVE_QUESTION))

    sizes = {}
    for image in images:
        out = run(["docker", "image", "inspect", image, "--format", "{{.Size}}"])
        sizes[image] = round(int(out.strip()) / 1e6, 1)
    clone_to_answer = args.clone_seconds + first_answer_at
    report = {
        "meta": {
            "single_run": True,
            "note": "fresh clone, no local images of the stack, .env with an API key; "
            "timed on the author's machine and network (the pull dominates)",
            "limit_seconds": LIMIT_S,
            "docker": run(["docker", "version", "--format", "{{.Server.Version}}"]).strip(),
            "compose": run(["docker", "compose", "version", "--short"]).strip(),
        },
        "images_pulled_mb_uncompressed": sizes,
        "seconds": {
            "clone": round(args.clone_seconds, 1),
            "compose_up_returned": round(t_up, 1),
            "api_ready": round(ready_at, 1),
            "first_answer": round(first_answer_at, 1),
            "clone_to_first_answer": round(clone_to_answer, 1),
        },
        "benchmark_question": {"question": bench_q, **bench},
        "live_question": {"question": LIVE_QUESTION, **live},
        "passed": clone_to_answer < LIMIT_S
        and bench["http_status"] == 200
        and bench["has_answer_text"]
        and bench["groundedness_score"] is not None
        and all(cached for _, cached in bench["model_calls"])
        and live["http_status"] == 200
        and len(live["model_calls"]) >= 1
        and not any(cached for _, cached in live["model_calls"]),
    }
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["seconds"]), f"passed={report['passed']}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
