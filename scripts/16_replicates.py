"""Two more runs of the two tied retrieved-condition Sonnet 5 arms (configs/replicates.yaml),
generated and judged exactly as run 0 was, so their numbers carry a measured spread.

Per run r: each run-0 generation request is rebuilt from its trace, checked to hash to the
key run 0 recorded, tagged `_replicate` and sent again (src/replication/replicates.py); the
answers are parsed and traced as in Phase 4, then evaluated by the unchanged Phase 5 harness
(src/evaluation/evaluate.py: the same judge model, prompts and ceilings; judge requests carry
the same tag, so no run reuses another's verdicts). All calls go through the Message Batches
API into a response cache and batch store of their own (a DVC root apart from Phase 4/5's).

Modes:
  --dry-run   cost estimate before any spend -> results/metrics/replicates_cost_estimate.json.
              Generation input is counted exactly with the token-counting endpoint (the
              requests are run 0's); the judge's calls depend on the new answers, so they use
              run 0's measured usage per arm. Output: run 0's measured usage; worst case: every
              call at its max_tokens.
  --runs R..  the runs to generate and evaluate (default: every configured run). After the
              checkpoint run, the script stops if twice that run's cost would not fit under
              the phase cap.
  --offline   everything from the caches (replay only); reproduces every output exactly.
Writes:
  results/replicates/traces/run{r}/retrieved/<model>__<prompt>.jsonl  Phase 4 trace schema
  results/metrics/replicates_per_trace.jsonl   Phase 5 records of runs 1.., with `replicate`
  results/metrics/replicates.json              per-run metrics, mean +- std over runs 0..,
                                               selection per run, per-question stability

Usage:
    uv run python scripts/16_replicates.py --dry-run
    uv run python scripts/16_replicates.py --runs 1
    uv run python scripts/16_replicates.py --runs 2
    uv run --env-file configs/replay.env python scripts/16_replicates.py --offline
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version as pkg_version
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluate import Judge, aggregate, evaluate_trace, judge_traces, load_inputs
from src.generation.batch import run_batch_cached
from src.generation.llm import (
    AnthropicBackend,
    BudgetExceeded,
    ResponseCache,
    SpendLedger,
    replay_only,
)
from src.generation.prompts import load_registry
from src.generation.trace import read_traces, write_traces
from src.replication.replicates import (
    pooled_difference,
    replicate_request,
    replicate_trace,
    run_metrics,
    selection_per_run,
    spread,
    stability,
)

CONFIG_PATH = Path("configs/replicates.yaml")
EVAL_CONFIG_PATH = Path("configs/evaluation.yaml")
RESULTS_DIR = Path("results/metrics")
RUN0_TRACES = Path("results/traces")
RUN0_RECORDS = RESULTS_DIR / "eval_per_trace.jsonl"
PER_TRACE_PATH = RESULTS_DIR / "replicates_per_trace.jsonl"
SUMMARY_PATH = RESULTS_DIR / "replicates.json"
ESTIMATE_PATH = RESULTS_DIR / "replicates_cost_estimate.json"
REPORTED = (  # the run_metrics fields given mean +- std
    "correct_and_grounded",
    "accuracy_all",
    "accuracy_answered",
    "declined_rate",
    "mean_groundedness",
    "fully_grounded_rate",
    "citation_precision",
    "citation_recall",
    "gold_cited_rate",
)


class ReplicateJudge(Judge):
    """The Phase 5 judge, with batches recorded in the replicates' own batch store."""

    def __init__(self, *args, batch_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_dir = batch_dir

    def run(self, requests):
        if self.mode != "batch":
            return super().run(requests)
        unique = list({r.cache_key: r for r in requests}.values())
        b = self.cfg["batch"]
        outcome = run_batch_cached(
            unique,
            self.cache,
            self.ledger,
            None if replay_only() else AnthropicBackend().client,
            max_requests_per_batch=b["max_requests_per_batch"],
            poll_seconds=b["poll_seconds"],
            batch_dir=self.batch_dir,
            log=self.log,
        )
        self.batch_outcomes.append(outcome)
        self.new_cost_usd += outcome.new_cost_usd
        found = {r.cache_key: resp for r in unique if (resp := self.cache.get(r.cache_key))}
        if len(found) < len(unique):
            self.n_missing += len(unique) - len(found)
            self.log(f"WARNING: {len(unique) - len(found)} judge results missing")
        return found


def arm_parts(arm: str) -> tuple[str, str, str]:
    condition, model, prompt = arm.split("__")
    return condition, model, prompt


def run0_traces(arm: str) -> list[dict]:
    condition, model, prompt = arm_parts(arm)
    return read_traces(RUN0_TRACES / condition / f"{model}__{prompt}.jsonl")


def make_ledger(cfg: dict) -> SpendLedger:
    b = cfg["budget"]
    return SpendLedger(
        Path(b["ledger"]),
        b["prices_usd_per_mtok"],
        b["project_cap_usd"],
        b["phase"],
        b["phase_cap_usd"],
        b["batch_discount"],
    )


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------------------
# One run


def generate_run(cfg, eval_cfg, registry, run, cache, ledger) -> float:
    tag = cfg["replicate_tag"].format(run=run)
    plan = []
    for arm in cfg["arms"]:
        for t in run0_traces(arm):
            template = registry[t["prompt"]["id"]]
            plan.append((arm, t, template, replicate_request(t, template, tag)))
    print(f"run {run}: {len(plan)} generation requests rebuilt, every one matching run 0's key")
    b = eval_cfg["batch"]
    outcome = run_batch_cached(
        [p[3] for p in plan],
        cache,
        ledger,
        None if replay_only() else AnthropicBackend().client,
        max_requests_per_batch=b["max_requests_per_batch"],
        poll_seconds=b["poll_seconds"],
        batch_dir=Path(cfg["batch_dir"]),
    )
    if outcome.failures:
        sys.exit(f"{len(outcome.failures)} generation requests failed; re-run to retry them")
    by_arm: dict[str, list[dict]] = {arm: [] for arm in cfg["arms"]}
    for arm, t, template, req in plan:
        resp = cache.get(req.cache_key)
        cost = round(
            ledger.cost(req.model, resp.input_tokens or 0, resp.output_tokens or 0, batch=True), 6
        )
        by_arm[arm].append(replicate_trace(t, template, req, resp, run, cost))
    for arm, traces in by_arm.items():
        condition, model, prompt = arm_parts(arm)
        path = Path(cfg["traces_dir"]) / f"run{run}" / condition / f"{model}__{prompt}.jsonl"
        write_traces(path, traces)
    return outcome.new_cost_usd


def evaluate_run(cfg, eval_cfg, run, cache, ledger) -> tuple[list[dict], float]:
    ecfg = copy.deepcopy(eval_cfg)
    ecfg["traces"] = {
        "dir": f"{cfg['traces_dir']}/run{run}",
        "conditions": sorted({arm_parts(a)[0] for a in cfg["arms"]}),
        "models": sorted({arm_parts(a)[1] for a in cfg["arms"]}),
        "prompts": [arm_parts(a)[2] for a in cfg["arms"]],
    }
    inputs = load_inputs(ecfg)  # also checks every trace's evidence against its metrics
    judge = ReplicateJudge(
        ecfg,
        ecfg["judge"]["model"],
        "offline" if replay_only() else "batch",
        cache,
        ledger,
        replicate=cfg["replicate_tag"].format(run=run),
        batch_dir=Path(cfg["batch_dir"]),
    )
    judged = judge_traces(inputs.traces, inputs, judge)
    c = ecfg["correctness"]
    tols = tuple(c["sensitivity_rel_tols"])
    records = [
        evaluate_trace(t, inputs, judged[t["trace_id"]], c["rel_tol"], tols) | {"replicate": run}
        for t in inputs.traces
    ]
    if judge.n_missing and not replay_only():
        print(f"WARNING: run {run}: {judge.n_missing} judge results missing (judge errors)")
    return sorted(records, key=lambda r: r["trace_id"]), judge.new_cost_usd


# --------------------------------------------------------------------------------------
# Summary


def summarise(cfg: dict, eval_cfg: dict, records_by_run: dict[int, list[dict]]) -> dict:
    nb, seed = eval_cfg["bootstrap"]["n_resamples"], eval_cfg["bootstrap"]["seed"]
    arms_out = {}
    for arm in cfg["arms"]:
        per_run = {}
        for run, recs in sorted(records_by_run.items()):
            rs = [r for r in recs if f"{r['condition']}__{r['model_key']}__{r['prompt_id']}" == arm]
            per_run[str(run)] = run_metrics(rs, aggregate(rs, eval_cfg))
        arms_out[arm] = {
            "runs": per_run,
            "spread": {k: spread([per_run[r][k] for r in per_run]) for k in REPORTED},
            "stability": stability(records_by_run, arm),
        }
    selections = {
        str(run): selection_per_run(recs, nb, seed) for run, recs in sorted(records_by_run.items())
    }
    a, b = cfg["arms"]
    return {
        "arms": arms_out,
        "selection_per_run": selections,
        "winner_counts": {
            arm: sum(s["winner"] == arm for s in selections.values()) for arm in cfg["arms"]
        },
        "pooled_difference": pooled_difference(records_by_run, a, b, nb, seed),
    }


def check_run0(cfg: dict, eval_cfg: dict, run0: list[dict]) -> None:
    """Run 0's metrics recomputed here must equal Phase 5's and the selection's, since they
    are the same records through the same functions."""
    main = json.loads((RESULTS_DIR / "eval_main.json").read_text(encoding="utf-8"))
    sel = json.loads((RESULTS_DIR / "pipeline_selection.json").read_text(encoding="utf-8"))
    for arm in cfg["arms"]:
        rs = [r for r in run0 if f"{r['condition']}__{r['model_key']}__{r['prompt_id']}" == arm]
        mine = aggregate(rs, eval_cfg)
        if mine != main["arms"][arm]:
            sys.exit(f"run 0 of {arm} does not reproduce eval_main.json")
        m = run_metrics(rs, mine)
        s = sel["arms"][arm]
        if (m["n_correct_and_grounded"], m["accuracy_all"]) != (
            s["n_correct_and_grounded"],
            s["accuracy_all"],
        ):
            sys.exit(f"run 0 of {arm} does not reproduce pipeline_selection.json")


def write_summary(cfg, eval_cfg, ledger, run0, replicate_records) -> None:
    records_by_run = {0: run0}
    for r in replicate_records:
        records_by_run.setdefault(r["replicate"], []).append(r)
    summary = summarise(cfg, eval_cfg, records_by_run)
    complete = sorted(records_by_run) == [0, *cfg["runs"]]
    out = {
        "meta": {
            "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "runs_included": sorted(records_by_run),
            "complete": complete,
            "run_0": "the Phase 4 traces and Phase 5 records (eval_per_trace.jsonl)",
            "std": "sample standard deviation (n-1) over the runs included",
            "config": cfg,
            "judge": {
                "model": eval_cfg["judge"]["model"],
                "prompts": eval_cfg["judge"]["prompts"],
                "max_tokens": eval_cfg["judge"]["max_tokens"],
            },
            "phase_spent_usd": round(ledger.phase_spent(), 6),
            "environment": {
                "python": platform.python_version(),
                "anthropic": pkg_version("anthropic"),
            },
        },
        **summary,
    }
    SUMMARY_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print_summary(out)
    print(f"wrote {SUMMARY_PATH}")


def print_summary(out: dict) -> None:
    for arm, a in out["arms"].items():
        print(f"\n{arm}")
        for k in ("correct_and_grounded", "accuracy_all", "declined_rate", "mean_groundedness"):
            s = a["spread"][k]
            vals = ", ".join("n/a" if v is None else f"{v:.3f}" for v in s["values"])
            std = "" if s.get("std") is None else f" +- {s['std']:.3f}"
            mean = "n/a" if s["mean"] is None else f"{s['mean']:.3f}"
            print(f"  {k:22s} {mean}{std}   runs: {vals}")
        print(f"  stability: {a['stability']}")
    print(f"\nwinner per run: {out['winner_counts']}")
    print(f"pooled difference: {out['pooled_difference']}")


# --------------------------------------------------------------------------------------
# Dry run


def dry_run(cfg: dict, eval_cfg: dict, registry) -> None:
    import anthropic

    client = anthropic.Anthropic()
    main_cache = ResponseCache()  # read only: run 0's measured usage
    judge_keys = {}
    for r in load_jsonl(RUN0_RECORDS):
        judge_keys[f"{r['condition']}__{r['model_key']}__{r['prompt_id']}::{r['trace_id']}"] = r[
            "judge"
        ]
    price = cfg["budget"]["prices_usd_per_mtok"]["claude-sonnet-5"]
    disc = cfg["budget"]["batch_discount"]
    ceilings = eval_cfg["judge"]["max_tokens"]

    def usd(i: int, o: int) -> float:
        return (i * price["input"] + o * price["output"]) / 1e6 * disc

    arms = {}
    for arm in cfg["arms"]:
        traces = run0_traces(arm)
        gen_in = gen_out = counted_equal = 0
        for t in traces:
            req = replicate_request(t, registry[t["prompt"]["id"]], "dry_run")
            params = AnthropicBackend.call_params(req)
            params.pop("max_tokens")
            n = client.messages.count_tokens(**params).input_tokens
            counted_equal += n == t["generation"]["input_tokens"]
            gen_in += n
            gen_out += t["generation"]["output_tokens"]
        stages = {"generate": {"calls": len(traces), "in": gen_in, "out": gen_out}}
        worst_out = {"generate": len(traces) * t["generation"]["max_tokens"]}
        for stage in ("decompose", "verify", "correctness"):
            calls = i = o = 0
            for t in traces:
                key = judge_keys[f"{arm}::{t['trace_id']}"].get(stage)
                if key and (resp := main_cache.get(key)) is not None:
                    calls += 1
                    i += resp.input_tokens
                    o += resp.output_tokens
            stages[stage] = {"calls": calls, "in": i, "out": o}
            worst_out[stage] = calls * ceilings[stage]
        for stage, s in stages.items():
            s["expected_usd"] = round(usd(s["in"], s["out"]), 4)
            s["worst_case_usd"] = round(usd(s["in"], worst_out[stage]), 4)
        arms[arm] = {
            "stages": stages,
            "generation_input_counted_equals_run0_usage": f"{counted_equal}/{len(traces)}",
            "per_run_expected_usd": round(sum(s["expected_usd"] for s in stages.values()), 4),
            "per_run_worst_case_usd": round(sum(s["worst_case_usd"] for s in stages.values()), 4),
        }
    n_runs = len(cfg["runs"])
    per_run = sum(a["per_run_expected_usd"] for a in arms.values())
    ledger = make_ledger(cfg)
    estimate = {
        "meta": {
            "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "estimate": True,
            "prices_as_of": cfg["budget"]["prices_as_of"],
            "price_basis": f"Message Batches, {disc} of the standard price",
            "input_tokens": "generation: counted with the token-counting endpoint on the "
            "rebuilt requests; judge: run 0's measured usage (its requests depend on the "
            "new answers)",
            "output_tokens": "run 0's measured usage; worst case: every call at max_tokens",
        },
        "arms": arms,
        "per_run_expected_usd": round(per_run, 4),
        "total_expected_usd": round(per_run * n_runs, 4),
        "total_worst_case_usd": round(
            n_runs * sum(a["per_run_worst_case_usd"] for a in arms.values()), 4
        ),
        "n_runs": n_runs,
        "phase_spent_usd": round(ledger.phase_spent(), 4),
        "project_spent_usd": round(ledger.state["total_usd"], 4),
        "caps": {"project_usd": ledger.project_cap_usd, "phase_usd": ledger.phase_cap_usd},
        "checkpoint": f"stop after run {cfg['budget']['checkpoint_after_run']} if twice its "
        "measured cost exceeds the phase cap",
    }
    ESTIMATE_PATH.write_text(json.dumps(estimate, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in estimate.items() if k != "arms"}, indent=2))
    for arm, a in arms.items():
        print(
            f"  {arm}: ${a['per_run_expected_usd']:.3f} per run (worst "
            f"${a['per_run_worst_case_usd']:.3f}); generation input counted = run 0 usage "
            f"for {a['generation_input_counted_equals_run0_usage']}"
        )
    print(f"wrote {ESTIMATE_PATH}")


# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--runs", nargs="+", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    if args.offline:
        os.environ["RAG_REPLAY_ONLY"] = "1"
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
    registry = load_registry()
    if args.dry_run:
        dry_run(cfg, eval_cfg, registry)
        return

    runs = args.runs or cfg["runs"]
    if not set(runs) <= set(cfg["runs"]):
        parser.error(f"configured runs: {cfg['runs']}")
    cache = ResponseCache(Path(cfg["cache_dir"]))
    ledger = make_ledger(cfg)
    run0 = [
        r
        for r in load_jsonl(RUN0_RECORDS)
        if f"{r['condition']}__{r['model_key']}__{r['prompt_id']}" in cfg["arms"]
    ]
    check_run0(cfg, eval_cfg, run0)

    kept = [r for r in load_jsonl(PER_TRACE_PATH) if r["replicate"] not in runs]
    for run in sorted(runs):
        try:
            gen_cost = generate_run(cfg, eval_cfg, registry, run, cache, ledger)
            records, judge_cost = evaluate_run(cfg, eval_cfg, run, cache, ledger)
        except BudgetExceeded as e:
            sys.exit(f"STOPPED: {e}\nCompleted calls are cached; nothing is lost.")
        kept = [r for r in kept if r["replicate"] != run] + records
        write_jsonl(PER_TRACE_PATH, sorted(kept, key=lambda r: (r["replicate"], r["trace_id"])))
        cost = gen_cost + judge_cost
        print(f"run {run}: generation ${gen_cost:.4f} + judge ${judge_cost:.4f} = ${cost:.4f}")
        if (
            not replay_only()
            and run == cfg["budget"]["checkpoint_after_run"]
            and cost > 0
            and 2 * cost > cfg["budget"]["phase_cap_usd"]
        ):
            write_summary(cfg, eval_cfg, ledger, run0, kept)
            sys.exit(
                f"CHECKPOINT: run {run} cost ${cost:.4f}; two runs would exceed the "
                f"${cfg['budget']['phase_cap_usd']:.2f} cap. Stopping before the next run."
            )
    write_summary(cfg, eval_cfg, ledger, run0, kept)


if __name__ == "__main__":
    main()
