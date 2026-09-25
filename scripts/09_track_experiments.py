"""Experiment tracking, rebuilt from the committed results files, and the pipeline registry.

  select   apply the selection rule fixed in DECISIONS.md (Phase 7) to the Phase 5 per-trace
           evaluations: the correct-and-grounded rate of every arm, the ranking of the
           `retrieved` arms, the winner's lead over the runner-up, and the winning pipeline's
           configuration -> results/metrics/pipeline_selection.json
  track    one MLflow run per configuration that was measured, with the parameters, the
           metrics and the artefacts behind it, then register the selected pipeline as a
           pyfunc model (src/tracking/pipeline_model.py) -> results/metrics/mlflow_runs.json
             phase2-index-build        18  chunking x embedder x vector store
             phase3-retrieval-grid     36  chunking x embedder x retrieval method
             phase5-generation-eval    16  condition x generator x prompt
             phase6-adversarial-eval   16  condition x generator x prompt
             phase6-reliability         1  the analysis, its plots and prediction checks
           Every number logged is read from a committed results file, so MLflow never holds
           a number the repository does not. Re-running replaces the runs of these
           experiments (the tracking store is gitignored and rebuilt, not accumulated).

Tracking goes to MLFLOW_TRACKING_URI (default sqlite:///mlflow.db; the compose stack's
server is http://localhost:5000). No API calls.

Usage:
    uv run python scripts/09_track_experiments.py select
    uv run python scripts/09_track_experiments.py track
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tracking.selection import SELECTION_CONDITION, arm_key, select  # noqa: E402

METRICS_DIR = Path("results/metrics")
PLOTS_DIR = Path("results/plots")
TRACES_DIR = Path("results/traces")
GEN_CONFIG = Path("configs/generation.yaml")
EVAL_CONFIG = Path("configs/evaluation.yaml")
SELECTION_PATH = METRICS_DIR / "pipeline_selection.json"
RUNS_MANIFEST = METRICS_DIR / "mlflow_runs.json"
REGISTERED_MODEL = "rag-groundedness-pipeline"
BOOTSTRAP = {"n_resamples": 10000, "seed": 0}  # the Phase 5/6 bootstrap settings

EXPERIMENTS = (
    "phase2-index-build",
    "phase3-retrieval-grid",
    "phase5-generation-eval",
    "phase6-adversarial-eval",
    "phase6-reliability",
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# select


def pipeline_config(gen: dict, winner: dict) -> dict:
    from src.generation.prompts import load_registry

    template = load_registry()[winner["prompt_id"]]
    return {
        "retrieval": gen["retrieval"],
        "generator": {"key": winner["model_key"], "model": gen["models"][winner["model_key"]]},
        "prompt": {
            "id": template.id,
            "version": template.version,
            "file_sha256": template.file_sha256,
        },
        "max_tokens": gen["max_tokens"],
    }


def cmd_select() -> None:
    per_trace = METRICS_DIR / "eval_per_trace.jsonl"
    rows = load_jsonl(per_trace)
    gen = yaml.safe_load(GEN_CONFIG.read_text(encoding="utf-8"))
    sel = select(rows, **BOOTSTRAP)
    winner = sel["arms"][sel["winner"]]
    out = {
        "meta": {
            "single_run": True,
            "rule": (
                f"the {SELECTION_CONDITION}-condition arm with the highest share of all its "
                "questions answered correct (strict) and fully grounded; ties: higher "
                "accuracy_all, then the earlier prompt version (DECISIONS.md, Phase 7)"
            ),
            "source": {"eval_per_trace": per_trace.as_posix(), "sha256": sha256_file(per_trace)},
            "bootstrap": BOOTSTRAP,
        },
        **sel,
        "pipeline_config": pipeline_config(gen, winner),
    }
    SELECTION_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"ranking ({SELECTION_CONDITION}):")
    for k in sel["ranking"]:
        a = sel["arms"][k]
        cg = a["correct_and_grounded"]
        print(
            f"  {k:45s} {cg['mean']:.3f} [{cg['ci95'][0]:.3f}, {cg['ci95'][1]:.3f}]"
            f"  accuracy {a['accuracy_all']:.3f}"
        )
    d = sel["winner_minus_runner_up"]
    lo, hi = d["ci95"]
    print(f"winner {sel['winner']}; lead over runner-up {d['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"wrote {SELECTION_PATH}")


# --------------------------------------------------------------------------------------
# track


def metric_name(key: str) -> str:
    """MLflow metric names allow letters, digits and _ - . / : and spaces."""
    return key.replace("@", "_at_")


def flatten(obj, prefix: str = "") -> dict[str, float]:
    """Numeric leaves of a results block as dotted names; a 95% interval [lo, hi] becomes
    `<name>.ci95_lo` / `.ci95_hi`. Booleans and non-finite values are skipped."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if prefix.endswith("ci95") and len(obj) == 2:
            for side, v in zip(("lo", "hi"), obj, strict=True):
                if isinstance(v, int | float) and math.isfinite(v):
                    out[f"{prefix}_{side}"] = float(v)
    elif isinstance(obj, int | float) and not isinstance(obj, bool) and math.isfinite(obj):
        out[prefix] = float(obj)
    return {metric_name(k): v for k, v in out.items()}


class Tracker:
    """Logs runs through the client API (batched), one experiment at a time."""

    def __init__(self):
        import mlflow
        from mlflow import MlflowClient

        from src.tracking.mlflow_utils import tracking_uri

        mlflow.set_tracking_uri(tracking_uri())
        self.mlflow = mlflow
        self.client = MlflowClient()
        self.manifest: dict[str, list[dict]] = {}

    def experiment(self, name: str) -> str:
        exp = self.client.get_experiment_by_name(name)
        if exp is None:
            return self.client.create_experiment(name)
        if exp.lifecycle_stage == "deleted":
            self.client.restore_experiment(exp.experiment_id)
        for run in self.client.search_runs([exp.experiment_id], max_results=50000):
            self.client.delete_run(run.info.run_id)
        return exp.experiment_id

    def log(
        self,
        experiment: str,
        run_name: str,
        params: dict,
        metrics: dict[str, float],
        tags: dict | None = None,
        artifacts: list[tuple[Path, str]] = (),
        texts: list[tuple[str, str]] = (),
    ) -> str:
        from mlflow.entities import Metric, Param, RunTag

        exp_id = self.experiment_ids[experiment]
        run = self.client.create_run(exp_id, run_name=run_name)
        rid = run.info.run_id
        params = {k: ("" if v is None else str(v)) for k, v in params.items()}
        tags = {"source": "results files (scripts/09_track_experiments.py)", **(tags or {})}
        items = sorted(metrics.items())
        for i in range(0, max(len(items), 1), 1000):
            self.client.log_batch(
                rid,
                metrics=[Metric(k, v, 0, 0) for k, v in items[i : i + 1000]],
                params=[Param(k, v) for k, v in params.items()] if i == 0 else [],
                tags=[RunTag(k, str(v)) for k, v in tags.items()] if i == 0 else [],
            )
        for path, dest in artifacts:
            self.client.log_artifact(rid, str(path), artifact_path=dest)
        for text, name in texts:
            self.client.log_text(rid, text, name)
        self.client.set_terminated(rid)
        self.manifest.setdefault(experiment, []).append(
            {"run_name": run_name, "n_params": len(params), "n_metrics": len(metrics)}
        )
        return rid


def track_index_build(t: Tracker) -> None:
    stats = load_json(METRICS_DIR / "index_stats.json")
    for config, backends in stats["configs"].items():
        chunking, embedder = config.split("__")
        eq = stats["equivalence_check"].get(config, {})
        for backend, s in backends.items():
            metrics = flatten(
                {
                    "build_time_sec": s["build_time_sec"],
                    "n_vectors": s["n_vectors"],
                    "index_size_bytes": s["index_size_bytes"],
                    "query_latency": s["query_latency"],
                    "backend_top_k_match_rate": eq.get("match_rate"),
                }
            )
            t.log(
                "phase2-index-build",
                f"{config}__{backend}",
                {
                    "chunking": chunking,
                    "embedder": embedder,
                    "vector_store": backend,
                    "embedding_dim": stats["meta"][embedder]["dim"],
                    "n_benchmark_queries": stats["n_benchmark_queries"],
                },
                metrics,
                {"phase": "2", "single_run": "true"},
            )


def track_retrieval_grid(t: Tracker) -> None:
    grid = load_json(METRICS_DIR / "retrieval_grid.json")
    cfg = grid["meta"]["config"]
    winner = grid["selection"]["winner"]
    for cell, c in grid["cells"].items():
        t.log(
            "phase3-retrieval-grid",
            cell,
            {
                "chunking": c["chunking"],
                "embedder": c["embedding"],
                "retriever": c["method"],
                "vector_store": cfg["backend"],
                "k_values": cfg["k_values"],
                "n_questions": c["n_questions"],
            },
            flatten({"metrics": c["metrics"], "latency": c["latency"]}),
            {"phase": "3", "single_run": "true", "selected": str(cell == winner).lower()},
        )


def track_generation_eval(t: Tracker, selection: dict) -> None:
    gen = yaml.safe_load(GEN_CONFIG.read_text(encoding="utf-8"))
    ev = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
    main = load_json(METRICS_DIR / "eval_main.json")
    runs = load_json(METRICS_DIR / "generation_runs.json")["arms"]
    per_trace = load_jsonl(METRICS_DIR / "eval_per_trace.jsonl")
    rc = gen["retrieval"]
    snapshot = yaml.safe_dump({"generation": gen, "evaluation": ev}, sort_keys=False)
    for arm, m in main["arms"].items():
        condition, model_key, prompt_id = arm.split("__")
        g = runs[arm]
        sel = selection["arms"][arm]
        metrics = flatten({k: v for k, v in m.items() if k != "judge_errors"}) | flatten(
            {
                "generation": {
                    k: g[k]
                    for k in (
                        "parse_ok_rate",
                        "abstained_token_rate",
                        "mean_citations",
                        "invalid_citation_rate",
                        "input_tokens_total",
                        "output_tokens_total",
                        "output_tokens_mean",
                        "cost_usd_total",
                        "latency_ms",
                    )
                },
                "correct_and_grounded": sel["correct_and_grounded"],
            }
        )
        rows = [r for r in per_trace if arm_key(r) == arm]
        with tempfile.TemporaryDirectory() as tmp:
            eval_rows = Path(tmp) / "eval_per_trace.jsonl"
            eval_rows.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
            )
            t.log(
                "phase5-generation-eval",
                arm,
                {
                    "condition": condition,
                    "chunking": rc["chunking"],
                    "embedder": rc["embedding"],
                    "retriever": rc["method"] if condition == "retrieved" else "oracle",
                    "vector_store": rc["backend"],
                    "k": rc["k"],
                    "prompt_id": prompt_id,
                    "prompt_version": g["prompt_version"],
                    "prompt_sha256": g["prompt_file_sha256"],
                    "generation_model": model_key,
                    "generation_model_id": gen["models"][model_key]["model"],
                    "max_tokens": gen["max_tokens"],
                    "judge_model": ev["judge"]["model"],
                    "n_questions": m["n"],
                },
                metrics,
                {
                    "phase": "5",
                    "single_run": "true",
                    "selected": str(arm == selection["winner"]).lower(),
                    "model_digest": g.get("model_digest", ""),
                },
                artifacts=[
                    (Path(g["trace_file"]), "traces"),
                    (eval_rows, "evaluation"),
                    (Path("prompts") / f"{prompt_id}.md", "prompts"),
                    (PLOTS_DIR / "reliability_diagram.png", "plots"),
                ],
                texts=[(snapshot, "config/config_snapshot.yaml")],
            )


def track_adversarial(t: Tracker) -> None:
    gen = yaml.safe_load(GEN_CONFIG.read_text(encoding="utf-8"))
    q4 = load_json(METRICS_DIR / "reliability_analysis.json")["q4"]
    runs = load_json(METRICS_DIR / "generation_runs_adversarial.json")["arms"]
    rc = gen["retrieval"]
    for arm, g in runs.items():
        condition, model_key, prompt_id = arm.split("__")
        rates: dict = {}
        for cat, by_cond in q4["by_category"].items():
            block = by_cond.get(condition, {}).get(model_key)
            if block is None:
                continue
            for name, v in block.items():
                if isinstance(v, dict) and prompt_id in v.get("per_prompt", {}):
                    rates[f"{cat}.{name}"] = v["per_prompt"][prompt_id]
        if condition == "retrieved":
            for rater, p in q4["premise"].get(model_key, {}).items():
                for name, v in p.items():
                    if isinstance(v, dict) and prompt_id in v.get("per_prompt", {}):
                        rates[f"d.premise_{name}.{rater}"] = v["per_prompt"][prompt_id]
        t.log(
            "phase6-adversarial-eval",
            arm,
            {
                "condition": condition,
                "chunking": rc["chunking"],
                "embedder": rc["embedding"],
                "retriever": rc["method"] if condition == "retrieved" else "oracle",
                "k": rc["k"],
                "prompt_id": prompt_id,
                "prompt_version": g["prompt_version"],
                "generation_model": model_key,
                "n_traces": g["n"],
            },
            flatten(rates),
            {"phase": "6", "single_run": "true"},
            artifacts=[(Path(g["trace_file"]), "traces")],
        )


def track_reliability(t: Tracker) -> None:
    res = load_json(METRICS_DIR / "reliability_analysis.json")
    preds = res["predictions"]
    checked = [p for p in preds if p.get("observed") not in (None, "pending")]
    t.log(
        "phase6-reliability",
        "reliability-analysis",
        {"n_predictions": len(preds), "bootstrap_resamples": BOOTSTRAP["n_resamples"]},
        {
            "predictions_checked": len(checked),
            "predictions_unexpected": sum(bool(p.get("unexpected")) for p in checked),
        },
        {"phase": "6", "single_run": "true"},
        artifacts=[
            (METRICS_DIR / "reliability_analysis.json", "results"),
            (METRICS_DIR / "reliability_analysis.md", "results"),
            (METRICS_DIR / "quadrant_examples.md", "results"),
            *[(p, "plots") for p in sorted(PLOTS_DIR.glob("*.png"))],
        ],
    )


def register_pipeline(t: Tracker, selection: dict) -> dict:
    from mlflow.exceptions import MlflowException

    from src.tracking.pipeline_model import RAGPipelineModel

    cfg = selection["pipeline_config"]
    exp_id = t.experiment_ids["phase5-generation-eval"]
    try:  # a rebuild replaces the registry entry, as it replaces the runs
        t.client.delete_registered_model(REGISTERED_MODEL)
    except MlflowException:
        pass  # not registered yet
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "pipeline_config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        with t.mlflow.start_run(experiment_id=exp_id, run_name="registered-pipeline") as run:
            t.mlflow.set_tags(
                {
                    "source": "results files (scripts/09_track_experiments.py)",
                    "selected_arm": selection["winner"],
                }
            )
            t.mlflow.log_params(
                {
                    "chunking": cfg["retrieval"]["chunking"],
                    "embedder": cfg["retrieval"]["embedding"],
                    "retriever": cfg["retrieval"]["method"],
                    "k": cfg["retrieval"]["k"],
                    "prompt_id": cfg["prompt"]["id"],
                    "generation_model": cfg["generator"]["key"],
                }
            )
            cg = selection["arms"][selection["winner"]]["correct_and_grounded"]
            t.mlflow.log_metrics(flatten({"correct_and_grounded": cg}))
            info = t.mlflow.pyfunc.log_model(
                name="pipeline",
                python_model=RAGPipelineModel(),
                artifacts={
                    "pipeline_config": str(cfg_path),
                    "prompt": str(Path("prompts") / f"{cfg['prompt']['id']}.md"),
                },
                code_paths=["src"],
                registered_model_name=REGISTERED_MODEL,
            )
    t.manifest.setdefault("phase5-generation-eval", []).append(
        {"run_name": "registered-pipeline", "registered_model": REGISTERED_MODEL}
    )
    return {"name": REGISTERED_MODEL, "run_id": run.info.run_id, "model_uri": info.model_uri}


def purge_deleted(t: Tracker) -> None:
    """Permanently remove the runs a rebuild replaced (`mlflow gc`), for the local store. A
    tracking server's store is garbage-collected where the server runs."""
    import subprocess

    uri = t.mlflow.get_tracking_uri()
    if not uri.startswith("sqlite:"):
        print(f"replaced runs are soft-deleted on {uri}; run `mlflow gc` on the server to purge")
        return
    env = {**os.environ, "MLFLOW_TRACKING_URI": uri}
    cmd = [sys.executable, "-m", "mlflow", "gc", "--backend-store-uri", uri]
    subprocess.run(cmd, env=env, check=True, capture_output=True)


def cmd_track() -> None:
    selection = load_json(SELECTION_PATH)
    t = Tracker()
    t.experiment_ids = {name: t.experiment(name) for name in EXPERIMENTS}
    track_index_build(t)
    track_retrieval_grid(t)
    track_generation_eval(t, selection)
    track_adversarial(t)
    track_reliability(t)
    registered = register_pipeline(t, selection)
    purge_deleted(t)
    manifest = {
        "meta": {
            "note": "run and model ids differ on every rebuild and are not recorded here",
            "registered_model": registered["name"],
            "selected_arm": selection["winner"],
        },
        "experiments": {
            name: {"n_runs": len(runs), "runs": runs} for name, runs in t.manifest.items()
        },
        "n_runs": sum(len(r) for r in t.manifest.values()),
    }
    RUNS_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, runs in t.manifest.items():
        print(f"  {name:28s} {len(runs)} runs")
    print(f"registered {registered['name']} from run {registered['run_id']}")
    print(f"wrote {RUNS_MANIFEST}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["select", "track"])
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    {"select": cmd_select, "track": cmd_track}[args.command]()


if __name__ == "__main__":
    main()
