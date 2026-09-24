"""Generate answers for every (condition, model, prompt, question) and log each as a trace.

Two context conditions (configs/generation.yaml):
  retrieved  the Phase 3 winning retriever over the whole corpus, all 150 questions. Every
             top-k is checked against the ids Phase 3 stored, and its metrics against
             Phase 3's, so generation provably sees the retrieval that was scored.
  oracle     the 126 questions with aligned gold evidence, given the chunks that contain
             it (src/retrieval/oracle.py): generation with the evidence guaranteed present.
Each trace carries its context's retrieval metrics against the gold evidence.

Writes:
  results/traces/<condition>/<model>__<prompt>.jsonl   one trace per question
                                                  (schema: src/generation/trace.py)
  results/metrics/generation_runs.json            per-(condition, model, prompt) summary:
                                                  parse rates, abstention-token rate,
                                                  tokens, cost, latency
  results/metrics/api_spend.json                  cumulative API spend ledger (all phases)
  results/metrics/generation_cost_estimate.json   --dry-run only
  results/metrics/generation_replay_check.json    --verify-replay only
Pilot runs (--pilot N) write to results/traces/pilot/<condition>/ and
generation_runs_pilot.json.
One MLflow run per (condition, model, prompt), with the prompt version and hash as params.

Every API response is cached on disk by request hash (data/cache/llm/), so re-running
costs nothing and rewrites identical traces.

Usage:
    uv run python scripts/05_generate.py --dry-run
    uv run python scripts/05_generate.py --pilot 10
    uv run python scripts/05_generate.py
    uv run python scripts/05_generate.py --models qwen2.5-3b --prompts v1_zero_shot
    uv run python scripts/05_generate.py --verify-replay
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import random
import statistics
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version as pkg_version
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.llm import (
    AnthropicBackend,
    BudgetExceeded,
    GenerationRequest,
    OllamaBackend,
    ResponseCache,
    SpendLedger,
    generate_cached,
)
from src.generation.parsing import parse_output
from src.generation.prompts import PromptTemplate, load_registry, render
from src.generation.trace import (
    TRACES_DIR,
    build_trace,
    read_traces,
    replay_trace,
    trace_chunks,
    write_traces,
)

CONFIG_PATH = Path("configs/generation.yaml")
FINANCEBENCH_PATH = Path("data/raw/financebench/financebench_merged.jsonl")
PARSED_DIR = Path("data/processed/parsed")
INDICES_DIR = Path("data/indices")
RESULTS_DIR = Path("results/metrics")
PHASE3_PER_QUESTION = RESULTS_DIR / "retrieval_per_question.jsonl"
PHASE3_CONFIG = Path("configs/retrieval_grid.yaml")
CHUNKS_DIR = Path("data/processed/chunks")
PHASE = "phase4_generation"
# Metrics that depend only on the top k=5, so the retrieved condition's recomputation must
# equal Phase 3's stored values exactly (MRR is excluded: Phase 3 computed it over 20).
PHASE3_METRICS_CHECKED = ("recall@5", "precision@5", "ndcg@5", "span_recall@5", "doc_hit@5")
QUESTION_FIELDS = (
    "financebench_id",
    "question",
    "question_type",
    "question_reasoning",
    "company",
    "doc_name",
    "doc_period",
    "doc_type",
)


# --------------------------------------------------------------------------------------
# Questions and retrieval


def load_questions() -> list[dict]:
    rows = [json.loads(line) for line in FINANCEBENCH_PATH.read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda r: r["financebench_id"])
    out = []
    for r in rows:
        q = {k: r.get(k) for k in QUESTION_FIELDS}
        q["in_corpus"] = (PARSED_DIR / f"{r['doc_name']}.json").exists()
        out.append(q)
    return out


def pilot_sample(questions: list[dict], n: int, seed: int) -> list[dict]:
    """Fixed sample, stratified so out-of-corpus questions appear in proportion (at
    least one) and the pilot exercises the no-relevant-context path."""
    rng = random.Random(seed)
    outside = [q for q in questions if not q["in_corpus"]]
    inside = [q for q in questions if q["in_corpus"]]
    n_out = min(len(outside), max(1, math.ceil(n * len(outside) / len(questions))))
    picked = rng.sample(outside, n_out) + rng.sample(inside, n - n_out)
    return sorted(picked, key=lambda q: q["financebench_id"])


def load_phase3_module():
    """Phase 3's own question loading and gold-evidence alignment, so both conditions are
    scored against exactly the gold spans Phase 3 used."""
    path = Path(__file__).resolve().parent / "04_eval_retrieval.py"
    spec = importlib.util.spec_from_file_location("eval_retrieval_phase3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_contexts(
    cfg: dict, questions: list[dict], conditions: list[str]
) -> tuple[dict[str, dict[str, dict]], dict]:
    """Return {condition: {financebench_id: retrieval record}} plus a check report.

    `retrieved`: the Phase 3 winner over the whole corpus, for every question. Its top-k
    must match the ids Phase 3 scored, and its metrics must equal Phase 3's.
    `oracle`: for questions with aligned gold evidence only, see src/retrieval/oracle.py.
    Both conditions' `context_metrics` are computed the same way, on the k chunks the
    model is shown (so `mrr` here is MRR within those k).
    """
    from src.evaluation.retrieval_metrics import count_relevant, question_metrics
    from src.retrieval.embeddings import EmbeddingModel
    from src.retrieval.oracle import build_oracle_context
    from src.retrieval.scoped import ScopedFaissIndex

    rc = cfg["retrieval"]
    if rc["method"] != "dense" or rc["backend"] != "faiss":
        raise NotImplementedError("generation is wired for the dense FAISS winner only")
    cell = f"{rc['chunking']}__{rc['embedding']}__{rc['method']}"
    k, depth = rc["k"], rc["search_depth"]
    if depth < k:
        raise ValueError("retrieval.search_depth must be >= retrieval.k")

    p3 = load_phase3_module()
    cfg3 = yaml.safe_load(PHASE3_CONFIG.read_text(encoding="utf-8"))
    min_ov = cfg3["relevance"]["min_overlap_frac"]
    wanted = {q["financebench_id"] for q in questions}
    p3_rows, _ = p3.load_questions(None)  # the raw benchmark rows, evidence included
    golds, _ = p3.align_all([r for r in p3_rows if r["financebench_id"] in wanted], cfg3)
    by_doc: dict[str, list] = {}
    chunk_file = CHUNKS_DIR / f"{rc['chunking']}.jsonl"
    for line in chunk_file.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        by_doc.setdefault(c["doc_id"], []).append(p3.chunk_span(c))

    phase3 = {}
    for line in PHASE3_PER_QUESTION.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["cell"] == cell:
            phase3[row["financebench_id"]] = row
    if not phase3:
        raise ValueError(f"no Phase 3 results for {cell}")

    embed = EmbeddingModel(rc["embedding"])
    index = ScopedFaissIndex(INDICES_DIR / f"{rc['chunking']}__{rc['embedding']}__faiss")

    def metrics_for(results, gs) -> dict | None:
        if not gs:
            return None
        spans = [p3.chunk_span(r.metadata) for r in results]
        return question_metrics(spans, gs, count_relevant(by_doc, gs, min_ov), [1, 3, 5], min_ov)

    contexts: dict[str, dict[str, dict]] = {c: {} for c in conditions}
    match_counts, mismatches, oracle_recall = Counter(), [], []
    for q in questions:
        qid = q["financebench_id"]
        gs = golds.get(qid, [])
        qv = embed.encode_queries([q["question"]])

        if "retrieved" in conditions:
            results = index.search_corpus(qv, depth)[:k]
            ids = [r.chunk_id for r in results]
            m = metrics_for(results, gs)
            match = None  # not scored in Phase 3 (filing not in corpus, or no aligned gold)
            if qid in phase3:
                stored = phase3[qid]["top_ids"][:k]
                match = (
                    "exact"
                    if ids == stored
                    else ("same_set_reordered" if set(ids) == set(stored) else "different")
                )
                same_metrics = all(
                    math.isclose(m[x], phase3[qid]["metrics"][x], abs_tol=1e-12)
                    for x in PHASE3_METRICS_CHECKED
                )
                if match == "different" or not same_metrics:
                    mismatches.append({"financebench_id": qid, "now": ids, "phase3": stored})
            match_counts[str(match)] += 1
            contexts["retrieved"][qid] = {
                "condition": "retrieved",
                "config": cell,
                "k": k,
                "phase3_top_k_match": match,
                "context_metrics": m,
                "chunks": trace_chunks(results),
            }

        if "oracle" in conditions and gs:
            ranked = index.search_filing(qv, q["doc_name"])  # every chunk of the filing
            chosen = build_oracle_context(
                ranked,
                gs,
                k,
                to_span=lambda r: p3.chunk_span(r.metadata),
                min_overlap_frac=min_ov,
            )
            m = metrics_for(chosen, gs)
            oracle_recall.append(m["recall@5"])
            contexts["oracle"][qid] = {
                "condition": "oracle",
                "config": f"oracle__{cell}",
                "k": k,
                "context_metrics": m,
                "chunks": trace_chunks(chosen),
            }

    if mismatches:
        raise RuntimeError(
            f"{len(mismatches)} questions retrieve a different top-{k} (or score differently) "
            f"than Phase 3; first: {mismatches[0]}"
        )
    report = {"cell": cell, "k": k, "search_depth": depth}
    if "retrieved" in conditions:
        report["retrieved_phase3_top_k_match_counts"] = dict(match_counts)
    if "oracle" in conditions:
        report["oracle"] = {
            "n_questions": len(oracle_recall),
            "mean_recall@5": statistics.fmean(oracle_recall) if oracle_recall else None,
            "n_recall@5_below_1": sum(r < 1 for r in oracle_recall),
        }
    return contexts, report


# --------------------------------------------------------------------------------------
# Generation


def make_backend(model_cfg: dict):
    if model_cfg["backend"] == "anthropic":
        return AnthropicBackend()
    if model_cfg["backend"] == "ollama":
        return OllamaBackend()
    raise ValueError(f"unknown backend {model_cfg['backend']!r}")


def request_params(model_cfg: dict) -> dict:
    if model_cfg["backend"] == "anthropic":
        return AnthropicBackend.request_params(thinking=model_cfg["thinking"])
    return OllamaBackend.request_params(
        temperature=model_cfg["temperature"], seed=model_cfg["seed"], num_ctx=model_cfg["num_ctx"]
    )


def build_request(
    model_cfg: dict, max_tokens: int, template: PromptTemplate, question: dict, chunks: list[dict]
) -> tuple[GenerationRequest, str]:
    rendered = render(template, question["question"], chunks)
    req = GenerationRequest(
        backend=model_cfg["backend"],
        model=model_cfg["model"],
        system=rendered.system,
        user=rendered.user,
        max_tokens=max_tokens,
        params=request_params(model_cfg),
    )
    return req, rendered.sha256


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[max(0, int(round(q * len(s))) - 1)]


def summarise_arm(traces: list[dict], n_cached: int, new_cost: float) -> dict:
    parsed = [t["parsed"] for t in traces]
    gens = [t["generation"] for t in traces]
    n = len(traces)
    confs = [p["confidence"] for p in parsed if p["confidence"] is not None]
    lat = [g["latency_ms"] for g in gens]
    status = Counter(p["status"] for p in parsed)
    return {
        "n": n,
        "parse_status": {s: status.get(s, 0) for s in ("ok", "partial", "failed")},
        "parse_ok_rate": status.get("ok", 0) / n,
        "abstained_token_rate": sum(p["abstained_token"] for p in parsed) / n,
        "mean_citations": statistics.fmean(len(p["citations"]) for p in parsed),
        "invalid_citation_rate": sum(bool(p["invalid_citations"]) for p in parsed) / n,
        "confidence": {
            "n_parsed": len(confs),
            "mean": statistics.fmean(confs) if confs else None,
            "median": statistics.median(confs) if confs else None,
        },
        "issues": dict(Counter(i for p in parsed for i in p["issues"]).most_common()),
        "stop_reasons": dict(Counter(str(g["stop_reason"]) for g in gens)),
        "models_reported": sorted({g["model_reported"] for g in gens}),
        "input_tokens_total": sum(g["input_tokens"] or 0 for g in gens),
        "output_tokens_total": sum(g["output_tokens"] or 0 for g in gens),
        "output_tokens_mean": statistics.fmean(g["output_tokens"] or 0 for g in gens),
        "cost_usd_total": sum(g["cost_usd"] or 0 for g in gens),
        "latency_ms": {"p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95)},
        "this_run": {"cache_hits": n_cached, "new_calls": n - n_cached, "new_cost_usd": new_cost},
    }


def run_arm(
    model_key, model_cfg, template, questions, retrievals, backend, cache, ledger, max_tokens
) -> tuple[list[dict], int, float]:
    def one(q: dict) -> tuple[dict, bool, float]:
        retrieval = retrievals[q["financebench_id"]]
        req, rendered_sha = build_request(model_cfg, max_tokens, template, q, retrieval["chunks"])
        resp, was_cached, cost = generate_cached(backend, req, cache, ledger)
        parsed = parse_output(resp.text, template.output_fields, len(retrieval["chunks"]))
        # Cost of the original call, recomputed from its recorded usage so a trace written
        # from the cache carries the same number as the one written when the call was made.
        call_cost = None
        if ledger is not None and resp.input_tokens is not None:
            call_cost = round(ledger.cost(req.model, resp.input_tokens, resp.output_tokens or 0), 6)
        trace = build_trace(
            question=q,
            retrieval=retrieval,
            template=template,
            request=req,
            model_key=model_key,
            response=resp,
            parsed=parsed,
            rendered_sha256=rendered_sha,
            cost_usd=call_cost,
        )
        return trace, was_cached, cost

    workers = max(1, int(model_cfg.get("concurrency", 1)))
    traces, n_cached, new_cost = [], 0, 0.0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for trace, was_cached, cost in pool.map(one, questions):
            traces.append(trace)
            n_cached += was_cached
            new_cost += cost
            done += 1
            if done % 25 == 0 or done == len(questions):
                print(f"    {done}/{len(questions)}  (cached {n_cached}, new ${new_cost:.4f})")
    return traces, n_cached, new_cost


def log_mlflow(
    cfg, condition, model_key, model_cfg, template, summary, trace_path, pilot, n_questions
):
    import mlflow

    from src.tracking.mlflow_utils import init_tracking

    init_tracking()
    rc = cfg["retrieval"]
    with mlflow.start_run(
        run_name=f"gen__{condition}__{model_key}__{template.id}" + ("__pilot" if pilot else "")
    ):
        mlflow.set_tags({"phase": PHASE, "pilot": str(pilot)})
        mlflow.log_params(
            {
                "context_condition": condition,
                "prompt_id": template.id,
                "prompt_version": template.version,
                "prompt_variant": template.variant,
                "prompt_file_sha256": template.file_sha256,
                "model_key": model_key,
                "backend": model_cfg["backend"],
                "model_requested": model_cfg["model"],
                "models_reported": ",".join(summary["models_reported"]),
                "chunking": rc["chunking"],
                "embedder": rc["embedding"],
                "retriever": rc["method"],
                "k": rc["k"],
                "max_tokens": cfg["max_tokens"],
                "request_params": json.dumps(request_params(model_cfg), sort_keys=True),
                "n_questions": n_questions,
            }
        )
        mlflow.log_metrics(
            {
                "parse_ok_rate": summary["parse_ok_rate"],
                "parse_failed": summary["parse_status"]["failed"],
                "abstained_token_rate": summary["abstained_token_rate"],
                "mean_citations": summary["mean_citations"],
                "invalid_citation_rate": summary["invalid_citation_rate"],
                "output_tokens_mean": summary["output_tokens_mean"],
                "input_tokens_total": summary["input_tokens_total"],
                "output_tokens_total": summary["output_tokens_total"],
                "cost_usd_total": summary["cost_usd_total"],
                **(
                    {"confidence_mean": summary["confidence"]["mean"]}
                    if summary["confidence"]["mean"] is not None
                    else {}
                ),
                **(
                    {
                        "latency_p50_ms": summary["latency_ms"]["p50"],
                        "latency_p95_ms": summary["latency_ms"]["p95"],
                    }
                    if summary["latency_ms"]["p50"] is not None
                    else {}
                ),
            }
        )
        mlflow.log_artifact(str(trace_path), artifact_path="traces")
        mlflow.log_artifact(str(CONFIG_PATH), artifact_path="config")
        mlflow.log_artifact(f"prompts/{template.id}.md", artifact_path="prompts")


def environment_meta(cfg: dict, selected_models: list[str]) -> dict:
    meta = {
        "python": platform.python_version(),
        "anthropic": pkg_version("anthropic"),
        "ollama_client": pkg_version("ollama"),
    }
    if any(cfg["models"][m]["backend"] == "ollama" for m in selected_models):
        try:
            import os

            host = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
            with urllib.request.urlopen(f"{host}/api/version", timeout=5) as r:
                meta["ollama_server"] = json.loads(r.read())["version"]
        except OSError as e:
            meta["ollama_server"] = f"unavailable: {e}"
    try:
        import torch

        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        meta["gpu"] = None
    return meta


# --------------------------------------------------------------------------------------
# Dry run and replay


# Assumed mean output tokens per prompt, used only when no pilot traces exist yet.
ASSUMED_OUTPUT_TOKENS = {
    "v1_zero_shot": 120,
    "v2_citation_required": 300,
    "v3_chain_of_thought": 700,
    "v4_abstention": 120,
}


def dry_run(cfg, registry, models, prompts, questions, contexts, cache) -> None:
    """Estimate API cost of the uncached calls: input tokens measured with the token
    counting endpoint on a sample and extrapolated by characters; output tokens from the
    pilot traces when they exist (else the stated assumptions); plus the worst case where
    every call uses its full max_tokens."""
    import anthropic

    client = anthropic.Anthropic()
    prices = cfg["budget"]["prices_usd_per_mtok"]
    estimate = {
        "meta": {
            "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_questions": len(questions),
            "prices_as_of": cfg["budget"]["prices_as_of"],
            "estimate": True,
        },
        "arms": {},
    }
    rng = random.Random(0)
    for model_key in models:
        mc = cfg["models"][model_key]
        if mc["backend"] != "anthropic":
            continue
        reqs = {
            (cond, p): [
                build_request(
                    mc, cfg["max_tokens"], registry[p], q, ctx[q["financebench_id"]]["chunks"]
                )[0]
                for q in questions
                if q["financebench_id"] in ctx
            ]
            for cond, ctx in contexts.items()
            for p in prompts
        }
        pool = [r for rs in reqs.values() for r in rs]
        sample = rng.sample(pool, min(20, len(pool)))
        counted = [
            client.messages.count_tokens(
                model=mc["model"], system=r.system, messages=[{"role": "user", "content": r.user}]
            ).input_tokens
            for r in sample
        ]
        tok_per_char = sum(counted) / sum(len(r.system) + len(r.user) for r in sample)
        p = prices[mc["model"]]
        for (cond, pid), rs in reqs.items():
            todo = [r for r in rs if cache.get(r.cache_key) is None]
            in_tok = sum((len(r.system) + len(r.user)) * tok_per_char for r in todo)
            # output length: this condition's pilot if run, else the retrieved-condition pilot
            pilot_path = next(
                (
                    p
                    for p in (
                        TRACES_DIR / "pilot" / cond / f"{model_key}__{pid}.jsonl",
                        TRACES_DIR / "pilot" / "retrieved" / f"{model_key}__{pid}.jsonl",
                    )
                    if p.exists()
                ),
                None,
            )
            if pilot_path is not None:
                outs = [t["generation"]["output_tokens"] for t in read_traces(pilot_path)]
                out_mean = statistics.fmean(outs)
                out_source = f"pilot mean over {len(outs)} ({pilot_path.parent.name})"
            else:
                out_mean, out_source = ASSUMED_OUTPUT_TOKENS[pid], "assumed"
            expected = (in_tok * p["input"] + len(todo) * out_mean * p["output"]) / 1e6
            worst = (in_tok * p["input"] + len(todo) * cfg["max_tokens"] * p["output"]) / 1e6
            estimate["arms"][f"{cond}__{model_key}__{pid}"] = {
                "calls_total": len(rs),
                "calls_uncached": len(todo),
                "input_tokens_est": round(in_tok),
                "output_tokens_mean_used": round(out_mean, 1),
                "output_tokens_source": out_source,
                "expected_usd": round(expected, 4),
                "worst_case_usd": round(worst, 4),
            }
        estimate["meta"][f"{model_key}_tokens_per_char_measured"] = round(tok_per_char, 4)
    arms = estimate["arms"].values()
    estimate["total_expected_usd"] = round(sum(a["expected_usd"] for a in arms), 4)
    estimate["total_worst_case_usd"] = round(sum(a["worst_case_usd"] for a in arms), 4)
    ledger_path = Path(cfg["budget"]["ledger"])
    spent = json.loads(ledger_path.read_text())["total_usd"] if ledger_path.exists() else 0.0
    estimate["already_spent_usd"] = round(spent, 4)
    estimate["caps"] = {
        "project_usd": cfg["budget"]["project_cap_usd"],
        "phase_usd": cfg["budget"]["phase_cap_usd"],
    }
    out = RESULTS_DIR / "generation_cost_estimate.json"
    out.write_text(json.dumps(estimate, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in estimate.items() if k != "arms"}, indent=2))
    for name, a in estimate["arms"].items():
        print(
            f"  {name:45s} uncached {a['calls_uncached']:4d}  expected ${a['expected_usd']:.3f}"
            f"  worst ${a['worst_case_usd']:.3f}  (output: {a['output_tokens_source']})"
        )
    print(f"wrote {out}")


def verify_replay(registry) -> None:
    cache = ResponseCache()
    report = {
        "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {},
        "failures": [],
    }
    total = ok = from_cache = 0
    for path in sorted(TRACES_DIR.rglob("*.jsonl")):
        counts = Counter()
        for trace in read_traces(path):
            r = replay_trace(trace, registry, cache)
            counts["n"] += 1
            counts["ok"] += r.ok
            counts[f"output_source_{r.output_source}"] += 1
            if not r.ok:
                report["failures"].append(
                    {"file": path.as_posix(), "trace_id": r.trace_id, "problems": r.problems}
                )
        report["files"][path.relative_to(TRACES_DIR).as_posix()] = dict(counts)
        total += counts["n"]
        ok += counts["ok"]
        from_cache += counts["output_source_cache"]
    report.update({"n_traces": total, "n_ok": ok, "n_output_checked_against_cache": from_cache})
    out = RESULTS_DIR / "generation_replay_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"replayed {total} traces: {ok} ok, {total - ok} failed; "
        f"{from_cache} outputs checked against the response cache"
    )
    print(f"wrote {out}")
    if ok != total:
        sys.exit(1)


# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        metavar="N",
        help="fixed stratified sample of N questions; separate output files",
    )
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--prompts", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true", help="estimate API cost, call nothing")
    parser.add_argument("--verify-replay", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    registry = load_registry()
    if args.verify_replay:
        verify_replay(registry)
        return

    conditions = args.conditions or cfg["conditions"]
    models = args.models or list(cfg["models"])
    prompts = args.prompts or cfg["prompts"]
    for c in conditions:
        if c not in cfg["conditions"]:
            parser.error(f"unknown condition {c!r}; configured: {cfg['conditions']}")
    for m in models:
        if m not in cfg["models"]:
            parser.error(f"unknown model {m!r}; configured: {list(cfg['models'])}")
    for p in prompts:
        if p not in registry:
            parser.error(f"unknown prompt {p!r}; registry: {list(registry)}")

    questions = load_questions()
    if cfg["questions"] != "all":
        raise ValueError("only `questions: all` is supported")
    pilot = args.pilot is not None
    if pilot:
        questions = pilot_sample(questions, args.pilot, cfg["pilot_seed"])
    print(f"{len(questions)} questions ({sum(q['in_corpus'] for q in questions)} in corpus)")

    contexts, context_check = build_contexts(cfg, questions, conditions)
    print(f"contexts: {context_check}")

    cache = ResponseCache()
    if args.dry_run:
        dry_run(cfg, registry, models, prompts, questions, contexts, cache)
        return

    b = cfg["budget"]
    ledger = SpendLedger(
        Path(b["ledger"]), b["prices_usd_per_mtok"], b["project_cap_usd"], PHASE, b["phase_cap_usd"]
    )
    summary_path = RESULTS_DIR / ("generation_runs_pilot.json" if pilot else "generation_runs.json")
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {"arms": {}}
    )

    for condition in conditions:
        ctx = contexts[condition]
        cond_questions = [q for q in questions if q["financebench_id"] in ctx]
        trace_dir = (TRACES_DIR / "pilot" if pilot else TRACES_DIR) / condition
        for model_key in models:
            mc = cfg["models"][model_key]
            backend = make_backend(mc)
            use_ledger = ledger if mc["backend"] == "anthropic" else None
            for pid in prompts:
                template = registry[pid]
                print(
                    f"\n=== {condition} x {model_key} x {pid} (v{template.version}), "
                    f"{len(cond_questions)} questions ==="
                )
                try:
                    traces, n_cached, new_cost = run_arm(
                        model_key,
                        mc,
                        template,
                        cond_questions,
                        ctx,
                        backend,
                        cache,
                        use_ledger,
                        cfg["max_tokens"],
                    )
                except BudgetExceeded as e:
                    print(f"STOPPED: {e}\nCompleted calls are cached; nothing is lost.")
                    sys.exit(2)
                trace_path = trace_dir / f"{model_key}__{pid}.jsonl"
                write_traces(trace_path, traces)
                arm = summarise_arm(traces, n_cached, new_cost)
                arm.update(
                    {
                        "condition": condition,
                        "prompt_version": template.version,
                        "prompt_file_sha256": template.file_sha256,
                        "trace_file": trace_path.as_posix(),
                    }
                )
                if mc["backend"] == "ollama":
                    arm["model_digest"] = backend.digest(mc["model"])
                summary["arms"][f"{condition}__{model_key}__{pid}"] = arm
                print(
                    f"    parse {arm['parse_status']}  "
                    f"abstain-token {arm['abstained_token_rate']:.2f}"
                    f"  out-tok mean {arm['output_tokens_mean']:.0f}"
                    f"  p50 {arm['latency_ms']['p50']:.0f} ms"
                )
                if not args.no_mlflow:
                    log_mlflow(
                        cfg,
                        condition,
                        model_key,
                        mc,
                        template,
                        arm,
                        trace_path,
                        pilot,
                        len(cond_questions),
                    )

    summary["meta"] = {
        "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "single_run": True,
        "pilot": pilot,
        "config": cfg,
        "n_questions": len(questions),
        "n_in_corpus": sum(q["in_corpus"] for q in questions),
        "n_questions_per_condition": {c: len(contexts[c]) for c in conditions},
        "context_check": context_check,
        "environment": environment_meta(cfg, models),
        "api_spend_to_date_usd": round(ledger.state["total_usd"], 6),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {summary_path}; API spend to date ${ledger.state['total_usd']:.4f}")


if __name__ == "__main__":
    main()
