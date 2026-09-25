"""Phase 6, Q4: generate and evaluate the adversarial question set with the Phase 4 and 5
pipeline (the retriever, generators, prompts, judge and metrics unchanged).

  contexts           build the contexts: `retrieved` (the Phase 3 winner over the whole
                     corpus) for every approved question, `oracle` for (a) and (b); gold
                     evidence aligned exactly as FinanceBench's; for a two-filing question the
                     oracle ranks both filings' chunks together
                     -> results/metrics/adversarial_contexts.json
  generate           both generators, all four prompts
                     -> results/traces/adversarial/<condition>/<model>__<prompt>.jsonl
                        results/metrics/generation_runs_adversarial.json
  estimate           cost of the uncached generation calls (input tokens counted with the
                     free token-counting endpoint, output lengths from the Phase 4 traces of
                     the same condition and prompt), and an upper bound for the judge
                     -> results/metrics/adversarial_cost_estimate.json
  premise-pilot      the premise judge on a fixed sample of (d) answers, synchronously, for
                     review before the full run -> results/metrics/premise_judge_pilot.md
  evaluate           the Phase 5 judge through Message Batches (--offline: response cache only),
                     plus the premise judge on every (d) answer
                     -> results/metrics/eval_adversarial_per_trace.jsonl
  label              the author's blind premise-labelling page for every (d) answer
                     -> results/labels/premise_sample.json, premise_labeling.html (not committed)

Traces keep the pipeline's question key, `financebench_id`; for these questions it holds the
adversarial id (adv_a01, ...). Every API call goes through the spend ledger under the Phase 6
cap in configs/adversarial.yaml, and every response is cached, so re-runs cost nothing.

Usage:
    uv run python scripts/07b_adversarial_run.py contexts
    uv run python scripts/07b_adversarial_run.py estimate
    uv run python scripts/07b_adversarial_run.py generate
    uv run python scripts/07b_adversarial_run.py premise-pilot
    uv run python scripts/07b_adversarial_run.py evaluate [--offline]
    uv run python scripts/07b_adversarial_run.py label
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.abstention import is_bare_token, needs_judge
from src.evaluation.evaluate import Inputs, Judge, check_evidence, evaluate_trace, judge_traces
from src.evaluation.gold_spans import GoldSpan, NormalisedText, align_evidence
from src.evaluation.judge import PREMISE_SCHEMA, parse_premise
from src.evaluation.labeling_page import render_premise_page
from src.generation.llm import BudgetExceeded, ResponseCache, SpendLedger
from src.generation.prompts import load_registry
from src.generation.trace import TRACES_DIR, read_traces, trace_chunks, write_traces

CONFIG_PATH = Path("configs/adversarial.yaml")
PARSED_DIR = Path("data/processed/parsed")
CHUNKS_DIR = Path("data/processed/chunks")
INDICES_DIR = Path("data/indices")
RESULTS_DIR = Path("results/metrics")
LABELS_DIR = Path("results/labels")
ADV_TRACES = TRACES_DIR / "adversarial"
CONTEXTS_PATH = RESULTS_DIR / "adversarial_contexts.json"
RUNS_PATH = RESULTS_DIR / "generation_runs_adversarial.json"
ESTIMATE_PATH = RESULTS_DIR / "adversarial_cost_estimate.json"
PILOT_PATH = RESULTS_DIR / "premise_judge_pilot.md"
# The pilot answers were shown with the judge's verdicts before the author labelled them, so
# their labels are not blind to the judge (excluded from the agreement statistic).
PILOT_IDS_PATH = RESULTS_DIR / "premise_judge_pilot.json"
EVAL_PATH = RESULTS_DIR / "eval_adversarial_per_trace.jsonl"
PREMISE_SAMPLE_PATH = LABELS_DIR / "premise_sample.json"
PREMISE_PAGE_PATH = LABELS_DIR / "premise_labeling.html"
PHASE5_PER_TRACE = RESULTS_DIR / "eval_per_trace.jsonl"


def load_configs() -> tuple[dict, dict, dict]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gen = yaml.safe_load(Path(cfg["generation_config"]).read_text(encoding="utf-8"))
    ev = yaml.safe_load(Path(cfg["evaluation_config"]).read_text(encoding="utf-8"))
    return cfg, gen, ev


def load_generate_module():
    """Phase 4's generation script, for its arm runner and helpers (not duplicated here)."""
    path = Path(__file__).resolve().parent / "05_generate.py"
    spec = importlib.util.spec_from_file_location("generate_phase4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_ledger(cfg: dict, ev: dict) -> SpendLedger:
    b = ev["budget"]
    return SpendLedger(
        Path(b["ledger"]),
        b["prices_usd_per_mtok"],
        b["project_cap_usd"],
        cfg["budget"]["phase"],
        cfg["budget"]["phase_cap_usd"],
        b["batch_discount"],
    )


# --------------------------------------------------------------------------------------
# Questions and gold evidence


def approved_records(cfg: dict) -> dict[str, dict]:
    lines = Path(cfg["questions"]).read_text(encoding="utf-8").splitlines()
    recs = [json.loads(x) for x in lines]
    return {r["id"]: r for r in recs if r["status"] == "approved"}


def trace_question(rec: dict) -> dict:
    """An approved record in the question format the Phase 4 traces use."""
    docs = [e["doc_id"] for e in rec["gold_evidence"] or rec["reference_evidence"]]
    return {
        "financebench_id": rec["id"],
        "question": rec["question"],
        "question_type": f"adversarial_{rec['category']}",
        "question_reasoning": None,
        "company": None,
        "doc_name": docs[0] if docs else None,
        "doc_period": None,
        "doc_type": None,
        "in_corpus": rec["subtype"] != "filing_not_in_corpus",
        "adversarial": {
            "category": rec["category"],
            "subtype": rec["subtype"],
            "expected_behaviour": rec["expected_behaviour"],
        },
    }


def gold_spans(recs: dict[str, dict]) -> dict[str, list[GoldSpan]]:
    """Each gold passage aligned in its filing; it must occur there exactly once (checked
    again here, so a changed corpus cannot silently move the evidence)."""
    docs: dict[str, NormalisedText] = {}
    out: dict[str, list[GoldSpan]] = {}
    for qid, r in sorted(recs.items()):
        spans = []
        for ev in r["gold_evidence"]:
            d = ev["doc_id"]
            if d not in docs:
                text = json.loads((PARSED_DIR / f"{d}.json").read_text(encoding="utf-8"))
                docs[d] = NormalisedText(text["full_text"])
            al = align_evidence(ev["text"], d, docs[d])
            if al.status != "exact" or al.n_occurrences != 1:
                raise RuntimeError(f"{qid}: evidence in {d} is {al.status} x{al.n_occurrences}")
            spans.append(al.span)
        if spans:
            out[qid] = spans
    return out


# --------------------------------------------------------------------------------------
# Contexts


def build_contexts(cfg: dict, gen: dict, recs: dict[str, dict]) -> tuple[dict, dict]:
    from src.evaluation.retrieval_metrics import count_relevant, question_metrics
    from src.retrieval.embeddings import EmbeddingModel
    from src.retrieval.oracle import build_oracle_context
    from src.retrieval.scoped import ScopedFaissIndex

    g5 = load_generate_module()
    p3 = g5.load_phase3_module()
    rc = gen["retrieval"]
    if rc["method"] != "dense" or rc["backend"] != "faiss":
        raise NotImplementedError("generation is wired for the dense FAISS winner only")
    cell = f"{rc['chunking']}__{rc['embedding']}__{rc['method']}"
    k, depth = rc["k"], rc["search_depth"]
    cfg3 = yaml.safe_load(g5.PHASE3_CONFIG.read_text(encoding="utf-8"))
    min_ov = cfg3["relevance"]["min_overlap_frac"]
    golds = gold_spans(recs)

    by_doc: dict[str, list] = {}
    for line in (CHUNKS_DIR / f"{rc['chunking']}.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        by_doc.setdefault(c["doc_id"], []).append(p3.chunk_span(c))
    embed = EmbeddingModel(rc["embedding"])
    index = ScopedFaissIndex(INDICES_DIR / f"{rc['chunking']}__{rc['embedding']}__faiss")

    def metrics_for(results, gs) -> dict | None:
        if not gs:
            return None
        spans = [p3.chunk_span(r.metadata) for r in results]
        return question_metrics(spans, gs, count_relevant(by_doc, gs, min_ov), [1, 3, 5], min_ov)

    contexts: dict[str, dict[str, dict]] = {c: {} for c in cfg["conditions"]}
    for qid, rec in sorted(recs.items()):
        cat, gs = rec["category"], golds.get(qid, [])
        qv = embed.encode_queries([rec["question"]])
        if cat in cfg["conditions"].get("retrieved", []):
            results = index.search_corpus(qv, depth)[:k]
            contexts["retrieved"][qid] = {
                "condition": "retrieved",
                "config": cell,
                "k": k,
                "phase3_top_k_match": None,
                "context_metrics": metrics_for(results, gs),
                "chunks": trace_chunks(results),
            }
        if cat in cfg["conditions"].get("oracle", []):
            if not gs:
                raise RuntimeError(f"{qid}: an oracle context needs gold evidence")
            ranked = sorted(
                (r for d in sorted({g.doc_id for g in gs}) for r in index.search_filing(qv, d)),
                key=lambda r: -r.score,
            )
            chosen = build_oracle_context(
                ranked, gs, k, to_span=lambda r: p3.chunk_span(r.metadata), min_overlap_frac=min_ov
            )
            contexts["oracle"][qid] = {
                "condition": "oracle",
                "config": f"oracle__{cell}",
                "k": k,
                "context_metrics": metrics_for(chosen, gs),
                "chunks": trace_chunks(chosen),
            }

    report: dict = {"cell": cell, "k": k, "search_depth": depth, "conditions": {}}
    for cond, ctx in contexts.items():
        by_cat: dict[str, dict] = {}
        for qid, c in ctx.items():
            cat = recs[qid]["category"]
            e = by_cat.setdefault(cat, {"n": 0, "with_gold": 0, "recall5": []})
            e["n"] += 1
            if c["context_metrics"] is not None:
                e["with_gold"] += 1
                e["recall5"].append(c["context_metrics"]["recall@5"])
        report["conditions"][cond] = {
            cat: {
                "n_questions": e["n"],
                "n_with_gold": e["with_gold"],
                "mean_recall@5": statistics.fmean(e["recall5"]) if e["recall5"] else None,
                "n_recall@5_gt_0": sum(x > 0 for x in e["recall5"]),
                "n_recall@5_eq_1": sum(x == 1 for x in e["recall5"]),
            }
            for cat, e in sorted(by_cat.items())
        }
    per_question = {
        cond: {
            qid: {
                "recall@5": (c["context_metrics"] or {}).get("recall@5"),
                "docs": sorted({ch["doc_id"] for ch in c["chunks"]}),
            }
            for qid, c in sorted(ctx.items())
        }
        for cond, ctx in contexts.items()
    }
    report["per_question"] = per_question
    return contexts, report


def cmd_contexts(cfg: dict, gen: dict) -> dict:
    recs = approved_records(cfg)
    contexts, report = build_contexts(cfg, gen, recs)
    report["meta"] = {
        "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_approved": len(recs),
        "questions_sha256": hashlib.sha256(Path(cfg["questions"]).read_bytes()).hexdigest(),
    }
    CONTEXTS_PATH.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["conditions"], indent=1))
    print(f"wrote {CONTEXTS_PATH}")
    return contexts


# --------------------------------------------------------------------------------------
# Generation, and its cost estimate


def arms(cfg: dict, gen: dict, contexts: dict, recs: dict) -> list[tuple]:
    out = []
    for condition, ctx in contexts.items():
        questions = [trace_question(recs[qid]) for qid in sorted(ctx)]
        for model_key in gen["models"]:
            for pid in gen["prompts"]:
                out.append((condition, model_key, pid, questions, ctx))
    return out


def cmd_estimate(cfg: dict, gen: dict, ev: dict) -> None:
    """Generation: input tokens counted on a sample of the real requests with the free
    token-counting endpoint and scaled by characters; output tokens from the Phase 4 traces
    of the same condition and prompt. Judge: an upper bound, every trace judged at the
    Phase 5 judge spend per judged trace (all Phase 5 spend, pilots included, over the
    traces the full run judged)."""
    import anthropic

    g5 = load_generate_module()
    recs = approved_records(cfg)
    contexts, _ = build_contexts(cfg, gen, recs)
    registry = load_registry()
    cache = ResponseCache()
    client = anthropic.Anthropic()
    prices = ev["budget"]["prices_usd_per_mtok"]
    est: dict = {"arms": {}}
    reqs_by_arm = {}
    for condition, model_key, pid, questions, ctx in arms(cfg, gen, contexts, recs):
        mc = gen["models"][model_key]
        if mc["backend"] != "anthropic":
            continue
        reqs = [
            g5.build_request(
                mc, gen["max_tokens"], registry[pid], q, ctx[q["financebench_id"]]["chunks"]
            )[0]
            for q in questions
        ]
        reqs_by_arm[(condition, model_key, pid)] = reqs
    pool = [r for rs in reqs_by_arm.values() for r in rs]
    sample = random.Random(0).sample(pool, min(20, len(pool)))
    counted = [
        client.messages.count_tokens(
            model=r.model, system=r.system, messages=[{"role": "user", "content": r.user}]
        ).input_tokens
        for r in sample
    ]
    tok_per_char = sum(counted) / sum(len(r.system) + len(r.user) for r in sample)
    total_expected = total_worst = 0.0
    for (condition, model_key, pid), reqs in reqs_by_arm.items():
        p = prices[gen["models"][model_key]["model"]]
        todo = [r for r in reqs if cache.get(r.cache_key) is None]
        in_tok = sum((len(r.system) + len(r.user)) * tok_per_char for r in todo)
        ref = read_traces(TRACES_DIR / condition / f"{model_key}__{pid}.jsonl")
        out_mean = statistics.fmean(t["generation"]["output_tokens"] or 0 for t in ref)
        expected = (in_tok * p["input"] + len(todo) * out_mean * p["output"]) / 1e6
        worst = (in_tok * p["input"] + len(todo) * gen["max_tokens"] * p["output"]) / 1e6
        total_expected += expected
        total_worst += worst
        est["arms"][f"{condition}__{model_key}__{pid}"] = {
            "calls": len(reqs),
            "calls_uncached": len(todo),
            "input_tokens_est": round(in_tok),
            "output_tokens_mean_phase4": round(out_mean, 1),
            "expected_usd": round(expected, 4),
            "worst_case_usd": round(worst, 4),
        }
    n_traces = sum(len(q) for _, _, _, q, _ in arms(cfg, gen, contexts, recs))
    spend = json.loads(Path(ev["budget"]["ledger"]).read_text(encoding="utf-8"))
    p5 = [json.loads(x) for x in PHASE5_PER_TRACE.read_text(encoding="utf-8").splitlines()]
    n_judged_p5 = sum(1 for r in p5 if r["judge"].get("decompose"))
    per_trace = spend["by_phase"]["phase5_evaluation"]["usd"] / n_judged_p5
    est.update(
        {
            "meta": {
                "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "estimate": True,
                "tokens_per_char_measured": round(tok_per_char, 4),
                "n_traces": n_traces,
            },
            "generation_expected_usd": round(total_expected, 4),
            "generation_worst_case_usd": round(total_worst, 4),
            "judge_upper_bound_usd": round(n_traces * per_trace, 4),
            "judge_upper_bound_basis": (
                f"every trace judged at ${per_trace:.5f}: Phase 5 spend "
                f"${spend['by_phase']['phase5_evaluation']['usd']:.2f} over {n_judged_p5} "
                "traces judged (pilots and validation included, so an overestimate)"
            ),
            "already_spent_usd": round(spend["total_usd"], 4),
            "caps": {
                "project_usd": ev["budget"]["project_cap_usd"],
                "phase6_usd": cfg["budget"]["phase_cap_usd"],
            },
        }
    )
    est["total_expected_upper_usd"] = round(total_expected + est["judge_upper_bound_usd"], 4)
    ESTIMATE_PATH.write_text(json.dumps(est, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in est.items() if k != "arms"}, indent=1))
    print(f"wrote {ESTIMATE_PATH}")


def cmd_generate(cfg: dict, gen: dict, ev: dict) -> None:
    g5 = load_generate_module()
    recs = approved_records(cfg)
    contexts = cmd_contexts(cfg, gen)
    registry = load_registry()
    cache = ResponseCache()
    ledger = make_ledger(cfg, ev)
    summary: dict = {"arms": {}}
    for condition, model_key, pid, questions, ctx in arms(cfg, gen, contexts, recs):
        mc = gen["models"][model_key]
        backend = g5.make_backend(mc)
        template = registry[pid]
        print(f"\n=== {condition} x {model_key} x {pid}, {len(questions)} questions ===")
        try:
            traces, n_cached, new_cost = g5.run_arm(
                model_key,
                mc,
                template,
                questions,
                ctx,
                backend,
                cache,
                ledger if mc["backend"] == "anthropic" else None,
                gen["max_tokens"],
            )
        except BudgetExceeded as e:
            print(f"STOPPED: {e}\nCompleted calls are cached; nothing is lost.")
            sys.exit(2)
        path = ADV_TRACES / condition / f"{model_key}__{pid}.jsonl"
        write_traces(path, traces)
        arm = g5.summarise_arm(traces, n_cached, new_cost)
        arm.update(
            {
                "condition": condition,
                "prompt_version": template.version,
                "prompt_file_sha256": template.file_sha256,
                "trace_file": path.as_posix(),
            }
        )
        if mc["backend"] == "ollama":
            arm["model_digest"] = g5.arm_model_digest(backend, mc["model"], traces)
        summary["arms"][f"{condition}__{model_key}__{pid}"] = arm
        print(f"    parse {arm['parse_status']}  abstain-token {arm['abstained_token_rate']:.2f}")
    summary["meta"] = {
        "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "single_run": True,
        "config": cfg,
        "generation_config": gen,
        "environment": g5.environment_meta(gen, list(gen["models"])),
        "api_spend_to_date_usd": round(ledger.state["total_usd"], 6),
    }
    RUNS_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {RUNS_PATH}; API spend to date ${ledger.state['total_usd']:.4f}")


# --------------------------------------------------------------------------------------
# Evaluation


def adversarial_traces() -> list[dict]:
    return [t for p in sorted(ADV_TRACES.rglob("*.jsonl")) for t in read_traces(p)]


def judge_config(cfg: dict, ev: dict) -> dict:
    """The Phase 5 judge configuration plus the premise prompt."""
    j = json.loads(json.dumps(ev))
    j["judge"]["prompts"]["premise"] = cfg["premise_judge"]["prompt"]
    j["judge"]["max_tokens"]["premise"] = cfg["premise_judge"]["max_tokens"]
    return j


def premise_note(rec: dict) -> str:
    return rec["answer"]


def premise_requests(traces: list[dict], recs: dict, judge: Judge) -> dict:
    return {
        t["trace_id"]: judge.request(
            "premise",
            {
                "question": t["question"]["question"],
                "premise_note": premise_note(recs[t["question"]["financebench_id"]]),
                "answer": t["parsed"]["answer"],
            },
            PREMISE_SCHEMA,
        )
        for t in traces
        if t["question"]["adversarial"]["category"] == "d" and needs_judge(t["parsed"])
    }


def premise_rule(parsed: dict) -> str | None:
    """Handling decided without the judge: no answer at all, or a bare decline token."""
    answer = parsed.get("answer")
    if answer is None:
        return "unparsed"
    if is_bare_token(answer):
        return "declines_without_addressing"
    return None


def cmd_premise_pilot(cfg: dict, ev: dict) -> None:
    from src.evaluation.evaluate import judge_outcome

    load_dotenv()
    recs = approved_records(cfg)
    traces = [
        t
        for t in adversarial_traces()
        if t["retrieval"]["condition"] == "retrieved"
        and t["question"]["adversarial"]["category"] == "d"
        and needs_judge(t["parsed"])
    ]
    pc = cfg["premise_judge"]
    sample = random.Random(pc["pilot_seed"]).sample(
        sorted(traces, key=lambda t: t["trace_id"]), min(pc["pilot_n"], len(traces))
    )
    judge = Judge(
        judge_config(cfg, ev), ev["judge"]["model"], "sync", ResponseCache(), make_ledger(cfg, ev)
    )
    reqs = premise_requests(sample, recs, judge)
    responses = judge.run(list(reqs.values()))
    out = [
        "# Premise judge pilot",
        "",
        f"Single run. {len(sample)} (d) answers drawn at random (seed {pc['pilot_seed']}), "
        f"judged synchronously with `{pc['prompt']}` for review before the full run. "
        f"Cost ${judge.new_cost_usd:.4f}.",
    ]
    for t in sample:
        o = judge_outcome(reqs[t["trace_id"]], responses, parse_premise)
        res = o.get("result") or {}
        out += [
            "",
            f"## {t['trace_id']}",
            "",
            f"- **Question:** {t['question']['question']}",
            f"- **Premise note:** {premise_note(recs[t['question']['financebench_id']])}",
            f"- **Answer:** {t['parsed']['answer']}",
            f"- **Judge:** {res.get('handling', o.get('error'))} — {res.get('reason', '')}",
        ]
    PILOT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    PILOT_IDS_PATH.write_text(
        json.dumps({"trace_ids": sorted(t["trace_id"] for t in sample)}, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {PILOT_PATH}, {PILOT_IDS_PATH}")


def cmd_evaluate(cfg: dict, gen: dict, ev: dict, offline: bool) -> None:
    from src.evaluation.evaluate import judge_outcome

    load_dotenv()
    recs = approved_records(cfg)
    traces = adversarial_traces()
    golds = gold_spans(recs)
    g5 = load_generate_module()
    cfg3 = yaml.safe_load(g5.PHASE3_CONFIG.read_text(encoding="utf-8"))
    gold_rows = {
        qid: {"answer": r["answer"], "justification": ""}
        for qid, r in recs.items()
        if r["category"] in ("a", "b")
    }
    inputs = Inputs(traces, gold_rows, {}, golds, cfg3["relevance"]["min_overlap_frac"])
    check_evidence(inputs)
    mode = "offline" if offline else "batch"
    judge = Judge(
        judge_config(cfg, ev),
        ev["judge"]["model"],
        mode,
        ResponseCache(),
        None if offline else make_ledger(cfg, ev),
    )
    judged = judge_traces(
        traces,
        inputs,
        judge,
        grade_if=lambda t: t["question"]["adversarial"]["category"] in ("a", "b"),
    )
    preq = premise_requests(traces, recs, judge)
    judge.log(f"premise judge: {len(preq)} answers")
    presp = judge.run(list(preq.values()))
    c = ev["correctness"]
    records = []
    for t in traces:
        rec = evaluate_trace(
            t, inputs, judged[t["trace_id"]], c["rel_tol"], tuple(c["sensitivity_rel_tols"])
        )
        adv = t["question"]["adversarial"]
        rec["adversarial"] = adv
        if adv["category"] == "d":
            rule = premise_rule(t["parsed"])
            if rule is not None:
                rec["premise"] = {"handling": rule, "source": "rule"}
            else:
                o = judge_outcome(preq[t["trace_id"]], presp, parse_premise)
                rec["premise"] = {
                    "handling": (o.get("result") or {}).get("handling"),
                    "reason": (o.get("result") or {}).get("reason"),
                    "source": "judge",
                    "cache_key": o["cache_key"],
                    **({"error": o["error"]} if "error" in o else {}),
                }
        records.append(rec)
    EVAL_PATH.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
        encoding="utf-8",
        newline="\n",
    )
    errors = sum(bool(r["judge_errors"]) for r in records) + sum(
        "error" in (r.get("premise") or {}) for r in records
    )
    print(
        f"wrote {EVAL_PATH}: {len(records)} traces, {errors} with judge errors; "
        f"new judge spend ${judge.new_cost_usd:.4f}"
    )


# --------------------------------------------------------------------------------------
# The author's blind premise labels


def cmd_label(cfg: dict) -> None:
    recs = approved_records(cfg)
    traces = [
        t
        for t in adversarial_traces()
        if t["question"]["adversarial"]["category"] == "d" and needs_judge(t["parsed"])
    ]
    items = [
        {
            "label_id": hashlib.sha256(("premise:" + t["trace_id"]).encode()).hexdigest()[:12],
            "trace_id": t["trace_id"],
        }
        for t in sorted(traces, key=lambda t: t["trace_id"])
    ]
    random.Random(cfg["label_seed"]).shuffle(items)
    sha = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
    if PREMISE_SAMPLE_PATH.exists():
        old = json.loads(PREMISE_SAMPLE_PATH.read_text(encoding="utf-8"))
        if old["items"] != items:
            raise RuntimeError(f"{PREMISE_SAMPLE_PATH} does not match the current traces")
    else:
        PREMISE_SAMPLE_PATH.write_text(
            json.dumps(
                {
                    "meta": {
                        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                        "rule": "every (d) answer with answer text; bare decline tokens and "
                        "failed parses are labelled by rule",
                        "seed": cfg["label_seed"],
                    },
                    "sample_sha256": sha,
                    "items": items,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    by_id = {t["trace_id"]: t for t in traces}
    page = [
        {
            "label_id": it["label_id"],
            "question": by_id[it["trace_id"]]["question"]["question"],
            "premise_note": premise_note(
                recs[by_id[it["trace_id"]]["question"]["financebench_id"]]
            ),
            "answer": by_id[it["trace_id"]]["parsed"]["answer"],
        }
        for it in items
    ]
    PREMISE_PAGE_PATH.write_text(render_premise_page(page, sha), encoding="utf-8", newline="\n")
    print(f"{len(items)} answers -> {PREMISE_PAGE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["contexts", "estimate", "generate", "premise-pilot", "evaluate", "label"],
    )
    parser.add_argument("--offline", action="store_true", help="evaluate: response cache only")
    args = parser.parse_args()
    load_dotenv()
    cfg, gen, ev = load_configs()
    if args.command == "contexts":
        cmd_contexts(cfg, gen)
    elif args.command == "estimate":
        cmd_estimate(cfg, gen, ev)
    elif args.command == "generate":
        cmd_generate(cfg, gen, ev)
    elif args.command == "premise-pilot":
        cmd_premise_pilot(cfg, ev)
    elif args.command == "evaluate":
        cmd_evaluate(cfg, gen, ev, args.offline)
    else:
        cmd_label(cfg)


if __name__ == "__main__":
    main()
