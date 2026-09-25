"""Evaluation harness: the four metric families for every Phase 4 trace, and per-arm
aggregates. Shared by scripts/06_eval_generation.py (pilot, full run) and
scripts/06b_judge_validation.py (the hand-labelled validation sample).

Per trace (one record in results/metrics/eval_per_trace.jsonl):
  abstention    final status (answered | partial | declined | unparsed) and its source, the
                rule detector's status, and the question's answerability in its condition
  correctness   label (correct | partially_correct | incorrect | unit_error | no_answer |
                abstained | unparsed) and which grader gave it (numeric | judge)
  groundedness  claims with verdicts, and the scores in src/evaluation/groundedness.py
  citations     judge-based precision/recall plus the judge-free checks
Judge calls run in two stages, because stage 2 depends on stage 1:
  1. decompose every trace with answer text
  2. verify the traces with >=1 claim; grade correctness for the traces not declined
A judge call that returns nothing, is cut off at max_tokens, is refused, or fails schema
validation is recorded as an error for that trace and that metric; it is never guessed.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.evaluation import groundedness
from src.evaluation.abstention import answerability, final_status, needs_judge, rule_status
from src.evaluation.agreement import agreement_summary, bootstrap_mean_ci
from src.evaluation.citations import (
    evidence_labels,
    judge_citation_metrics,
    numeric_support,
    quote_fidelity,
)
from src.evaluation.correctness import NumericGold, grade_numeric, numeric_gold
from src.evaluation.gold_spans import GoldSpan
from src.evaluation.judge import (
    CORRECTNESS_SCHEMA,
    DECOMPOSE_SCHEMA,
    JudgeOutputError,
    JudgePrompt,
    build_request,
    format_claims,
    load_judge_registry,
    parse_correctness,
    parse_decomposition,
    parse_verification,
    verify_schema,
)
from src.generation.batch import BatchOutcome, run_batch_cached
from src.generation.llm import (
    AnthropicBackend,
    GenerationRequest,
    GenerationResponse,
    ResponseCache,
    SpendLedger,
    generate_cached,
)
from src.generation.prompts import format_context
from src.generation.trace import read_traces

PHASE = "phase5_evaluation"
FINANCEBENCH_PATH = Path("data/raw/financebench/financebench_merged.jsonl")
GOLD_ALIGNMENT_PATH = Path("results/metrics/gold_span_alignment.json")
RETRIEVAL_CONFIG_PATH = Path("configs/retrieval_grid.yaml")
# Phase 6 audit of the out-of-corpus questions; applied only once its status is "approved".
OUT_OF_CORPUS_AUDIT_PATH = Path("results/labels/out_of_corpus_audit.json")
CORRECTNESS_LABELS = (
    "correct",
    "partially_correct",
    "unit_error",
    "incorrect",
    "no_answer",
    "abstained",
    "unparsed",
)


# --------------------------------------------------------------------------------------
# Inputs


@dataclass
class Inputs:
    traces: list[dict]
    gold_rows: dict[str, dict]
    numeric_golds: dict[str, NumericGold]
    gold_spans: dict[str, list[GoldSpan]]
    min_overlap_frac: float
    evidence: dict[str, list[str]] = field(default_factory=dict)  # trace_id -> labels
    # out-of-corpus questions a filing in the corpus answers (approved audit), else empty
    answerable_elsewhere: frozenset[str] = frozenset()


def trace_paths(cfg: dict) -> list[Path]:
    tc = cfg["traces"]
    return [
        Path(tc["dir"]) / cond / f"{model}__{prompt}.jsonl"
        for cond in tc["conditions"]
        for model in tc["models"]
        for prompt in tc["prompts"]
    ]


def load_gold_spans(path: Path = GOLD_ALIGNMENT_PATH) -> dict[str, list[GoldSpan]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    spans: dict[str, list[GoldSpan]] = {}
    for it in report["items"]:
        if it["char_start"] is not None:
            spans.setdefault(it["financebench_id"], []).append(
                GoldSpan(it["doc_id"], it["char_start"], it["char_end"])
            )
    return spans


def load_answerable_elsewhere(path: Path = OUT_OF_CORPUS_AUDIT_PATH) -> frozenset[str]:
    """Question ids whose source filing is not in the corpus but whose answer another
    filing in the corpus holds. Empty unless the audit exists and has been approved."""
    if not path.exists():
        return frozenset()
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit["meta"]["status"] != "approved":
        return frozenset()
    return frozenset(
        it["financebench_id"] for it in audit["items"] if it["status"] == "answerable_elsewhere"
    )


def load_inputs(cfg: dict) -> Inputs:
    traces = [t for p in trace_paths(cfg) for t in read_traces(p)]
    rows = [json.loads(x) for x in FINANCEBENCH_PATH.read_text(encoding="utf-8").splitlines()]
    gold_rows = {r["financebench_id"]: r for r in rows}
    numeric = {
        fid: g
        for fid, r in gold_rows.items()
        if (g := numeric_gold(r["answer"], r["question"])) is not None
    }
    min_ov = yaml.safe_load(RETRIEVAL_CONFIG_PATH.read_text(encoding="utf-8"))["relevance"][
        "min_overlap_frac"
    ]
    inputs = Inputs(
        traces, gold_rows, numeric, load_gold_spans(), min_ov, {}, load_answerable_elsewhere()
    )
    check_evidence(inputs)
    return inputs


def check_evidence(inputs: Inputs) -> None:
    """Mark each trace's evidence chunks, and check them against the context metrics Phase 4
    stored in the trace: recall@k > 0 exactly when some chunk holds gold evidence, and in the
    oracle condition every trace has evidence. A mismatch means the gold spans here are not
    the ones the traces were scored against, so it stops the run."""
    bad = []
    for t in inputs.traces:
        fid = t["question"]["financebench_id"]
        labels = evidence_labels(
            t["retrieval"]["chunks"], inputs.gold_spans.get(fid, []), inputs.min_overlap_frac
        )
        inputs.evidence[t["trace_id"]] = labels
        m = t["retrieval"]["context_metrics"]
        if m is not None:
            if (m[f"recall@{t['retrieval']['k']}"] > 0) != bool(labels):
                bad.append(t["trace_id"])
        elif labels:
            bad.append(t["trace_id"])
        if t["retrieval"]["condition"] == "oracle" and not labels:
            bad.append(t["trace_id"])
    if bad:
        raise RuntimeError(f"{len(bad)} traces disagree with their stored evidence; first {bad[0]}")


# --------------------------------------------------------------------------------------
# Judge calls


def make_ledger(cfg: dict) -> SpendLedger:
    b = cfg["budget"]
    return SpendLedger(
        Path(b["ledger"]),
        b["prices_usd_per_mtok"],
        b["project_cap_usd"],
        PHASE,
        b["phase_cap_usd"],
        b["batch_discount"],
    )


class Judge:
    """Sends judge requests synchronously (pilot, validation sample), through the Message
    Batches API (full run), or not at all (`offline`: results come from the response cache
    only, as when re-running the analysis). In every mode a request with no result is
    recorded as a judge error for its trace, so an offline re-run reproduces a run that had
    failures instead of diverging from it; the number missing is logged and returned in
    `n_missing`."""

    def __init__(
        self,
        cfg: dict,
        model_key: str,
        mode: str,
        cache: ResponseCache,
        ledger: SpendLedger | None,
        replicate: int | None = None,
        log: Callable[[str], None] = print,
    ):
        if mode not in ("sync", "batch", "offline"):
            raise ValueError(f"unknown judge mode {mode!r}")
        if mode != "offline" and ledger is None:
            raise ValueError("API calls require a SpendLedger")
        self.cfg = cfg
        self.model_key = model_key
        self.model_cfg = cfg["judge"]["models"][model_key]
        registry = load_judge_registry()
        self.prompts: dict[str, JudgePrompt] = {
            purpose: registry[pid] for purpose, pid in cfg["judge"]["prompts"].items()
        }
        self.max_tokens = cfg["judge"]["max_tokens"]
        self.mode = mode
        self.cache = cache
        self.ledger = ledger
        self.replicate = replicate
        self.log = log
        self.new_cost_usd = 0.0
        self.n_missing = 0
        self.batch_outcomes: list[BatchOutcome] = []

    def request(self, purpose: str, values: dict[str, str], schema: dict) -> GenerationRequest:
        return build_request(
            self.model_cfg,
            self.prompts[purpose],
            values,
            schema,
            self.max_tokens[purpose],
            self.replicate,
        )

    def run(self, requests: list[GenerationRequest]) -> dict[str, GenerationResponse]:
        unique = list({r.cache_key: r for r in requests}.values())
        if self.mode == "sync":
            backend = AnthropicBackend()
            workers = max(1, int(self.cfg["judge"]["sync_concurrency"]))

            def one(r: GenerationRequest) -> float:
                return generate_cached(backend, r, self.cache, self.ledger)[2]

            with ThreadPoolExecutor(max_workers=workers) as pool:
                self.new_cost_usd += sum(pool.map(one, unique))
        elif self.mode == "batch":
            b = self.cfg["batch"]
            outcome = run_batch_cached(
                unique,
                self.cache,
                self.ledger,
                AnthropicBackend().client,
                max_requests_per_batch=b["max_requests_per_batch"],
                poll_seconds=b["poll_seconds"],
                log=self.log,
            )
            self.batch_outcomes.append(outcome)
            self.new_cost_usd += outcome.new_cost_usd
        found = {r.cache_key: resp for r in unique if (resp := self.cache.get(r.cache_key))}
        missing = len(unique) - len(found)
        if missing:
            self.n_missing += missing
            self.log(f"WARNING: {missing} of {len(unique)} judge results missing (judge errors)")
        return found


def decompose_values(trace: dict) -> dict[str, str]:
    return {"question": trace["question"]["question"], "answer": trace["parsed"]["answer"]}


def verify_values(trace: dict, claims: list[dict]) -> dict[str, str]:
    return {
        "context": format_context(trace["retrieval"]["chunks"]),
        "claims": format_claims(claims),
    }


def correctness_values(trace: dict, gold_row: dict) -> dict[str, str]:
    return {
        "question": trace["question"]["question"],
        "gold_answer": gold_row["answer"].strip(),
        "gold_justification": (gold_row.get("justification") or "").strip() or "(none given)",
        "answer": trace["parsed"]["answer"],
    }


def judge_outcome(
    request: GenerationRequest, responses: dict[str, GenerationResponse], parse: Callable
) -> dict:
    out: dict[str, Any] = {"cache_key": request.cache_key}
    resp = responses.get(request.cache_key)
    if resp is None:
        out["error"] = "no_response"
    elif resp.stop_reason in ("max_tokens", "refusal"):
        out["error"] = f"stop_reason_{resp.stop_reason}"
    else:
        try:
            out["result"] = parse(resp.text)
        except JudgeOutputError as e:
            out["error"] = f"invalid_output: {e}"
    return out


def judge_traces(
    traces: list[dict],
    inputs: Inputs,
    judge: Judge,
    grade_if: Callable[[dict], bool] | None = None,
) -> dict[str, dict]:
    """Run both judge stages over `traces`; return {trace_id: {stage: outcome}}. `grade_if`
    limits the correctness grade to the traces it accepts (Phase 6's adversarial set grades
    only questions with a gold answer); by default every trace not declined is graded."""
    out: dict[str, dict] = {t["trace_id"]: {} for t in traces}
    stage1 = {
        t["trace_id"]: judge.request("decompose", decompose_values(t), DECOMPOSE_SCHEMA)
        for t in traces
        if needs_judge(t["parsed"])
    }
    judge.log(f"judge stage 1: {len(stage1)} decompositions")
    responses = judge.run(list(stage1.values()))
    for tid, req in stage1.items():
        out[tid]["decompose"] = judge_outcome(req, responses, parse_decomposition)

    verify, grade = {}, {}
    for t in traces:
        dec = out[t["trace_id"]].get("decompose", {}).get("result")
        if dec is None:
            continue
        labels = [c["label"] for c in t["retrieval"]["chunks"]]
        if dec["claims"]:
            verify[t["trace_id"]] = (
                judge.request("verify", verify_values(t, dec["claims"]), verify_schema(labels)),
                len(dec["claims"]),
                labels,
            )
        if dec["response_type"] != "declined" and (grade_if is None or grade_if(t)):
            row = inputs.gold_rows[t["question"]["financebench_id"]]
            grade[t["trace_id"]] = judge.request(
                "correctness", correctness_values(t, row), CORRECTNESS_SCHEMA
            )
    judge.log(f"judge stage 2: {len(verify)} verifications, {len(grade)} correctness grades")
    responses = judge.run([v[0] for v in verify.values()] + list(grade.values()))
    for tid, (req, n, labels) in verify.items():
        out[tid]["verify"] = judge_outcome(
            req, responses, lambda text, n=n, labels=labels: parse_verification(text, n, labels)
        )
    for tid, req in grade.items():
        out[tid]["correctness"] = judge_outcome(req, responses, parse_correctness)
    return out


# --------------------------------------------------------------------------------------
# One trace


def evaluate_trace(
    trace: dict,
    inputs: Inputs,
    judged: dict,
    rel_tol: float,
    sensitivity_tols: tuple[float, ...],
) -> dict:
    parsed, q, r = trace["parsed"], trace["question"], trace["retrieval"]
    fid = q["financebench_id"]
    answer = parsed["answer"]
    dec = judged.get("decompose", {}).get("result")
    verdicts = judged.get("verify", {}).get("result")
    graded = judged.get("correctness", {}).get("result")

    status, source = final_status(parsed, dec["response_type"] if dec else None)
    answered = status in ("answered", "partial")

    numeric = None
    if fid in inputs.numeric_golds and answer is not None and answered:
        numeric = grade_numeric(answer, inputs.numeric_golds[fid], rel_tol, sensitivity_tols)
    if status == "declined":
        label, grader = "abstained", None
    elif status == "unparsed":
        label, grader = "unparsed", None
    elif numeric is not None and numeric.label is not None:
        label, grader = numeric.label, "numeric"
    elif graded is not None:
        label, grader = graded["grade"], "judge"
    else:
        label, grader = None, None  # judge failed or not run

    ground = None
    if dec is not None:
        claims = dec["claims"]
        if not claims:
            ground = groundedness.score([], [])
            ground["claims"] = []
        elif verdicts is not None:
            ground = groundedness.score(claims, verdicts)
            ground["claims"] = [
                {**c, **{k: v[k] for k in ("verdict", "reason", "supporting_excerpts")}}
                for c, v in zip(claims, verdicts, strict=True)
            ]

    chunks = r["chunks"]
    cited = parsed["citations"]
    evidence = inputs.evidence[trace["trace_id"]]
    cit: dict[str, Any] = {"cited": cited, "evidence_labels": evidence}
    if verdicts is not None:
        cit.update(judge_citation_metrics(cited, dec["claims"], verdicts))
    cit["gold_cited"] = bool(set(cited) & set(evidence)) if (evidence and answered) else None
    cit["quotes"] = quote_fidelity(parsed["quotes"], chunks) if parsed["quotes"] else None
    cit["numeric_support"] = (
        numeric_support(answer, [c for c in chunks if c["label"] in cited], chunks)
        if answered
        else None
    )

    errors = {stage: o["error"] for stage, o in judged.items() if "error" in o}
    return {
        "trace_id": trace["trace_id"],
        "condition": r["condition"],
        "model_key": trace["generation"]["model_key"],
        "prompt_id": trace["prompt"]["id"],
        "prompt_version": trace["prompt"]["version"],
        "financebench_id": fid,
        "question_type": q["question_type"],
        "in_corpus": q["in_corpus"],
        "answerability": answerability(
            r["condition"],
            q["in_corpus"],
            r["context_metrics"],
            r["k"],
            answerable_elsewhere=fid in inputs.answerable_elsewhere,
        ),
        "answer": answer,
        "parse_status": parsed["status"],
        "confidence": parsed["confidence"],
        "abstention": {
            "status": status,
            "source": source,
            "rule": rule_status(parsed),
            "judge": dec["response_type"] if dec else None,
            "token": parsed["abstained_token"],
        },
        "correctness": {
            "label": label,
            "grader": grader,
            "numeric": numeric.to_dict() if numeric is not None else None,
            "judge": graded,
        },
        "groundedness": ground,
        "citations": cit,
        "judge": {stage: o["cache_key"] for stage, o in sorted(judged.items())},
        "judge_errors": errors,
    }


# --------------------------------------------------------------------------------------
# Aggregates


def _rate(flags: list[bool], n_resamples: int, seed: int) -> dict:
    vals = [float(f) for f in flags]
    return {
        "n": len(vals),
        "rate": sum(vals) / len(vals) if vals else None,
        "ci95": bootstrap_mean_ci(vals, n_resamples, seed) if vals else None,
    }


def _mean(values: list[float], n_resamples: int, seed: int) -> dict:
    return {
        "n": len(values),
        "mean": sum(values) / len(values) if values else None,
        "ci95": bootstrap_mean_ci(values, n_resamples, seed) if values else None,
    }


def calibration(pairs: list[tuple[int, bool]], n_bins: int) -> dict:
    """Reliability-diagram bins over stated confidence (0-100) and the expected calibration
    error: sum over bins of (bin share) x |observed rate - mean stated confidence|."""
    bins = [[] for _ in range(n_bins)]
    for conf, ok in pairs:
        bins[min(int(conf * n_bins / 100), n_bins - 1)].append((conf, ok))
    out, ece, n = [], 0.0, len(pairs)
    for i, b in enumerate(bins):
        lo, hi = 100 * i / n_bins, 100 * (i + 1) / n_bins
        if not b:
            out.append({"lo": lo, "hi": hi, "n": 0, "mean_confidence": None, "observed": None})
            continue
        mc = sum(c for c, _ in b) / len(b)
        obs = sum(ok for _, ok in b) / len(b)
        ece += len(b) / n * abs(obs - mc / 100)
        out.append({"lo": lo, "hi": hi, "n": len(b), "mean_confidence": mc, "observed": obs})
    return {"n": n, "ece": ece if n else None, "bins": out}


def calibration_pairs(records: list[dict]) -> dict[str, list[tuple[int, bool]]]:
    """(stated confidence, outcome) on answered questions only: the Phase 4 analysis rule
    (DECISIONS.md), since confidence on a decline is not interpretable."""
    correct, grounded = [], []
    for rec in records:
        conf = rec["confidence"]
        if conf is None or rec["abstention"]["status"] not in ("answered", "partial"):
            continue
        if rec["correctness"]["label"] is not None:
            correct.append((conf, rec["correctness"]["label"] == "correct"))
        g = rec["groundedness"]
        if g is not None and g["fully_grounded"] is not None:
            grounded.append((conf, g["fully_grounded"]))
    return {"correct": correct, "fully_grounded": grounded}


def aggregate(records: list[dict], cfg: dict) -> dict:
    nb, seed = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["seed"]
    n = len(records)
    status = Counter(r["abstention"]["status"] for r in records)
    answered = [r for r in records if r["abstention"]["status"] in ("answered", "partial")]

    by_ans: dict[str, dict] = {}
    for group in sorted({r["answerability"] for r in records}):
        rs = [r for r in records if r["answerability"] == group]
        by_ans[group] = {
            "n": len(rs),
            "declined": _rate([r["abstention"]["status"] == "declined" for r in rs], nb, seed),
            "partial_rate": sum(r["abstention"]["status"] == "partial" for r in rs) / len(rs),
            "answered_rate": sum(r["abstention"]["status"] == "answered" for r in rs) / len(rs),
        }

    labels = Counter(r["correctness"]["label"] for r in records)
    graded = [r for r in records if r["correctness"]["label"] is not None]
    ans_graded = [r for r in answered if r["correctness"]["label"] is not None]
    numeric = [r for r in records if r["correctness"]["numeric"] is not None]
    sens = {}
    for tol in cfg["correctness"]["sensitivity_rel_tols"]:
        with_fig = [r for r in numeric if r["correctness"]["numeric"]["label"] is not None]
        sens[str(tol)] = (
            sum(r["correctness"]["numeric"]["sensitivity"][str(tol)] == "correct" for r in with_fig)
            / len(with_fig)
            if with_fig
            else None
        )
    with_fig = [r for r in numeric if r["correctness"]["numeric"]["label"] is not None]

    scored = [
        r
        for r in answered
        if r["groundedness"] is not None and r["groundedness"]["groundedness"] is not None
    ]
    declined_scored = [
        r
        for r in records
        if r["abstention"]["status"] == "declined"
        and r["groundedness"] is not None
        and r["groundedness"]["groundedness"] is not None
    ]
    n_doc = sum(r["groundedness"]["n_document_claims"] for r in scored)
    with_ctx = [r for r in records if r["groundedness"] and r["groundedness"]["n_context_claims"]]
    n_ctx = sum(r["groundedness"]["n_context_claims"] for r in with_ctx)
    quotes = [
        s for r in records if r["citations"]["quotes"] for s in r["citations"]["quotes"]["statuses"]
    ]
    precision = [
        r["citations"]["citation_precision"]
        for r in answered
        if r["citations"].get("citation_precision") is not None
    ]
    recall = [
        r["citations"]["citation_recall"]
        for r in answered
        if r["citations"].get("citation_recall") is not None
    ]
    ns = [r["citations"]["numeric_support"] for r in answered if r["citations"]["numeric_support"]]
    pairs = calibration_pairs(records)
    n_bins = cfg["reliability"]["n_bins"]

    return {
        "n": n,
        "abstention": {
            "status_counts": {
                s: status.get(s, 0) for s in ("answered", "partial", "declined", "unparsed")
            },
            "by_answerability": by_ans,
        },
        "correctness": {
            "label_counts": {lab: labels.get(lab, 0) for lab in CORRECTNESS_LABELS},
            "n_ungraded": labels.get(None, 0),
            "accuracy_all": _rate(
                [r["correctness"]["label"] == "correct" for r in graded], nb, seed
            ),
            "accuracy_answered": _rate(
                [r["correctness"]["label"] == "correct" for r in ans_graded], nb, seed
            ),
            "unit_error_rate_answered": (
                sum(r["correctness"]["label"] == "unit_error" for r in ans_graded) / len(ans_graded)
                if ans_graded
                else None
            ),
            "graders": dict(Counter(r["correctness"]["grader"] or "none" for r in records)),
            "numeric_questions": {
                "n_answered": len(numeric),
                "n_with_figure": len(with_fig),
                "accuracy_with_figure": _rate(
                    [r["correctness"]["numeric"]["label"] == "correct" for r in with_fig], nb, seed
                ),
                "accuracy_any_figure": (
                    sum(
                        r["correctness"]["numeric"]["any_figure_label"] == "correct"
                        for r in with_fig
                    )
                    / len(with_fig)
                    if with_fig
                    else None
                ),
                "accuracy_by_rel_tol": sens,
            },
        },
        "groundedness": {
            "n_scored": len(scored),
            "n_answered_without_document_claims": sum(
                1
                for r in answered
                if r["groundedness"] is not None and r["groundedness"]["groundedness"] is None
            ),
            "mean_groundedness": _mean(
                [r["groundedness"]["groundedness"] for r in scored], nb, seed
            ),
            "fully_grounded": _rate(
                [r["groundedness"]["fully_grounded"] for r in scored], nb, seed
            ),
            "any_contradicted": _rate(
                [r["groundedness"]["any_contradicted"] for r in scored], nb, seed
            ),
            "claim_level": {
                "n_document_claims": n_doc,
                "supported_rate": (
                    sum(r["groundedness"]["n_supported"] for r in scored) / n_doc if n_doc else None
                ),
                "contradicted_rate": (
                    sum(r["groundedness"]["n_contradicted"] for r in scored) / n_doc
                    if n_doc
                    else None
                ),
                "n_general_claims": sum(r["groundedness"]["n_general_claims"] for r in scored),
            },
            # Claims about the excerpts ("X is not given"), over every response that makes
            # them; a contradicted one says information is missing when an excerpt holds it.
            "context_claims": {
                "n_claims": n_ctx,
                "n_responses": len(with_ctx),
                "contradicted_rate": (
                    sum(r["groundedness"]["n_context_contradicted"] for r in with_ctx) / n_ctx
                    if n_ctx
                    else None
                ),
                "responses_with_false_absence": sum(
                    r["groundedness"]["n_context_contradicted"] > 0 for r in with_ctx
                ),
            },
            "declined_with_claims": {
                "n": len(declined_scored),
                "mean_groundedness": (
                    sum(r["groundedness"]["groundedness"] for r in declined_scored)
                    / len(declined_scored)
                    if declined_scored
                    else None
                ),
            },
        },
        "citations": {
            "citation_precision": _mean(precision, nb, seed),
            "citation_recall": _mean(recall, nb, seed),
            "answered_without_citations": sum(1 for r in answered if not r["citations"]["cited"]),
            "gold_cited": _rate(
                [
                    r["citations"]["gold_cited"]
                    for r in answered
                    if r["citations"]["gold_cited"] is not None
                ],
                nb,
                seed,
            ),
            "quotes": {
                "n": len(quotes),
                "status_counts": dict(Counter(quotes)),
                "faithful_rate": (
                    sum(s in ("exact", "near") for s in quotes) / len(quotes) if quotes else None
                ),
            },
            "numeric_support": {
                "n_answers_with_figures": sum(1 for x in ns if x["n_figures"]),
                "figures_in_context_rate": (
                    sum(x["n_in_context"] for x in ns) / sum(x["n_figures"] for x in ns)
                    if sum(x["n_figures"] for x in ns)
                    else None
                ),
                "figures_in_cited_rate": (
                    sum(x["n_in_cited"] for x in ns if x["in_cited_rate"] is not None)
                    / sum(x["n_figures"] for x in ns if x["in_cited_rate"] is not None)
                    if sum(x["n_figures"] for x in ns if x["in_cited_rate"] is not None)
                    else None
                ),
            },
        },
        "calibration": {
            "correct": {
                k: v for k, v in calibration(pairs["correct"], n_bins).items() if k != "bins"
            },
            "fully_grounded": {
                k: v for k, v in calibration(pairs["fully_grounded"], n_bins).items() if k != "bins"
            },
        },
        "judge_errors": dict(
            Counter(
                f"{stage}:{err.split(':')[0]}"
                for r in records
                for stage, err in r["judge_errors"].items()
            )
        ),
    }


def detector_agreement(records: list[dict], n_resamples: int, seed: int) -> dict:
    """Rule-based vs judge abstention status, where both exist."""
    both = [r for r in records if r["abstention"]["judge"] is not None]
    rule = [r["abstention"]["rule"] for r in both]
    judge = [r["abstention"]["judge"] for r in both]
    summary = agreement_summary(
        rule, judge, ["answered", "partial", "declined"], None, n_resamples, seed
    )
    summary["rater_a"], summary["rater_b"] = "rule", "judge"
    summary["binary_declined"] = agreement_summary(
        [x == "declined" for x in rule],
        [x == "declined" for x in judge],
        [True, False],
        None,
        n_resamples,
        seed,
    )
    return summary


def correctness_crosscheck(records: list[dict], n_resamples: int, seed: int) -> dict:
    """The deterministic numeric matcher against the LLM correctness judge, on the numeric
    questions where both graded the same answer."""
    both = [
        r
        for r in records
        if r["correctness"]["numeric"] is not None
        and r["correctness"]["numeric"]["label"] is not None
        and r["correctness"]["judge"] is not None
    ]
    num = [r["correctness"]["numeric"]["label"] for r in both]
    jud = [r["correctness"]["judge"]["grade"] for r in both]
    table = Counter(zip(num, jud, strict=True))
    binary = agreement_summary(
        [x == "correct" for x in num],
        [x == "correct" for x in jud],
        [True, False],
        None,
        n_resamples,
        seed,
    )
    unit = [
        (x == "unit_error", r["correctness"]["judge"]["unit_error"])
        for x, r in zip(num, both, strict=True)
    ]
    disagreements = [
        {
            "trace_id": r["trace_id"],
            "answer": r["answer"],
            "numeric": r["correctness"]["numeric"]["label"],
            "numeric_figure": r["correctness"]["numeric"]["primary"]["figure"],
            "judge": r["correctness"]["judge"]["grade"],
            "judge_unit_error": r["correctness"]["judge"]["unit_error"],
            "judge_reason": r["correctness"]["judge"]["reason"],
        }
        for r, x, y in zip(both, num, jud, strict=True)
        if (x == "correct") != (y == "correct")
    ]
    return {
        "n": len(both),
        "table_numeric_by_judge": {f"{a}|{b}": c for (a, b), c in sorted(table.items())},
        "binary_correct": binary,
        "unit_error_flags": {
            "both": sum(a and b for a, b in unit),
            "numeric_only": sum(a and not b for a, b in unit),
            "judge_only": sum(b and not a for a, b in unit),
        },
        "disagreements": disagreements,
    }
