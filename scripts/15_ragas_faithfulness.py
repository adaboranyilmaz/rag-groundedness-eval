"""RAGAS faithfulness on the 50 hand-labelled validation answers, compared with the project's
own groundedness judge and the author's labels.

The spec (§4) asks for the custom metric first and RAGAS second: the judge was built and
validated in Phase 5 (results/metrics/judge_agreement.json); this scores the same 50 answers
with RAGAS's Faithfulness (ragas.metrics.collections, the metric RAGAS 0.4 documents) on the
same model as the judge, and compares the three raters per answer
(src/external/faithfulness_comparison.py). RAGAS sees what the generator and the judge saw:
the question, the ANSWER field, and the five excerpts with their headers (the header is the
only place an excerpt names its company and period).

Modes:
  --dry-run   cost estimate before any spend -> results/metrics/ragas_cost_estimate.json.
              Statement-generation prompts are rendered by RAGAS and counted exactly with the
              token-counting endpoint (free). The NLI step's prompts depend on RAGAS's own
              statements, so they are counted with the judge's claims standing in for them.
              Output lengths are the judge's measured outputs on the same answers (the NLI
              output x1.5, since RAGAS echoes each statement word by word); the worst case is
              every call at max_tokens.
  (default)   score every answer (standard API calls through the response cache and ledger)
              -> results/metrics/ragas_faithfulness.json            comparison and meta
                 results/metrics/ragas_faithfulness_per_answer.jsonl statements and verdicts
  --offline   the default outputs from the response cache only; no API calls. Reproduces
              both files exactly.

Runs in the `ragas` environment (.venv-ragas, pyproject.toml group `ragas`, resolved apart
from the pipeline's). Started from the default environment, it re-runs itself there:
    uv run python scripts/15_ragas_faithfulness.py --dry-run
    uv run python scripts/15_ragas_faithfulness.py
    uv run --env-file configs/replay.env python scripts/15_ragas_faithfulness.py --offline
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["RAGAS_DO_NOT_TRACK"] = "true"  # RAGAS otherwise reports usage to its servers

CONFIG_PATH = Path("configs/ragas.yaml")
EVAL_CONFIG_PATH = Path("configs/evaluation.yaml")
RESULTS_DIR = Path("results/metrics")
OUT_PATH = RESULTS_DIR / "ragas_faithfulness.json"
PER_ANSWER_PATH = RESULTS_DIR / "ragas_faithfulness_per_answer.jsonl"
ESTIMATE_PATH = RESULTS_DIR / "ragas_cost_estimate.json"
RAGAS_ENV = ".venv-ragas"
NLI_OUTPUT_ECHO_FACTOR = 1.5  # dry run only: RAGAS repeats each statement in its verdicts


def ensure_ragas_env() -> None:
    """Re-run this script in the ragas environment if ragas is not importable here."""
    if importlib.util.find_spec("ragas") is not None:
        return
    if os.environ.get("RAG_IN_RAGAS_ENV") == "1":
        sys.exit(f"ragas is not installed in {RAGAS_ENV}")
    try:
        from dotenv import load_dotenv

        load_dotenv()  # the API key, for the child as for every other script
    except ImportError:
        pass
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": RAGAS_ENV, "RAG_IN_RAGAS_ENV": "1"}
    env.pop("VIRTUAL_ENV", None)
    cmd = ["uv", "run", "--frozen", "--no-default-groups", "--group", "ragas", "python"]
    print(f"re-running in {RAGAS_ENV} (the ragas dependency group)", flush=True)
    sys.exit(subprocess.call([*cmd, str(Path(__file__).resolve()), *sys.argv[1:]], env=env))


def contexts(item: dict) -> list[str]:
    """Each excerpt as src/generation/prompts.py `format_context` renders it."""
    return [f"{x['header']}\n{x['text'].strip()}" for x in item["excerpts"]]


def load_inputs(cfg: dict) -> tuple[dict, list[dict], list[dict]]:
    sample = json.loads(Path(cfg["sample"]).read_text(encoding="utf-8"))
    agreement = json.loads(Path(cfg["agreement"]).read_text(encoding="utf-8"))
    if agreement["meta"]["sample_sha256"] != sample["sample_sha256"]:
        sys.exit("judge_agreement.json was computed on a different sample")
    return sample, sample["items"], agreement["claims"]


def make_ledger(cfg: dict):
    from src.generation.llm import SpendLedger

    b = cfg["budget"]
    return SpendLedger(
        Path(b["ledger"]),
        b["prices_usd_per_mtok"],
        b["project_cap_usd"],
        b["phase"],
        b["phase_cap_usd"],
        b["batch_discount"],
    )


def prompt_meta() -> dict:
    """RAGAS's two prompts as it renders them, hashed on a fixed input, so a RAGAS upgrade
    that changes them shows in the results file."""
    import hashlib

    from ragas.metrics.collections.faithfulness.util import (
        NLIStatementInput,
        NLIStatementPrompt,
        StatementGeneratorInput,
        StatementGeneratorPrompt,
    )

    def h(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    return {
        "statement_generator": h(
            StatementGeneratorPrompt().to_string(StatementGeneratorInput(question="q", answer="a"))
        ),
        "nli_statement": h(
            NLIStatementPrompt().to_string(NLIStatementInput(context="c", statements=["s"]))
        ),
    }


# --------------------------------------------------------------------------------------
# Dry run


def dry_run(cfg: dict, items: list[dict]) -> None:
    import anthropic
    from ragas.metrics.collections.faithfulness.util import (
        NLIStatementInput,
        NLIStatementOutput,
        NLIStatementPrompt,
        StatementGeneratorInput,
        StatementGeneratorOutput,
        StatementGeneratorPrompt,
    )

    from src.external.ragas_llm import ProjectRagasLLM
    from src.generation.llm import AnthropicBackend, ResponseCache

    client = anthropic.Anthropic()
    llm = ProjectRagasLLM(cfg["model"], cfg["max_tokens"], ResponseCache(cfg["cache_dir"]), None)
    main_cache = ResponseCache()  # read only: the judge's measured outputs on these answers
    judge_keys = {}
    for line in (RESULTS_DIR / "eval_per_trace.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        judge_keys[r["trace_id"]] = r["judge"]

    def count(req) -> int:
        params = AnthropicBackend.call_params(req)
        params.pop("max_tokens")
        return client.messages.count_tokens(**params).input_tokens

    steps = {"statements": {"in": 0, "out": 0}, "nli": {"in": 0, "out": 0}}
    for it in items:
        s_req = llm.request(
            StatementGeneratorPrompt().to_string(
                StatementGeneratorInput(question=it["question"], answer=it["answer"])
            ),
            StatementGeneratorOutput,
        )
        n_req = llm.request(
            NLIStatementPrompt().to_string(
                NLIStatementInput(
                    context="\n".join(contexts(it)), statements=[c["claim"] for c in it["claims"]]
                )
            ),
            NLIStatementOutput,
        )
        keys = judge_keys[it["trace_id"]]
        steps["statements"]["in"] += count(s_req)
        steps["statements"]["out"] += main_cache.get(keys["decompose"]).output_tokens
        steps["nli"]["in"] += count(n_req)
        steps["nli"]["out"] += round(
            main_cache.get(keys["verify"]).output_tokens * NLI_OUTPUT_ECHO_FACTOR
        )

    price = cfg["budget"]["prices_usd_per_mtok"][cfg["model"]["model"]]
    n = len(items)
    for s in steps.values():
        s["calls"] = n
        s["expected_usd"] = round((s["in"] * price["input"] + s["out"] * price["output"]) / 1e6, 4)
        s["worst_case_usd"] = round(
            (s["in"] * price["input"] + n * cfg["max_tokens"] * price["output"]) / 1e6, 4
        )
    ledger = make_ledger(cfg)
    estimate = {
        "meta": {
            "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "estimate": True,
            "prices_as_of": cfg["budget"]["prices_as_of"],
            "price_basis": "standard (RAGAS calls are synchronous; no batch discount)",
            "n_answers": n,
            "input_tokens": "counted with the token-counting endpoint; the NLI step with the "
            "judge's claims standing in for RAGAS's statements",
            "output_tokens": "the judge's measured outputs on the same answers (decompose for "
            f"statements; verify x{NLI_OUTPUT_ECHO_FACTOR} for NLI)",
        },
        "steps": steps,
        "total_expected_usd": round(sum(s["expected_usd"] for s in steps.values()), 4),
        "total_worst_case_usd": round(sum(s["worst_case_usd"] for s in steps.values()), 4),
        "phase_spent_usd": round(ledger.phase_spent(), 4),
        "project_spent_usd": round(ledger.state["total_usd"], 4),
        "caps": {"project_usd": ledger.project_cap_usd, "phase_usd": ledger.phase_cap_usd},
    }
    ESTIMATE_PATH.write_text(json.dumps(estimate, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(estimate, indent=2))
    print(f"wrote {ESTIMATE_PATH}")


# --------------------------------------------------------------------------------------
# Scoring


async def score_all(cfg: dict, items: list[dict], ledger) -> dict[str, dict]:
    from ragas.metrics.collections import Faithfulness

    from src.external.ragas_llm import ProjectRagasLLM, RagasCallError
    from src.generation.llm import ResponseCache

    cache = ResponseCache(cfg["cache_dir"])
    sem = asyncio.Semaphore(max(1, int(cfg["concurrency"])))

    async def one(it: dict) -> tuple[str, dict]:
        async with sem:
            llm = ProjectRagasLLM(cfg["model"], cfg["max_tokens"], cache, ledger)
            score, error = None, None
            try:
                result = await Faithfulness(llm=llm).ascore(
                    user_input=it["question"],
                    response=it["answer"],
                    retrieved_contexts=contexts(it),
                )
                if math.isnan(result.value):
                    error = "no_statements"
                else:
                    score = float(result.value)
            except RagasCallError as e:
                error = str(e).split(":")[0]
            outputs = {c["step"]: c.get("output") for c in llm.calls}
            statements = (outputs.get("StatementGeneratorOutput") or {}).get("statements")
            verdicts = (outputs.get("NLIStatementOutput") or {}).get("statements")
            return it["trace_id"], {
                "score": score,
                "error": error,
                "n_statements": len(statements) if statements is not None else None,
                # RAGAS scores over the verdicts it gets back and does not check that there
                # is one per statement; recorded so a mismatch is visible
                "n_verdicts": len(verdicts) if verdicts is not None else None,
                "statements": statements,
                "verdicts": verdicts,
                "calls": [
                    {k: c.get(k) for k in ("step", "cache_key", "input_tokens", "output_tokens")}
                    | {"stop_reason": c.get("stop_reason"), "error": c.get("error")}
                    for c in llm.calls
                ],
                "_new_cost_usd": sum(c["new_cost_usd"] for c in llm.calls),
            }

    return dict(await asyncio.gather(*(one(it) for it in items)))


def score_and_compare(cfg: dict, sample: dict, items: list[dict], claim_rows: list[dict]) -> None:
    import yaml

    from src.external.faithfulness_comparison import build_per_answer, compare_all
    from src.generation.llm import BudgetExceeded, replay_only

    ledger = make_ledger(cfg)
    try:
        ragas = asyncio.run(score_all(cfg, items, ledger))
    except BudgetExceeded as e:
        sys.exit(f"STOPPED: {e}\nCompleted calls are cached; nothing is lost.")
    new_cost = sum(r.pop("_new_cost_usd") for r in ragas.values())
    print(f"this run: ${new_cost:.4f} new spend ({'replay only' if replay_only() else 'live'})")

    eval_cfg = yaml.safe_load(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
    model_keys = list(eval_cfg["validation"]["n_per_generator"])
    per_answer = build_per_answer(items, claim_rows, ragas)
    nb, seed = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["seed"]
    comparison = compare_all(per_answer, model_keys, nb, seed)

    # The judge's answer-level kappa with the author must equal the Phase 5 number, since
    # both come from the same verdicts; a mismatch means the inputs are not the ones validated.
    agreement = json.loads(Path(cfg["agreement"]).read_text(encoding="utf-8"))
    phase5 = agreement["agreement"]["human__vs__judge"]["document_claims"]["answer_fully_grounded"]
    here = comparison["pairs"]["human__vs__judge"]["all"]["fully_grounded"]
    if not math.isclose(here["kappa"], phase5["kappa"], abs_tol=1e-12):
        sys.exit(f"human-vs-judge kappa {here['kappa']} differs from Phase 5's {phase5['kappa']}")

    from importlib.metadata import version as pkg_version

    calls = [c for r in ragas.values() for c in r["calls"]]
    out = {
        "meta": {
            "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "single_run": True,
            "sample_sha256": sample["sample_sha256"],
            "config": cfg,
            "ragas_version": pkg_version("ragas"),
            "ragas_prompt_sha256": prompt_meta(),
            "metric": "ragas.metrics.collections.Faithfulness",
            "inputs": "question, the ANSWER field, the five excerpts with their headers "
            "(as the generator and the judge saw them)",
            "rater_definitions": {
                "human": "the author's blind claim labels (judge_agreement.json), on the "
                "judge's own claim split; score = supported / document claims",
                "judge": "the Phase 5 judge (claude-sonnet-5, thinking off)",
                "second_judge": "claude-haiku-4-5, the Phase 5 second judge",
                "ragas": "RAGAS Faithfulness on the same model as the judge; score = "
                "faithful statements / statements (RAGAS has no claim kinds)",
                "fully_grounded": "score == 1",
            },
            "caveat": "the human labels were given on the judge's claims, not on RAGAS's "
            "statements, which favours the judge in any comparison with RAGAS",
            "calls": {
                "n": len(calls),
                "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
                "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
                "stop_reasons": {
                    s: sum(c["stop_reason"] == s for c in calls)
                    for s in sorted({str(c["stop_reason"]) for c in calls})
                },
            },
            "phase_spent_usd": round(ledger.phase_spent(), 6),
            "environment": {
                "python": platform.python_version(),
                "anthropic": pkg_version("anthropic"),
            },
            "phase5_human_vs_judge_kappa_reproduced": True,
        },
        "comparison": comparison,
        "per_answer": per_answer,
    }
    OUT_PATH.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    lines = [
        json.dumps({"trace_id": tid, **r}, ensure_ascii=False, sort_keys=True)
        for tid, r in sorted(ragas.items())
    ]
    PER_ANSWER_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print_summary(comparison)
    print(f"wrote {OUT_PATH} and {PER_ANSWER_PATH}")


def print_summary(comparison: dict) -> None:
    print(f"\nanswers {comparison['n_answers']}, RAGAS unscored {comparison['n_ragas_unscored']}")
    print(f"{'pair':28s} {'subset':16s} {'n':>3s} {'kappa (fully grounded)':>26s} {'rho':>6s}")
    for pair, subsets in comparison["pairs"].items():
        for name, c in subsets.items():
            k, s = c["fully_grounded"], c["score"]
            ci = k["kappa_ci95"]
            kt = "undefined" if k["kappa"] is None else f"{k['kappa']:.2f}"
            if ci:
                kt += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
            rho = "n/a" if s["spearman"] is None else f"{s['spearman']:.2f}"
            print(f"{pair:28s} {name:16s} {k['n']:3d} {kt:>26s} {rho:>6s}")
    d = comparison["human_judge_minus_human_ragas"]
    if d["kappa_diff"] is not None:
        ci = d["kappa_diff_ci95"]
        print(f"kappa(human, judge) - kappa(human, ragas) = {d['kappa_diff']:.2f} {ci}")


# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    ensure_ragas_env()

    import yaml
    from dotenv import load_dotenv

    load_dotenv()
    if args.offline:
        os.environ["RAG_REPLAY_ONLY"] = "1"
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    sample, items, claim_rows = load_inputs(cfg)
    print(f"{len(items)} validation answers, {len(claim_rows)} labelled claims")
    if args.dry_run:
        dry_run(cfg, items)
        return
    score_and_compare(cfg, sample, items, claim_rows)


if __name__ == "__main__":
    main()
