"""Judge validation: the author's hand labels against the LLM groundedness judge.

  sample     Draw the validation sample (configs/evaluation.yaml `validation`): a quota per
             (condition, generator) cell, prompts balanced within each cell, one trace per
             question, no pilot question (the pilot's verdicts have been read). Each candidate
             is decomposed with the frozen judge prompt, as the same request the full run
             sends (so the full run finds it cached), and kept if the response is answered or
             partial with >=1 document claim. Only the decomposition is run: no verdict for a
             sample item exists before the labels do.
             -> results/labels/validation_sample.json      (committed)
                results/labels/groundedness_labeling.html  (generated; not committed)
  agreement  After the labels are exported to results/labels/groundedness_human.json:
             the primary judge's verdicts for the sample (the full run's cached calls), the
             second judge's (Haiku 4.5, temperature 0), and the primary judge re-run once
             bypassing the cache (test-retest). Cohen's kappa for each pair, at claim level
             (three-class and supported-vs-not, cluster bootstrap over answers) and answer
             level (fully grounded), overall and per generator.
             -> results/metrics/judge_agreement.json, results/metrics/judge_agreement.md

Usage:
    uv run python scripts/06b_judge_validation.py sample
    uv run python scripts/06b_judge_validation.py agreement
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.agreement import agreement_summary
from src.evaluation.evaluate import (
    Judge,
    decompose_values,
    judge_outcome,
    load_inputs,
    make_ledger,
    verify_values,
)
from src.evaluation.judge import (
    DECOMPOSE_SCHEMA,
    VERDICTS,
    parse_decomposition,
    parse_verification,
    verify_schema,
)
from src.evaluation.labeling_page import render_labeling_page
from src.evaluation.sampling import pilot_questions, validation_candidates, validation_quotas
from src.generation.llm import ResponseCache

CONFIG_PATH = Path("configs/evaluation.yaml")
LABELS_DIR = Path("results/labels")
SAMPLE_PATH = LABELS_DIR / "validation_sample.json"
PAGE_PATH = LABELS_DIR / "groundedness_labeling.html"
HUMAN_PATH = LABELS_DIR / "groundedness_human.json"
ADJUDICATION_PATH = LABELS_DIR / "groundedness_adjudication.json"
RESULTS_DIR = Path("results/metrics")
ROUND_SIZE = 8  # candidates decomposed per round (standard API, concurrent)


def excerpt_header(chunk: dict) -> str:
    """The header src/generation/prompts.py `format_context` gives each excerpt."""
    header = f"[{chunk['label']}] {chunk['doc_id']} | page {chunk['page']}"
    return header + (f" | {chunk['section']}" if chunk.get("section") else "")


def item_id(trace_id: str) -> str:
    """Opaque id for the labelling page: the trace id names the model and condition."""
    return hashlib.sha256(trace_id.encode("utf-8")).hexdigest()[:12]


def sample_hash(items: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(items, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------------------
# sample


def cmd_sample(cfg: dict) -> None:
    if SAMPLE_PATH.exists():
        sys.exit(f"{SAMPLE_PATH} exists; the sample is drawn once. Delete it to redraw.")
    inputs = load_inputs(cfg)
    cache = ResponseCache()
    ledger = make_ledger(cfg)
    judge = Judge(cfg, cfg["judge"]["model"], "sync", cache, ledger)
    p, v = cfg["pilot"], cfg["validation"]
    pilot_fids = set(
        pilot_questions(
            inputs.traces, p["n_questions_in_corpus"], p["n_questions_out_of_corpus"], p["seed"]
        )
    )
    quotas = validation_quotas(v["n_per_generator"], cfg["traces"]["conditions"])
    # Materialised, so a candidate deferred because its cell's open slots are already pending
    # in this round is tried in a later round instead of being dropped.
    queue = list(validation_candidates(inputs.traces, pilot_fids, quotas, v["seed"]))
    taken: set[int] = set()
    accepted: dict[tuple, list] = {cell: [] for cell in quotas}
    used: set[str] = set()
    rejected: list[dict] = []
    n_tried = 0

    def open_slots(cell) -> int:
        return quotas[cell] - len(accepted[cell])

    while any(open_slots(c) > 0 for c in quotas):
        batch, pending = [], Counter()
        for idx, (cell, t) in enumerate(queue):
            fid = t["question"]["financebench_id"]
            if idx in taken:
                continue
            if open_slots(cell) <= 0 or fid in used:
                taken.add(idx)  # never needed
                continue
            if open_slots(cell) - pending[cell] <= 0:
                continue  # deferred to a later round
            if any(fid == x["question"]["financebench_id"] for _, x in batch):
                continue  # deferred: one trace per question
            batch.append((cell, t))
            pending[cell] += 1
            taken.add(idx)
            if len(batch) == ROUND_SIZE:
                break
        if not batch:
            filled = {f"{c[0]}/{c[1]}": len(a) for c, a in accepted.items()}
            sys.exit(f"candidates exhausted before quotas were met: {filled}")
        requests = [
            judge.request("decompose", decompose_values(t), DECOMPOSE_SCHEMA) for _, t in batch
        ]
        responses = judge.run(requests)
        for (cell, t), req in zip(batch, requests, strict=True):
            n_tried += 1
            out = judge_outcome(req, responses, parse_decomposition)
            reason = None
            if "error" in out:
                reason = out["error"]
            elif out["result"]["response_type"] == "declined":
                reason = "declined"
            elif not any(c["kind"] == "document" for c in out["result"]["claims"]):
                reason = "no_document_claims"
            elif open_slots(cell) <= 0 or t["question"]["financebench_id"] in used:
                reason = "quota_filled"
            if reason:
                rejected.append({"trace_id": t["trace_id"], "reason": reason})
                continue
            used.add(t["question"]["financebench_id"])
            accepted[cell].append(
                {
                    "item_id": item_id(t["trace_id"]),
                    "trace_id": t["trace_id"],
                    "condition": cell[0],
                    "model_key": cell[1],
                    "prompt_id": t["prompt"]["id"],
                    "financebench_id": t["question"]["financebench_id"],
                    "question": t["question"]["question"],
                    "answer": t["parsed"]["answer"],
                    "excerpts": [
                        {"label": c["label"], "header": excerpt_header(c), "text": c["text"]}
                        for c in t["retrieval"]["chunks"]
                    ],
                    "claims": out["result"]["claims"],
                    "response_type": out["result"]["response_type"],
                    "decompose_cache_key": out["cache_key"],
                }
            )
        progress = ", ".join(f"{c[0]}/{c[1]} {len(a)}/{quotas[c]}" for c, a in accepted.items())
        print(f"tried {n_tried}: {progress}")

    items = [it for cell in sorted(accepted) for it in accepted[cell]]
    random.Random(f"{v['seed']}:presentation").shuffle(items)  # hide cell structure
    digest = sample_hash(items)
    dp = judge.prompts["decompose"]
    sample = {
        "meta": {
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "seed": v["seed"],
            "quotas": {f"{c[0]}|{c[1]}": n for c, n in sorted(quotas.items())},
            "pilot_questions_excluded": sorted(pilot_fids),
            "decompose_prompt": {"id": dp.id, "version": dp.version, "file_sha256": dp.file_sha256},
            "n_candidates_tried": n_tried,
            "rejected_reasons": dict(Counter(r["reason"].split(":")[0] for r in rejected)),
            "n_claims": sum(len(it["claims"]) for it in items),
            "claim_kinds": dict(Counter(c["kind"] for it in items for c in it["claims"])),
            "api_cost_usd": round(judge.new_cost_usd, 6),
        },
        "sample_sha256": digest,
        "items": items,
        "rejected": rejected,
    }
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(sample, indent=1, ensure_ascii=False), encoding="utf-8")
    page_items = [
        {k: it[k] for k in ("item_id", "question", "answer", "claims")}
        | {"excerpts": [{"header": x["header"], "text": x["text"]} for x in it["excerpts"]]}
        for it in items
    ]
    PAGE_PATH.write_text(render_labeling_page(page_items, digest), encoding="utf-8")
    print(json.dumps(sample["meta"], indent=2))
    print(f"wrote {SAMPLE_PATH} and {PAGE_PATH}")


# --------------------------------------------------------------------------------------
# agreement


def judge_verdicts(judge: Judge, traces: dict, items: list[dict]) -> tuple[dict, list]:
    """{trace_id: [verdict dicts]} for the sample, plus failures."""
    reqs = {}
    for it in items:
        t = traces[it["trace_id"]]
        labels = [c["label"] for c in t["retrieval"]["chunks"]]
        reqs[it["trace_id"]] = (
            judge.request("verify", verify_values(t, it["claims"]), verify_schema(labels)),
            len(it["claims"]),
            labels,
        )
    responses = judge.run([r for r, _, _ in reqs.values()])
    out, failures = {}, []
    for tid, (req, n, labels) in reqs.items():
        o = judge_outcome(
            req, responses, lambda text, n=n, labels=labels: parse_verification(text, n, labels)
        )
        if "error" in o:
            failures.append({"trace_id": tid, "error": o["error"]})
        else:
            out[tid] = o["result"]
    return out, failures


def compare(rows: list[dict], a: str, b: str, nb: int, seed: int) -> dict:
    """Claim-level (3-class, binary) and answer-level agreement between raters a and b on
    `rows` (one per claim). Answers where either rater lacks a verdict are dropped."""
    rows = [r for r in rows if r[a] is not None and r[b] is not None]
    clusters = [r["trace_id"] for r in rows]
    three = agreement_summary(
        [r[a] for r in rows], [r[b] for r in rows], VERDICTS, clusters, nb, seed
    )
    binary = agreement_summary(
        [r[a] == "supported" for r in rows],
        [r[b] == "supported" for r in rows],
        [True, False],
        clusters,
        nb,
        seed,
    )
    by_answer: dict[str, list[dict]] = {}
    for r in rows:
        if r["kind"] == "document":
            by_answer.setdefault(r["trace_id"], []).append(r)
    fa = [all(x[a] == "supported" for x in rs) for rs in by_answer.values()]
    fb = [all(x[b] == "supported" for x in rs) for rs in by_answer.values()]
    answer = agreement_summary(fa, fb, [True, False], None, nb, seed)
    return {"claim_3class": three, "claim_supported": binary, "answer_fully_grounded": answer}


def cmd_agreement(cfg: dict) -> None:
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    human = json.loads(HUMAN_PATH.read_text(encoding="utf-8"))
    if human["sample_sha256"] != sample["sample_sha256"]:
        sys.exit("the labels were made on a different sample")
    items = sample["items"]
    missing = [
        it["trace_id"]
        for it in items
        if it["item_id"] not in human["items"]
        or len(human["items"][it["item_id"]]["claims"]) != len(it["claims"])
        or any(c["verdict"] not in VERDICTS for c in human["items"][it["item_id"]]["claims"])
    ]
    if missing:
        sys.exit(f"{len(missing)} items are not fully labelled; first: {missing[0]}")

    inputs = load_inputs(cfg)
    traces = {t["trace_id"]: t for t in inputs.traces}
    cache, ledger = ResponseCache(), make_ledger(cfg)
    v = cfg["validation"]
    judges = {
        "judge": Judge(cfg, cfg["judge"]["model"], "sync", cache, ledger),
        "second_judge": Judge(cfg, v["second_judge"], "sync", cache, ledger),
        "judge_retest": Judge(
            cfg, cfg["judge"]["model"], "sync", cache, ledger, replicate=v["retest_replicate"]
        ),
    }
    verdicts, failures = {}, {}
    for name, j in judges.items():
        verdicts[name], failures[name] = judge_verdicts(j, traces, items)

    # Adjudicated labels (unblinded, the labeller's decisions after seeing the verdicts) are
    # reported beside the blind labels, never in place of them.
    adjudicated, adjudication = None, None
    if ADJUDICATION_PATH.exists():
        adjudication = json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8"))
        if adjudication["sample_sha256"] != sample["sample_sha256"]:
            sys.exit("the adjudication was made on a different sample")
        adjudicated = {
            (c["item_id"], c["claim_index"]): c["adjudicated_label"]
            for c in adjudication["changes"]
        }

    rows = []
    for it in items:
        h = human["items"][it["item_id"]]
        for i, c in enumerate(it["claims"]):
            row = {
                "trace_id": it["trace_id"],
                "model_key": it["model_key"],
                "condition": it["condition"],
                "claim_index": i + 1,
                "claim": c["claim"],
                "kind": c["kind"],
                "human": h["claims"][i]["verdict"],
                "human_misrepresents": h["claims"][i]["misrepresents"],
            }
            for name in judges:
                vs = verdicts[name].get(it["trace_id"])
                row[name] = vs[i]["verdict"] if vs else None
                if name == "judge" and vs:
                    row["judge_reason"] = vs[i]["reason"]
            if adjudicated is not None:
                row["human_adjudicated"] = adjudicated.get((it["item_id"], i + 1), row["human"])
            rows.append(row)

    nb, seed = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["seed"]
    doc = [r for r in rows if r["kind"] == "document"]
    subsets = {
        "document_claims": doc,
        "all_claims": rows,
        "document_claims_faithful": [r for r in doc if not r["human_misrepresents"]],
        **{
            f"document_claims__{m}": [r for r in doc if r["model_key"] == m]
            for m in v["n_per_generator"]
        },
    }
    pairs = [
        ("human", "judge"),
        ("human", "second_judge"),
        ("judge", "second_judge"),
        ("judge", "judge_retest"),
    ]
    if adjudicated is not None:
        pairs += [("human_adjudicated", "judge"), ("human_adjudicated", "second_judge")]
    results = {
        f"{a}__vs__{b}": {name: compare(rs, a, b, nb, seed) for name, rs in subsets.items()}
        for a, b in pairs
    }
    out = {
        "meta": {
            "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "single_run": True,
            "sample_sha256": sample["sample_sha256"],
            "n_answers": len(items),
            "n_claims": len(rows),
            "n_document_claims": len(doc),
            "raters": {
                "human": "the author, blind to judge verdicts, gold answers, citations and model",
                "judge": cfg["judge"]["models"][cfg["judge"]["model"]],
                "second_judge": cfg["judge"]["models"][v["second_judge"]],
                "judge_retest": "the primary judge re-run once with the cache bypassed",
            },
            "judge_prompts": {
                purpose: {"id": p.id, "version": p.version, "file_sha256": p.file_sha256}
                for purpose, p in judges["judge"].prompts.items()
            },
            "judge_failures": failures,
            "api_cost_usd": round(sum(j.new_cost_usd for j in judges.values()), 6),
            "adjudication": (
                {
                    "file": ADJUDICATION_PATH.as_posix(),
                    "blind": False,
                    "n_claims_changed": len(adjudicated),
                    "scope": adjudication["scope"],
                    "basis": adjudication["basis"],
                }
                if adjudicated is not None
                else None
            ),
        },
        "decomposition_quality": {
            "claims_flagged_misrepresenting": sum(r["human_misrepresents"] for r in rows),
            "answers_flagged_missing_content": sum(
                human["items"][it["item_id"]]["misses_content"] for it in items
            ),
        },
        "agreement": results,
        "claims": rows,
    }
    (RESULTS_DIR / "judge_agreement.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RESULTS_DIR / "judge_agreement.md").write_text(markdown_table(out), encoding="utf-8")
    print(markdown_table(out))


def _k(s: dict) -> str:
    if s["kappa"] is None:
        return "undefined"
    ci = s["kappa_ci95"]
    return f"{s['kappa']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{s['kappa']:.2f}"


def markdown_table(out: dict) -> str:
    lines = [
        "# Judge agreement",
        "",
        f"Single run. {out['meta']['n_answers']} answers, {out['meta']['n_claims']} claims "
        f"({out['meta']['n_document_claims']} document claims); generated by "
        "`scripts/06b_judge_validation.py agreement` from `judge_agreement.json`.",
        "",
        "`human` is the blind labelling, the primary measurement."
        + (
            f" `human_adjudicated` is not blind: {out['meta']['adjudication']['n_claims_changed']}"
            " claims were changed to the primary judge's verdict by the labeller's decision after"
            " seeing the verdicts (`results/labels/groundedness_adjudication.json`), so its"
            " agreement with the primary judge is an upper bound; the second judge was not"
            " consulted in the changes."
            if out["meta"]["adjudication"]
            else ""
        ),
        "",
        "| Comparison | Claims | Level | n | kappa [95% CI] | raw agreement "
        "| share supported (A / B) |",
        "|---|---|---|---|---|---|---|",
    ]
    for pair, subsets in out["agreement"].items():
        a, b = pair.split("__vs__")
        for subset, levels in subsets.items():
            for level, s in levels.items():
                if s["n"] == 0:
                    continue
                pos = True if level != "claim_3class" else "supported"
                sa = s["marginals_a"].get(pos, 0) / s["n"]
                sb = s["marginals_b"].get(pos, 0) / s["n"]
                lines.append(
                    f"| {a} vs {b} | {subset} | {level} | {s['n']} | {_k(s)} | "
                    f"{s['raw_agreement']:.2f} | {sa:.2f} / {sb:.2f} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["sample", "agreement"])
    args = parser.parse_args()
    load_dotenv()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.command == "sample":
        cmd_sample(cfg)
    else:
        cmd_agreement(cfg)


if __name__ == "__main__":
    main()
