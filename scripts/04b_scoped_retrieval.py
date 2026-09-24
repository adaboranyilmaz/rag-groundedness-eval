"""Retrieval quality of the generation configuration under two search scopes:

  corpus  search all filings (the Phase 3 setting; generation's main condition)
  filing  search only the chunks of the question's own filing, as a user filtering to
          one 10-K would. Every FinanceBench evidence item lies in its question's own
          filing, so the gold evidence is always reachable in this scope.

Same retriever, same k, same gold spans and relevance rule as Phase 3: the question
loading and evidence alignment are Phase 3's own functions, imported from
04_eval_retrieval.py. As a correctness check, the corpus scope is recomputed here through
the same FAISS path and must reproduce Phase 3's stored top ids for every question before
anything is written.

Writes results/metrics/retrieval_scoped.json.

Usage:
    uv run python scripts/04b_scoped_retrieval.py
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.retrieval_metrics import (
    ChunkSpan,
    count_relevant,
    mean_metrics,
    paired_bootstrap_diff,
    question_metrics,
)
from src.retrieval.embeddings import MODEL_REGISTRY, EmbeddingModel
from src.retrieval.scoped import ScopedFaissIndex

GEN_CONFIG = Path("configs/generation.yaml")
PHASE3_CONFIG = Path("configs/retrieval_grid.yaml")
CHUNKS_DIR = Path("data/processed/chunks")
INDICES_DIR = Path("data/indices")
RESULTS_DIR = Path("results/metrics")
PHASE3_PER_QUESTION = RESULTS_DIR / "retrieval_per_question.jsonl"


def load_phase3_module():
    path = Path(__file__).resolve().parent / "04_eval_retrieval.py"
    spec = importlib.util.spec_from_file_location("eval_retrieval_phase3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p3 = load_phase3_module()
    rc = yaml.safe_load(GEN_CONFIG.read_text(encoding="utf-8"))["retrieval"]
    cfg3 = yaml.safe_load(PHASE3_CONFIG.read_text(encoding="utf-8"))
    if rc["method"] != "dense" or rc["backend"] != "faiss":
        raise NotImplementedError("scoped retrieval is implemented for dense FAISS only")
    k_values, min_ov = cfg3["k_values"], cfg3["relevance"]["min_overlap_frac"]
    k_max, corpus_depth = max(k_values), rc["search_depth"]
    cell = f"{rc['chunking']}__{rc['embedding']}__{rc['method']}"

    questions, n_benchmark = p3.load_questions(None)
    golds, _ = p3.align_all(questions, cfg3)
    scored = [q for q in questions if golds[q["financebench_id"]]]
    print(f"{len(scored)} scored questions (of {n_benchmark})")

    chunks = [
        json.loads(line)
        for line in (CHUNKS_DIR / f"{rc['chunking']}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_doc: dict[str, list[ChunkSpan]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(p3.chunk_span(c))

    index = ScopedFaissIndex(INDICES_DIR / f"{rc['chunking']}__{rc['embedding']}__{rc['backend']}")

    phase3_top = {}
    for line in PHASE3_PER_QUESTION.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["cell"] == cell:
            phase3_top[row["financebench_id"]] = row["top_ids"]

    embed = EmbeddingModel(rc["embedding"])
    per_q: dict[str, list[dict]] = {"corpus": [], "filing": []}
    rows = []
    for q in scored:
        qid = q["financebench_id"]
        gs = golds[qid]
        n_relevant = count_relevant(by_doc, gs, min_ov)
        qv = embed.encode_queries([q["question"]])

        # corpus scope: same depth-then-truncate path as generation (see generation.yaml)
        corpus_ranked = index.search_corpus(qv, corpus_depth)[:k_max]
        stored = phase3_top[qid][:k_max]
        if [r.chunk_id for r in corpus_ranked][: len(stored)] != stored:
            raise RuntimeError(f"corpus scope does not reproduce Phase 3 for {qid}")

        # filing scope: exact inner-product search restricted to the question's filing
        filing_ranked = index.search_filing(qv, q["doc_name"], k_max)

        row = {
            "financebench_id": qid,
            "doc_name": q["doc_name"],
            "n_filing_chunks": len(index.positions_by_doc[q["doc_name"]]),
        }
        for scope, ranked in (("corpus", corpus_ranked), ("filing", filing_ranked)):
            spans = [p3.chunk_span(r.metadata) for r in ranked]
            m = question_metrics(spans, gs, n_relevant, k_values, min_ov)
            per_q[scope].append(m)
            row[scope] = {"metrics": m, "top_ids": [r.chunk_id for r in ranked[:10]]}
        rows.append(row)

    summary = {}
    for scope in ("corpus", "filing"):
        means = mean_metrics(per_q[scope])
        summary[scope] = {
            "metrics": means,
            "n_questions_recall5_gt0": sum(m["recall@5"] > 0 for m in per_q[scope]),
            "n_questions_all_gold_in_top5": sum(m["recall@5"] == 1.0 for m in per_q[scope]),
        }
    comparisons = {
        metric: paired_bootstrap_diff(
            [m[metric] for m in per_q["filing"]],
            [m[metric] for m in per_q["corpus"]],
            n_resamples=cfg3["selection"]["bootstrap_resamples"],
            seed=cfg3["selection"]["seed"],
        )
        for metric in ("recall@5", "span_recall@5", "mrr")
    }
    out = {
        "meta": {
            "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "single_run": True,
            "config": cell,
            "embedding_model": MODEL_REGISTRY[rc["embedding"]].hf_id,
            "k_values": k_values,
            "corpus_search_depth": corpus_depth,
            "relevance_min_overlap_frac": min_ov,
            "n_questions_scored": len(scored),
            "corpus_scope_reproduces_phase3": True,  # enforced above; the script stops if not
            "filing_chunks_per_question": {
                "min": min(r["n_filing_chunks"] for r in rows),
                "median": statistics.median(r["n_filing_chunks"] for r in rows),
                "max": max(r["n_filing_chunks"] for r in rows),
            },
            "comparison_note": "paired bootstrap over questions of filing minus corpus",
        },
        "scopes": summary,
        "filing_minus_corpus": comparisons,
        "per_question": rows,
    }
    path = RESULTS_DIR / "retrieval_scoped.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for scope in ("corpus", "filing"):
        m = summary[scope]["metrics"]
        print(
            f"{scope:7s} recall@5 {m['recall@5']:.3f}  recall@20 {m['recall@20']:.3f}  "
            f"span_recall@5 {m['span_recall@5']:.3f}  mrr {m['mrr']:.3f}  "
            f"questions with recall@5>0: {summary[scope]['n_questions_recall5_gt0']}"
        )
    c = comparisons["recall@5"]
    print(
        f"filing - corpus recall@5: {c['diff']:+.3f} [{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}]"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
