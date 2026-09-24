"""Evaluate retrieval over the full Phase 3 grid — chunking strategy x embedding model x
retrieval method (dense, BM25, hybrid RRF, hybrid + cross-encoder rerank) — against
FinanceBench gold evidence aligned to character spans in the parsed filings.

Writes:
  results/metrics/gold_span_alignment.json    how every gold evidence item was aligned
  results/metrics/retrieval_grid.json         per-cell mean metrics + latency, winner
                                              selection with paired bootstrap, and a
                                              main-effect decomposition of the grid
  results/metrics/retrieval_per_question.jsonl  per-(cell, question) metrics + top ids

All settings come from configs/retrieval_grid.yaml and are copied into the output. Every
number is a single deterministic run over the question set (no seeds are involved in
retrieval itself; the seed only drives the bootstrap).

Usage:
    uv run python scripts/04_eval_retrieval.py
    uv run python scripts/04_eval_retrieval.py --max-questions 5   # smoke run
    uv run python scripts/04_eval_retrieval.py --analyse-only       # re-derive analysis
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import rank_bm25
import sentence_transformers
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.gold_spans import GoldSpan, NormalisedText, align_evidence
from src.evaluation.retrieval_metrics import (
    ChunkSpan,
    factor_decomposition,
    is_relevant,
    mean_metrics,
    paired_bootstrap_diff,
    question_metrics,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.embeddings import MODEL_REGISTRY, EmbeddingModel
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.rerank import CrossEncoderReranker
from src.retrieval.vectorstore import SearchResult, get_vectorstore

CONFIG_PATH = Path("configs/retrieval_grid.yaml")
FINANCEBENCH_PATH = Path("data/raw/financebench/financebench_merged.jsonl")
PARSED_DIR = Path("data/processed/parsed")
CHUNKS_DIR = Path("data/processed/chunks")
INDICES_DIR = Path("data/indices")
RESULTS_DIR = Path("results/metrics")

FACTORS = ("chunking", "embedding", "method")


def load_questions(max_questions: int | None) -> tuple[list[dict], int]:
    rows = [json.loads(line) for line in FINANCEBENCH_PATH.read_text(encoding="utf-8").splitlines()]
    in_corpus = [r for r in rows if (PARSED_DIR / f"{r['doc_name']}.json").exists()]
    in_corpus.sort(key=lambda r: r["financebench_id"])
    if max_questions is not None:
        in_corpus = in_corpus[:max_questions]
    return in_corpus, len(rows)


def align_all(questions: list[dict], cfg: dict) -> tuple[dict[str, list[GoldSpan]], dict]:
    """Align every evidence item; return gold spans per question plus the alignment report."""
    normalised: dict[str, NormalisedText] = {}
    golds: dict[str, list[GoldSpan]] = {}
    items = []
    for q in questions:
        spans = []
        for idx, ev in enumerate(q["evidence"]):
            doc_id = ev.get("doc_name", q["doc_name"])
            if doc_id not in normalised:
                parsed = json.loads((PARSED_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
                normalised[doc_id] = NormalisedText(parsed["full_text"])
            al = align_evidence(
                ev["evidence_text"],
                doc_id,
                normalised[doc_id],
                window_chars=cfg["alignment"]["window_chars"],
                min_window_coverage=cfg["alignment"]["min_window_coverage"],
            )
            items.append(
                {
                    "financebench_id": q["financebench_id"],
                    "evidence_index": idx,
                    "doc_id": doc_id,
                    "status": al.status,
                    "window_coverage": round(al.window_coverage, 4),
                    "n_occurrences": al.n_occurrences,
                    "char_start": al.span.char_start if al.span else None,
                    "char_end": al.span.char_end if al.span else None,
                    "evidence_chars": len(ev["evidence_text"]),
                }
            )
            if al.span is not None:
                spans.append(al.span)
        golds[q["financebench_id"]] = spans

    status_counts = {
        s: sum(1 for it in items if it["status"] == s) for s in ("exact", "partial", "unaligned")
    }
    n_all = sum(1 for q in questions if len(golds[q["financebench_id"]]) == len(q["evidence"]))
    n_none = sum(1 for q in questions if not golds[q["financebench_id"]])
    report = {
        "method": (
            "text alignment in a normalised space (lowercase ASCII letters+digits); "
            "see src/evaluation/gold_spans.py"
        ),
        "settings": cfg["alignment"],
        "n_questions": len(questions),
        "n_evidence_items": len(items),
        "status_counts": status_counts,
        "exact_with_multiple_occurrences": sum(1 for it in items if it["n_occurrences"] > 1),
        "questions_all_items_aligned": n_all,
        "questions_some_items_aligned": len(questions) - n_all - n_none,
        "questions_no_items_aligned": n_none,
        "items": items,
    }
    return golds, report


def chunk_span(c: dict) -> ChunkSpan:
    return ChunkSpan(c["chunk_id"], c["doc_id"], c["char_start"], c["char_end"])


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[max(0, int(round(q * len(s))) - 1)]


def latency_summary(values_ms: list[float]) -> dict:
    return {
        "p50_ms": statistics.median(values_ms),
        "p95_ms": percentile(values_ms, 0.95),
        "mean_ms": statistics.fmean(values_ms),
    }


# Subsets of the grid for the main-effect decomposition, with the factors that vary in each.
DECOMPOSITION_SUBSETS = (
    ("all_cells", lambda c: True, FACTORS),
    # BM25 ignores the embedding model by construction, so in the full grid it dilutes the
    # embedding effect; this subset measures it without that.
    ("embedding_dependent_methods", lambda c: c["method"] != "bm25", FACTORS),
    # Holding the method fixed at dense isolates the chunking-vs-embedding question the
    # spec poses, without the retrieval-method effect swamping both.
    ("dense_only", lambda c: c["method"] == "dense", ("chunking", "embedding")),
)
DECOMPOSITION_METRICS = ("recall@5", "span_recall@5", "mrr")


def analyse(cells: dict, per_question_selection: dict[str, list[float]], cfg: dict) -> dict:
    """Winner selection with paired bootstrap against the runners-up, plus the main-effect
    decomposition. Pure function of the cell metrics and per-question selection scores
    (paired by question order), so it can be recomputed from saved results."""
    sel = cfg["selection"]
    metric = sel["metric"]
    ranking = sorted(
        cells, key=lambda n: (-cells[n]["metrics"][metric], -cells[n]["metrics"]["mrr"], n)
    )
    winner = ranking[0]
    runners_up = []
    for other in ranking[1 : 1 + sel["n_runners_up"]]:
        boot = paired_bootstrap_diff(
            per_question_selection[winner],
            per_question_selection[other],
            n_resamples=sel["bootstrap_resamples"],
            seed=sel["seed"],
        )
        runners_up.append({"cell": other, metric: cells[other]["metrics"][metric], **boot})

    decomposition: dict = {}
    for label, keep, factors in DECOMPOSITION_SUBSETS:
        subset = [n for n in cells if keep(cells[n])]
        decomposition[label] = {}
        for m in DECOMPOSITION_METRICS:
            d = factor_decomposition(
                [{f: cells[n][f] for f in factors} for n in subset],
                [cells[n]["metrics"][m] for n in subset],
                factors,
            )
            d["n_cells"] = len(subset)
            d["largest_factor_by_ss_share"] = max(
                d["factors"], key=lambda f: d["factors"][f]["ss_share"]
            )
            decomposition[label][m] = d

    return {
        "ranking": [
            {
                "rank": i + 1,
                "cell": n,
                metric: cells[n]["metrics"][metric],
                "mrr": cells[n]["metrics"]["mrr"],
            }
            for i, n in enumerate(ranking)
        ],
        "selection": {"metric": metric, "winner": winner, "runners_up": runners_up},
        "factor_decomposition": decomposition,
    }


def analyse_only(cfg: dict) -> None:
    """Recompute the analysis sections of retrieval_grid.json from the saved per-question
    results, without re-running retrieval."""
    grid_path = RESULTS_DIR / "retrieval_grid.json"
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    metric = cfg["selection"]["metric"]
    per_q: dict[str, list[float]] = {}
    lines = (RESULTS_DIR / "retrieval_per_question.jsonl").read_text(encoding="utf-8")
    for line in lines.splitlines():
        row = json.loads(line)
        per_q.setdefault(row["cell"], []).append(row["metrics"][metric])
    grid.update(analyse(grid["cells"], per_q, cfg))
    grid["meta"]["analysis_recomputed_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    grid_path.write_text(json.dumps(grid, indent=2), encoding="utf-8")
    print(f"recomputed analysis in {grid_path}; winner {grid['selection']['winner']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=None, help="smoke-run subset")
    parser.add_argument(
        "--analyse-only",
        action="store_true",
        help="recompute selection + decomposition from saved per-question results",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.analyse_only:
        analyse_only(cfg)
        return
    k_values = cfg["k_values"]
    k_max = max(k_values)
    depth = cfg["hybrid"]["candidate_depth"]
    rerank_depth = cfg["rerank"]["depth"]
    min_ov = cfg["relevance"]["min_overlap_frac"]
    if not k_max <= rerank_depth <= depth:
        raise ValueError("config needs max(k_values) <= rerank.depth <= hybrid.candidate_depth")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    smoke = args.max_questions is not None
    suffix = "_smoke" if smoke else ""

    questions, n_benchmark = load_questions(args.max_questions)
    print(f"{len(questions)} questions with filings in the corpus (of {n_benchmark})")
    golds, alignment_report = align_all(questions, cfg)
    print(f"alignment: {alignment_report['status_counts']}")
    (RESULTS_DIR / f"gold_span_alignment{suffix}.json").write_text(
        json.dumps(alignment_report, indent=2), encoding="utf-8"
    )
    scored = [q for q in questions if golds[q["financebench_id"]]]
    print(f"{len(scored)} questions have >=1 aligned gold span and are scored")

    reranker = CrossEncoderReranker(
        cfg["rerank"]["model"],
        max_length=cfg["rerank"]["max_length"],
        batch_size=cfg["rerank"]["batch_size"],
    )

    cells: dict[str, dict] = {}
    per_question_lines: list[str] = []
    strategy_stats: dict[str, dict] = {}

    sel_metric = cfg["selection"]["metric"]

    def record_cell(strategy, model_name, method, rankings, latencies, n_relevant, extra=None):
        name = f"{strategy}__{model_name}__{method}"
        per_q = []
        for q in scored:
            qid = q["financebench_id"]
            ranked = rankings[qid][:k_max]
            m = question_metrics(
                [chunk_span(r.metadata) for r in ranked],
                golds[qid],
                n_relevant[qid],
                k_values,
                min_ov,
            )
            per_q.append(m)
            per_question_lines.append(
                json.dumps(
                    {
                        "cell": name,
                        "financebench_id": qid,
                        "metrics": m,
                        "top_ids": [r.chunk_id for r in ranked[: cfg["store_top_n_ids"]]],
                    }
                )
            )
        cells[name] = {
            "chunking": strategy,
            "embedding": model_name,
            "method": method,
            "n_questions": len(per_q),
            "metrics": mean_metrics(per_q),
            "latency": latency_summary(latencies),
            **(extra or {}),
        }
        cells[name]["_per_question_selection"] = [m[sel_metric] for m in per_q]

    for strategy in cfg["chunking_strategies"]:
        print(f"\n=== {strategy} ===")
        chunks = [
            json.loads(line)
            for line in (CHUNKS_DIR / f"{strategy}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        by_doc: dict[str, list[ChunkSpan]] = {}
        for c in chunks:
            by_doc.setdefault(c["doc_id"], []).append(chunk_span(c))

        # nDCG ideal and the relevance ceiling: how many chunks of this strategy *could*
        # count as relevant, and whether each gold span is reachable at all under the
        # relevance rule (a strategy whose chunks straddle a gold span badly can make it
        # unreachable, which is a chunking failure the retriever cannot fix).
        n_relevant: dict[str, int] = {}
        reachable = total_golds = 0
        for q in scored:
            qid = q["financebench_id"]
            gs = golds[qid]
            cands = [c for d in {g.doc_id for g in gs} for c in by_doc.get(d, [])]
            n_relevant[qid] = sum(1 for c in cands if any(is_relevant(c, g, min_ov) for g in gs))
            for g in gs:
                total_golds += 1
                reachable += any(is_relevant(c, g, min_ov) for c in by_doc.get(g.doc_id, []))
        lengths = [c["char_end"] - c["char_start"] for c in chunks]
        strategy_stats[strategy] = {
            "n_chunks": len(chunks),
            "mean_chunk_chars": statistics.fmean(lengths),
            "median_chunk_chars": statistics.median(lengths),
            "mean_relevant_chunks_per_question": statistics.fmean(n_relevant.values()),
            "gold_spans_reachable": reachable,
            "gold_spans_total": total_golds,
        }
        print(f"  {strategy_stats[strategy]}")

        print("  building BM25...")
        t0 = time.perf_counter()
        bm25 = BM25Retriever(chunks, k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"])
        strategy_stats[strategy]["bm25_build_sec"] = time.perf_counter() - t0
        bm25_rank: dict[str, list[SearchResult]] = {}
        bm25_lat: dict[str, float] = {}
        for q in scored:
            t0 = time.perf_counter()
            bm25_rank[q["financebench_id"]] = bm25.retrieve(q["question"], depth)
            bm25_lat[q["financebench_id"]] = (time.perf_counter() - t0) * 1000
        del bm25
        gc.collect()

        for model_name in cfg["embedding_models"]:
            print(f"  --- {model_name} ---")
            embed_model = EmbeddingModel(model_name)
            store = get_vectorstore(cfg["backend"], dim=embed_model.dim)
            store.load(INDICES_DIR / f"{strategy}__{model_name}__{cfg['backend']}")
            dense = DenseRetriever(store, embed_model)
            dense.retrieve("warm-up query", 1)  # exclude CUDA/lazy-init cost from latency
            reranker.rerank("warm-up query", dense.retrieve("warm-up query", 2), 2)

            rank = {m: {} for m in ("dense", "hybrid", "hybrid_rerank")}
            lat = {m: [] for m in ("dense", "bm25", "hybrid", "hybrid_rerank")}
            for q in scored:
                qid, text = q["financebench_id"], q["question"]
                t0 = time.perf_counter()
                dense_res = dense.retrieve(text, depth)
                t_dense = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                fused = reciprocal_rank_fusion([dense_res, bm25_rank[qid]], cfg["hybrid"]["k_rrf"])
                t_fuse = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                reranked = reranker.rerank(text, fused[:rerank_depth], rerank_depth)
                t_rerank = (time.perf_counter() - t0) * 1000

                rank["dense"][qid] = dense_res
                rank["hybrid"][qid] = fused
                rank["hybrid_rerank"][qid] = reranked
                # Latency of each method as a pipeline run sequentially on this machine:
                # hybrid = dense + BM25 + fusion; rerank adds the cross-encoder pass.
                t_hybrid = t_dense + bm25_lat[qid] + t_fuse
                lat["dense"].append(t_dense)
                lat["bm25"].append(bm25_lat[qid])
                lat["hybrid"].append(t_hybrid)
                lat["hybrid_rerank"].append(t_hybrid + t_rerank)

            for method in cfg["methods"]:
                if method == "bm25":
                    record_cell(
                        strategy,
                        model_name,
                        "bm25",
                        bm25_rank,
                        lat["bm25"],
                        n_relevant,
                        {"embedding_independent": True},
                    )
                else:
                    record_cell(strategy, model_name, method, rank[method], lat[method], n_relevant)
                c = cells[f"{strategy}__{model_name}__{method}"]
                print(
                    f"    {method:14s} recall@5={c['metrics']['recall@5']:.3f} "
                    f"mrr={c['metrics']['mrr']:.3f} p50={c['latency']['p50_ms']:.0f}ms"
                )

            del dense, store, embed_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    per_question_selection = {n: c.pop("_per_question_selection") for n, c in cells.items()}
    analysis = analyse(cells, per_question_selection, cfg)

    output = {
        "meta": {
            "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "single_run": True,
            "smoke_run": smoke,
            "config": cfg,
            "n_benchmark_questions": n_benchmark,
            "n_questions_in_corpus": len(questions),
            "n_questions_scored": len(scored),
            "mrr_cutoff": k_max,
            "embedding_models": {m: MODEL_REGISTRY[m].hf_id for m in cfg["embedding_models"]},
            "reranker": cfg["rerank"]["model"],
            "device": reranker.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "rank_bm25": getattr(rank_bm25, "__version__", "0.2.2 (no __version__)"),
            },
            "latency_note": (
                "per-query wall-clock on this machine, methods run sequentially; bm25 is "
                "rank_bm25's pure-Python scoring over all chunks of the strategy; hybrid = "
                "dense + bm25 + fusion; hybrid_rerank adds the cross-encoder over the top "
                f"{rerank_depth} fused candidates"
            ),
        },
        "strategy_stats": strategy_stats,
        "cells": cells,
        **analysis,
    }
    (RESULTS_DIR / f"retrieval_grid{suffix}.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / f"retrieval_per_question{suffix}.jsonl").write_text(
        "\n".join(per_question_lines) + "\n", encoding="utf-8"
    )
    winner = analysis["selection"]["winner"]
    print(f"\nwinner by {sel_metric}: {winner} ({cells[winner]['metrics'][sel_metric]:.3f})")
    print(f"wrote {RESULTS_DIR / f'retrieval_grid{suffix}.json'}")


if __name__ == "__main__":
    main()
