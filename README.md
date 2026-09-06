# RAG Reliability & Evaluation Framework for Financial Documents

*When a RAG system answers a financial question confidently, can we tell whether the
answer is actually grounded in the documents it retrieved?*

> Status: Phase 0 (scaffolding). This is a stub — the full README is written last, per
> `PROJECT_SPEC.md` §8, once results exist to report. Every number below is `TBD` until a
> committed script in `results/` produces it.

## Results (placeholder)

| Metric | Value | Source |
|---|---|---|
| Corpus size (docs / chunks) | TBD | `results/metrics/corpus_stats.json` |
| Retrieval recall@5 (best config) | TBD | `results/metrics/retrieval_grid.json` |
| Groundedness score (mean) | TBD | `results/metrics/eval_main.json` |
| Judge agreement (Cohen's κ) | TBD | `results/metrics/eval_main.json` |
| Retrieval↔groundedness correlation | TBD | `results/metrics/eval_main.json` |
| Load test p95 latency | TBD | `results/metrics/load_test.json` |

See `PROJECT_SPEC.md` for the full project design and phase plan.
