# Chunking strategies and financial tables

Computed from `results/metrics/chunking_stats.json` — regenerate this file by
re-running `scripts/02_chunk.py` rather than editing it by hand.

## fixed_size

75951 chunks, mean length 507 chars (median 508). Of 7966 source table blocks in the corpus: 6831 were split across more than one chunk, 1135 were merged into a chunk alongside adjacent narrative text, and 0 were preserved as their own standalone chunk.

## recursive_structural

39593 chunks, mean length 848 chars (median 849). Of 7966 source table blocks in the corpus: 0 were split across more than one chunk, 5146 were merged into a chunk alongside adjacent narrative text, and 2820 were preserved as their own standalone chunk.

## table_aware

44657 chunks, mean length 752 chars (median 754). Of 7966 source table blocks in the corpus: 0 were split across more than one chunk, 0 were merged into a chunk alongside adjacent narrative text, and 7966 were preserved as their own standalone chunk.

`fixed_size` is a blind character window, so a table that happens to fall near a window boundary gets sliced mid-row — that split count is the direct cost of ignoring document structure. `recursive_structural` respects block boundaries so it never cuts a table mid-row, but it can still pack a small table into the same chunk as surrounding prose, diluting it. `table_aware` forces every detected table into its own chunk regardless of size, at the cost of sometimes producing a very small or very large standalone chunk.

## Gold-figure spot check

Found 8/10 FinanceBench gold figures verbatim in the `table_aware` chunks for their source document, restricted to questions FinanceBench tags `question_reasoning: Information extraction` (a single figure stated directly in the filing, as opposed to a ratio or growth rate computed from several figures — a computed answer was never printed anywhere in the source text, so checking for it verbatim would be a false negative, not a parsing failure).

Misses in this sample: AMCOR_2020_10K (answer $1616.00); BLOCK_2020_10K (answer $382.00). In each case checked by hand, the underlying figure is present in the filing but differs from the gold answer's exact string by rounding (e.g. a filing's `1,615.9` vs. a gold answer rounded to `1616`) or by scale (a filing reporting in thousands against a gold answer stated in millions) — exactly the class of discrepancy PROJECT_SPEC.md's Phase 5 numeric-tolerance matching exists to handle, not evidence of corrupted parsing. See `results/metrics/chunking_stats.json`'s `gold_figure_spot_check` for the full per-question detail.
