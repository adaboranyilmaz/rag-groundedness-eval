"""Build FAISS and Qdrant indices for every (chunking_strategy, embedding_model) config in
configs/index_configs.yaml, and measure build time, index size, and query latency (p50/p95)
for each. Writes results/metrics/index_stats.json.

Embeddings are computed once per (chunking_strategy, embedding_model) and cached to disk
(data/processed/embeddings/), then reused to build both backends -- re-embedding ~500k
chunks on every rerun would be the single most wasteful thing this script could do.

Usage:
    uv run python scripts/03_build_index.py --all
    uv run python scripts/03_build_index.py --chunking-strategy fixed_size \
        --embedding-model bge-small-en-v1.5 --backend faiss
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.embeddings import MODEL_REGISTRY, EmbeddingModel
from src.retrieval.vectorstore import get_vectorstore

CONFIG_PATH = Path("configs/index_configs.yaml")
CHUNKS_DIR = Path("data/processed/chunks")
EMBEDDINGS_DIR = Path("data/processed/embeddings")
INDICES_DIR = Path("data/indices")
RESULTS_DIR = Path("results/metrics")
FINANCEBENCH_PATH = Path("data/raw/financebench/financebench_merged.jsonl")
QDRANT_CONTAINER = "rag-groundedness-eval-qdrant-1"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_chunks(strategy: str) -> list[dict]:
    path = CHUNKS_DIR / f"{strategy}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def get_or_build_embeddings(
    strategy: str, model_name: str, chunks: list[dict], embed_model: EmbeddingModel
) -> np.ndarray:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EMBEDDINGS_DIR / f"{strategy}__{model_name}.npy"
    if cache_path.exists():
        vectors = np.load(cache_path)
        if vectors.shape[0] == len(chunks):
            return vectors
        print(f"  cache size mismatch for {cache_path.name}, recomputing")

    print(f"  embedding {len(chunks)} chunks with {model_name}...")
    texts = [c["text"] for c in chunks]
    vectors = embed_model.encode_corpus(texts)
    np.save(cache_path, vectors)
    return vectors


def load_benchmark_queries(n: int, known_doc_ids: set[str]) -> list[str]:
    rows = [json.loads(line) for line in FINANCEBENCH_PATH.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["doc_name"] in known_doc_ids]
    rows.sort(key=lambda r: r["financebench_id"])
    return [r["question"] for r in rows[:n]]


def _docker_exe() -> str:
    found = shutil.which("docker")
    if found:
        return found
    fallback = Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe")
    if fallback.exists():
        return str(fallback)
    return "docker"


def measure_qdrant_disk_bytes(collection_name: str) -> tuple[int | None, str | None]:
    """Real measured bytes via `docker exec ... du`, not an analytical estimate --
    Qdrant's client API doesn't expose on-disk size directly, and a naive
    n_vectors*dim*4 estimate would understate real HNSW/payload/WAL overhead
    (Hard Constraint #1: no fabricated numbers)."""
    try:
        result = subprocess.run(
            [
                _docker_exe(),
                "exec",
                QDRANT_CONTAINER,
                "du",
                "-sb",
                f"/qdrant/storage/collections/{collection_name}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return int(result.stdout.split()[0]), None
    except Exception as e:  # noqa: BLE001 - measurement is best-effort, must not crash the build
        return None, f"docker exec du failed: {e}"


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def percentiles(latencies_ms: list[float]) -> dict:
    s = sorted(latencies_ms)
    return {
        "p50_ms": statistics.median(s),
        "p95_ms": s[max(0, int(len(s) * 0.95) - 1)],
    }


def build_one(
    strategy: str,
    model_name: str,
    backend: str,
    chunks: list[dict],
    vectors: np.ndarray,
    embed_model: EmbeddingModel,
    query_texts: list[str],
    query_vectors: np.ndarray,
    bench_cfg: dict,
) -> dict:
    ids = [c["chunk_id"] for c in chunks]
    config_name = f"{strategy}__{model_name}"

    t0 = time.perf_counter()
    if backend == "faiss":
        store = get_vectorstore("faiss", dim=embed_model.dim)
        store.add(ids, vectors, chunks)
        build_time = time.perf_counter() - t0
        persist_dir = INDICES_DIR / f"{config_name}__faiss"
        store.persist(persist_dir)
        size_bytes = dir_size_bytes(persist_dir)
        size_note = None
    else:
        collection_name = config_name.replace(".", "_")
        store = get_vectorstore(
            "qdrant", dim=embed_model.dim, collection_name=collection_name, recreate=True
        )
        store.add(ids, vectors, chunks)
        build_time = time.perf_counter() - t0
        persist_dir = INDICES_DIR / f"{config_name}__qdrant"
        store.persist(persist_dir)
        size_bytes, size_note = measure_qdrant_disk_bytes(collection_name)

    k = bench_cfg["k"]
    n_repeats = bench_cfg["n_repeats"]
    latencies_ms: list[float] = []
    all_results = []
    for _ in range(n_repeats):
        for qv in query_vectors:
            t0 = time.perf_counter()
            results = store.search(qv, k=k)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            all_results.append(results)

    return {
        "build_time_sec": build_time,
        "n_vectors": len(ids),
        "index_size_bytes": size_bytes,
        "index_size_note": size_note,
        "query_latency": percentiles(latencies_ms),
        "n_latency_samples": len(latencies_ms),
        # first-repeat results only, used by the equivalence check
        "top_k_per_query": [[r.chunk_id for r in res] for res in all_results[: len(query_vectors)]],
    }


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunking-strategy", choices=cfg["chunking_strategies"])
    parser.add_argument("--embedding-model", choices=list(MODEL_REGISTRY))
    parser.add_argument("--backend", choices=["faiss", "qdrant"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        strategies = cfg["chunking_strategies"]
        model_names = cfg["embedding_models"]
        backends = cfg["backends"]
    else:
        if not (args.chunking_strategy and args.embedding_model and args.backend):
            parser.error("specify --all, or all of --chunking-strategy/--embedding-model/--backend")
        strategies = [args.chunking_strategy]
        model_names = [args.embedding_model]
        backends = [args.backend]

    bench_cfg = cfg["latency_benchmark"]

    # doc_ids actually present in the corpus, so benchmark queries have real candidate chunks
    any_strategy_chunks = load_chunks(strategies[0])
    known_doc_ids = {c["doc_id"] for c in any_strategy_chunks}
    query_texts = load_benchmark_queries(bench_cfg["n_queries"], known_doc_ids)
    print(f"Loaded {len(query_texts)} benchmark queries")

    results: dict = {"meta": {}, "configs": {}}
    equivalence: dict = {}

    for strategy in strategies:
        chunks = load_chunks(strategy)
        for model_name in model_names:
            print(f"\n=== {strategy} / {model_name} ===")
            embed_model = EmbeddingModel(model_name)
            vectors = get_or_build_embeddings(strategy, model_name, chunks, embed_model)
            query_vectors = embed_model.encode_queries(query_texts)

            results["meta"][model_name] = {
                "hf_id": MODEL_REGISTRY[model_name].hf_id,
                "dim": MODEL_REGISTRY[model_name].dim,
                "device": embed_model.device,
            }

            per_backend = {}
            for backend in backends:
                print(f"  building {backend}...")
                stats = build_one(
                    strategy,
                    model_name,
                    backend,
                    chunks,
                    vectors,
                    embed_model,
                    query_texts,
                    query_vectors,
                    bench_cfg,
                )
                per_backend[backend] = stats
                print(
                    f"    build_time={stats['build_time_sec']:.1f}s "
                    f"size={stats['index_size_bytes']} "
                    f"p50={stats['query_latency']['p50_ms']:.2f}ms "
                    f"p95={stats['query_latency']['p95_ms']:.2f}ms"
                )

            config_key = f"{strategy}__{model_name}"
            results["configs"][config_key] = {
                backend: {k: v for k, v in stats.items() if k != "top_k_per_query"}
                for backend, stats in per_backend.items()
            }

            if "faiss" in per_backend and "qdrant" in per_backend:
                faiss_tk = per_backend["faiss"]["top_k_per_query"]
                qdrant_tk = per_backend["qdrant"]["top_k_per_query"]
                matches = sum(1 for a, b in zip(faiss_tk, qdrant_tk, strict=True) if a == b)
                equivalence[config_key] = {
                    "n_queries": len(faiss_tk),
                    "exact_top_k_matches": matches,
                    "match_rate": matches / len(faiss_tk) if faiss_tk else None,
                }

            # Write after every (strategy, model) pair, not just at the end -- this grid
            # takes long enough that losing all progress to a late failure would be costly.
            results["equivalence_check"] = equivalence
            results["n_benchmark_queries"] = len(query_texts)
            out_path = RESULTS_DIR / "index_stats.json"
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"  wrote (partial) {out_path}")

    print(f"\nDone. Final results at {RESULTS_DIR / 'index_stats.json'}")


if __name__ == "__main__":
    main()
