"""The serving bundle, and the check that the service reproduces the evaluated answers.

  bundle   assemble build/serving/ (src/serving/service.py documents the layout): the
           selected FAISS index, the embedding weights at the pinned revision, the cached
           responses of the selected arm's 150 benchmark questions (generation plus the two
           groundedness judge calls), the selection and the judge-agreement report, with a
           manifest of sha256 hashes. The Docker image bakes it in, so the container needs
           neither DVC data nor a download at start-up. Deterministic: no timestamps.
  check    load the service from the bundle alone (replay-only, no API key, embeddings on
           CPU as in the container) and send it every benchmark question of the selected
           arm. Each answer must match its Phase 4 trace (retrieved chunks and their order,
           the generation request's cache key, the parsed answer and citations) and its
           Phase 5 evaluation (final status, both judge requests' cache keys, groundedness
           score, counts and per-claim verdicts). Retrieval scores may differ from the GPU
           ones by float rounding, up to 1e-5 -> results/metrics/serving_check.json

No API calls: `check` runs with RAG_REPLAY_ONLY=1 and with the API key removed from its own
environment, so a request missing from the bundle fails instead of being paid for.

Usage:
    uv run python scripts/12_serving.py bundle
    uv run python scripts/12_serving.py check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.embeddings import MODEL_REGISTRY  # noqa: E402

SERVING_CONFIG = Path("configs/serving.yaml")
CACHE_DIR = Path("data/cache/llm")
CHECK_PATH = Path("results/metrics/serving_check.json")
SCORE_TOL = 1e-5  # CPU vs GPU query embeddings (DECISIONS.md, Phase 7: max |diff| 1.5e-6)
GROUNDEDNESS_FIELDS = (
    "groundedness",
    "fully_grounded",
    "any_contradicted",
    "n_claims",
    "n_document_claims",
    "n_context_claims",
    "n_general_claims",
    "n_supported",
    "n_unsupported",
    "n_contradicted",
    "n_context_contradicted",
)
CLAIM_FIELDS = ("claim", "kind", "verdict", "supporting_excerpts")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_cfg() -> dict:
    return yaml.safe_load(SERVING_CONFIG.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]


def arm_records(cfg: dict) -> tuple[dict, list[dict], dict[str, dict]]:
    """(selection, the arm's traces, their evaluations by trace id), checked to be the
    selected arm's."""
    selection = json.loads(Path(cfg["selection"]).read_text(encoding="utf-8"))
    traces = read_jsonl(Path(cfg["arm_traces"]))
    prefix = selection["winner"] + "__"
    wrong = [t["trace_id"] for t in traces if not t["trace_id"].startswith(prefix)]
    if wrong:
        raise ValueError(f"{cfg['arm_traces']} is not the selected arm ({wrong[0]})")
    ids = {t["trace_id"] for t in traces}
    evals = {}
    with Path(cfg["arm_evaluations"]).open(encoding="utf-8") as f:
        for line in f:
            if prefix in line:
                rec = json.loads(line)
                if rec["trace_id"] in ids:
                    evals[rec["trace_id"]] = rec
    if set(evals) != ids:
        raise ValueError(f"{len(ids - set(evals))} traces have no evaluation")
    return selection, traces, evals


# --------------------------------------------------------------------------------------
# bundle


def copy_model(hf_id: str, revision: str, dest: Path) -> None:
    """The pinned snapshot, from the local Hugging Face cache when it holds it (as after
    Phase 2), else downloaded. Links are followed, so the bundle holds plain files."""
    from huggingface_hub import snapshot_download

    try:
        src = snapshot_download(hf_id, revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached locally: fetch that exact revision
        src = snapshot_download(hf_id, revision=revision)
    shutil.copytree(src, dest, symlinks=False)


def cmd_bundle() -> None:
    cfg = load_cfg()
    selection, traces, evals = arm_records(cfg)
    rc = selection["pipeline_config"]["retrieval"]
    out = Path(cfg["bundle_dir"])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    index_src = Path("data/indices") / f"{rc['chunking']}__{rc['embedding']}__faiss"
    (out / "index").mkdir()
    for name in ("index.faiss", "meta.json"):
        shutil.copyfile(index_src / name, out / "index" / name)

    spec = MODEL_REGISTRY[rc["embedding"]]
    copy_model(spec.hf_id, cfg["embedding_revision"], out / "model")

    keys = []
    for t in traces:
        keys.append(t["generation"]["cache_key"])
        judged = evals[t["trace_id"]]["judge"]
        keys += [judged[s] for s in ("decompose", "verify") if s in judged]
    missing = [k for k in keys if not (CACHE_DIR / k[:2] / f"{k}.json").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} responses not in {CACHE_DIR}; first {missing[0]}")
    for k in sorted(set(keys)):
        dest = out / "cache" / k[:2] / f"{k}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CACHE_DIR / k[:2] / f"{k}.json", dest)

    shutil.copyfile(cfg["selection"], out / "pipeline_selection.json")
    shutil.copyfile(cfg["judge_agreement"], out / "judge_agreement.json")

    files = {
        p.relative_to(out).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    listing = "\n".join(f"{name} {f['sha256']}" for name, f in files.items())
    manifest = {
        "bundle_sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(),
        "selected_arm": selection["winner"],
        "sources": {
            "index": index_src.as_posix(),
            "embedding_model": {"hf_id": spec.hf_id, "revision": cfg["embedding_revision"]},
            "response_cache": CACHE_DIR.as_posix(),
            **{
                name: {"path": cfg[name], "sha256": sha256_file(Path(cfg[name]))}
                for name in ("selection", "arm_traces", "arm_evaluations", "judge_agreement")
            },
        },
        "counts": {
            "questions": len(traces),
            "cached_responses": len(set(keys)),
            "files": len(files),
            "bytes": sum(f["bytes"] for f in files.values()),
        },
        "files": files,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    c = manifest["counts"]
    print(
        f"wrote {out}: {c['questions']} questions, {c['cached_responses']} cached responses, "
        f"{c['files']} files, {c['bytes'] / 1e6:.1f} MB; bundle {manifest['bundle_sha256'][:12]}"
    )


# --------------------------------------------------------------------------------------
# check


def compare(trace: dict, ev: dict, served: dict) -> tuple[dict[str, bool], float]:
    """Per-field agreement of one served answer with its trace and evaluation, and the
    largest retrieval-score difference."""
    t_chunks = trace["retrieval"]["chunks"]
    calls = {c["purpose"]: c for c in served["llm_calls"]}
    ctx = served["context"]
    score_diff = max(abs(a["score"] - b["score"]) for a, b in zip(t_chunks, ctx, strict=True))
    g_eval = ev["groundedness"]
    g_srv = served["groundedness"]
    if g_eval is None:
        ground_ok = g_srv["score"] is None and not g_srv.get("claims")
    else:
        ground_ok = all(g_eval[k] == g_srv.get(k) for k in GROUNDEDNESS_FIELDS) and [
            [c[k] for k in CLAIM_FIELDS] for c in g_eval["claims"]
        ] == [[c[k] for k in CLAIM_FIELDS] for c in g_srv["claims"]]
    return {
        "chunks": [c["chunk_id"] for c in t_chunks] == [c["chunk_id"] for c in ctx],
        "scores_within_tol": score_diff <= SCORE_TOL,
        "generation_request": calls["generate"]["cache_key"] == trace["generation"]["cache_key"],
        "answer": served["answer"] == trace["parsed"]["answer"],
        "citations": [c["label"] for c in served["citations"]] == trace["parsed"]["citations"],
        "status": served["status"] == ev["abstention"]["status"],
        "judge_requests": all(
            calls.get(f"judge_{s}", {}).get("cache_key") == ev["judge"].get(s)
            for s in ("decompose", "verify")
        ),
        "groundedness": ground_ok,
        "all_cached": all(c["cached"] for c in served["llm_calls"]),
    }, score_diff


def cmd_check() -> None:
    cfg = load_cfg()
    os.environ["RAG_REPLAY_ONLY"] = "1"
    os.environ.pop("ANTHROPIC_API_KEY", None)  # in-process: NoKeyBackend, no .env loaded
    from src.serving.service import NoKeyBackend, QueryService

    _, traces, evals = arm_records(cfg)
    bundle = Path(cfg["bundle_dir"])
    with tempfile.TemporaryDirectory() as state_dir:
        service = QueryService.from_bundle(bundle, Path(state_dir), device=cfg["check_device"])
        assert isinstance(service.pipeline.backend, NoKeyBackend)
        fields: dict[str, int] = {}
        mismatches, max_diff = [], 0.0
        for i, t in enumerate(traces, 1):
            served = service.query(t["question"]["question"], include_context=True)
            ok, diff = compare(t, evals[t["trace_id"]], served)
            max_diff = max(max_diff, diff)
            for k, v in ok.items():
                fields[k] = fields.get(k, 0) + int(v)
            if not all(ok.values()):
                mismatches.append(
                    {"trace_id": t["trace_id"], "failed": [k for k, v in ok.items() if not v]}
                )
            if i % 25 == 0:
                print(f"  {i}/{len(traces)} checked, {len(mismatches)} mismatches")
        spent = service.pipeline.ledger.state["total_usd"]
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "meta": {
            "selected_arm": manifest["selected_arm"],
            "bundle_sha256": manifest["bundle_sha256"],
            "embedding_device": cfg["check_device"],
            "replay_only": True,
            "score_tolerance": SCORE_TOL,
            "api_spend_usd": spent,
            "note": "the service, loaded from the bundle alone, answering every benchmark "
            "question of the selected arm; compared with the Phase 4 traces and the Phase 5 "
            "evaluations",
        },
        "n_questions": len(traces),
        "n_matching": {k: v for k, v in fields.items()},
        "max_score_abs_diff": max_diff,
        "mismatches": mismatches,
        "passed": not mismatches and spent == 0.0,
    }
    CHECK_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("n_matching", "max_score_abs_diff", "passed")}))
    print(f"wrote {CHECK_PATH}")
    if not report["passed"]:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["bundle", "check"])
    args = parser.parse_args()
    {"bundle": cmd_bundle, "check": cmd_check}[args.command]()


if __name__ == "__main__":
    main()
