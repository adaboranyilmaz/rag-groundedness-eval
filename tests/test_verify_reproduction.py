"""Unit tests for scripts/11_verify_reproduction.py: field differences, the exemption rules
(including their numeric tolerances and the rank restriction), and file comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "verify_reproduction", REPO / "scripts" / "11_verify_reproduction.py"
)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(o) + "\n" for o in obj), encoding="utf-8")
    else:
        path.write_text(json.dumps(obj), encoding="utf-8")
    return path


class TestDifferences:
    def test_paths_values_and_structure(self):
        a = {"x": {"y": 1, "only_a": 0}, "l": [1, 2], "n": 1}
        b = {"x": {"y": 2}, "l": [1, 3], "n": 1.0}
        paths = {p for p, *_ in v.differences(a, b)}
        assert paths == {".x.y", ".x.only_a", ".l[1]"}  # 1 == 1.0 is not a difference

    def test_length_mismatch_reported_once(self):
        assert [p for p, *_ in v.differences([1, 2], [1])] == [""]


class TestExemptions:
    def test_timestamps_anywhere(self):
        assert v.exemption("metrics/eval_main.json", ".meta.updated_utc")
        assert v.exemption("metrics/eval_main.json", ".meta.n_traces") is None

    def test_rule_is_bound_to_its_files(self):
        path = ".configs.x.qdrant.index_size_bytes"
        assert v.exemption("metrics/index_stats.json", path)
        assert v.exemption("metrics/retrieval_grid.json", path) is None
        assert v.exemption("metrics/index_stats.json", ".configs.x.faiss.index_size_bytes") is None

    def test_score_tolerance_allows_rounding_noise_only(self):
        rel, path = "traces/retrieved/m__p.jsonl", "[3].retrieval.chunks[1].score"
        assert v.exemption(rel, path, 0.719329, 0.71933)
        assert v.exemption(rel, path, 0.7193, 0.7194) is None  # 1e-4 > tolerance
        assert v.exemption(rel, "[3].retrieval.chunks[1].chunk_id", "a", "b") is None

    def test_ranking_swap_allowed_only_beyond_rank_5(self):
        rel = "metrics/retrieval_per_question.jsonl"
        assert v.exemption(rel, "[12].top_ids[5]", "a", "b")  # rank 6
        assert v.exemption(rel, "[12].top_ids[4]", "a", "b") is None  # rank 5
        assert v.exemption(rel, "[12].metrics.recall@10", 0.1, 0.2) is None


class TestCompareFile:
    def test_identical_semantic_and_exempt(self, tmp_path):
        a = write(tmp_path / "a" / "metrics" / "x.json", {"n": 1, "meta": {"run_utc": "t0"}})
        b = write(tmp_path / "b" / "metrics" / "x.json", {"meta": {"run_utc": "t1"}, "n": 1})
        assert v.compare_file("metrics/x.json", a, b)["status"] == "exempt_only"
        c = write(tmp_path / "c" / "metrics" / "x.json", {"n": 2, "meta": {"run_utc": "t0"}})
        out = v.compare_file("metrics/x.json", a, c)
        assert out["status"] == "different" and out["n"] == 1

    def test_binary_files_need_a_byte_rule(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        a.write_bytes(b"\x89PNG1")
        b.write_bytes(b"\x89PNG2")
        assert v.compare_file("plots/retrieval_vs_groundedness.png", a, b)["status"] == (
            "exempt_only"
        )
        assert v.compare_file("plots/prompt_effect.png", a, b)["status"] == "different"
