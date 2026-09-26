"""The public data release (scripts/19_data_release.py): deterministic archives, the scan that
keeps an email or key out of them, and the checksum a restore verifies."""

from __future__ import annotations

import hashlib
import importlib.util
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "data_release", Path(__file__).resolve().parent.parent / "scripts/19_data_release.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "data/cache/llm/ab").mkdir(parents=True)
    (root / "data/cache/llm/ab/ab12.json").write_text('{"x": 1}', encoding="utf-8")
    (root / "data/cache/llm/cd.json").write_bytes(b"\x00binary\r\n")
    monkeypatch.setattr(release, "ROOT", root)
    return root


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_archive_is_byte_identical_when_packed_twice(tree, tmp_path):
    a, b = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    assert release.deterministic_tar(a, ["data/cache/llm"]) == 2
    (tree / "data/cache/llm/cd.json").touch()  # a newer mtime must not change the bytes
    release.deterministic_tar(b, ["data/cache/llm"])
    assert sha(a) == sha(b)


def test_archive_round_trips_exact_bytes(tree, tmp_path):
    out = tmp_path / "x.tar.gz"
    release.deterministic_tar(out, ["data/cache/llm"])
    dest = tmp_path / "dest"
    with tarfile.open(out, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    for rel in ("data/cache/llm/ab/ab12.json", "data/cache/llm/cd.json"):
        assert (dest / rel).read_bytes() == (tree / rel).read_bytes()


def test_pack_refuses_a_file_holding_the_contact_email(tree, monkeypatch):
    monkeypatch.setenv("EDGAR_USER_AGENT", "Some Name someone@example.org")
    monkeypatch.setattr(release, "ARCHIVES", {"c.tar.gz": ["data/cache/llm"]})
    (tree / "data/cache/llm/leak.json").write_text("from someone@example.org", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to pack"):
        release.cmd_pack()


def test_pack_refuses_an_api_key(tree, monkeypatch):
    monkeypatch.setenv("EDGAR_USER_AGENT", "Some Name someone@example.org")
    monkeypatch.setattr(release, "ARCHIVES", {"c.tar.gz": ["data/cache/llm"]})
    (tree / "data/cache/llm/k.json").write_text("sk-ant-abcdefghijklmnop", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to pack"):
        release.cmd_pack()


def test_restore_rejects_an_archive_whose_checksum_differs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.tar.gz").write_bytes(b"tampered")
    manifest = {"archives": {"a.tar.gz": {"sha256": "0" * 64}}}
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(SystemExit, match="does not match"):
        release.fetch("a.tar.gz", manifest, src, work)
