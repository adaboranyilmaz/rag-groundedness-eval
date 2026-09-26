"""The data a public clone needs to replay every stage without calling a model API: the raw
EDGAR filings and every cached model response, published as two archives on a GitHub release
(the DVC remote itself is local to the development machine).

  pack      writes build/release/<archive>.tar.gz and results/metrics/data_release.json (the
            manifest: each archive's SHA-256, size and contents, and the DVC roots it holds).
            Archives are deterministic (sorted members, zeroed times and owners), so packing
            the same data gives the same bytes. Refuses if any file contains an email address
            taken from EDGAR_USER_AGENT or an API-key prefix.
  restore   downloads the archives from the release (or reads them from --from DIR), checks
            each SHA-256, and extracts them into place. FinanceBench is not redistributed: it
            is downloaded from Hugging Face at the pinned revision, checked, and the three
            small metadata files the Hub client wrote are rewritten from the manifest, so the
            directory hashes exactly as DVC recorded it. Finally `dvc status` must report
            every root unchanged; after that, `dvc repro` rebuilds every stage from the roots
            with no model call.

Usage:
    uv run python scripts/19_data_release.py pack
    uv run python scripts/19_data_release.py restore
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build/release"
MANIFEST = ROOT / "results/metrics/data_release.json"
TAG = "data-v1"
REPO = "adaboranyilmaz/rag-groundedness-eval"
ARCHIVES = {
    "raw-edgar.tar.gz": ["data/raw/edgar_cache"],
    "llm-caches.tar.gz": [
        "data/cache/llm",
        "data/cache/llm_batches",
        "data/cache/llm_live",
        "data/cache/llm_ragas",
        "data/cache/replicates",
    ],
}
FINANCEBENCH = {
    "dir": "data/raw/financebench",
    "repo_id": "PatronusAI/financebench",
    "file": "financebench_merged.jsonl",
    "license": "CC BY-NC 4.0 (downloaded from the Hub, not redistributed)",
}
KEY_PATTERN = re.compile(rb"sk-ant-[A-Za-z0-9_-]{8,}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files_under(rel: str) -> list[Path]:
    return sorted(p for p in (ROOT / rel).rglob("*") if p.is_file())


def forbidden_strings() -> list[bytes]:
    ua = os.environ.get("EDGAR_USER_AGENT", "")
    return [e.encode() for e in re.findall(r"[\w.%+-]+@[\w.-]+\.\w+", ua)]


# --------------------------------------------------------------------------------------
# pack


def deterministic_tar(out: Path, roots: list[str]) -> int:
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(out, "wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for rel in roots:
            for path in files_under(rel):
                info = tar.gettarinfo(str(path), arcname=path.relative_to(ROOT).as_posix())
                info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
                info.mode = 0o644
                with open(path, "rb") as f:
                    tar.addfile(info, f)
                n += 1
    return n


def cmd_pack() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    bad = forbidden_strings()
    hits = []
    for rels in ARCHIVES.values():
        for rel in rels:
            for path in files_under(rel):
                data = path.read_bytes()
                if KEY_PATTERN.search(data) or any(b in data for b in bad):
                    hits.append(path.relative_to(ROOT).as_posix())
    if hits:
        sys.exit(f"refusing to pack: {len(hits)} files hold an email or key, e.g. {hits[0]}")
    if not bad:
        sys.exit("set EDGAR_USER_AGENT (in .env) so the email scan has something to look for")

    fb_dir = ROOT / FINANCEBENCH["dir"]
    manifest = {
        "tag": TAG,
        "repo": REPO,
        "download_base": f"https://github.com/{REPO}/releases/download/{TAG}/",
        "scanned_for": "email addresses in EDGAR_USER_AGENT and API-key prefixes: none found",
        "archives": {},
        "financebench": {
            **FINANCEBENCH,
            "sha256": sha256_file(fb_dir / FINANCEBENCH["file"]),
            # the Hub client's own files, rewritten verbatim so the DVC hash matches
            "hub_files": {
                p.relative_to(fb_dir).as_posix(): p.read_bytes().decode("utf-8")
                for p in sorted(fb_dir.rglob("*"))
                if p.is_file() and p.name != FINANCEBENCH["file"]
            },
        },
        "dvc_roots": {},
    }
    hub_files = manifest["financebench"]["hub_files"]
    meta = next(v for k, v in hub_files.items() if k.endswith(".metadata"))
    manifest["financebench"]["revision"] = meta.splitlines()[0].strip()
    for name, roots in ARCHIVES.items():
        out = BUILD / name
        n = deterministic_tar(out, roots)
        manifest["archives"][name] = {
            "sha256": sha256_file(out),
            "bytes": out.stat().st_size,
            "n_files": n,
            "contains": roots,
        }
        print(f"packed {name}: {n} files, {out.stat().st_size / 1e6:.1f} MB")
    for dvc in sorted((ROOT / "data").rglob("*.dvc")):
        text = dvc.read_text(encoding="utf-8")
        manifest["dvc_roots"][dvc.relative_to(ROOT).as_posix()] = re.search(
            r"md5: (\S+)", text
        ).group(1)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST}")


# --------------------------------------------------------------------------------------
# restore


def fetch(name: str, manifest: dict, source: Path | None, into: Path) -> Path:
    path = into / name
    if source is not None:
        shutil.copyfile(source / name, path)
    else:
        url = manifest["download_base"] + name
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
    got = sha256_file(path)
    if got != manifest["archives"][name]["sha256"]:
        sys.exit(f"{name}: SHA-256 {got} does not match the manifest")
    return path


def cmd_restore(source: Path | None, force: bool) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    targets = [r for rs in ARCHIVES.values() for r in rs] + [FINANCEBENCH["dir"]]
    present = [t for t in targets if (ROOT / t).exists() and any((ROOT / t).iterdir())]
    if present and not force:
        sys.exit(f"already present (use --force to overwrite): {present}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, entry in manifest["archives"].items():
            archive = fetch(name, manifest, source, tmp)
            for rel in entry["contains"]:
                shutil.rmtree(ROOT / rel, ignore_errors=True)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(ROOT, filter="data")
            print(f"extracted {name}")

        from huggingface_hub import hf_hub_download

        fb = manifest["financebench"]
        got = hf_hub_download(
            repo_id=fb["repo_id"],
            repo_type="dataset",
            filename=fb["file"],
            revision=fb["revision"],
            cache_dir=tmp / "hf",
        )
        if sha256_file(Path(got)) != fb["sha256"]:
            sys.exit("FinanceBench at the pinned revision does not match the manifest")
        fb_dir = ROOT / fb["dir"]
        shutil.rmtree(fb_dir, ignore_errors=True)
        fb_dir.mkdir(parents=True)
        shutil.copyfile(got, fb_dir / fb["file"])
        for rel, text in fb["hub_files"].items():
            (fb_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (fb_dir / rel).write_bytes(text.encode("utf-8"))
        print(f"FinanceBench restored at revision {fb['revision'][:12]}")

    roots = list(manifest["dvc_roots"])
    out = subprocess.run(
        [sys.executable, "-m", "dvc", "status", *roots], cwd=ROOT, capture_output=True, text=True
    )
    if out.returncode != 0 or "up to date" not in out.stdout.lower():
        sys.exit(f"DVC roots do not match their .dvc hashes:\n{out.stdout}{out.stderr}")
    print("every DVC root matches its recorded hash; `dvc repro` now needs no model call")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pack")
    r = sub.add_parser("restore")
    r.add_argument("--from", dest="source", type=Path, default=None, help="local archive dir")
    r.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.cmd == "pack":
        cmd_pack()
    else:
        cmd_restore(args.source, args.force)


if __name__ == "__main__":
    main()
