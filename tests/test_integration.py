"""Integration test: the full pipeline on the 10 smoke questions (scripts/10_smoke_eval.py),
run in a temporary workspace and compared with the committed traces and evaluations.

Deselected by default (pyproject.toml); run with `uv run pytest -m integration`. Uses the local
EDGAR cache when there is one (no network), else fetches the filings (needs EDGAR_USER_AGENT).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.integration
def test_smoke_eval_reproduces_committed_results(tmp_path):
    cmd = [sys.executable, "scripts/10_smoke_eval.py", "run", "--workspace", str(tmp_path / "ws")]
    edgar = REPO / "data" / "raw" / "edgar_cache"
    if edgar.exists():
        cmd += ["--edgar-cache", str(edgar)]
    elif not os.environ.get("EDGAR_USER_AGENT"):
        pytest.skip("no local EDGAR cache and no EDGAR_USER_AGENT to fetch the filings")
    assert subprocess.run(cmd, cwd=REPO).returncode == 0
