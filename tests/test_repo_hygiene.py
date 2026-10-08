"""Guards for publishing: nothing proprietary, personal or generated may be tracked."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tracked() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [line for line in out.splitlines() if (ROOT / line).is_file()]


def test_no_raw_or_generated_data_is_tracked():
    bad = [f for f in tracked() if re.match(r"^(data/|outputs/|models/)", f) or f.endswith((".parquet", ".pkl", ".joblib"))]
    assert not bad, f"raw/generated data is tracked: {bad}"


def test_no_pff_exports_are_tracked():
    bad = [f for f in tracked() if f.lower().endswith(".csv") and re.search(r"pff|summary|_stats", f.lower())
           and not f.startswith(("docs/", "ledger/"))]
    assert not bad, f"possible PFF exports tracked: {bad}"


def test_notebooks_have_no_outputs():
    for f in tracked():
        if f.endswith(".ipynb"):
            nb = json.loads((ROOT / f).read_text())
            assert not any(c.get("outputs") for c in nb["cells"]), f"{f} has saved outputs (may hold licensed data)"


def test_no_local_absolute_paths_in_text():
    pat = re.compile(r"/Users/[A-Za-z0-9._-]+|Mobile Documents|CloudDocs")
    hits = []
    for f in tracked():
        if f.endswith((".png", ".jpg", ".pdf")) or f in ("tests/test_repo_hygiene.py", "scripts/sync_public.sh"):   # both hold the patterns they search for
            continue
        try:
            text = (ROOT / f).read_text()
        except UnicodeDecodeError:
            continue
        if pat.search(text):
            hits.append(f)
    assert not hits, f"local paths found in: {hits}"
