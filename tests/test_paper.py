"""Structural checks on paper/paper.qmd that a render would otherwise catch late.

Quarto is not required: these tests parse the source, so they also run in CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
QMD = PAPER / "paper.qmd"


@pytest.fixture(scope="module")
def text() -> str:
    if not QMD.exists():
        pytest.skip("paper not present")
    return QMD.read_text()


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    path = PAPER / "_variables.yml"
    if not path.exists():
        pytest.skip("run python -m src.report.paper_assets")
    out = {}
    for line in path.read_text().splitlines():
        key, _, val = line.partition(":")
        out[key.strip()] = val.strip().strip('"')
    return out


def test_every_variable_is_defined_and_finite(text, variables):
    used = set(re.findall(r"\{\{<\s*var\s+(\w+)\s*>\}\}", text))
    assert used, "paper uses no variables"
    assert not used - set(variables), f"undefined variables: {sorted(used - set(variables))}"
    assert all("nan" not in variables[k].lower() for k in used), "a variable is NaN"


def test_every_cross_reference_has_a_target(text):
    refs = set(re.findall(r"@((?:fig|tbl|sec)-[\w-]+)", text))
    targets = set(re.findall(r"\{#((?:fig|tbl|sec)-[\w-]+)[^}]*\}", text))
    assert not refs - targets, f"dangling references: {sorted(refs - targets)}"


def test_included_files_and_figures_exist(text):
    for rel in re.findall(r"\{\{<\s*include\s+([^\s>]+)\s*>\}\}", text):
        assert (PAPER / rel).exists(), f"missing include {rel}"
    for rel in re.findall(r"\]\((figures/[^)\s]+)\)", text):
        assert (PAPER / rel).exists(), f"missing figure {rel}"


def test_every_citation_is_in_the_bibliography(text):
    bib = (PAPER / "references.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set(re.findall(r"(?<![\w-])@((?!fig-|tbl-|sec-)[a-z]+\d{4}[a-z]?)", text))
    assert not cited - keys, f"citations missing from references.bib: {sorted(cited - keys)}"


def test_no_unfilled_placeholders(text):
    assert "TODO" not in text and "XXX" not in text
