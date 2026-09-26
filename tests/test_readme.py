"""The README renderer (scripts/17_readme.py): number formats, value resolution, and the check
that no number is typed into the template by hand."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "readme", Path(__file__).resolve().parent.parent / "scripts/17_readme.py"
)
readme = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readme)


@pytest.mark.parametrize(
    "value, spec, expected",
    [
        (0.17857, "f3", "0.179"),
        (-0.0357, "f2", "−0.04"),
        (-0.001, "f2", "0.00"),  # no negative zero
        (0.036, "sf3", "+0.036"),
        (-0.04, "sf2", "−0.04"),
        (0.0001, "sf2", "0.00"),
        (0.789, "pct1", "78.9%"),
        (0.0033, "pp1", "+0.3 pp"),
        (-0.01, "pp1", "−1.0 pp"),
        ([-0.01, 0.0167], "ppci1", "[−1.0, +1.7]"),
        ([0.27, 0.751], "ci2", "[0.27, 0.75]"),
        ({"kappa": 0.516, "kappa_ci95": [0.27, 0.75]}, "kci2", "0.52 [0.27, 0.75]"),
        ({"mean": 0.0733, "std": 0.0092}, "pm3", "0.073 ± 0.009"),
        (1007.5, "int", "1,008"),
        (0.62, "usd2", "$0.62"),
        (8.42, "s1", "8.4 s"),
    ],
)
def test_formats(value, spec, expected):
    assert readme.fmt(value, spec) == expected


def test_lookup_uses_slashes_because_keys_contain_dots():
    obj = {"cells": {"fixed_size__bge-small-en-v1.5__dense": {"m": [0.1, 0.2]}}}
    assert readme.lookup(obj, "cells/fixed_size__bge-small-en-v1.5__dense/m/1") == 0.2
    with pytest.raises(KeyError):
        readme.lookup(obj, "cells/missing")


def test_expr_is_arithmetic_only():
    env = {"a__b": 3.0, "c": 1.0}
    assert readme._eval("a.b / (1 - c / 2)", env) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        readme._eval("__import__('os')", env)


def test_values_resolve_in_dependency_order(monkeypatch):
    monkeypatch.setattr(readme, "load", lambda f: {"x": {"y": 0.25, "items": [{"k": 1}, {"k": 2}]}})
    spec = {
        "values": {
            "d": {"expr": "a + b.c", "fmt": "f2"},  # defined before its inputs
            "a": {"file": "f", "path": "x/y", "fmt": "f2"},
            "b.c": {"file": "f", "path": "x/items", "count_where": {"k": 2}, "fmt": "int"},
            "m": {"file": "f", "path": "x/items", "mean_of": "k", "fmt": "f1"},
        }
    }
    assert readme.resolve_values(spec) == {"a": "0.25", "b.c": "1", "d": "1.25", "m": "1.5"}


def test_typed_numbers_are_caught_outside_placeholders_and_code(tmp_path, monkeypatch):
    template = tmp_path / "t.md"
    template.write_text(
        "Recall is {{r}}, not 0.18 or 12%.\n"
        "Setting `chunk_size=512, rel_tol=0.01` and model bge-small-en-v1.5 are fine.\n"
        "[a link](results/v1.5/plot.png) is fine; 3 questions is an integer.\n",
        encoding="utf-8",
    )
    values = tmp_path / "v.yaml"
    values.write_text("values: {}\nallowed_literals: []\n", encoding="utf-8")
    monkeypatch.setattr(readme, "TEMPLATE", template)
    monkeypatch.setattr(readme, "VALUES", values)
    found = readme.typed_literals()
    assert len(found) == 2
    assert "'0.18'" in found[0] and "'12%'" in found[1]
