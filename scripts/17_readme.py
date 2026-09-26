"""Render README.md from docs/readme/README.template.md, so every number in it comes from a
committed results file (spec §3.1-3.2, §10: README numbers are never edited by hand).

The template holds the prose. Numbers are placeholders:
  {{name}}         a value defined in docs/readme/values.yaml: a results file, a path inside
                   it (`/`-separated, since keys contain dots), and a format; or an `expr` over
                   other values (differences, ratios)
  {{table:name}}   a table built below from results files, ending with its source line
Modes:
  (default)   write README.md
  --check     render in memory and fail if README.md differs (a hand edit, or a results file
              changed without a re-render), or if the template types a decimal or a
              percentage outside a placeholder that values.yaml does not allow. CI runs this.

Usage:
    uv run python scripts/17_readme.py
    uv run python scripts/17_readme.py --check
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "docs/readme/README.template.md"
VALUES = ROOT / "docs/readme/values.yaml"
README = ROOT / "README.md"
RESULTS = ROOT / "results"
PLACEHOLDER = re.compile(r"\{\{\s*([\w.:-]+)\s*\}\}")
# a decimal or a percentage typed into the prose (integers are left to review: years,
# phase numbers and k=5 are not results)
LITERAL = re.compile(r"(?<![\w.-])\d+\.\d+(?![\w.])|(?<![\w.-])\d+(?:\.\d+)?\s?%")

_files: dict[str, Any] = {}


def load(rel: str) -> Any:
    """A results file, by its path under results/ (JSON, or JSONL as a list)."""
    if rel not in _files:
        path = RESULTS / rel
        text = path.read_text(encoding="utf-8")
        if rel.endswith(".jsonl"):
            _files[rel] = [json.loads(x) for x in text.splitlines() if x]
        else:
            _files[rel] = json.loads(text)
    return _files[rel]


def lookup(obj: Any, path: str) -> Any:
    for part in path.split("/") if path else []:
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            if part not in obj:
                raise KeyError(f"no {part!r} in path {path!r}")
            obj = obj[part]
    return obj


# --------------------------------------------------------------------------------------
# Formats


def _num(x: float, digits: int) -> str:
    s = f"{x:.{digits}f}"
    return s.replace("-", "−") if s.startswith("-") and float(s) != 0 else s.lstrip("-")


def fmt(value: Any, spec: str) -> str:
    """f3 -> 0.179 | sf2 -> +0.04 | pct1 -> 17.9% | pp1 -> +1.2 pp | ppci1 -> [−1.0, +1.7] |
    int -> 1,234 | usd2 -> $0.62 |
    ci2 -> [0.27, 0.75] | kci2 -> 0.52 [0.27, 0.75] (a kappa summary) | pm3 -> 0.080 ± 0.012
    (a spread) | s1 -> seconds with one decimal | str"""
    kind, digits = re.fullmatch(r"([a-z]+)(\d*)", spec).groups()
    d = int(digits) if digits else 0
    if kind == "str":
        return str(value)
    if kind == "int":
        return f"{int(round(value)):,}"
    if kind == "f":
        return _num(value, d)
    if kind == "sf":  # signed: a difference
        s = _num(value, d)
        return f"+{s}" if round(value, d) > 0 else s
    if kind == "pct":
        return f"{_num(100 * value, d)}%"
    if kind == "pp":
        s = _num(100 * value, d)
        return f"{'+' if round(100 * value, d) > 0 else ''}{s} pp"
    if kind == "usd":
        return f"${value:,.{d}f}"
    if kind == "ci":
        return f"[{_num(value[0], d)}, {_num(value[1], d)}]"
    if kind == "ppci":  # an interval of a difference, in percentage points
        lo, hi = (fmt(v, f"pp{d}").removesuffix(" pp") for v in value)
        return f"[{lo}, {hi}]"
    if kind == "kci":
        ci = value.get("kappa_ci95") or value.get("ci95")
        k = value["kappa"] if "kappa" in value else value["value"]
        return f"{_num(k, d)} [{_num(ci[0], d)}, {_num(ci[1], d)}]"
    if kind == "pm":
        return f"{_num(value['mean'], d)} ± {_num(value['std'], d)}"
    if kind == "s":
        return f"{value:.{d}f} s"
    raise ValueError(f"unknown format {spec!r}")


# --------------------------------------------------------------------------------------
# Values

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _eval(expr: str, env: dict[str, float]) -> float:
    """Arithmetic over named values only: + - * / and parentheses."""

    def walk(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](walk(n.left), walk(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](walk(n.operand))
        if isinstance(n, ast.Constant) and isinstance(n.value, int | float):
            return n.value
        if isinstance(n, ast.Name):
            return env[n.id]
        raise ValueError(f"not allowed in expr: {ast.dump(n)}")

    return walk(ast.parse(expr.replace(".", "__"), mode="eval"))


def resolve_values(spec: dict) -> dict[str, str]:
    raw: dict[str, Any] = {}
    out: dict[str, str] = {}
    pending = dict(spec["values"])
    while pending:
        progressed = False
        for name, v in list(pending.items()):
            if "expr" in v:
                deps = set(re.findall(r"[A-Za-z_][\w.]*", v["expr"]))
                if not deps <= set(raw):
                    continue
                env = {k.replace(".", "__"): raw[k] for k in deps}
                raw[name] = _eval(v["expr"], env)
            elif "count_where" in v:  # items of a list (or values of a dict) matching all
                items = lookup(load(v["file"]), v["path"])
                items = items.values() if isinstance(items, dict) else items
                raw[name] = sum(
                    all(x.get(k) == want for k, want in v["count_where"].items()) for x in items
                )
            elif "mean_of" in v:  # mean of one field over a list of records
                items = [lookup(x, v["mean_of"]) for x in lookup(load(v["file"]), v["path"])]
                raw[name] = sum(items) / len(items)
            else:
                raw[name] = lookup(load(v["file"]), v["path"])
            out[name] = fmt(raw[name], v["fmt"])
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError(f"unresolvable values (unknown names in expr?): {sorted(pending)}")
    return out


# --------------------------------------------------------------------------------------
# Tables

TABLES: dict[str, Any] = {}


def table(fn):
    TABLES[fn.__name__] = fn
    return fn


def md(header: list[str], rows: list[list[str]], source: str) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + f"\n\n<sub>Source: {source}</sub>"


GEN = {"claude-sonnet-5": "Claude Sonnet 5", "qwen2.5-3b": "Qwen2.5 3B"}
PROMPT = {
    "v1_zero_shot": "v1 zero-shot",
    "v2_citation_required": "v2 citation-required",
    "v3_chain_of_thought": "v3 chain-of-thought",
    "v4_abstention": "v4 abstention",
}


@table
def retrieval_methods() -> str:
    g = load("metrics/retrieval_grid.json")
    rows = []
    for method in ("dense", "hybrid_rerank", "hybrid", "bm25"):
        cells = {k: c for k, c in g["cells"].items() if c["method"] == method}
        best = max(cells, key=lambda k: cells[k]["metrics"]["recall@5"])
        m = cells[best]["metrics"]
        mean = g["factor_decomposition"]["all_cells"]["recall@5"]["factors"]["method"][
            "level_means"
        ][method]
        rows.append(
            [
                method.replace("_", " + "),
                fmt(mean, "f3"),
                f"{cells[best]['chunking']} + {cells[best]['embedding']}",
                fmt(m["recall@5"], "f3"),
                fmt(m["recall@20"], "f3"),
                fmt(m["doc_hit@5"], "f3"),
                fmt(m["mrr"], "f3"),
            ]
        )
    return md(
        [
            "Method",
            "Mean recall@5 (9 cells)",
            "Best cell",
            "recall@5",
            "recall@20",
            "Right filing in top 5",
            "MRR",
        ],
        rows,
        "`results/metrics/retrieval_grid.json` (126 questions; deterministic)",
    )


@table
def retrieval_factors() -> str:
    fd = load("metrics/retrieval_grid.json")["factor_decomposition"]
    rows = []
    for scope, label in (("all_cells", "All 36 cells"), ("dense_only", "Dense only (9 cells)")):
        f = fd[scope]["recall@5"]["factors"]
        rows.append(
            [label]
            + [
                fmt(f[k]["ss_share"], "pct1") if k in f else "–"
                for k in ("method", "chunking", "embedding")
            ]
        )
    return md(
        ["Share of between-cell variance in recall@5", "Method", "Chunking", "Embedding"],
        rows,
        "`results/metrics/retrieval_grid.json` (`factor_decomposition`)",
    )


@table
def main_eval() -> str:
    e = load("metrics/eval_main.json")["by_condition_model"]
    rows = []
    for cond in ("oracle", "retrieved"):
        for gen in ("claude-sonnet-5", "qwen2.5-3b"):
            a = e[f"{cond}__{gen}"]
            c, g, ci = a["correctness"], a["groundedness"], a["citations"]
            rows.append(
                [
                    cond,
                    GEN[gen],
                    fmt(a["n"], "int"),
                    fmt(c["accuracy_all"]["rate"], "f2"),
                    fmt(c["accuracy_answered"]["rate"], "f2"),
                    fmt(g["mean_groundedness"]["mean"], "f2"),
                    fmt(g["fully_grounded"]["rate"], "f2"),
                    fmt(ci["citation_precision"]["mean"], "f2"),
                    fmt(a["abstention"]["status_counts"]["declined"] / a["n"], "f2"),
                ]
            )
    return md(
        [
            "Context",
            "Generator",
            "Answers",
            "Accuracy (all)",
            "Accuracy (answered)",
            "Groundedness",
            "Fully grounded",
            "Citation precision",
            "Declined (share of answers)",
        ],
        rows,
        "`results/metrics/eval_main.json` (`by_condition_model`; four prompts pooled; single "
        "run; groundedness and citation precision are judge-measured)",
    )


@table
def quadrant() -> str:
    q2 = load("metrics/reliability_analysis.json")["q2"]["cells"]
    rows = []
    for cell in (
        "oracle__claude-sonnet-5",
        "oracle__qwen2.5-3b",
        "retrieved__claude-sonnet-5",
        "retrieved__qwen2.5-3b",
    ):
        cond, gen = cell.split("__")
        p = q2[cell]["primary"]
        n = p["counts"]
        u = p["ungrounded_given_correct"]
        rows.append(
            [
                cond,
                GEN[gen],
                fmt(sum(n.values()), "int"),
                fmt(n["correct_grounded"], "int"),
                fmt(n["correct_ungrounded"], "int"),
                fmt(n["incorrect_grounded"], "int"),
                fmt(n["incorrect_ungrounded"], "int"),
                f"{fmt(u['mean'], 'f2')} {fmt(u['ci95'], 'ci2')}",
            ]
        )
    return md(
        [
            "Context",
            "Generator",
            "Graded answers",
            "Correct + grounded",
            "Correct + ungrounded",
            "Incorrect + grounded",
            "Incorrect + ungrounded",
            "Correct but not fully grounded",
        ],
        rows,
        "`results/metrics/reliability_analysis.json` (`q2`; single run; 95% intervals "
        "resample questions)",
    )


@table
def judge_agreement() -> str:
    ja = load("metrics/judge_agreement.json")["agreement"]
    rf = load("metrics/ragas_faithfulness.json")["comparison"]["pairs"]

    def k(s: dict) -> str:
        return fmt(s, "kci2") if s["kappa"] is not None else "undefined"

    rows = []
    for pair, label in (
        ("human__vs__judge", "Author vs project judge (Sonnet 5)"),
        ("human__vs__second_judge", "Author vs second judge (Haiku 4.5)"),
        ("judge__vs__second_judge", "Project judge vs second judge"),
        ("judge__vs__judge_retest", "Project judge vs itself (cache bypassed)"),
    ):
        d = ja[pair]["document_claims"]
        rows.append(
            [
                label,
                k(d["claim_supported"]),
                k(d["answer_fully_grounded"]),
                fmt(d["answer_fully_grounded"]["raw_agreement"], "f2"),
            ]
        )
    for pair, label in (
        ("human__vs__ragas", "Author vs RAGAS faithfulness"),
        ("judge__vs__ragas", "Project judge vs RAGAS faithfulness"),
    ):
        a = rf[pair]["all"]["fully_grounded"]
        rows.append([label, "– (own statements)", k(a), fmt(a["raw_agreement"], "f2")])
    return md(
        [
            "Raters (50 answers)",
            "Claim-level κ (supported or not)",
            "Answer-level κ (fully grounded)",
            "Answer-level raw agreement",
        ],
        rows,
        "`results/metrics/judge_agreement.json`, `results/metrics/ragas_faithfulness.json` "
        "(single run; 95% bootstrap intervals; claim level resamples answers)",
    )


@table
def replicates() -> str:
    r = load("metrics/replicates.json")
    rows = []
    labels = (
        ("correct_and_grounded", "Correct and fully grounded (all 150)", "f3"),
        ("accuracy_all", "Accuracy (all 150)", "f3"),
        ("accuracy_answered", "Accuracy (answered)", "f3"),
        ("declined_rate", "Declined", "f3"),
        ("mean_groundedness", "Groundedness (scored answers)", "f3"),
        ("citation_precision", "Citation precision", "f3"),
    )
    arms = list(r["arms"])
    for key, label, f in labels:
        row = [label]
        for arm in arms:
            s = r["arms"][arm]["spread"][key]
            vals = " / ".join(fmt(v, f) for v in s["values"])
            row.append(f"{fmt(s, 'pm' + f[1:])} ({vals})")
        rows.append(row)
    return md(
        ["Metric"] + [f"{PROMPT[a.split('__')[2]]}: mean ± std (runs 0 / 1 / 2)" for a in arms],
        rows,
        "`results/metrics/replicates.json` (three runs; std is the sample standard deviation)",
    )


@table
def adversarial() -> str:
    q4 = load("metrics/reliability_analysis.json")["q4"]["by_category"]
    names = {
        "a": "(a) one filing",
        "b": "(b) two filings",
        "c": "(c) unanswerable",
        "d": "(d) false premise",
    }
    rows = []
    for cat in ("a", "b", "c", "d"):
        for cond in ("oracle", "retrieved"):
            if cond not in q4[cat]:
                continue
            for gen in ("claude-sonnet-5", "qwen2.5-3b"):
                c = q4[cat][cond][gen]
                acc = c.get("accuracy_all")
                rows.append(
                    [
                        names[cat],
                        cond,
                        GEN[gen],
                        fmt(acc["pooled"]["mean"], "f2") if acc else "–",
                        fmt(c["declined"]["pooled"]["mean"], "f2"),
                    ]
                )
    return md(
        ["Category (10 questions each)", "Context", "Generator", "Accuracy", "Declined"],
        rows,
        "`results/metrics/reliability_analysis.json` (`q4`; four prompts pooled; single run)",
    )


@table
def load_levels() -> str:
    lt = load("metrics/load_test.json")["replay"]["levels"]
    rows = [
        [
            fmt(lv["users"], "int"),
            fmt(lv["throughput_rps"], "f1"),
            fmt(lv["latency_ms"]["p50"], "int"),
            fmt(lv["latency_ms"]["p95"], "int"),
            fmt(lv["latency_ms"]["p99"], "int"),
            fmt(lv["n_errors"], "int"),
        ]
        for lv in lt
    ]
    return md(
        ["Concurrent users", "Requests/s", "p50 ms", "p95 ms", "p99 ms", "Errors"],
        rows,
        "`results/metrics/load_test.json` (`replay`; single run; the load generator shares "
        "the machine)",
    )


STAGE_WHAT = {
    "ingest": "parse the filings into text with page and section metadata",
    "chunk": "the three chunking strategies",
    "embed": "every embedding set, computed from scratch on the GPU",
    "index": "every FAISS and Qdrant index, and the backend equivalence check",
    "retrieve": "the retrieval grid against the gold evidence",
    "retrieve_scoped": "retrieval restricted to the question's own filing",
    "generate": "every generation trace, checked against Phase 3's rankings",
    "replay_check": "every trace re-rendered, re-keyed and re-parsed",
    "evaluate": "the four metric families on every trace",
    "judge_validation": "judge agreement with the author's labels",
    "adversarial_generate": "the adversarial questions: answers",
    "adversarial_evaluate": "the adversarial questions: judging",
    "analyse": "the pre-registered reliability analysis and its plots",
    "select": "the pipeline selection rule",
    "track": "MLflow rebuilt from the results; the pipeline registered",
    "serving_bundle": "the serving bundle baked into the API image",
    "serving_check": "the service reproduces every evaluated answer",
    "ragas_faithfulness": "RAGAS faithfulness on the labelled answers",
    "replicates": "the two extra runs of the tied arms",
}


def _duration(s: float) -> str:
    if s >= 5400:
        return f"{s / 3600:.1f} h"
    return f"{s / 60:.0f} min" if s >= 90 else f"{s:.0f} s"


@table
def runtimes() -> str:
    if not (RESULTS / "metrics/runtimes.json").exists():  # not measured yet: say so (spec §3.1)
        return (
            "| Script | Time |\n|---|---|\n| every stage | TBD |\n\n"
            "<sub>Source: `results/metrics/runtimes.json`, not yet measured "
            "(`scripts/18_stage_runtimes.py`)</sub>"
        )
    r = load("metrics/runtimes.json")
    rows, i = [], 0
    for part, mode in (("from_scratch", "from scratch"), ("replayed", "replayed")):
        for stage, t in r[part].items():
            i += 1
            rows.append(
                [
                    str(i),
                    f"`{t['script']}`",
                    STAGE_WHAT.get(stage, stage),
                    mode,
                    _duration(t["seconds"]),
                ]
            )
    total = sum(t["seconds"] for part in ("from_scratch", "replayed") for t in r[part].values())
    rows.append(["", "**all stages**", "", "", f"**{_duration(total)}**"])
    return md(
        ["", "Script", "What it does", "Mode", "Time"],
        rows,
        "`results/metrics/runtimes.json` (the development machine on mains power; from-scratch "
        "stages timed once, replayed stages the slower of two runs)",
    )


# --------------------------------------------------------------------------------------
# Figure


AGREEMENT_FIGURE = RESULTS / "plots/rater_agreement.png"


def plot_rater_agreement(path: Path = AGREEMENT_FIGURE) -> None:
    """Kappa with its 95% interval for every pair of raters on the 50 labelled answers, at
    claim and answer level. Rows against the author (validity) in the house blue; rows
    between model raters (consistency) in the house grey; each group is also labelled."""
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT))
    from src.evaluation.plots import (
        INK_2,
        MUTED,
        SERIES,
        SURFACE,
        ZERO,
        _legend,
        _save,
        _style,
        _title,
    )

    ja = load("metrics/judge_agreement.json")["agreement"]
    rf = load("metrics/ragas_faithfulness.json")["comparison"]["pairs"]
    blue = SERIES["claude-sonnet-5"]
    rows = [  # label, claim-level summary or None, answer-level summary, colour
        ("Project judge", "human__vs__judge", None, blue),
        ("Second judge (Haiku 4.5)", "human__vs__second_judge", None, blue),
        ("RAGAS faithfulness", None, "human__vs__ragas", blue),
        ("Judge vs itself, re-run", "judge__vs__judge_retest", None, MUTED),
        ("Judge vs second judge", "judge__vs__second_judge", None, MUTED),
        ("Judge vs RAGAS", None, "judge__vs__ragas", MUTED),
    ]

    def summaries(ja_key, rf_key):
        if ja_key:
            d = ja[ja_key]["document_claims"]
            return d["claim_supported"], d["answer_fully_grounded"]
        return None, rf[rf_key]["all"]["fully_grounded"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    panels = [
        ("Claim level", "supported or not, document claims"),
        ("Answer level", "fully grounded or not"),
    ]
    for j, (ax, (title, note)) in enumerate(zip(axes, panels, strict=True)):
        _style(ax, grid_axis="x")
        ax.set_xlim(-0.3, 1.02)
        ax.axvline(0, color=ZERO, lw=1, zorder=1)
        ax.axhline(2.5, color=ZERO, lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        for y, (_, ja_key, rf_key, colour) in enumerate(rows):
            s = summaries(ja_key, rf_key)[j]
            if s is None:
                ax.text(
                    0.35,
                    y,
                    "not comparable: RAGAS writes its own statements",
                    va="center",
                    ha="center",
                    fontsize=8,
                    color=MUTED,
                )
                continue
            lo, hi = s["kappa_ci95"]
            ax.plot([lo, hi], [y, y], color=colour, lw=2, solid_capstyle="round", zorder=3)
            ax.scatter(
                [s["kappa"]], [y], s=55, color=colour, edgecolor=SURFACE, linewidth=2, zorder=4
            )
        ax.set_xlabel("Cohen's κ, 95% interval", fontsize=9, color=INK_2)
        _title(ax, title, note)
    handles = [
        plt.Line2D(
            [],
            [],
            color=blue,
            lw=0,
            marker="o",
            markersize=7,
            label="Against the author's blind labels",
        ),
        plt.Line2D(
            [], [], color=MUTED, lw=0, marker="o", markersize=7, label="Between model raters"
        ),
    ]
    _legend(fig, handles)
    _save(fig, path, top=0.9)
    plt.close(fig)


# --------------------------------------------------------------------------------------


def render() -> str:
    spec = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    values = resolve_values(spec)
    text = TEMPLATE.read_text(encoding="utf-8")

    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name.startswith("table:"):
            return TABLES[name.split(":", 1)[1]]()
        if name not in values:
            raise KeyError(f"template placeholder {{{{{name}}}}} has no value in values.yaml")
        return values[name]

    in_exprs = {
        n
        for v in spec["values"].values()
        if "expr" in v
        for n in re.findall(r"[A-Za-z_][\w.]*", v["expr"])
    }
    unused = set(values) - set(PLACEHOLDER.findall(text)) - in_exprs
    if unused:
        print(f"note: values defined but not used: {sorted(unused)}")
    header = "<!-- Generated by scripts/17_readme.py from docs/readme/; edit those, not this. -->\n"
    return header + PLACEHOLDER.sub(sub, text)


def typed_literals() -> list[str]:
    spec = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    allowed = set(spec.get("allowed_literals", []))
    text = PLACEHOLDER.sub("", TEMPLATE.read_text(encoding="utf-8"))
    text = re.sub(r"`[^`\n]*`", "", text)  # code spans: settings, versions, paths
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"\]\([^)]*\)", "]", text)  # link targets
    found = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in LITERAL.finditer(line):
            if m.group(0) not in allowed:
                found.append(f"line {i}: {m.group(0)!r} in: {line.strip()[:90]}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render()
    literals = typed_literals()
    if args.check:
        problems = []
        if not README.exists() or README.read_text(encoding="utf-8") != rendered:
            problems.append("README.md differs from a fresh render: run scripts/17_readme.py")
        problems += [f"typed number (not from a results file): {x}" for x in literals]
        if not AGREEMENT_FIGURE.exists():
            problems.append(f"{AGREEMENT_FIGURE.relative_to(ROOT)} is missing")
        if problems:
            sys.exit("\n".join(problems))
        print("README.md matches the results files")
        return
    if literals:
        sys.exit("typed numbers in the template:\n" + "\n".join(literals))
    README.write_text(rendered, encoding="utf-8", newline="\n")
    plot_rater_agreement()
    print(f"wrote {README} and {AGREEMENT_FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
