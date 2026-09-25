"""Figures for the evaluation results. Colours follow a fixed categorical order (one slot per
generator model, validated for colour-vision deficiency), and text stays in ink colours."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {"claude-sonnet-5": "#2a78d6", "qwen2.5-3b": "#eb6834"}  # categorical slots 1, 2
MODEL_LABELS = {"claude-sonnet-5": "Claude Sonnet 5", "qwen2.5-3b": "Qwen2.5 3B"}
CONDITION_LABELS = {"retrieved": "Retrieved context", "oracle": "Oracle context"}
OUTCOME_LABELS = {"correct": "Answer correct", "fully_grounded": "Answer fully grounded"}


def plot_reliability(
    reliability: dict[str, dict], conditions: list[str], models: list[str], path: Path
):
    """Reliability diagram: stated confidence (binned) against the observed rate of a
    correct / fully grounded answer, answered questions only. `reliability` maps
    "<condition>__<model>" to {"correct": calibration, "fully_grounded": calibration}
    (src/evaluation/evaluate.py `calibration`). Marker area follows the bin's count."""
    outcomes = list(OUTCOME_LABELS)
    fig, axes = plt.subplots(
        len(conditions), len(outcomes), figsize=(9.6, 4.3 * len(conditions)), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    max_n = max(
        (b["n"] for cm in reliability.values() for cal in cm.values() for b in cal["bins"]),
        default=1,
    )
    for i, cond in enumerate(conditions):
        for j, outcome in enumerate(outcomes):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            ax.plot([0, 100], [0, 1], ls=(0, (4, 3)), lw=1.2, color=INK_2, zorder=1)
            ece_parts = []
            for model in models:
                cal = reliability.get(f"{cond}__{model}", {}).get(outcome)
                if not cal or not cal["n"]:
                    continue
                pts = [b for b in cal["bins"] if b["n"]]
                xs = [b["mean_confidence"] for b in pts]
                ys = [b["observed"] for b in pts]
                sizes = [40 + 360 * (b["n"] / max_n) ** 0.5 for b in pts]
                ax.plot(xs, ys, lw=2, color=SERIES[model], zorder=2)
                ax.scatter(
                    xs, ys, s=sizes, color=SERIES[model], edgecolor=SURFACE, linewidth=2, zorder=3
                )
                ece_parts.append(f"{MODEL_LABELS[model]} {cal['ece']:.2f} (n={cal['n']})")
            ax.set_xlim(-2, 102)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks(range(0, 101, 20))
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax.grid(color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=9, length=0)
            ax.set_title(
                f"{CONDITION_LABELS[cond]} — {OUTCOME_LABELS[outcome].lower()}",
                fontsize=10.5,
                color=INK,
                loc="left",
                pad=20,
            )
            if i == len(conditions) - 1:
                ax.set_xlabel("Stated confidence", fontsize=9.5, color=INK_2)
            if j == 0:
                ax.set_ylabel("Observed rate", fontsize=9.5, color=INK_2)
            # Above the plot area, under the title: inside it, the note collides with data.
            ax.text(
                0.0,
                1.02,
                "ECE: " + " · ".join(ece_parts),
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=8.5,
                color=INK_2,
            )
    handles = [
        plt.Line2D([], [], color=SERIES[m], lw=2, marker="o", markersize=7, label=MODEL_LABELS[m])
        for m in models
    ] + [plt.Line2D([], [], color=INK_2, lw=1.2, ls=(0, (4, 3)), label="Perfect calibration")]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=9.5,
        labelcolor=INK,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Phase 6 figures. Every plotted value is read from results/metrics/reliability_analysis.json;
# nothing is computed here but positions and colours.

MUTED = "#898781"
BAND = "#e1e0d9"  # the random-baseline band: the hairline-grid grey, recessive
ZERO = "#c3c2b7"  # baseline / axis ink
# Sequential blue (palette steps 100-650) for shares in the 2x2 tiles: a magnitude, one hue.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]
PROMPT_SHORT = {
    "v2_citation_required": "v2 citation-required",
    "v3_chain_of_thought": "v3 chain-of-thought",
    "v4_abstention": "v4 abstention",
}


def _style(ax, grid_axis: str = "both") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8, axis=grid_axis)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def _title(ax, title: str, note: str | None = None) -> None:
    n_lines = note.count("\n") + 1 if note else 0
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8 + 12 * n_lines)
    if note:
        # Above the plot area, under the title: inside it, a note collides with data.
        ax.text(
            0.0,
            1.02,
            note,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8.5,
            color=INK_2,
        )


def _legend(fig, handles: list) -> None:
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=9.5,
        labelcolor=INK,
        bbox_to_anchor=(0.5, 1.0),
    )


def _model_handles(models: list[str]) -> list:
    return [
        plt.Line2D([], [], color=SERIES[m], lw=0, marker="o", markersize=7, label=MODEL_LABELS[m])
        for m in models
    ]


def _save(fig, path: Path, top: float, bottom: float = 0.0) -> None:
    fig.tight_layout(rect=(0, bottom, 1, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _fmt_ci(v: float | None, ci: list[float] | None) -> str:
    if v is None:
        return "–"
    return f"{v:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def plot_retrieval_vs_groundedness(q1: dict, models: list[str], path: Path) -> None:
    """Q1. Left: each question's mean groundedness by whether retrieval found its gold
    evidence, with the group mean and its interval. Right: the same questions against the
    retriever's own top-1 score, the one signal available without gold labels."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)  # horizontal jitter only
    offsets = {m: (i - (len(models) - 1) / 2) * 0.32 for i, m in enumerate(models)}
    notes_l, notes_r = [], []
    for m in models:
        b = q1["by_model"].get(m)
        if not b:
            continue
        for x, key, hit in ((0, "recall5_eq_0", False), (1, "recall5_gt_0", True)):
            ys = [u["groundedness"] for u in b["units"] if (u["recall5"] > 0) == hit]
            xs = x + offsets[m] + rng.uniform(-0.08, 0.08, size=len(ys))
            ax_l.scatter(xs, ys, s=16, color=SERIES[m], alpha=0.35, linewidth=0, zorder=2)
            g = b["groundedness_by_recall"][key]
            if g["mean"] is None:
                continue
            if g["ci95"]:
                ax_l.plot(
                    [x + offsets[m]] * 2,
                    g["ci95"],
                    color=SERIES[m],
                    lw=2,
                    solid_capstyle="round",
                    zorder=3,
                )
            ax_l.scatter(
                [x + offsets[m]],
                [g["mean"]],
                s=70,
                color=SERIES[m],
                edgecolor=SURFACE,
                linewidth=2,
                zorder=4,
            )
        s = b["spearman_recall5_groundedness"]
        notes_l.append(f"Spearman rho, {MODEL_LABELS[m]}: {_fmt_ci(s['rho'], s['ci95'])}")
        units = b["units"]
        ax_r.scatter(
            [u["top1_score"] for u in units],
            [u["groundedness"] for u in units],
            s=26,
            color=SERIES[m],
            alpha=0.7,
            edgecolor=SURFACE,
            linewidth=1,
            zorder=2,
        )
        t = b["exploratory"]["top1_score"]["groundedness"]
        notes_r.append(f"Spearman rho, {MODEL_LABELS[m]}: {_fmt_ci(t['rho'], t['ci95'])}")
    for ax in (ax_l, ax_r):
        _style(ax)
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_l.grid(False, axis="x")
    ax_l.set_xlim(-0.6, 1.6)
    ax_l.set_xticks([0, 1])
    ax_l.set_xticklabels(["Not retrieved\n(recall@5 = 0)", "Retrieved\n(recall@5 > 0)"])
    ax_l.set_xlabel("Gold evidence", fontsize=9.5, color=INK_2)
    ax_l.set_ylabel("Groundedness (question mean)", fontsize=9.5, color=INK_2)
    ax_r.set_xlabel("Top-1 retrieval score", fontsize=9.5, color=INK_2)
    _title(ax_l, "Groundedness by retrieval outcome", "\n".join(notes_l))
    _title(ax_r, "Groundedness against the retriever's score", "\n".join(notes_r))
    mean_key = plt.Line2D(
        [],
        [],
        color=INK_2,
        lw=2,
        marker="o",
        markersize=7,
        markeredgecolor=SURFACE,
        label="Group mean, 95% CI",
    )
    _legend(fig, _model_handles(models) + [mean_key])
    _save(fig, path, top=0.93)


def _ramp_colour(share: float) -> str:
    """Piecewise-linear through the ramp steps: share 0 -> lightest, 1 -> darkest."""
    pos = min(max(share, 0.0), 1.0) * (len(RAMP) - 1)
    i = min(int(pos), len(RAMP) - 2)
    a, b = mcolors.to_rgb(RAMP[i]), mcolors.to_rgb(RAMP[i + 1])
    t = pos - i
    return mcolors.to_hex(tuple(x + (y - x) * t for x, y in zip(a, b, strict=True)))


def _luminance(colour: str) -> float:
    """WCAG relative luminance, to put white or ink text on a tile."""

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in mcolors.to_rgb(colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def plot_quadrants(q2: dict, conditions: list[str], models: list[str], path: Path) -> None:
    """Q2. One 2x2 per condition x generator: tile shade = share of that cell's scored,
    graded answers, and each tile prints its count and share. The note under each title is
    the pre-registered primary statistic: the share of correct answers not fully grounded."""
    quads = [
        ["correct_grounded", "correct_ungrounded"],
        ["incorrect_grounded", "incorrect_ungrounded"],
    ]
    fig, axes = plt.subplots(
        len(conditions),
        len(models),
        figsize=(4.8 * len(models), 4.0 * len(conditions)),
        squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)
    gap = 0.02  # half the surface gap between tiles, in data units
    for i, cond in enumerate(conditions):
        for j, m in enumerate(models):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(colors=INK_2, labelsize=9, length=0)
            cell = q2["cells"].get(f"{cond}__{m}")
            if not cell:
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            counts = cell["primary"]["counts"]
            n = sum(counts.values())
            for r, row in enumerate(quads):
                for c, q in enumerate(row):
                    share = counts[q] / n if n else 0.0
                    fill = _ramp_colour(share)
                    ax.add_patch(
                        FancyBboxPatch(
                            (c + gap, 1 - r + gap),
                            1 - 2 * gap,
                            1 - 2 * gap,
                            boxstyle="round,pad=0,rounding_size=0.04",
                            facecolor=fill,
                            edgecolor="none",
                        )
                    )
                    ink = "#ffffff" if _luminance(fill) < 0.3 else INK
                    ax.text(
                        c + 0.5,
                        1 - r + 0.57,
                        f"{counts[q]}",
                        ha="center",
                        va="center",
                        fontsize=15,
                        fontweight="bold",
                        color=ink,
                    )
                    ax.text(
                        c + 0.5,
                        1 - r + 0.35,
                        f"{share:.0%} of answers",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color=ink,
                    )
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels(["Fully grounded", "Not fully grounded"])
            ax.set_yticks([1.5, 0.5])
            ax.set_yticklabels(["Correct", "Incorrect"])
            u = cell["primary"]["ungrounded_given_correct"]
            _title(
                ax,
                f"{CONDITION_LABELS[cond]} — {MODEL_LABELS[m]}",
                f"Correct but not grounded: {_fmt_ci(u['mean'], u['ci95'])} · n = {n}",
            )
    _save(fig, path, top=1.0)


def plot_signal_agreement(q3: dict, conditions: list[str], models: list[str], path: Path) -> None:
    """Q3. Rows: condition x generator. Columns: Spearman rho of the continuous signals, and
    the Jaccard overlap of the answers each signal flags. Grey band: 95% of the statistic
    under within-prompt permutation (the random baseline), tick at its mean. Dot and line:
    the observed value and its question-bootstrap interval."""
    stats = [("spearman", "Spearman rho"), ("jaccard", "Jaccard overlap of flagged answers")]
    cells = [(c, m) for c in conditions for m in models if f"{c}__{m}" in q3]
    if not cells:
        return
    pairs = list(q3[f"{cells[0][0]}__{cells[0][1]}"]["pairs"])
    fig, axes = plt.subplots(
        len(cells), len(stats), figsize=(9.6, 2.5 * len(cells) + 0.6), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for j, (stat, stat_label) in enumerate(stats):
        ends = [
            v
            for c in q3.values()
            for p in c["pairs"].values()
            if p["status"] == "ok" and p[stat].get("status", "ok") == "ok"
            for v in (p[stat]["ci95"] or [])
            + [p[stat]["baseline"]["p2_5"], p[stat]["baseline"]["p97_5"]]
            if v is not None
        ]
        lo, hi = min(ends + [0.0]) - 0.05, max(ends + [0.6]) + 0.05
        for i, (cond, m) in enumerate(cells):
            ax = axes[i][j]
            _style(ax, grid_axis="x")
            ax.set_xlim(lo, hi)
            ax.set_ylim(len(pairs) - 0.4, -0.6)  # first pair on top
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels(pairs)
            if stat == "spearman":
                ax.axvline(0, color=ZERO, lw=1, zorder=1)
            for y, name in enumerate(pairs):
                p = q3[f"{cond}__{m}"]["pairs"][name]
                if p["status"] != "ok" or p[stat].get("status", "ok") != "ok":
                    ax.text(
                        (lo + hi) / 2,
                        y,
                        "no variance",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color=MUTED,
                    )
                    continue
                s, b = p[stat], p[stat]["baseline"]
                if b["p2_5"] is not None:
                    ax.barh(
                        y,
                        b["p97_5"] - b["p2_5"],
                        left=b["p2_5"],
                        height=0.5,
                        color=BAND,
                        zorder=1,
                    )
                    ax.plot([b["mean"]] * 2, [y - 0.25, y + 0.25], color=INK_2, lw=1, zorder=2)
                if s["ci95"]:
                    ax.plot(
                        s["ci95"], [y, y], color=SERIES[m], lw=2, solid_capstyle="round", zorder=3
                    )
                if s["value"] is not None:
                    ax.scatter(
                        [s["value"]],
                        [y],
                        s=60,
                        color=SERIES[m],
                        edgecolor=SURFACE,
                        linewidth=2,
                        zorder=4,
                    )
            _title(ax, f"{CONDITION_LABELS[cond]} — {MODEL_LABELS[m]}", stat_label)
    band_key = plt.Rectangle(
        (0, 0), 1, 1, color=BAND, label="Random baseline (95% of permutations)"
    )
    _legend(fig, _model_handles(models) + [band_key])
    fig.text(
        0.01,
        0.004,
        "G groundedness (judge) · CP citation precision (same judge as G)\n"
        "NS share of the answer's figures found in its cited excerpts (judge-free) · "
        "C stated confidence",
        fontsize=8.5,
        color=INK_2,
        ha="left",
        va="bottom",
    )
    _save(fig, path, top=0.97, bottom=0.03)


def plot_prompt_effect(q5: dict, conditions: list[str], models: list[str], path: Path) -> None:
    """Q5. Paired change against v1 (same question, generator and condition) in groundedness
    and in citation precision, with question-bootstrap intervals. The v2 row, in bold, is
    the pre-registered primary contrast."""
    outcomes = [
        ("groundedness", "Change in groundedness"),
        ("citation_precision", "Change in citation precision"),
    ]
    contrasts = list(PROMPT_SHORT)
    fig, axes = plt.subplots(
        len(conditions), len(outcomes), figsize=(9.6, 2.6 * len(conditions) + 0.6), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    offsets = {m: (k - (len(models) - 1) / 2) * 0.22 for k, m in enumerate(models)}
    for j, (field, label) in enumerate(outcomes):
        ends = [
            abs(v)
            for cell in q5.values()
            for e in cell.values()
            for v in (e["differences"][field]["ci95"] or [])
        ]
        span = max(ends + [0.1]) + 0.05
        for i, cond in enumerate(conditions):
            ax = axes[i][j]
            _style(ax, grid_axis="x")
            ax.set_xlim(-span, span)
            ax.axvline(0, color=ZERO, lw=1, zorder=1)
            ax.set_ylim(len(contrasts) - 0.4, -0.6)  # v2 on top
            ax.set_yticks(range(len(contrasts)))
            if j == 0:
                ax.set_yticklabels([PROMPT_SHORT[c] + " vs v1" for c in contrasts])
                ax.get_yticklabels()[0].set_fontweight("bold")
            else:
                ax.set_yticklabels([])
            for y, treat in enumerate(contrasts):
                for m in models:
                    e = q5.get(f"{cond}__{m}", {}).get(f"{treat}_vs_v1_zero_shot")
                    if not e:
                        continue
                    d = e["differences"][field]
                    yy = y + offsets[m]
                    if d["ci95"]:
                        ax.plot(
                            d["ci95"],
                            [yy, yy],
                            color=SERIES[m],
                            lw=2,
                            solid_capstyle="round",
                            zorder=3,
                        )
                    if d["mean"] is not None:
                        ax.scatter(
                            [d["mean"]],
                            [yy],
                            s=55,
                            color=SERIES[m],
                            edgecolor=SURFACE,
                            linewidth=2,
                            zorder=4,
                        )
            _title(ax, CONDITION_LABELS[cond], f"{label} against v1, paired by question")
    _legend(fig, _model_handles(models))
    _save(fig, path, top=0.95)


def plot_adversarial(q4: dict, models: list[str], path: Path) -> None:
    """Q4. One panel per adversarial category, each showing the behaviour that category
    rewards: accuracy for (a) and (b), declining for (c), rejecting the premise for (d)
    (the author's labels only; the panel says so while they are pending). Rows are generator
    x context condition; intervals resample questions (10 per category)."""
    panels = [
        ("a", "accuracy_all", "(a) One filing: answered correctly"),
        ("b", "accuracy_all", "(b) Two filings: answered correctly"),
        ("c", "declined", "(c) Unanswerable from the corpus: declined"),
        ("d", None, "(d) False premise: rejected (author's labels)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    fig.patch.set_facecolor(SURFACE)
    # the right-hand panels have the same rows as their left neighbours
    for i, (ax, (cat, key, title)) in enumerate(zip(axes.flat, panels, strict=True)):
        _style(ax, grid_axis="x")
        rows = []
        for m in models:
            for cond in ("retrieved", "oracle"):
                if cat == "d":
                    if cond != "retrieved":
                        continue
                    block = q4.get("premise", {}).get(m, {}).get("human", {})
                    est = None if block.get("status") == "not_done" else block["rejected"]["pooled"]
                else:
                    e = q4["by_category"].get(cat, {}).get(cond, {}).get(m)
                    if e is None:
                        continue
                    est = e[key]["pooled"]
                rows.append((f"{MODEL_LABELS[m]}, {cond}", m, est))
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows] if i % 2 == 0 else [])
        for y, (_, m, est) in enumerate(rows):
            if est is None:
                ax.text(
                    0.5, y, "labelling pending", ha="center", va="center", fontsize=8.5, color=MUTED
                )
                continue
            if est["ci95"]:
                ax.plot(
                    est["ci95"], [y, y], color=SERIES[m], lw=2, solid_capstyle="round", zorder=3
                )
            ax.scatter(
                [est["mean"]], [y], s=60, color=SERIES[m], edgecolor=SURFACE, linewidth=2, zorder=4
            )
        _title(ax, title)
    _legend(fig, _model_handles(models))
    _save(fig, path, top=0.95)
