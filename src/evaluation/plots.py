"""Figures for the evaluation results. Colours follow a fixed categorical order (one slot per
generator model, validated for colour-vision deficiency), and text stays in ink colours."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
