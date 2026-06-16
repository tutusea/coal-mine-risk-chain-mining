# -*- coding: utf-8 -*-
"""Generate Figure 4: three-panel sensitivity analysis (publication-ready)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

fig06 = __import__("src.06_make_figures", fromlist=["_setup_english_matplotlib_font"])
_setup_english_matplotlib_font = fig06._setup_english_matplotlib_font

OUT_DIR = ROOT / "figures_real"
STEM = "Figure4_sensitivity_analysis_results"

# Fixed sensitivity results (do not alter)
CHAIN_FREQ = [2, 3, 4]
CHAIN_COUNTS = [66, 21, 12]
CHAIN_ANNOT = "Top-10 Jaccard = 1.000"

RULE_THRESH = [5, 8, 10]
RULE_COUNTS = [100, 51, 38]
RULE_ANNOT = "Top-10 Jaccard = 0.053\u20130.176"

JACCARD = 0.818
SPEARMAN_RHO = 0.994

BAR_COLOR = "#2E6E9E"
CARD_FILL = "#FAFAFA"
CARD_EDGE = "#C8C8C8"
ANNOT_FILL = "#F5F5F0"
ANNOT_EDGE = "#C8C0B0"

FS_TITLE = 12
FS_SUPTITLE = 14
FS_AXIS = 11
FS_TICK = 10
FS_VALUE = 9
FS_ANNOT = 9
FS_CARD = 11


def _style_axis(ax, font_name: str) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(font_name)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)


def _add_annotation_box(ax, text: str, font_name: str) -> None:
    ax.text(
        0.97,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FS_ANNOT,
        fontfamily=font_name,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=ANNOT_FILL, edgecolor=ANNOT_EDGE, linewidth=0.8),
        zorder=5,
    )


def _draw_bar_panel(
    ax,
    *,
    x_labels: list[str],
    values: list[int],
    xlabel: str,
    ylabel: str,
    title: str,
    annot: str,
    font_name: str,
) -> None:
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=BAR_COLOR, width=0.55, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontfamily=font_name)
    ax.set_xlabel(xlabel, fontsize=FS_AXIS, fontfamily=font_name)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS, fontfamily=font_name)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", fontfamily=font_name, pad=12)
    ymax = max(values) * 1.2
    ax.set_ylim(0, ymax)
    _style_axis(ax, font_name)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.025,
            str(val),
            ha="center",
            va="bottom",
            fontsize=FS_VALUE,
            fontfamily=font_name,
            zorder=4,
        )
    _add_annotation_box(ax, annot, font_name)


def _draw_summary_card_panel(ax, *, title: str, font_name: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", fontfamily=font_name, pad=12)

    lines = [
        "Top-10 overlap: 9/10",
        f"Jaccard similarity: {JACCARD:.3f}",
        f"Spearman \u03c1: {SPEARMAN_RHO:.3f}",
    ]

    n = len(lines)
    box_h = 0.22
    gap = 0.055
    y_bottom = 0.14
    x, w = 0.08, 0.84

    for i, line in enumerate(lines):
        y = y_bottom + (n - 1 - i) * (box_h + gap)
        patch = FancyBboxPatch(
            (x, y),
            w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=0.9,
            edgecolor=CARD_EDGE,
            facecolor=CARD_FILL,
            transform=ax.transAxes,
            zorder=1,
        )
        ax.add_patch(patch)
        ax.text(
            0.5,
            y + box_h / 2,
            line,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FS_CARD,
            fontfamily=font_name,
            color="#222222",
            zorder=2,
        )


def make_figure4() -> None:
    font_name = _setup_english_matplotlib_font()
    fig = plt.figure(figsize=(12, 4.5), facecolor="white")
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.08, right=0.97, top=0.80, bottom=0.18)

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_bar_panel(
        ax_a,
        x_labels=[str(v) for v in CHAIN_FREQ],
        values=CHAIN_COUNTS,
        xlabel="Minimum chain frequency",
        ylabel="Number of high-risk local chains",
        title="(a) Local-chain threshold",
        annot=CHAIN_ANNOT,
        font_name=font_name,
    )

    ax_b = fig.add_subplot(gs[0, 1])
    _draw_bar_panel(
        ax_b,
        x_labels=[f"\u2265{t}" for t in RULE_THRESH],
        values=RULE_COUNTS,
        xlabel="Support-count threshold",
        ylabel="Number of robust rules",
        title="(b) Rule support-count threshold",
        annot=RULE_ANNOT,
        font_name=font_name,
    )

    ax_c = fig.add_subplot(gs[0, 2])
    _draw_summary_card_panel(ax_c, title="(c) Node-weight sensitivity", font_name=font_name)

    fig.suptitle(
        "Sensitivity analysis results",
        fontsize=FS_SUPTITLE,
        fontweight="bold",
        fontfamily=font_name,
        y=0.96,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{STEM}.png"
    pdf_path = OUT_DIR / f"{STEM}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"OK: {png_path}")
    print(f"OK: {pdf_path}")


if __name__ == "__main__":
    make_figure4()
