# -*- coding: utf-8 -*-
"""Figures for papers: bars, Sankey, network, association rules."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from . import config


def _configure_matplotlib_cjk():
    """
    Prefer common Chinese fonts on Windows so node labels render in PNG figures.
    If glyphs are still missing, install a CJK font or set MPLCONFIGDIR / font path.
    """
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_cjk()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_top_nodes(node_df: pd.DataFrame, top_k: int, out_path: Path) -> None:
    _ensure_out()
    sub = node_df.sort_values("risk_score", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.25)))
    y = np.arange(len(sub))
    ax.barh(y, sub["risk_score"].values, color="#2c7fb8")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["node"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("risk_score (frequency * severity_rate)")
    ax.set_title("Top risk nodes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIG_DPI)
    plt.close(fig)


def plot_top_edges(edge_df: pd.DataFrame, top_k: int, out_path: Path) -> None:
    _ensure_out()
    edge_df = edge_df.copy()
    edge_df["label"] = edge_df["source"].astype(str) + " -> " + edge_df["target"].astype(str)
    sub = edge_df.sort_values(["edge_risk_score", "lift_to_severe", "weight"], ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.28)))
    y = np.arange(len(sub))
    ax.barh(y, sub["edge_risk_score"].values, color="#e34a33")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("edge_risk_score (weight * edge_severity_rate)")
    ax.set_title("Top risk edges")
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIG_DPI)
    plt.close(fig)


def _sankey_display(layer: int, node_id_str: str) -> str:
    """Human-readable label for Sankey: layer + field name + value."""
    if "|" in node_id_str:
        fk, val = node_id_str.split("|", 1)
    else:
        fk, val = "", node_id_str
    dn = config.FIELD_DISPLAY_NAME.get(fk, fk)
    return "L%d %s:%s" % (layer, dn, val)


def _sankey_layer_counts(chain_df: pd.DataFrame, max_rows: int = 5000) -> Tuple[List[str], List[int], List[int], List[int]]:
    """Build Sankey nodes and links from consecutive layer positions."""
    node_index = {}
    sources = []
    targets = []
    values = []
    counts = Counter()

    cdf = chain_df.head(max_rows) if len(chain_df) > max_rows else chain_df

    def nid(layer: int, nid_str: str) -> int:
        label = _sankey_display(layer, nid_str)
        if label not in node_index:
            node_index[label] = len(node_index)
        return node_index[label]

    for _, r in cdf.iterrows():
        ch: List[str] = r["chain_list"]
        for i in range(len(ch) - 1):
            a, b = ch[i], ch[i + 1]
            ia, ib = nid(i, a), nid(i + 1, b)
            counts[(ia, ib)] += 1

    for (ia, ib), w in counts.items():
        sources.append(ia)
        targets.append(ib)
        values.append(w)

    labels = [""] * len(node_index)
    for lab, idx in node_index.items():
        labels[idx] = lab
    return labels, sources, targets, values


def plot_sankey(chain_df: pd.DataFrame, out_html: Path, out_png: Optional[Path] = None) -> None:
    _ensure_out()
    labels, sources, targets, values = _sankey_layer_counts(chain_df)
    if go is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.1, 0.5, "plotly not installed; pip install plotly kaleido for Sankey", fontsize=12)
        ax.axis("off")
        note_path = (out_png or out_html).parent / "sankey_note.png"
        fig.savefig(note_path, dpi=config.FIG_DPI)
        plt.close(fig)
        return

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, pad=15, thickness=20),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(title_text="Risk chain Sankey (layered transitions)", font_size=11, height=700)
    fig.write_html(str(out_html))
    try:
        if out_png:
            fig.write_image(str(out_png), width=1200, height=700, scale=2)
    except Exception:
        pass


def plot_network(G: nx.DiGraph, node_df: pd.DataFrame, top_n: int, out_path: Path) -> None:
    _ensure_out()
    sub_nodes = set(node_df.sort_values("risk_score", ascending=False).head(top_n)["node"])
    H = G.subgraph(sub_nodes).copy()
    if H.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "empty subgraph", ha="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=config.FIG_DPI)
        plt.close(fig)
        return

    pos = nx.spring_layout(H, seed=config.RANDOM_LAYOUT_SEED, k=0.5 / np.sqrt(max(len(H), 1)))
    fig, ax = plt.subplots(figsize=(12, 10))
    deg = dict(H.degree())
    max_deg = max(deg.values()) if deg else 1
    ns = [80 + 400 * (deg.get(n, 1) / max_deg) for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_size=ns, node_color="#9ecae1", ax=ax)
    nx.draw_networkx_edges(H, pos, arrows=True, arrowsize=12, width=0.8, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=6, ax=ax)
    ax.set_axis_off()
    ax.set_title("Risk network (top nodes by risk_score, spring layout)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIG_DPI)
    plt.close(fig)


def plot_association_rules(rules_df: pd.DataFrame, out_path: Path) -> None:
    _ensure_out()
    if rules_df is None or rules_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no association rules to plot", ha="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=config.FIG_DPI)
        plt.close(fig)
        return

    sub = rules_df.head(min(40, len(rules_df))).copy()
    sub["rule"] = sub["antecedents"].astype(str) + " => " + sub["consequents"].astype(str)
    fig, ax = plt.subplots(figsize=(10, max(6, len(sub) * 0.22)))
    y = np.arange(len(sub))
    ax.barh(y, sub["lift"].values, color="#31a354")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["rule"].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("lift")
    ax.set_title("Association rules toward severe outcome")
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIG_DPI)
    plt.close(fig)
