# -*- coding: utf-8 -*-
"""阶段 06：论文用图（节点与边排名、子网络、关联规则、Sankey）。"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

FIGURES_DIR = config.FIGURES_DIR
NODE_METRICS = config.OUTPUT_DIR / "node_metrics.csv"
EDGE_METRICS = config.OUTPUT_DIR / "edge_metrics.csv"
ASSOC_SEVERE_PREVENTION = config.OUTPUT_DIR / "association_rules_severe_prevention.csv"
ASSOC_SEVERE_PREVENTION_ROBUST = config.OUTPUT_DIR / "association_rules_severe_prevention_robust.csv"
ASSOC_ALL = config.OUTPUT_DIR / "association_rules_all.csv"
CHAINS = config.OUTPUT_DIR / "accident_chains.csv"

CONSEQ_SEVERE_MARK = "SevereBinary:较严重伤害"

TOP_NODES = 20
TOP_EDGES = 20
TOP_LOCAL_NETWORK_NODES = 40
TOP_ASSOC = 20
TOP_SANKEY_CHAINS = 20

# 论文正文用简化局部网络（完整图仍为 network_overview_local_chain）
SIMPLIFIED_LOCAL_TOP_NODES = 15
SIMPLIFIED_LOCAL_TOP_EDGES = 20

# multipartite_layout：同层节点纵向排列，减轻与下一层边的交叉
_NODE_LAYER_FOR_LAYOUT: dict[str, int] = {
    "Team": 0,
    "Job": 1,
    "Shift": 2,
    "Cause": 2,
    "Loc": 3,
    "Act": 4,
    "HazardMode": 5,
    "HazardSource": 5,
    "Body": 6,
    "InjuryForm": 7,
    "Sev": 8,
}

PREVENTION_NODE_TYPES = frozenset({"Team", "Job", "Shift", "Loc", "Act", "HazardMode", "HazardSource"})
CONSEQUENCE_NODE_TYPES = frozenset({"Body", "InjuryForm", "Sev"})

CHAIN_SEP = re.compile(r"\s*->\s*")

# -----------------------------------------------------------------------------
# Display-only English translations for JSR figures (data CSVs unchanged).
# Extend NODE_TYPE_PREFIX_EN and CHINESE_VALUE_TO_EN as new labels appear.
# -----------------------------------------------------------------------------

NODE_TYPE_PREFIX_EN: dict[str, str] = {
    "Team": "Team",
    "Job": "Job",
    "Shift": "Shift",
    "Loc": "Location",
    "Act": "Activity",
    "HazardMode": "Hazard mode",
    "HazardSource": "Hazard source",
    "Body": "Body part",
    "InjuryForm": "Injury form",
    "Sev": "Injury severity",
    "Cause": "Cause",
}

# Rule antecedent item keys in mined CSV (e.g. Act:, Loc:) -> English label for publication.
RULE_ITEM_KEY_EN: dict[str, str] = {
    "Team": "Team",
    "Job": "Job",
    "Shift": "Shift",
    "Loc": "Location",
    "Act": "Activity",
    "HazardMode": "Hazard mode",
    "HazardSource": "Hazard source",
    "Body": "Body part",
    "InjuryForm": "Injury form",
    "Sev": "Injury severity",
    "Cause": "Cause",
}

CHINESE_VALUE_TO_EN: dict[str, str] = {
    # Shifts / activities / locations (user + dataset)
    "早班": "Day shift",
    "中班": "Middle shift",
    "夜班": "Night shift",
    "采煤工作面": "Coalface",
    "猴车巷": "Chairlift roadway",
    "掘进头": "Heading face",
    "采煤": "Coal mining",
    "掘进": "Tunneling",
    "运输": "Transportation",
    "安装回收": "Installation and recovery",
    "采煤工": "Coal miner",
    "掘进工": "Tunneling worker",
    "砸": "Struck by object",
    "碰": "Collision",
    "挤": "Squeezing",
    "摔": "Fall/slip",
    "刮": "Scraping",
    "崴": "Sprain/twist",
    # Body / injury / severity (user + common nodes)
    "脚": "Foot",
    "腿": "Leg",
    "手": "Hand",
    "头": "Head",
    "多处骨头": "Multiple bones",
    "骨折": "Fracture",
    "内脏破裂": "Internal organ rupture",
    "内脏": "Internal organs",
    "离断": "Amputation",
    "聋": "Hearing loss",
    "轻微伤": "Minor injury",
    "轻伤": "Mild injury",
    "重伤": "Severe injury",
    "肋骨": "Ribs",
    "脊椎": "Spine",
    "眼": "Eye",
    "腰": "Lower back",
    "手臂": "Arm",
    "背": "Back",
    "胸": "Chest",
    "下身": "Lower body",
    # Activities / jobs / teams / locations (dataset-specific; unknowns stay Chinese)
    "巷修": "Roadway repair",
    "开拓": "Development driving",
    "推车": "Cart pushing",
    "搬运": "Material handling",
    "检修": "Maintenance",
    "清理": "Cleanup",
    "行走": "Walking",
    "猴车事故": "Chairlift-related incident",
    "准备工": "Preparation worker",
    "副区长": "Deputy section chief",
    "开拓工": "Development worker",
    "支架工": "Support worker",
    "机电工": "Mechanical and electrical worker",
    "检修工": "Repair worker",
    "班长": "Work team leader",
    "电瓶车司机": "Battery-locomotive driver",
    "运输工": "Transport worker",
    "通风工": "Ventilation worker",
    "主井": "Main shaft",
    "副井": "Auxiliary shaft",
    "各类硐室": "Various chambers",
    "回风大巷": "Return airway",
    "开拓迎头": "Development face",
    "抽采巷": "Gas drainage roadway",
    "石门": "Rock cross-cut",
    "联络巷": "Connection roadway",
    "车场": "Siding yard",
    "轨道上山": "Track uphill roadway",
    "轨道大巷": "Main haulage roadway",
    "轨道顺槽": "Track gate road",
    "运输上山": "Transport uphill roadway",
    "运输大巷": "Main transport roadway",
    "运输顺槽": "Transport gate road",
    "修护队": "Roadway support team",
    "准备队": "Preparation team",
    "安装队": "Installation team",
    "开拓队": "Development team",
    "抽采队": "Gas drainage team",
    "掘进队": "Tunneling team",
    "机电队": "Mechanical and electrical team",
    "通风队": "Ventilation team",
    "采煤队": "Coal mining team",
}


def translate_node_label(label: str) -> str:
    """
    Map a node_id / node_label like ``Act:运输`` to English display, e.g. ``Activity: Transportation``.
    - Prefix is always mapped via NODE_TYPE_PREFIX_EN when recognized.
    - Value part uses CHINESE_VALUE_TO_EN; unknown Chinese substrings are left unchanged.
    """
    s = str(label).strip()
    if not s or s.lower() in {"nan", "none"}:
        return s
    if ":" not in s:
        return s
    prefix, rest = s.split(":", 1)
    p = prefix.strip()
    val = rest.strip()
    en_p = NODE_TYPE_PREFIX_EN.get(p, p)
    en_v = CHINESE_VALUE_TO_EN.get(val, val)
    return f"{en_p}: {en_v}"


def translate_rule_label(antecedent: str) -> str:
    """
    Translate a semicolon-separated antecedent string (e.g. ``Act:运输;Shift:早班``) for figure text.
    Each token must look like ``Key:value``; unrecognized keys keep the original key text.
    """
    s = str(antecedent).strip()
    if not s or s.lower() in {"nan", "none"}:
        return s
    parts_out: list[str] = []
    for piece in re.split(r"\s*;\s*", s):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            parts_out.append(piece)
            continue
        k, v = piece.split(":", 1)
        k = k.strip()
        v = v.strip()
        en_k = RULE_ITEM_KEY_EN.get(k, k)
        en_v = CHINESE_VALUE_TO_EN.get(v, v)
        parts_out.append(f"{en_k}: {en_v}")
    return "; ".join(parts_out)


def _compact_rule_label(antecedent: str) -> str:
    """Short antecedent for Figure 4: English values joined by ' + ' (display only)."""
    s = str(antecedent).strip()
    if not s or s.lower() in {"nan", "none"}:
        return s
    parts_out: list[str] = []
    for piece in re.split(r"\s*;\s*", s):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            parts_out.append(piece)
            continue
        _k, v = piece.split(":", 1)
        v = v.strip()
        parts_out.append(CHINESE_VALUE_TO_EN.get(v, v))
    return " + ".join(parts_out) if parts_out else translate_rule_label(antecedent)


def _compact_node_label_figure3(label: str) -> str:
    """Shorter English node labels for Figure 3 (display only)."""
    en = translate_node_label(label)
    prefix_map = {
        "Body part:": "Body:",
        "Injury form:": "Injury:",
        "Injury severity:": "Severity:",
    }
    for old, new in prefix_map.items():
        if en.startswith(old):
            en = new + en[len(old) :]
            break
    if ": " in en:
        p, v = en.split(": ", 1)
        v_short = {
            "Minor injury": "Minor",
            "Mild injury": "Mild",
            "Severe injury": "Severe",
            "Internal organ rupture": "Internal rupture",
        }.get(v, v)
        return f"{p}: {v_short}"
    return en


def _pick_english_font() -> str:
    """Arial / Helvetica / DejaVu Sans only — avoid CJK fonts on English figures."""
    want = ("Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans", "Liberation Sans")
    try:
        names = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        names = set()
    for w in want:
        if w in names:
            return w
    for w in want:
        for n in names:
            if w.lower() in n.lower():
                return n
    return "DejaVu Sans"


def _setup_english_matplotlib_font() -> str:
    fam = _pick_english_font()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [fam]
    plt.rcParams["axes.unicode_minus"] = False
    return fam


def _pick_cjk_font() -> str:
    """优先 SimHei 或 Microsoft YaHei，否则回退到系统已有无衬线字体。"""
    want = ("SimHei", "Microsoft YaHei", "MS YaHei", "PingFang SC", "Noto Sans CJK SC")
    try:
        names = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        names = set()
    for w in want:
        if w in names:
            return w
    for w in want:
        for n in names:
            if w.lower() in n.lower():
                return n
    return "DejaVu Sans"


def _setup_matplotlib_font() -> str:
    fam = _pick_cjk_font()
    plt.rcParams["font.sans-serif"] = [fam, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return fam


def _save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    try:
        fig.savefig(pdf, bbox_inches="tight")
    except Exception as e:
        print(f"WARNING: PDF save failed for {pdf.name}: {e}")
    plt.close(fig)


def _parse_layer_token(token: str) -> tuple[str, str] | None:
    t = token.strip()
    for prefix, key in (
        ("Loc:", "Loc"),
        ("Act:", "Act"),
        ("HazardMode:", "HazardMode"),
        ("HazardSource:", "HazardSource"),
        ("Body:", "Body"),
        ("InjuryForm:", "InjuryForm"),
        ("Sev:", "Sev"),
    ):
        if t.startswith(prefix):
            return key, t
    return None


def _figure_node_risk_ranking_subset(
    nm: pd.DataFrame,
    font_name: str,
    *,
    allowed_types: frozenset[str],
    stem: str,
    title: str,
    xlabel: str,
) -> None:
    sub = nm[nm["node_type"].astype(str).str.strip().isin(allowed_types)].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无可用节点", ha="center", va="center", fontsize=12, fontfamily=font_name)
        ax.axis("off")
        _save_figure(fig, stem)
        return
    sub = sub.sort_values("risk_score", ascending=False).head(TOP_NODES)
    sub = sub.sort_values("risk_score", ascending=True)
    labels = sub["node_label"].astype(str).tolist()
    vals = sub["risk_score"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(labels, vals, color="#2E6E9E", height=0.65)
    ax.set_xlabel(xlabel, fontsize=11, fontfamily=font_name)
    ax.set_title(title, fontsize=12, fontweight="bold", fontfamily=font_name)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    _save_figure(fig, stem)


def _figure_edge_risk_ranking(em: pd.DataFrame, font_name: str) -> None:
    em = em.copy()
    if "edge_risk_score" not in em.columns:
        w = pd.to_numeric(em["weight"], errors="coerce").fillna(0.0)
        esr = pd.to_numeric(em["edge_severity_rate"], errors="coerce").fillna(0.0)
        em["edge_risk_score"] = w * esr
    sub = em.sort_values("edge_risk_score", ascending=False).head(TOP_EDGES)
    sub = sub.sort_values("edge_risk_score", ascending=True)
    labels = (sub["source"].astype(str) + " -> " + sub["target"].astype(str)).tolist()
    vals = sub["edge_risk_score"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(labels, vals, color="#C44E52", height=0.65)
    ax.set_xlabel("边风险得分 weight x edge_severity_rate", fontsize=11, fontfamily=font_name)
    ax.set_title("有向边风险得分排名前 20", fontsize=12, fontweight="bold", fontfamily=font_name)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    _save_figure(fig, "edge_risk_ranking")


def _figure_network_overview_local(nm: pd.DataFrame, font_name: str) -> None:
    """局部风险链核心网络：Loc→Act→HazardMode→Body/InjuryForm→Sev 等局部链上的边。"""
    if not CHAINS.is_file():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "缺少事故链条表", ha="center", va="center", fontsize=12, fontfamily=font_name)
        ax.axis("off")
        _save_figure(fig, "network_overview_local_chain")
        return

    cdf = pd.read_csv(CHAINS, encoding="utf-8-sig")
    edge_w: dict[tuple[str, str], float] = {}
    for col in ("chain_local_1", "chain_local_2"):
        if col not in cdf.columns:
            continue
        for s in cdf[col].astype(str):
            if not s or s.lower() in {"nan", "none"}:
                continue
            parts = [p.strip() for p in CHAIN_SEP.split(s) if p.strip()]
            for a, b in zip(parts[:-1], parts[1:]):
                edge_w[(a, b)] = edge_w.get((a, b), 0.0) + 1.0

    if not edge_w:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无局部链边", ha="center", va="center", fontsize=12, fontfamily=font_name)
        ax.axis("off")
        _save_figure(fig, "network_overview_local_chain")
        return

    node_strength: dict[str, float] = {}
    for (u, v), w in edge_w.items():
        node_strength[u] = node_strength.get(u, 0.0) + w
        node_strength[v] = node_strength.get(v, 0.0) + w

    top_ids = sorted(node_strength.keys(), key=lambda k: node_strength[k], reverse=True)[
        :TOP_LOCAL_NETWORK_NODES
    ]
    top_set = set(top_ids)
    sub_edges = [(u, v, w) for (u, v), w in edge_w.items() if u in top_set and v in top_set]

    G = nx.DiGraph()
    risk_map = nm.set_index("node_id")["risk_score"].astype(float).to_dict()
    label_map = nm.set_index("node_id")["node_label"].astype(str).to_dict()
    ntype_map = nm.set_index("node_id")["node_type"].astype(str).to_dict()

    for nid in top_ids:
        G.add_node(nid, risk=risk_map.get(nid, 0.0), lab=label_map.get(nid, nid), nt=ntype_map.get(nid, ""))

    for u, v, w in sub_edges:
        if u in G and v in G and w > 0:
            if G.has_edge(u, v):
                G[u][v]["weight"] = float(G[u][v].get("weight", 0.0)) + w
            else:
                G.add_edge(u, v, weight=w)

    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无可用节点", ha="center", va="center", fontsize=14, fontfamily=font_name)
        ax.axis("off")
        _save_figure(fig, "network_overview_local_chain")
        return

    pos = nx.spring_layout(G, seed=42, k=0.9 / max(1, np.sqrt(G.number_of_nodes())))

    risks = np.array([float(G.nodes[n].get("risk", 0.0)) for n in G.nodes()])
    rmin, rmax = float(risks.min()), float(risks.max())
    if rmax <= rmin:
        sizes = np.full(len(risks), 280.0)
    else:
        sizes = 120.0 + 520.0 * (risks - rmin) / (rmax - rmin)

    type_color = {
        "Team": "#7B68EE",
        "Job": "#4682B4",
        "Shift": "#5F9EA0",
        "Loc": "#2E8B57",
        "Act": "#DAA520",
        "HazardMode": "#CD853F",
        "HazardSource": "#B87333",
        "Body": "#BC8F8F",
        "InjuryForm": "#D2691E",
        "Sev": "#A0522D",
        "Cause": "#708090",
    }
    node_colors = [type_color.get(str(G.nodes[n].get("nt", "")), "#696969") for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(10, 8))
    widths = []
    for u, v in G.edges():
        w = float(G[u][v].get("weight", 1.0))
        widths.append(max(0.3, min(6.0, 0.15 * np.sqrt(w))))
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=widths,
        edge_color="#555555",
        arrows=True,
        arrowsize=12,
        alpha=0.55,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=node_colors, alpha=0.92, linewidths=0.4)
    draw_labels = {n: str(G.nodes[n].get("lab", n)) for n in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=draw_labels,
        ax=ax,
        font_size=7,
        font_family=font_name,
    )
    ax.set_title(
        "局部风险链条核心网络（Loc→Act→HazardMode→Body/InjuryForm→Sev）",
        fontsize=12,
        fontweight="bold",
        fontfamily=font_name,
    )
    ax.axis("off")
    fig.tight_layout()
    _save_figure(fig, "network_overview_local_chain")


def _edge_metric_score(row: pd.Series) -> float:
    """与阶段 03 一致：优先 edge_risk_score，否则 weight × edge_severity_rate。"""
    if "edge_risk_score" in row.index and pd.notna(row.get("edge_risk_score")):
        try:
            return float(row["edge_risk_score"])
        except (TypeError, ValueError):
            pass
    w = float(pd.to_numeric(row.get("weight"), errors="coerce") or 0.0)
    esr = float(pd.to_numeric(row.get("edge_severity_rate"), errors="coerce") or 0.0)
    return w * esr


def _wrap_label_by_words(text: str, max_chars_per_line: int) -> str:
    """Word-wrap English labels on spaces; fallback to prefix:value split."""
    s = str(text).strip().replace("\r", "")
    if not s or len(s) <= max_chars_per_line:
        return s
    if ": " in s:
        prefix, rest = s.split(": ", 1)
        head = f"{prefix}:"
        words = rest.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            trial = f"{cur} {w}".strip() if cur else w
            if len(trial) <= max_chars_per_line:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        if not lines:
            return head
        return head + " " + lines[0] + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else "")
    words = s.split()
    lines = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip() if cur else w
        if len(trial) <= max_chars_per_line:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines) if lines else s


def _wrap_cjk_label(text: str, max_chars_per_line: int) -> str:
    """对过长标签按字符宽度换行，保留「前缀:取值」结构的可读性。"""
    s = str(text).strip().replace("\r", "")
    if not s:
        return ""
    if len(s) <= max_chars_per_line:
        return s
    if ":" in s:
        prefix, rest = s.split(":", 1)
        head = f"{prefix}:"
        lines: list[str] = []
        chunk = max(6, int(max_chars_per_line) - 1)
        t = rest.strip()
        while t:
            lines.append(t[:chunk])
            t = t[chunk:].strip()
        if not lines:
            return head
        return head + lines[0] + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else "")
    lines = []
    chunk = max_chars_per_line
    t = s
    while t:
        lines.append(t[:chunk])
        t = t[chunk:].strip()
    return "\n".join(lines)


def _save_figure_custom(fig: plt.Figure, stem: str, *, dpi: int = 300) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    try:
        fig.savefig(pdf, bbox_inches="tight")
    except Exception as e:
        print(f"WARNING: PDF save failed for {pdf.name}: {e}")
    plt.close(fig)


def _save_figure_to_stems(fig: plt.Figure, stems: list[str], *, dpi: int = 300) -> None:
    """同一画布写入多个 stem（PNG+PDF），避免重复绘图。"""
    if not stems:
        plt.close(fig)
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dpi_i = int(dpi)
    for stem in stems:
        png = FIGURES_DIR / f"{stem}.png"
        pdf = FIGURES_DIR / f"{stem}.pdf"
        fig.savefig(png, dpi=dpi_i, bbox_inches="tight")
        try:
            fig.savefig(pdf, bbox_inches="tight")
        except Exception as e:
            print(f"WARNING: PDF save failed for {pdf.name}: {e}")
    plt.close(fig)


def _build_local_chain_simplified_graph(nm: pd.DataFrame, em: pd.DataFrame) -> nx.DiGraph | None:
    """
    与论文图3一致：risk_score 前 SIMPLIFIED_LOCAL_TOP_NODES 个节点；
    诱导子图上按 edge_risk_score（或 weight×edge_severity_rate）取前 SIMPLIFIED_LOCAL_TOP_EDGES 条边。
    """
    nm2 = nm.sort_values("risk_score", ascending=False).head(int(SIMPLIFIED_LOCAL_TOP_NODES))
    top_ids = [str(x) for x in nm2["node_id"].tolist()]
    top_set = set(top_ids)
    ntype_map = nm.set_index("node_id")["node_type"].astype(str).str.strip().to_dict()
    risk_map = nm.set_index("node_id")["risk_score"].astype(float).to_dict()
    lab_map: dict[str, str] = {}
    if "node_label" in nm.columns:
        lab_map = nm.set_index("node_id")["node_label"].astype(str).to_dict()

    em2 = em.copy()
    em2["_e_score"] = em2.apply(_edge_metric_score, axis=1)
    sub = em2[em2["source"].astype(str).isin(top_set) & em2["target"].astype(str).isin(top_set)].copy()
    sub = sub.sort_values("_e_score", ascending=False).head(int(SIMPLIFIED_LOCAL_TOP_EDGES))

    G = nx.DiGraph()
    for nid in top_ids:
        nt = ntype_map.get(nid, "")
        layer = int(_NODE_LAYER_FOR_LAYOUT.get(nt, 5))
        disp = str(lab_map.get(nid, nid)).strip() or nid
        G.add_node(nid, nt=nt, subset=layer, risk=float(risk_map.get(nid, 0.0)), lab=disp)

    for _, r in sub.iterrows():
        u, v = str(r["source"]), str(r["target"])
        if u not in G or v not in G:
            continue
        w = float(pd.to_numeric(r.get("weight"), errors="coerce") or 0.0)
        if G.has_edge(u, v):
            G[u][v]["weight"] = float(G[u][v].get("weight", 0.0)) + w
        else:
            G.add_edge(u, v, weight=w)
    return G if G.number_of_nodes() else None


def _layout_local_chain_simplified(
    G: nx.DiGraph,
    *,
    layout_scale: float,
    spring_k_scale: float = 1.0,
) -> dict:
    """分层布局优先；失败时用 spring_layout 并加大节点间距。"""
    try:
        pos = nx.multipartite_layout(G, subset_key="subset", align="vertical", scale=float(layout_scale))
    except Exception:
        n = max(1, G.number_of_nodes())
        k = float(spring_k_scale) * 4.2 / max(1.0, np.sqrt(n))
        pos = nx.spring_layout(G, seed=44, k=k, iterations=120)
    # 略微纵向拉伸，减轻同层标签上下重叠
    sy = 1.12
    return {nid: (float(x), float(y) * sy) for nid, (x, y) in pos.items()}


def _label_positions_with_offset(
    G: nx.DiGraph,
    pos: dict,
    *,
    y_push: float,
) -> dict:
    """按节点大致象限将标签略向外推，减轻与邻接节点文字重叠。"""
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    mx = float(np.mean(xs)) if xs else 0.0
    my = float(np.mean(ys)) if ys else 0.0
    out: dict = {}
    for n in G.nodes():
        x, y = pos[n]
        dx = 0.02 if x >= mx else -0.02
        dy = float(y_push) if y >= my else -float(y_push)
        out[n] = (x + dx, y + dy)
    return out


def _draw_local_chain_simplified_variant(
    G: nx.DiGraph,
    pos: dict,
    font_name: str,
    *,
    figsize: tuple[float, float],
    label_font: float,
    title_font: float,
    footnote_font: float,
    arrowsize: float,
    layout_scale_used: float,
    title_override: str | None = None,
    footnote_override: str | None = None,
    label_transform: Callable[[str], str] | None = None,
    label_wrap: str = "cjk",
) -> plt.Figure:
    type_color = {
        "Team": "#7B68EE",
        "Job": "#4682B4",
        "Shift": "#5F9EA0",
        "Loc": "#2E8B57",
        "Act": "#DAA520",
        "HazardMode": "#CD853F",
        "HazardSource": "#B87333",
        "Body": "#BC8F8F",
        "InjuryForm": "#D2691E",
        "Sev": "#A0522D",
        "Cause": "#708090",
    }
    risks = np.array([float(G.nodes[n].get("risk", 0.0)) for n in G.nodes()])
    rmin, rmax = float(risks.min()), float(risks.max())
    # 与画布协调：略放大节点面积，仍保持 risk 相对比例
    base_small, span = 420.0, 1100.0
    if rmax <= rmin:
        sizes = np.full(len(risks), base_small + 0.55 * span)
    else:
        sizes = base_small + span * (risks - rmin) / (rmax - rmin)

    node_colors = [type_color.get(str(G.nodes[n].get("nt", "")), "#696969") for n in G.nodes()]

    fig, ax = plt.subplots(figsize=figsize)
    widths = []
    for u, v in G.edges():
        w = float(G[u][v].get("weight", 1.0))
        widths.append(max(0.4, min(6.2, 0.2 * np.sqrt(max(w, 0.01)))))

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=widths,
        edge_color="#444444",
        arrows=True,
        arrowsize=arrowsize,
        alpha=0.55,
        connectionstyle="arc3,rad=0.11",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=sizes, node_color=node_colors, alpha=0.93, linewidths=0.55
    )
    max_line = 22 if figsize[0] >= 17 else 18
    wrap_fn = _wrap_label_by_words if label_wrap == "word" else _wrap_cjk_label

    def _lab_for_node(nid) -> str:
        raw = str(G.nodes[nid].get("lab", nid))
        if label_transform is not None:
            try:
                raw = str(label_transform(raw))
            except Exception:
                pass
        return wrap_fn(raw, max_chars_per_line=max_line)

    draw_labels = {n: _lab_for_node(n) for n in G.nodes()}
    label_pos = _label_positions_with_offset(
        G, pos, y_push=0.028 * float(layout_scale_used)
    )
    nx.draw_networkx_labels(
        G,
        label_pos,
        labels=draw_labels,
        ax=ax,
        font_size=label_font,
        font_family=font_name,
        bbox=dict(
            boxstyle="round,pad=0.32",
            facecolor="white",
            edgecolor="#bbbbbb",
            linewidth=0.65,
            alpha=0.95,
        ),
    )
    title_txt = title_override or "煤矿伤害事故局部风险链条核心网络"
    ax.set_title(
        title_txt,
        fontsize=title_font,
        fontweight="bold",
        fontfamily=font_name,
        pad=18,
    )
    foot = footnote_override or (
        "节点大小表示综合风险得分，边宽表示链条聚合权重或边风险得分，完整网络用于补充分析。"
        "（Core network of local injury risk chains）"
    )
    fig.text(0.5, 0.02, foot, ha="center", va="bottom", fontsize=footnote_font, fontfamily=font_name)
    ax.axis("off")
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    return fig


def _figure_network_overview_local_chain_simplified(nm: pd.DataFrame, em: pd.DataFrame, font_name: str) -> None:
    """
    论文正文用简化局部网络：risk_score 前 15 节点；在诱导子图上按边风险得分取前 20 边。
    完整局部网络仍由 network_overview_local_chain 输出（附录）。
    另输出 _v2（正文版）与 _large（超清大图）以提升标签可读性。
    """
    stem = "network_overview_local_chain_simplified"
    if nm is None or nm.empty or "risk_score" not in nm.columns or "node_id" not in nm.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "缺少节点指标表", ha="center", va="center", fontsize=12, fontfamily=font_name)
        ax.axis("off")
        _save_figure(fig, stem)
        return
    if em is None or em.empty or "source" not in em.columns or "target" not in em.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "缺少边指标表", ha="center", va="center", fontsize=12, fontfamily=font_name)
        ax.axis("off")
        _save_figure(fig, stem)
        return

    G = _build_local_chain_simplified_graph(nm, em)
    if G is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无可用节点", ha="center", va="center", fontsize=12, fontfamily=font_name)
        ax.axis("off")
        _save_figure(fig, stem)
        return

    # 正文展示版：较大画布 + 大字号 + multipartite 较大 scale
    body_scale = 1.55
    pos_body = _layout_local_chain_simplified(G, layout_scale=body_scale, spring_k_scale=1.15)
    fig_body = _draw_local_chain_simplified_variant(
        G,
        pos_body,
        font_name,
        figsize=(14.0, 10.0),
        label_font=11.0,
        title_font=16.0,
        footnote_font=10.5,
        arrowsize=16.0,
        layout_scale_used=body_scale,
    )
    _save_figure_to_stems(fig_body, [stem, f"{stem}_v2"], dpi=300)

    large_scale = 1.82
    pos_large = _layout_local_chain_simplified(G, layout_scale=large_scale, spring_k_scale=1.22)
    fig_large = _draw_local_chain_simplified_variant(
        G,
        pos_large,
        font_name,
        figsize=(22.0, 15.5),
        label_font=13.5,
        title_font=18.5,
        footnote_font=12.0,
        arrowsize=19.0,
        layout_scale_used=large_scale,
    )
    _save_figure_custom(fig_large, f"{stem}_large", dpi=450)


def _figure_workflow_jsr(*, font_name: str | None = None) -> None:
    """
    Figure 1 (JSR): horizontal two-row workflow; display-only, no data dependency.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fam = font_name or _pick_english_font()
    steps: list[tuple[str, str | None]] = [
        ("Accident register data", None),
        ("Data cleaning and\nreal-sample selection", None),
        ("Semantic reconstruction", "HazardMode / InjuryForm / HazardSource"),
        ("Risk node coding", None),
        ("Local risk-chain construction", "Loc -> Act -> HazardMode -> Body/InjuryForm -> Sev"),
        ("Directed weighted risk\nnetwork", None),
        ("Node, edge, chain\nand rule analysis", None),
        ("Interpretable risk\nevidence", None),
    ]

    fig, ax = plt.subplots(figsize=(14.0, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Workflow of semantic reconstruction and local risk-chain mining",
        fontsize=12,
        fontweight="bold",
        fontfamily=fam,
        pad=10,
    )

    ncols = 4
    box_w, box_h = 0.21, 0.30
    xs = [0.02 + i * 0.245 for i in range(ncols)]
    y_top, y_bot = 0.58, 0.14
    ys = [y_top if i < ncols else y_bot for i in range(len(steps))]

    def _draw_box(xc: float, yc: float, title: str, sub: str | None) -> None:
        x0 = xc - box_w / 2
        y0 = yc - box_h / 2
        patch = FancyBboxPatch(
            (x0, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.9,
            edgecolor="#555555",
            facecolor="#f8f8f8",
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(patch)
        body = title if not sub else f"{title}\n{sub}"
        ax.text(
            xc,
            yc,
            body,
            ha="center",
            va="center",
            fontsize=8.6,
            linespacing=1.12,
            fontfamily=fam,
            transform=ax.transAxes,
            clip_on=False,
        )

    for i, (title, sub) in enumerate(steps):
        col = i % ncols
        row = i // ncols
        xc = xs[col] + box_w / 2
        yc = y_top if row == 0 else y_bot
        _draw_box(xc, yc, title, sub)

    arrow_kw = dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color="#333333")

    for row in range(2):
        yc = y_top if row == 0 else y_bot
        for col in range(ncols - 1):
            x0 = xs[col] + box_w
            x1 = xs[col + 1]
            ax.add_patch(
                FancyArrowPatch(
                    (x0 + 0.01, yc),
                    (x1 - 0.01, yc),
                    transform=ax.transAxes,
                    **arrow_kw,
                )
            )

    # Row break: step 4 -> step 5
    x_end_r1 = xs[3] + box_w / 2
    x_start_r2 = xs[0] + box_w / 2
    ax.add_patch(
        FancyArrowPatch(
            (x_end_r1, y_top - box_h / 2 - 0.02),
            (x_start_r2, y_bot + box_h / 2 + 0.02),
            transform=ax.transAxes,
            connectionstyle="arc3,rad=-0.35",
            **arrow_kw,
        )
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.06)
    _save_figure_to_stems(fig, ["Figure1_workflow"], dpi=300)


def _figure_workflow_vertical_manuscript(*, font_name: str | None = None) -> None:
    """Figure 1 vertical variant (hand-drawn manuscript style)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fam = font_name or _pick_english_font()
    steps: list[tuple[str, str | None, str]] = [
        ("Accident register data", None, "#E8E8E8"),
        ("Data cleaning and\nreal-sample selection", None, "#D6EAF8"),
        ("Semantic reconstruction\nof hazard field", "HazardMode / InjuryForm / HazardSource", "#D5F5E3"),
        ("Risk node coding", None, "#FDEBD0"),
        ("Local risk-chain\nconstruction", None, "#FCF3CF"),
        ("Directed weighted risk\nnetwork", None, "#FADBD8"),
        ("Node, edge, chain and\nassociation-rule analysis", None, "#E8DAEF"),
        ("Interpretable risk\nevidence", None, "#FADBD8"),
    ]

    fig_h = 0.92 * len(steps) + 0.8
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    box_w, box_h = 0.78, 0.085
    xc = 0.5
    gap = 0.018
    total_h = len(steps) * box_h + (len(steps) - 1) * gap
    y_top = 0.5 + total_h / 2 - box_h / 2
    ys = [y_top - i * (box_h + gap) for i in range(len(steps))]

    def _draw_box(yc: float, title: str, sub: str | None, face: str) -> None:
        x0 = xc - box_w / 2
        y0 = yc - box_h / 2
        patch = FancyBboxPatch(
            (x0, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=0.9,
            edgecolor="#555555",
            facecolor=face,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(patch)
        body = title if not sub else f"{title}\n• {sub}"
        ax.text(
            xc,
            yc,
            body,
            ha="center",
            va="center",
            fontsize=10.5,
            linespacing=1.15,
            fontfamily=fam,
            transform=ax.transAxes,
            clip_on=False,
        )

    arrow_kw = dict(arrowstyle="-|>", mutation_scale=14, linewidth=1.0, color="#333333")
    for i, (title, sub, face) in enumerate(steps):
        _draw_box(ys[i], title, sub, face)
        if i < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (xc, ys[i] - box_h / 2 - 0.004),
                    (xc, ys[i + 1] + box_h / 2 + 0.004),
                    transform=ax.transAxes,
                    **arrow_kw,
                )
            )

    fig.subplots_adjust(left=0.06, right=0.94, top=0.98, bottom=0.02)
    _save_figure_to_stems(fig, ["Figure1_workflow_vertical"], dpi=300)


def _figure_node_risk_ranking_prevention_english(nm: pd.DataFrame, font_name: str) -> None:
    """Figure 2 (JSR): same subset logic as node_risk_ranking_prevention; English labels only on canvas."""
    stem = "Figure2_prevention_risk_node_ranking"
    allowed_types = PREVENTION_NODE_TYPES
    sub = nm[nm["node_type"].astype(str).str.strip().isin(allowed_types)].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No nodes available", ha="center", va="center", fontsize=12, fontfamily=font_name)
        ax.axis("off")
        _save_figure_to_stems(fig, [stem], dpi=300)
        return
    sub = sub.sort_values("risk_score", ascending=False).head(TOP_NODES)
    sub = sub.sort_values("risk_score", ascending=True)
    labels = [translate_node_label(x) for x in sub["node_label"].astype(str).tolist()]
    vals = sub["risk_score"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.barh(labels, vals, color="#2E6E9E", height=0.65)
    ax.set_xlabel("Risk score", fontsize=11, fontfamily=font_name)
    ax.set_ylabel("Node", fontsize=11, fontfamily=font_name)
    ax.set_title("Ranking of prevention-side risk nodes", fontsize=12, fontweight="bold", fontfamily=font_name)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.subplots_adjust(left=0.38)
    fig.tight_layout()
    _save_figure_to_stems(fig, [stem], dpi=300)


def _figure_core_local_risk_chain_network_english(nm: pd.DataFrame, em: pd.DataFrame, font_name: str) -> None:
    """Figure 3 (JSR): same graph filter as simplified v2; English title, caption, translated node labels."""
    stem = "Figure3_core_local_risk_chain_network"
    if nm is None or nm.empty or "risk_score" not in nm.columns or "node_id" not in nm.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Missing node metrics", ha="center", va="center", fontsize=12, fontfamily=font_name)
        ax.axis("off")
        _save_figure_to_stems(fig, [stem], dpi=300)
        return
    if em is None or em.empty or "source" not in em.columns or "target" not in em.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Missing edge metrics", ha="center", va="center", fontsize=12, fontfamily=font_name)
        ax.axis("off")
        _save_figure_to_stems(fig, [stem], dpi=300)
        return

    G = _build_local_chain_simplified_graph(nm, em)
    if G is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No nodes available", ha="center", va="center", fontsize=12, fontfamily=font_name)
        ax.axis("off")
        _save_figure_to_stems(fig, [stem], dpi=300)
        return

    body_scale = 1.55
    pos_body = _layout_local_chain_simplified(G, layout_scale=body_scale, spring_k_scale=1.15)
    title_en = "Core network of local injury risk chains"
    foot_en = (
        "Node size indicates the composite risk score. Edge width indicates aggregated chain weight "
        "or edge risk score. Only core nodes and high-risk edges are shown; the full network is used "
        "for supplementary analysis."
    )
    fig_body = _draw_local_chain_simplified_variant(
        G,
        pos_body,
        font_name,
        figsize=(15.0, 10.5),
        label_font=11.0,
        title_font=14.0,
        footnote_font=10.5,
        arrowsize=16.0,
        layout_scale_used=body_scale,
        title_override=title_en,
        footnote_override=foot_en,
        label_transform=_compact_node_label_figure3,
        label_wrap="word",
    )
    _save_figure_to_stems(fig_body, [stem], dpi=300)


def _load_severe_prevention_rules_df() -> pd.DataFrame | None:
    path = ASSOC_SEVERE_PREVENTION
    if ASSOC_SEVERE_PREVENTION_ROBUST.is_file():
        try:
            tdf = pd.read_csv(ASSOC_SEVERE_PREVENTION_ROBUST, encoding="utf-8-sig")
            if not tdf.empty:
                path = ASSOC_SEVERE_PREVENTION_ROBUST
        except Exception:
            pass
    ar: pd.DataFrame | None = None
    if path.is_file():
        ar = pd.read_csv(path, encoding="utf-8-sig")
    elif ASSOC_ALL.is_file():
        all_df = pd.read_csv(ASSOC_ALL, encoding="utf-8-sig")
        if "consequents" in all_df.columns:
            mask = all_df["consequents"].astype(str).str.contains(
                CONSEQ_SEVERE_MARK, regex=False, na=False
            )
            ar = all_df.loc[mask].copy()
            print(f"INFO: Using subset of {ASSOC_ALL.name} for English prevention rules bar.")
    if ar is None or ar.empty or "lift" not in ar.columns:
        return None
    return ar


def _figure_association_rules_severe_prevention_bar_english(
    font_name: str, *, top_n: int, stem: str, title: str, use_compact_labels: bool
) -> None:
    """Figure 4 (JSR): same CSV selection as Chinese bar; English display only."""
    ar = _load_severe_prevention_rules_df()
    if ar is None:
        print(f"WARNING: No association rules for English bar chart ({stem}), skipped.")
        return
    ar = ar.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).head(int(top_n))
    ar = ar.sort_values("lift", ascending=True)
    if use_compact_labels:
        ylabs = [_compact_rule_label(str(r.get("antecedents", ""))) for _, r in ar.iterrows()]
    else:
        ylabs = [translate_rule_label(_truncate(str(r.get("antecedents", "")))) for _, r in ar.iterrows()]
    xvals = ar["lift"].astype(float).tolist()

    h = max(4.2, 0.52 * len(ylabs) + 1.6)
    w = 8.2 if use_compact_labels else 10.5
    fig, ax = plt.subplots(figsize=(w, h))
    ax.barh(ylabs, xvals, color="#117733", height=0.62)
    ax.set_xlabel("Lift", fontsize=11, fontfamily=font_name)
    ax.set_ylabel("Rule antecedent", fontsize=11, fontfamily=font_name)
    ax.set_title(title, fontsize=12, fontweight="bold", fontfamily=font_name)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.subplots_adjust(left=0.32 if use_compact_labels else 0.38)
    fig.tight_layout()
    _save_figure_to_stems(fig, [stem], dpi=300)


def run_english_jsr_figures(nm: pd.DataFrame | None = None, em: pd.DataFrame | None = None) -> None:
    """Generate JSR English figures (Figures 1–4) without altering result tables."""
    font_name = _setup_english_matplotlib_font()
    print(f"INFO: English JSR figures, matplotlib font family: {font_name}")
    _figure_workflow_jsr(font_name=font_name)
    _figure_workflow_vertical_manuscript(font_name=font_name)
    if nm is None:
        if not NODE_METRICS.is_file():
            print("WARNING: node_metrics.csv missing, skipped Figures 2–3.")
            _figure_association_rules_severe_prevention_bar_english(
                font_name,
                top_n=8,
                stem="Figure4_top_robust_prevention_rules",
                title="Top robust prevention-side rules by lift",
                use_compact_labels=True,
            )
            _figure_association_rules_severe_prevention_bar_english(
                font_name,
                top_n=TOP_ASSOC,
                stem="Figure4_robust_prevention_association_rules",
                title="Robust prevention-side association rules for more severe injuries",
                use_compact_labels=False,
            )
            print(f"OK: English JSR figures written under {FIGURES_DIR.resolve()}")
            return
        nm = pd.read_csv(NODE_METRICS, encoding="utf-8-sig")
    if em is None:
        if not EDGE_METRICS.is_file():
            print("WARNING: edge_metrics.csv missing, skipped Figure 3.")
            em = pd.DataFrame()
        else:
            em = pd.read_csv(EDGE_METRICS, encoding="utf-8-sig")
    _figure_node_risk_ranking_prevention_english(nm, font_name)
    if not em.empty:
        _figure_core_local_risk_chain_network_english(nm, em, font_name)
    _figure_association_rules_severe_prevention_bar_english(
        font_name,
        top_n=8,
        stem="Figure4_top_robust_prevention_rules",
        title="Top robust prevention-side rules by lift",
        use_compact_labels=True,
    )
    _figure_association_rules_severe_prevention_bar_english(
        font_name,
        top_n=TOP_ASSOC,
        stem="Figure4_robust_prevention_association_rules",
        title="Robust prevention-side association rules for more severe injuries",
        use_compact_labels=False,
    )
    print(f"OK: English JSR figures (Figures 1-4) written under {FIGURES_DIR.resolve()}")


def _truncate(s: str, max_len: int = 72) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _figure_association_rules_severe_prevention_bar(font_name: str) -> None:
    path = ASSOC_SEVERE_PREVENTION
    if ASSOC_SEVERE_PREVENTION_ROBUST.is_file():
        try:
            tdf = pd.read_csv(ASSOC_SEVERE_PREVENTION_ROBUST, encoding="utf-8-sig")
            if not tdf.empty:
                path = ASSOC_SEVERE_PREVENTION_ROBUST
        except Exception:
            pass
    ar: pd.DataFrame | None = None
    if path.is_file():
        ar = pd.read_csv(path, encoding="utf-8-sig")
    elif ASSOC_ALL.is_file():
        all_df = pd.read_csv(ASSOC_ALL, encoding="utf-8-sig")
        if "consequents" in all_df.columns:
            mask = all_df["consequents"].astype(str).str.contains(
                CONSEQ_SEVERE_MARK, regex=False, na=False
            )
            ar = all_df.loc[mask].copy()
            print(f"INFO: Using subset of {ASSOC_ALL.name} for prevention severe rules bar.")
    if ar is None or ar.empty or "lift" not in ar.columns:
        print("WARNING: No association_rules_severe_prevention.csv, skipped bar chart.")
        return
    ar = ar.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).head(TOP_ASSOC)
    ar = ar.sort_values("lift", ascending=True)
    ylabs = []
    for _, r in ar.iterrows():
        ant = _truncate(str(r.get("antecedents", "")))
        con = _truncate(str(r.get("consequents", "")), 40)
        ylabs.append(f"{ant} => {con}")
    xvals = ar["lift"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(ylabs, xvals, color="#117733", height=0.6)
    ax.set_xlabel("提升度 lift", fontsize=11, fontfamily=font_name)
    use_robust = path.name.endswith("_robust.csv")
    ax.set_title(
        "预防型较严重伤害稳健关联规则（前 20，按 lift）" if use_robust else "预防型较严重伤害关联规则（前 20，按 lift）",
        fontsize=12,
        fontweight="bold",
        fontfamily=font_name,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    _save_figure(fig, "association_rules_severe_prevention_bar")


def _write_sankey_html() -> None:
    if not CHAINS.is_file():
        print(f"WARNING: Missing {CHAINS.name}, skipped Sankey.")
        return
    df = pd.read_csv(CHAINS, encoding="utf-8-sig")
    if "chain_local_1" not in df.columns:
        print("WARNING: accident_chains.csv has no chain_local_1 column, skipped Sankey.")
        return

    weight: dict[tuple[str, str], float] = defaultdict(float)
    severe_w: dict[tuple[str, str], float] = defaultdict(float)
    sb = pd.to_numeric(df.get("severe_binary"), errors="coerce")

    for i in range(len(df)):
        is_sev = bool(sb.iloc[i] == 1) if i < len(sb) and pd.notna(sb.iloc[i]) else False
        for col in ("chain_local_1", "chain_local_2"):
            if col not in df.columns:
                continue
            raw = str(df.iloc[i].get(col, ""))
            parts = [p.strip() for p in CHAIN_SEP.split(raw) if p.strip()]
            for a, b in zip(parts[:-1], parts[1:]):
                if not a or not b:
                    continue
                weight[(a, b)] += 1.0
                if is_sev:
                    severe_w[(a, b)] += 1.0

    if not weight:
        print("WARNING: Sankey edges empty, skipped.")
        return

    top_pairs = sorted(weight.items(), key=lambda kv: kv[1], reverse=True)[: min(120, len(weight))]

    labels: list[str] = []
    index: dict[str, int] = {}

    def idx(x: str) -> int:
        if x not in index:
            index[x] = len(labels)
            labels.append(x)
        return index[x]

    source, target, value = [], [], []
    for (a, b), w in top_pairs:
        if w <= 0:
            continue
        source.append(idx(a))
        target.append(idx(b))
        value.append(w)

    if not value:
        print("WARNING: Sankey edges empty after selection, skipped.")
        return

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=16,
                    line=dict(color="black", width=0.35),
                    label=labels,
                ),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )
    fig.update_layout(
        title_text="高风险局部链条 Sankey（Loc→Act→HazardMode→Body/InjuryForm→Sev 局部链相邻边，高频段）",
        font=dict(family="Microsoft YaHei, SimHei, sans-serif", size=12),
        margin=dict(l=30, r=30, t=50, b=30),
        height=720,
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_html = FIGURES_DIR / "top_risk_chains_sankey.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"OK: Sankey HTML {out_html}")


def run_local_chain_simplified_figures_only() -> None:
    """仅重绘图3（局部风险链条核心网络简化版），不跑其他图。"""
    if not NODE_METRICS.is_file():
        raise FileNotFoundError(f"未找到 {NODE_METRICS}，请先运行阶段 03。")
    if not EDGE_METRICS.is_file():
        raise FileNotFoundError(f"未找到 {EDGE_METRICS}，请先运行阶段 03。")
    font_name = _setup_matplotlib_font()
    print(f"INFO: Matplotlib CJK font family: {font_name}")
    nm = pd.read_csv(NODE_METRICS, encoding="utf-8-sig")
    em = pd.read_csv(EDGE_METRICS, encoding="utf-8-sig")
    _figure_network_overview_local_chain_simplified(nm, em, font_name)
    print(
        f"OK: network_overview_local_chain_simplified / _v2 / _large written under {FIGURES_DIR.resolve()}"
    )


def run() -> None:
    if not NODE_METRICS.is_file():
        raise FileNotFoundError(f"未找到 {NODE_METRICS}，请先运行阶段 03。")
    if not EDGE_METRICS.is_file():
        raise FileNotFoundError(f"未找到 {EDGE_METRICS}，请先运行阶段 03。")

    font_name = _setup_matplotlib_font()
    print(f"INFO: Matplotlib CJK font family: {font_name}")

    nm = pd.read_csv(NODE_METRICS, encoding="utf-8-sig")
    em = pd.read_csv(EDGE_METRICS, encoding="utf-8-sig")

    _figure_node_risk_ranking_subset(
        nm,
        font_name,
        allowed_types=PREVENTION_NODE_TYPES,
        stem="node_risk_ranking_prevention",
        title="预防型语境节点风险得分排名前 20（Team–HazardMode/HazardSource）",
        xlabel="综合风险得分 risk_score",
    )
    print(f"OK: node_risk_ranking_prevention written under {FIGURES_DIR}")

    _figure_node_risk_ranking_subset(
        nm,
        font_name,
        allowed_types=CONSEQUENCE_NODE_TYPES,
        stem="node_risk_ranking_consequence",
        title="后果型节点风险得分排名前 20（Body–InjuryForm–Sev）",
        xlabel="综合风险得分 risk_score",
    )
    print(f"OK: node_risk_ranking_consequence written under {FIGURES_DIR}")

    _figure_edge_risk_ranking(em, font_name)
    print(f"OK: edge_risk_ranking written under {FIGURES_DIR}")

    _figure_network_overview_local(nm, font_name)
    print(f"OK: network_overview_local_chain written under {FIGURES_DIR}")

    _figure_network_overview_local_chain_simplified(nm, em, font_name)
    print(f"OK: network_overview_local_chain_simplified (+_v2, +_large) written under {FIGURES_DIR}")

    _figure_association_rules_severe_prevention_bar(font_name)
    print(f"OK: association_rules_severe_prevention_bar written under {FIGURES_DIR}")

    _write_sankey_html()

    run_english_jsr_figures(nm, em)

    print(f"OK: Stage 06 finished, figures directory: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    import sys

    if "--only-local-chain-simplified" in sys.argv:
        run_local_chain_simplified_figures_only()
    elif "--english-jsr-only" in sys.argv:
        run_english_jsr_figures()
    else:
        run()
