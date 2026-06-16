# -*- coding: utf-8 -*-
"""阶段 07：汇总各阶段产出表，自动生成中文论文风格结果摘要 report_summary.md。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

CLEANED_XLSX = config.OUTPUT_DIR / "cleaned_data.xlsx"
NODE_METRICS_CSV = config.OUTPUT_DIR / "node_metrics.csv"
EDGE_METRICS_CSV = config.OUTPUT_DIR / "edge_metrics.csv"
CHAIN_FULL_SUMMARY_CSV = config.OUTPUT_DIR / "chain_full_summary.csv"
CHAIN_FULL_SUMMARY_XLSX = config.OUTPUT_DIR / "full_chain_summary.xlsx"
CHAIN_LOCAL_SUMMARY_XLSX = config.OUTPUT_DIR / "local_chain_summary.xlsx"
TOP_LOCAL_SEVERE_DEDUP_XLSX = config.OUTPUT_DIR / "top_30_local_severe_chains_dedup.xlsx"
TOP_LOCAL_SEVERE_XLSX = config.OUTPUT_DIR / "top_30_local_severe_chains.xlsx"
AR_SEVERE_PREV = config.OUTPUT_DIR / "association_rules_severe_prevention.csv"
AR_SEVERE_PREV_ROBUST = config.OUTPUT_DIR / "association_rules_severe_prevention_robust.csv"
AR_SEVERE_CONS = config.OUTPUT_DIR / "association_rules_severe_consequence.csv"
AR_SEVERE_CONS_ROBUST = config.OUTPUT_DIR / "association_rules_severe_consequence_robust.csv"
AR_SERIOUS_PREV = config.OUTPUT_DIR / "association_rules_serious_prevention.csv"
REPORT_MD = config.OUTPUT_DIR / "report_summary.md"

NODE_TYPE_ZH: dict[str, str] = {
    "Team": "队别",
    "Job": "工种（岗位）",
    "Shift": "班次",
    "Loc": "作业地点",
    "Act": "作业活动",
    "Haz": "致害物（原始台账列，未作为图谱节点前缀）",
    "HazardMode": "致害方式（接触方式）",
    "InjuryForm": "伤害形态（伤害后果）",
    "HazardSource": "致害源（物理来源，仅在有明确判定时）",
    "Body": "受伤部位",
    "Cause": "事故原因",
    "Sev": "伤害程度",
}


def _pct(x: float | None, digits: int = 2) -> str:
    if x is None or not np.isfinite(float(x)):
        return "—"
    return f"{100.0 * float(x):.{digits}f}%"


def _num(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e6:
        return str(int(round(v)))
    return f"{v:.{digits}g}"


def _node_type_zh(nt: str) -> str:
    t = str(nt).strip()
    return NODE_TYPE_ZH.get(t, t if t else "未标注类型")


def _split_prefixed_node(node_id: str) -> tuple[str, str]:
    s = str(node_id).strip()
    if ":" in s:
        p, rest = s.split(":", 1)
        return p.strip(), rest.strip()
    return "", s


def _human_label(node_id: str, node_label: Any = None) -> str:
    """用于正文表述：优先返回去前缀后的取值文本。"""
    _, raw_id = _split_prefixed_node(str(node_id))
    if node_label is None or (isinstance(node_label, float) and np.isnan(node_label)):
        return raw_id or str(node_id)
    lab = str(node_label).strip()
    if not lab:
        return raw_id or str(node_id)
    _, raw_lab = _split_prefixed_node(lab)
    return raw_lab or raw_id or lab


def _md_escape_inline(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def _load_cleaned() -> pd.DataFrame | None:
    if not CLEANED_XLSX.is_file():
        return None
    try:
        return pd.read_excel(CLEANED_XLSX, engine="openpyxl")
    except Exception:
        return None


def _section_data_overview(df: pd.DataFrame | None) -> list[str]:
    lines: list[str] = ["## 1. 数据概况", ""]
    if df is None or df.empty:
        lines.append("清洗数据表（`cleaned_data.xlsx`）暂不可用，本节略。")
        lines.append("")
        return lines

    n_total = len(df)
    syn = df["is_synthetic"] if "is_synthetic" in df.columns else pd.Series([np.nan] * n_total)
    n_real = int((syn == 0).sum())
    n_synth = int((syn.notna() & (syn != 0)).sum())
    n_syn_na = int(syn.isna().sum())

    lines.append(
        f"在现行筛选规则下，分析样本共 **{n_total}** 条；其中判定为真实事故记录 **{n_real}** 条，"
        f"合成或其他标记样本 **{n_synth}** 条"
        + (f"（另有 `is_synthetic` 缺失 **{n_syn_na}** 条）" if n_syn_na else "")
        + "。"
    )
    lines.append("")

    lines.append(
        "关于原始台账中的「致害物」字段：本数据里该列**主要记录致害方式与伤害形态**，并不总是物理意义上的致害物体来源。"
        "因此分析中将该列拆分为 **HazardMode（致害方式）**、**InjuryForm（伤害形态）** 与 **HazardSource（致害源，仅在可明确判定时）**，"
        "以避免把骨折、离断、内脏破裂、聋等**结果性表述**误读为事故发生前的独立风险因素。"
    )
    lines.append("")

    if "Sev" in df.columns:
        sev_vc = df["Sev"].value_counts(dropna=False)
        lines.append("伤害程度（`Sev`）分布如下：")
        for lab, cnt in sev_vc.items():
            prop = float(cnt) / float(n_total) if n_total else float("nan")
            lab_s = "缺失" if pd.isna(lab) else str(lab)
            lines.append(f"- {lab_s}：{int(cnt)} 条（占全部分析记录的 {_pct(prop)}）")
        lines.append("")
    else:
        lines.append("（表中缺少 `Sev` 列，无法列出伤害程度分布。）")
        lines.append("")

    if "severe_binary" in df.columns:
        sb = pd.to_numeric(df["severe_binary"], errors="coerce")
        valid = sb.notna()
        n_v = int(valid.sum())
        if n_v > 0:
            severe_n = int((sb == 1).sum())
            severe_p = float(severe_n) / float(n_v)
            lines.append(
                f"按项目定义，将轻伤及以上视为「较严重伤害」（`severe_binary=1`）。"
                f"在 `severe_binary` 有效填答的 **{n_v}** 条记录中，较严重伤害 **{severe_n}** 条，"
                f"占比 **{_pct(severe_p)}**。"
            )
            if n_v < n_total:
                lines.append(
                    f"另有 **{n_total - n_v}** 条记录因伤类标签未纳入二值映射而未计入上述比例，"
                    "可在数据字典中扩展映射或回溯原始标注后重算。"
                )
        else:
            lines.append("`severe_binary` 无有效取值，无法计算较严重伤害比例。")
    else:
        lines.append("（缺少 `severe_binary` 列。）")
    lines.append("")
    return lines


def _build_graph_from_edges(em: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    if em is None or em.empty:
        return G
    for _, r in em.iterrows():
        u, v = str(r["source"]), str(r["target"])
        w = float(r.get("weight", 1.0))
        if not np.isfinite(w) or w < 0:
            w = 0.0
        if G.has_edge(u, v):
            G[u][v]["weight"] = float(G[u][v].get("weight", 0.0)) + w
        else:
            G.add_edge(u, v, weight=w)
    return G


def _section_kg_overview(nm: pd.DataFrame | None, em: pd.DataFrame | None) -> list[str]:
    lines: list[str] = ["## 2. 风险知识图谱概况", ""]
    if em is None or em.empty:
        lines.append("`edge_metrics.csv` 缺失或为空，无法汇总网络结构指标。")
        lines.append("")
        return lines

    G = _build_graph_from_edges(em)
    if nm is not None and not nm.empty and "node_id" in nm.columns:
        for nid in nm["node_id"].astype(str):
            if nid not in G:
                G.add_node(nid)
    n_n = G.number_of_nodes()
    n_e = G.number_of_edges()
    if n_n == 0:
        lines.append("边表为空或无法构图，网络指标略。")
        lines.append("")
        return lines

    dens = float(nx.density(G))
    degs = [int(G.degree(n)) for n in G.nodes()]
    avg_deg = float(np.mean(degs)) if degs else 0.0

    try:
        wccs = list(nx.weakly_connected_components(G))
        max_wcc = max(len(c) for c in wccs) if wccs else 0
    except Exception:
        max_wcc = 0

    lines.append(
        f"以事故链条相邻关系与原因指向关系构建的有向风险网络中，共抽象 **{n_n}** 个节点、**{n_e}** 条有向边。"
        f"网络密度采用有向简单图常用定义 **|E| / (N(N-1))**，数值为 **{_num(dens, digits=6)}**，"
        f"节点的平均（出度+入度）度为 **{_num(avg_deg, digits=4)}**。"
    )
    lines.append(
        f"在弱连通意义下，最大弱连通子图包含 **{max_wcc}** 个节点，"
        "可理解为多数风险因素在同一结构块中相互可达的程度。"
    )
    lines.append("")
    return lines


def _section_top_nodes(nm: pd.DataFrame | None) -> list[str]:
    lines: list[str] = ["## 3. 关键风险节点", ""]
    if nm is None or nm.empty or "risk_score" not in nm.columns:
        lines.append("节点指标表不可用，本节略。")
        lines.append("")
        return lines

    prev_types = {"Team", "Job", "Shift", "Loc", "Act", "HazardMode", "HazardSource"}
    cons_types = {"Body", "InjuryForm", "Sev"}

    lines.append("### 3.1 预防型风险因素（Team、Job、Shift、Loc、Act、HazardMode、HazardSource）")
    lines.append("")
    lines.append(
        "**HazardMode** 表示致害方式或接触方式；**HazardSource** 仅在原始文本可映射为明确物体/环境来源时给出，否则为 `NotAvailable`。"
        "下列节点按 `risk_score` 在预防型类型内排序（前十）。"
    )
    lines.append("")
    sub_p = nm[nm["node_type"].astype(str).str.strip().isin(prev_types)].sort_values(
        "risk_score", ascending=False
    ).head(10)
    if sub_p.empty:
        lines.append("（当前无预防型节点指标行。）")
    else:
        for _, r in sub_p.iterrows():
            nt = str(r.get("node_type", "")).strip()
            label = _human_label(str(r.get("node_id", "")), r.get("node_label")) or str(r.get("node_id", ""))
            lines.append(
                f"- **{_node_type_zh(nt)}** — `{_md_escape_inline(label)}` "
                f"（`risk_score`={_num(r.get('risk_score'))}，严重率≈{_pct(float(r.get('severity_rate', np.nan))) if pd.notna(r.get('severity_rate')) else '—'}）"
            )
    lines.append("")

    lines.append("### 3.2 后果型因素（Body、InjuryForm、Sev）")
    lines.append("")
    lines.append(
        "**InjuryForm** 为伤害形态或伤害后果；**Sev** 为伤害程度结果；**Body** 为承伤部位。"
        "三者**不能**解释为事故发生前的预防因素，仅用于刻画伤害表现与结局语境。"
    )
    lines.append("")
    sub_c = nm[nm["node_type"].astype(str).str.strip().isin(cons_types)].sort_values(
        "risk_score", ascending=False
    ).head(10)
    if sub_c.empty:
        lines.append("（当前无后果型节点指标行。）")
    else:
        for _, r in sub_c.iterrows():
            nt = str(r.get("node_type", "")).strip()
            label = _human_label(str(r.get("node_id", "")), r.get("node_label")) or str(r.get("node_id", ""))
            lines.append(
                f"- **{_node_type_zh(nt)}** — `{_md_escape_inline(label)}` "
                f"（`risk_score`={_num(r.get('risk_score'))}，严重率≈{_pct(float(r.get('severity_rate', np.nan))) if pd.notna(r.get('severity_rate')) else '—'}）"
            )
    lines.append("")
    return lines


def _section_top_edges(em: pd.DataFrame | None) -> list[str]:
    lines: list[str] = ["## 4. 关键风险边", ""]
    if em is None or em.empty:
        lines.append("边指标表缺失，本节略。")
        lines.append("")
        return lines

    df = em.copy()
    wcol = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    esr = pd.to_numeric(df["edge_severity_rate"], errors="coerce").fillna(0.0)
    df["_edge_risk"] = wcol * esr
    df = df.sort_values("_edge_risk", ascending=False).head(10)

    lines.append(
        "以下按 **边风险得分**（`weight × edge_severity_rate`，与阶段 03 中边排序一致）"
        "列出前十条高风险有向边，用于识别事故记录中**共现最强且较严重伤害占比较高**的相邻风险转移。"
    )
    lines.append("")
    for _, r in df.iterrows():
        su, sv = str(r["source"]), str(r["target"])
        pu, _ = _split_prefixed_node(su)
        pv, _ = _split_prefixed_node(sv)
        lift = r.get("lift_to_severe", np.nan)
        lift_s = _num(lift, digits=4) if pd.notna(lift) and np.isfinite(float(lift)) else "—"
        wt = pd.to_numeric(r.get("weight"), errors="coerce")
        lines.append(
            f"- `{_md_escape_inline(su)}` → `{_md_escape_inline(sv)}`："
            f"聚合权重 **{_num(wt, digits=0)}**，"
            f"边上较严重伤害比例 **{_pct(float(r.get('edge_severity_rate', np.nan)))}**，"
            f"相对全局较严重伤害率的提升度（lift）≈ **{lift_s}** "
            f"（{_node_type_zh(pu)}→{_node_type_zh(pv)}）。"
        )
    lines.append("")
    lines.append(
        "综合解读： lift 明显高于 1 的边表示「在该邻接关系出现条件下，较严重伤害的发生强度」"
        "高于样本整体基准；结合较大的聚合权重，可视为统计意义上**联系最强、最值得优先排查的相邻风险组合**。"
    )
    lines.append("")
    return lines


def _load_chain_full_summary() -> pd.DataFrame | None:
    if CHAIN_FULL_SUMMARY_XLSX.is_file():
        try:
            return pd.read_excel(CHAIN_FULL_SUMMARY_XLSX, engine="openpyxl")
        except Exception:
            pass
    if CHAIN_FULL_SUMMARY_CSV.is_file():
        try:
            return pd.read_csv(CHAIN_FULL_SUMMARY_CSV, encoding="utf-8-sig")
        except Exception:
            pass
    return None


def _load_local_chain_summary() -> pd.DataFrame | None:
    if not CHAIN_LOCAL_SUMMARY_XLSX.is_file():
        return None
    try:
        return pd.read_excel(CHAIN_LOCAL_SUMMARY_XLSX, engine="openpyxl")
    except Exception:
        return None


def _load_high_risk_local_severe_for_section_52() -> tuple[pd.DataFrame | None, str]:
    """
    第 5.2 节正文表：优先读取去重版 dedup；仅保留 severe_rate > 0。
    若 dedup 文件缺失或读入后无满足条件的行，回退读取原版 top_30（仍仅展示 severe_rate > 0），
    不把轻微伤链条（严重率为 0）当作高风险链条展示。
    """
    order = [
        (TOP_LOCAL_SEVERE_DEDUP_XLSX, "`top_30_local_severe_chains_dedup.xlsx`"),
        (TOP_LOCAL_SEVERE_XLSX, "`top_30_local_severe_chains.xlsx`（回退，已筛 severe_rate>0）"),
    ]
    for p, lab in order:
        if not p.is_file():
            continue
        try:
            t = pd.read_excel(p, engine="openpyxl")
        except Exception:
            continue
        if t.empty or "severe_rate" not in t.columns:
            continue
        t = t.copy()
        sr = pd.to_numeric(t["severe_rate"], errors="coerce")
        t = t.loc[sr > 0].reset_index(drop=True)
        if not t.empty:
            return t, lab
    return None, ""


def _try_read_rules_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None


def _section_top_chains() -> list[str]:
    lines: list[str] = ["## 5. 高风险链条（完整链为补充，主分析看局部链）", ""]

    ch_full = _load_chain_full_summary()
    if ch_full is None or ch_full.empty:
        lines.append("`full_chain_summary.xlsx` / `chain_full_summary.csv` 缺失或为空，完整链汇总略。")
        lines.append("")
    else:
        col = "chain_full" if "chain_full" in ch_full.columns else ch_full.columns[0]
        min_c = int(config.MIN_CHAIN_COUNT)
        t = ch_full.copy()
        t["count"] = pd.to_numeric(t["count"], errors="coerce").fillna(0).astype(int)
        t["severe_rate"] = pd.to_numeric(t["severe_rate"], errors="coerce")
        t = t.loc[t["count"] >= min_c].sort_values(
            by=["severe_rate", "count", "average_severity_score"],
            ascending=[False, False, False],
            na_position="last",
        ).head(5)
        lines.append(
            f"完整链条（Team→…→Sev）在 `count>={min_c}` 下仅作补充；当前列出至多 **5** 条（按 `severe_rate` 优先）。"
        )
        lines.append("")
        if t.empty:
            lines.append("在现行频次阈值下无满足条件的完整链条，这在小样本场景下属正常，不代表程序错误。")
        else:
            for _, r in t.iterrows():
                cnt = int(r["count"])
                sr = float(r["severe_rate"]) if pd.notna(r.get("severe_rate")) else float("nan")
                lines.append(
                    f"- `{_md_escape_inline(str(r.get(col, '')))}`：出现 **{cnt}** 次，较严重伤害占比 **{_pct(sr)}**。"
                )
        lines.append("")

    loc = _load_local_chain_summary()
    lines.append("### 5.1 局部风险链条（论文主分析）")
    lines.append("")
    lines.append(
        "主分析关注以下语境片段：**Loc → Act → HazardMode → Body / InjuryForm → Sev**（无致害方式时可为 **Loc → Act → Body → InjuryForm → Sev**）。"
        "其中 **HazardMode** 表示致害方式；**InjuryForm** 表示伤害形态；**Sev** 为伤害程度结局。"
        "**Body、InjuryForm、Sev 不应被解释为事故发生前的预防因素**。"
    )
    lines.append("")
    if loc is None or loc.empty:
        lines.append("`local_chain_summary.xlsx` 缺失或为空（方法复核表不可用）。")
        lines.append("")
    else:
        min_c = int(config.MIN_CHAIN_COUNT)
        loc2 = loc.copy()
        loc2["count"] = pd.to_numeric(loc2["count"], errors="coerce").fillna(0).astype(int)
        loc2 = loc2.loc[loc2["count"] >= min_c].sort_values(
            by=["severe_rate", "count", "average_severity_score"],
            ascending=[False, False, False],
            na_position="last",
        ).head(10)
        if loc2.empty:
            lines.append("局部链在现行阈值下无满足条件的汇总行。")
        else:
            for _, r in loc2.iterrows():
                layer = str(r.get("chain_layer", ""))
                chs = str(r.get("chain", ""))
                cnt = int(r["count"])
                sr = float(r["severe_rate"]) if pd.notna(r.get("severe_rate")) else float("nan")
                lines.append(
                    f"- **{layer}**：`{_md_escape_inline(chs)}` — 出现 **{cnt}** 次，较严重伤害占比 **{_pct(sr)}**。"
                )
        lines.append("")
    top_tbl, top_src = _load_high_risk_local_severe_for_section_52()
    lines.append("### 5.2 高风险局部链条，按严重伤害率和频次筛选")
    lines.append("")
    if top_tbl is None or top_tbl.empty:
        lines.append(
            "`top_30_local_severe_chains_dedup.xlsx` 不可用，或经 `severe_rate>0` 筛选后无可用行；"
            "本节不展示高风险局部链条摘录。"
        )
    else:
        lines.append(
            "经过去重和严重伤害筛选后，正文仅展示 **severe_rate 大于 0** 且满足最小频次阈值（`count>="
            f"{int(config.MIN_CHAIN_COUNT)}`）的局部链条；下列摘自 **{top_src}**，"
            "排序与阶段 04 中 dedup 表一致（严重率、严重条数、频次、平均伤害分降序）。"
        )
        lines.append("")
        t2 = top_tbl.copy()
        if "count" in t2.columns:
            t2["count"] = pd.to_numeric(t2["count"], errors="coerce").fillna(0).astype(int)
        if "severe_rate" in t2.columns:
            t2["severe_rate"] = pd.to_numeric(t2["severe_rate"], errors="coerce")
        t2 = t2.head(10)
        ch_col = "chain" if "chain" in t2.columns else t2.columns[0]
        for _, r in t2.iterrows():
            chs = str(r.get(ch_col, ""))
            layer = str(r.get("chain_layer", "")) if "chain_layer" in t2.columns else ""
            cnt = int(r["count"]) if "count" in t2.columns else 0
            sr = float(r["severe_rate"]) if "severe_rate" in t2.columns and pd.notna(r.get("severe_rate")) else float("nan")
            layer_s = f"**{layer}**：" if layer else ""
            lines.append(
                f"- {layer_s}`{_md_escape_inline(chs)}` — 出现 **{cnt}** 次，较严重伤害占比 **{_pct(sr)}**。"
            )
    lines.append("")
    return lines


def _rule_lines(title: str, path: Path, k: int = 10) -> list[str]:
    lines: list[str] = [f"### {title}", ""]
    if not path.is_file():
        lines.append(f"`{path.name}` 缺失。")
        lines.append("")
        return lines
    ru = pd.read_csv(path, encoding="utf-8-sig")
    if ru.empty:
        lines.append("规则表为空（可调低支持度/置信度/提升度阈值后重跑阶段 05）。")
        lines.append("")
        return lines
    ru = ru.sort_values("lift", ascending=False).head(k).reset_index(drop=True)
    for _, r in ru.iterrows():
        lines.append(
            f"- **IF** `{_md_escape_inline(str(r.get('antecedents', '')))}` **THEN** "
            f"`{_md_escape_inline(str(r.get('consequents', '')))}` — "
            f"lift={_num(r.get('lift'))}, confidence={_num(r.get('confidence'))}, support={_num(r.get('support'))}"
        )
    lines.append("")
    return lines


def _section_association_rules() -> list[str]:
    lines: list[str] = ["## 6. 关联规则（分三类）", ""]
    lines.append(
        "关联规则在「后件为单一结局项」约束下分为：**severe_prevention**（预防型较严重伤害）、"
        "**severe_consequence**（后果语境下的较严重伤害）、**serious_prevention**（预防型重伤）。"
        "正文优先展示 **稳健子集**（`antecedent_count`≥8、`rule_count`≥8、`lift`≥1.20、`confidence`≥0.60），"
        "以降低小样本下高置信但低频次规则带来的偶然性；**原始规则文件仍保留**作探索性对照。"
    )
    lines.append("")
    lines.append(
        "预防型规则前项仅含组织—作业语境与 **HazardMode、HazardSource**，**不得**将 **Body、InjuryForm、Sev、SevereBinary** 当作前项；"
        "后果型规则前项仅含 **Body、InjuryForm、HazardMode、Loc、Act**；前项中**严禁**包含 `Sev:`、`SevereBinary:` 与原始台账「致害物」列对应的图谱前缀项。"
    )
    lines.append("")
    lines.append(
        "**Body、InjuryForm、Sev** 属于**后果语境变量**，只能用于后果型规则或网络中的结局侧讨论，"
        "**不能**解释为事故发生前的预防因素；**HazardMode、Loc、Act、Shift、Job、Team** 等更适合进入预防型讨论。"
    )
    lines.append("")

    rob_prev = _try_read_rules_csv(AR_SEVERE_PREV_ROBUST)
    raw_prev = _try_read_rules_csv(AR_SEVERE_PREV)
    rob_cons = _try_read_rules_csv(AR_SEVERE_CONS_ROBUST)
    raw_cons = _try_read_rules_csv(AR_SEVERE_CONS)

    n_rob_p = len(rob_prev) if rob_prev is not None else 0
    disp_prev = rob_prev if n_rob_p > 0 else raw_prev
    prev_label = (
        "6.1 预防型较严重伤害（稳健：`association_rules_severe_prevention_robust.csv`）"
        if n_rob_p > 0
        else "6.1 预防型较严重伤害（稳健规则为空，回退：`association_rules_severe_prevention.csv`）"
    )
    if disp_prev is None or disp_prev.empty:
        lines.extend(_rule_lines(prev_label, AR_SEVERE_PREV, 10))
    else:
        p_used = AR_SEVERE_PREV_ROBUST if n_rob_p > 0 else AR_SEVERE_PREV
        lines.extend(_rule_lines(prev_label, p_used, 10))
    if n_rob_p > 0 and n_rob_p < 5:
        lines.append(
            "**说明**：稳健预防型规则少于 5 条，提示小样本下统计稳定规则有限；"
            f"`{AR_SEVERE_PREV.name}` 中其余规则仅宜作为**探索性补充**，不宜单独作为强证据表述。"
        )
        lines.append("")
        if raw_prev is not None and not raw_prev.empty:
            lines.append("以下为原始预防型规则节选（探索性，按 lift 排序至多 5 条）：")
            lines.append("")
            rv = raw_prev.sort_values("lift", ascending=False).head(5).reset_index(drop=True)
            for _, r in rv.iterrows():
                lines.append(
                    f"- **IF** `{_md_escape_inline(str(r.get('antecedents', '')))}` **THEN** "
                    f"`{_md_escape_inline(str(r.get('consequents', '')))}` — "
                    f"lift={_num(r.get('lift'))}, confidence={_num(r.get('confidence'))}, support={_num(r.get('support'))}"
                )
            lines.append("")

    n_rob_c = len(rob_cons) if rob_cons is not None else 0
    disp_cons = rob_cons if n_rob_c > 0 else raw_cons
    cons_label = (
        "6.2 后果型较严重伤害（稳健：`association_rules_severe_consequence_robust.csv`）"
        if n_rob_c > 0
        else "6.2 后果型较严重伤害（稳健规则为空，回退：`association_rules_severe_consequence.csv`）"
    )
    if disp_cons is None or disp_cons.empty:
        lines.extend(_rule_lines(cons_label, AR_SEVERE_CONS, 10))
    else:
        c_used = AR_SEVERE_CONS_ROBUST if n_rob_c > 0 else AR_SEVERE_CONS
        lines.extend(_rule_lines(cons_label, c_used, 10))
    if n_rob_c > 0 and n_rob_c < 5:
        lines.append(
            "**说明**：稳健后果型规则少于 5 条，小样本下稳定规则有限；"
            f"`{AR_SEVERE_CONS.name}` 中其余规则仅作**探索性对照**。"
        )
        lines.append("")
        if raw_cons is not None and not raw_cons.empty:
            lines.append("以下为原始后果型规则节选（探索性，按 lift 排序至多 5 条）：")
            lines.append("")
            rv = raw_cons.sort_values("lift", ascending=False).head(5).reset_index(drop=True)
            for _, r in rv.iterrows():
                lines.append(
                    f"- **IF** `{_md_escape_inline(str(r.get('antecedents', '')))}` **THEN** "
                    f"`{_md_escape_inline(str(r.get('consequents', '')))}` — "
                    f"lift={_num(r.get('lift'))}, confidence={_num(r.get('confidence'))}, support={_num(r.get('support'))}"
                )
            lines.append("")

    lines.extend(
        _rule_lines("6.3 预防型重伤（`association_rules_serious_prevention.csv`，不设稳健过滤）", AR_SERIOUS_PREV, 10)
    )
    lines.append(
        "解读提示：较高的 lift 表示前项与后项共现强度高于随机期望；"
        "对 **severe_prevention** 与 **serious_prevention** 可结合工程与管理干预讨论；"
        "对 **severe_consequence** 应明确其为**伤害表现与语境共现**，不可反向解释为事故前可控风险清单，"
        "更不应把骨折、离断、内脏破裂、聋等结果性词语写成事故前风险因素。"
    )
    lines.append("")
    return lines


def _section_discussion(
    top_locs: list[str],
    top_acts: list[str],
    top_modes: list[str],
    top_sources: list[str],
    top_bodies: list[str],
    top_inj: list[str],
) -> list[str]:
    lines: list[str] = ["## 7. 管理启示", ""]
    lines.append(
        "结合上述网络结构、链条聚合与关联规则结果，从事故记录可提炼出若干**面向防控讨论**的观察，"
        "供论文「讨论」部分参考表述（具体措辞应结合本单位制度与现场核验结果调整）。"
    )
    lines.append("")

    def _bul(items: list[str], label: str) -> str:
        xs = [f"「{x}」" for x in items[:5] if x]
        if not xs:
            return f"样本中{label}的高分实体较为分散，宜在报告中以表格为准展开个案。"
        return "、".join(xs)

    lines.append(
        f"第一，**作业地点—作业活动**的耦合往往决定了暴露情境与行为触发方式。"
        f"在高分节点中，地点侧要素如 {_bul(top_locs, '地点')} 与活动侧要素如 {_bul(top_acts, '活动')} "
        "若同时出现在链条前段，提示现场布局、通行/交叉作业组织与「何处、在做什么」并表管理的重要性。"
    )
    lines.append("")

    lines.append(
        f"第二，**致害方式（HazardMode）—致害源（HazardSource）—受伤部位/伤害形态**共同刻画暴露与后果之间的结构关系。"
        f"致害方式侧如 {_bul(top_modes, '致害方式')}；在确有判定时，致害源侧如 {_bul(top_sources, '致害源')}；"
        f"部位侧如 {_bul(top_bodies, '受伤部位')}、伤害形态如 {_bul(top_inj, '伤害形态')} 若在局部链中反复共现，宜区分："
        "**HazardMode / HazardSource（在可判定时）** 可作为防控讨论对象；**Body / InjuryForm** 主要反映伤害表现，不宜反向解释为事故前风险清单。"
    )
    lines.append("")

    lines.append(
        "第三，**伤害程度作为链条末端**，在方法上被用于度量路径的「后果严重性」。"
        "当多条路径在相同或相近的前端语境下收敛到更高的较严重伤害比例时，应优先审视该语境下是否存在系统性防护失效，"
        "例如规程缺口、联保互保不到位、应急与急救准备不足等组织因素（需结合定性调查进一步论证）。"
    )
    lines.append("")

    lines.append(
        "第四，**关联规则**所揭示的多维共现模式，可用于提示「多个低风险信号叠加」时的脆弱性，"
        "适合作为培训案例库与检查清单更新的证据来源；但其解释应始终与现场工艺、设备状态与管理记录交叉验证。"
    )
    lines.append("")
    return lines


def _extract_top_labels(nm: pd.DataFrame, node_type: str, k: int = 5) -> list[str]:
    if nm is None or nm.empty or "node_type" not in nm.columns:
        return []
    sub = nm[nm["node_type"].astype(str).str.strip() == node_type].copy()
    if sub.empty or "risk_score" not in sub.columns:
        return []
    sub = sub.sort_values("risk_score", ascending=False).head(k)
    labs: list[str] = []
    for _, r in sub.iterrows():
        lab = _human_label(str(r.get("node_id", "")), r.get("node_label"))
        if lab:
            labs.append(lab)
    return labs


def _section_limits() -> list[str]:
    return [
        "## 8. 方法边界",
        "",
        "本流水线从事故台账字段中抽取**有序链条与多维项集**，在图模型与关联规则框架下识别"
        "统计上的高频结构、较高严重率的路径片段及提升度较高的共现规则。",
        "",
        "需要强调的是：上述结果刻画的是记录内的**统计关联与风险传导路径的可计算表征**，"
        "并不构成对单一事件致因机制的司法或工程意义上的**因果关系证明**；"
        "任何干预决策仍应结合法规要求、工艺安全分析、人机工效学与现场观察证据进行综合判断。",
        "",
    ]


def run() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = _load_cleaned()
    nm = None
    em = None
    if NODE_METRICS_CSV.is_file():
        try:
            nm = pd.read_csv(NODE_METRICS_CSV, encoding="utf-8-sig")
        except Exception:
            nm = None
    if EDGE_METRICS_CSV.is_file():
        try:
            em = pd.read_csv(EDGE_METRICS_CSV, encoding="utf-8-sig")
        except Exception:
            em = None

    parts: list[str] = [
        "# 风险分析结果摘要",
        "",
        "以下为基于本项目流水线产出表自动整理的结果性描述，措辞对齐学术论文「结果」小节；"
        "模型参数与筛选阈值以项目根目录 `config.py` 为准。",
        "",
    ]
    parts.extend(_section_data_overview(cleaned))
    parts.extend(_section_kg_overview(nm, em))
    parts.extend(_section_top_nodes(nm))
    parts.extend(_section_top_edges(em))

    parts.extend(_section_top_chains())
    parts.extend(_section_association_rules())

    top_locs = _extract_top_labels(nm, "Loc") if nm is not None else []
    top_acts = _extract_top_labels(nm, "Act") if nm is not None else []
    top_modes = _extract_top_labels(nm, "HazardMode") if nm is not None else []
    top_sources = _extract_top_labels(nm, "HazardSource") if nm is not None else []
    top_bodies = _extract_top_labels(nm, "Body") if nm is not None else []
    top_inj = _extract_top_labels(nm, "InjuryForm") if nm is not None else []
    parts.extend(_section_discussion(top_locs, top_acts, top_modes, top_sources, top_bodies, top_inj))
    parts.extend(_section_limits())

    text = "\n".join(parts).rstrip() + "\n"
    REPORT_MD.write_text(text, encoding="utf-8")
    print(f"OK: Wrote {REPORT_MD.resolve()}")


if __name__ == "__main__":
    run()
