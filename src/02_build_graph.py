# -*- coding: utf-8 -*-
"""阶段 02：由清洗表构建风险节点、有向边与事故链条（不含网络指标、关联规则与作图）。"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

CLEANED_PATH = config.OUTPUT_DIR / "cleaned_data.xlsx"
NODES_CSV = config.OUTPUT_DIR / "nodes.csv"
EDGES_CSV = config.OUTPUT_DIR / "edges.csv"
CHAINS_CSV = config.OUTPUT_DIR / "accident_chains.csv"


def _prefixed_node(node_type: str, raw: str) -> str:
    s = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw).strip()
    return f"{node_type}:{s}"


def _raw_display(prefixed: str) -> str:
    if ":" not in prefixed:
        return prefixed
    return prefixed.split(":", 1)[1]


def _node_type_from_id(prefixed: str) -> str:
    if ":" not in prefixed:
        return ""
    return prefixed.split(":", 1)[0]


def _cause_valid(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "unknown"}:
        return False
    return True


def _skip_unknown_dim(v) -> bool:
    """HazardMode / InjuryForm 为 Unknown、空、占位时跳过该节点。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    s = str(v).strip()
    if s == "":
        return True
    low = s.lower()
    return low in {"unknown", "notapplicable", "nan", "none", "-"}


def _skip_hazard_source(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    s = str(v).strip()
    if s == "":
        return True
    low = s.lower()
    return low in {"unknown", "notapplicable", "nan", "none", "-", "notavailable"}


def _chain_str(nodes: list[str]) -> str:
    return " -> ".join(nodes)


def _row_prefixed(row: pd.Series, node_type: str, col: str) -> str:
    v = row.get(col)
    s = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
    return _prefixed_node(node_type, s)


def _main_chain_nodes(row: pd.Series) -> list[str]:
    """
    Team -> Job -> Shift -> Loc -> Act -> [HazardMode] -> Body -> [InjuryForm] -> Sev
    HazardMode / InjuryForm 为 Unknown 时跳过对应段。
    """
    order_types = ["Team", "Job", "Shift", "Loc", "Act"]
    order_cols = ["Team", "Job", "Shift", "Loc", "Act"]
    nodes = [
        _prefixed_node(t, str(row[c]) if row[c] is not None and not pd.isna(row[c]) else "")
        for t, c in zip(order_types, order_cols)
    ]
    if not _skip_unknown_dim(row.get("HazardMode")):
        nodes.append(
            _prefixed_node("HazardMode", str(row.get("HazardMode", "")).strip())
        )
    nodes.append(_prefixed_node("Body", str(row.get("Body", "")) if row.get("Body") is not None and not pd.isna(row.get("Body")) else ""))
    if not _skip_unknown_dim(row.get("InjuryForm")):
        nodes.append(_prefixed_node("InjuryForm", str(row.get("InjuryForm", "")).strip()))
    nodes.append(_prefixed_node("Sev", str(row.get("Sev", "")) if row.get("Sev") is not None and not pd.isna(row.get("Sev")) else ""))
    return nodes


def run() -> None:
    if not CLEANED_PATH.is_file():
        raise FileNotFoundError(f"未找到清洗表: {CLEANED_PATH.resolve()} — 请先运行阶段 01。")

    df = pd.read_excel(CLEANED_PATH, engine="openpyxl")
    required = [
        "Team",
        "Job",
        "Shift",
        "Loc",
        "Act",
        "HazardMode",
        "InjuryForm",
        "HazardSource",
        "Body",
        "Sev",
        "Cause",
        "severe_binary",
        "severity_score",
    ]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"cleaned_data.xlsx 缺少列: {miss}")

    df = df.reset_index(drop=True)

    node_freq: dict[str, int] = defaultdict(int)
    node_severe: dict[str, int] = defaultdict(int)
    node_sev_sum: dict[str, float] = defaultdict(float)
    node_sev_n: dict[str, int] = defaultdict(int)

    edge_w: dict[tuple[str, str], int] = defaultdict(int)
    edge_severe: dict[tuple[str, str], int] = defaultdict(int)
    edge_sev_sum: dict[tuple[str, str], float] = defaultdict(float)
    edge_sev_n: dict[tuple[str, str], int] = defaultdict(int)

    chain_rows: list[dict] = []

    for i, row in df.iterrows():
        record_id = int(i)
        main_nodes = _main_chain_nodes(row)

        sev_raw = row["Sev"]
        sb = row["severe_binary"]
        sc = row["severity_score"]

        is_severe = int(sb) == 1 if pd.notna(sb) and sb is not None else False
        sc_f = float(sc) if pd.notna(sc) and sc is not None else np.nan

        nodes_in_rec: set[str] = set(main_nodes)
        edges_in_rec: list[tuple[str, str]] = []

        for a, b in zip(main_nodes[:-1], main_nodes[1:]):
            edges_in_rec.append((a, b))

        type_to_node = {_node_type_from_id(n): n for n in main_nodes}
        loc_n = type_to_node.get("Loc")
        act_n = type_to_node.get("Act")
        body_n = type_to_node.get("Body")
        sev_n = type_to_node.get("Sev")
        if not _skip_hazard_source(row.get("HazardSource")) and loc_n and act_n and body_n and sev_n:
            src_n = _prefixed_node("HazardSource", str(row.get("HazardSource", "")).strip())
            nodes_in_rec.add(src_n)
            for u, v in ((loc_n, act_n), (act_n, src_n), (src_n, body_n), (body_n, sev_n)):
                if u is not None and v is not None:
                    edges_in_rec.append((u, v))

        cause_v = row["Cause"]
        cause_node: str | None = None
        if _cause_valid(cause_v):
            cause_node = _prefixed_node("Cause", str(cause_v).strip())
            nodes_in_rec.add(cause_node)
            hazm_n = type_to_node.get("HazardMode")
            for u, v in ((cause_node, act_n), (cause_node, hazm_n), (cause_node, sev_n)):
                if u is not None and v is not None:
                    edges_in_rec.append((u, v))

        for nid in nodes_in_rec:
            node_freq[nid] += 1
            if is_severe:
                node_severe[nid] += 1
            if not np.isnan(sc_f):
                node_sev_sum[nid] += sc_f
                node_sev_n[nid] += 1

        seen_e = set(edges_in_rec)
        for u, v in seen_e:
            edge_w[(u, v)] += 1
            if is_severe:
                edge_severe[(u, v)] += 1
            if not np.isnan(sc_f):
                edge_sev_sum[(u, v)] += sc_f
                edge_sev_n[(u, v)] += 1

        loc_n = type_to_node.get("Loc", "")
        act_n = type_to_node.get("Act", "")
        body_n = type_to_node.get("Body", "")
        sev_n = type_to_node.get("Sev", "")
        hm_n = type_to_node.get("HazardMode") if "HazardMode" in type_to_node else None
        inj_n = type_to_node.get("InjuryForm") if "InjuryForm" in type_to_node else None
        job_n = type_to_node.get("Job", "")
        shift_n = type_to_node.get("Shift", "")

        if hm_n:
            chain_local_1 = _chain_str([loc_n, act_n, hm_n, body_n, sev_n])
        else:
            parts1 = [loc_n, act_n, body_n]
            if inj_n:
                parts1.append(inj_n)
            parts1.append(sev_n)
            chain_local_1 = _chain_str(parts1)

        if hm_n:
            parts2 = [loc_n, act_n, hm_n, body_n]
            if inj_n:
                parts2.append(inj_n)
            parts2.append(sev_n)
            chain_local_2 = _chain_str(parts2)
        else:
            parts2b = [loc_n, act_n, body_n]
            if inj_n:
                parts2b.append(inj_n)
            parts2b.append(sev_n)
            chain_local_2 = _chain_str(parts2b)

        if hm_n:
            chain_local_3 = _chain_str([act_n, hm_n, body_n, sev_n])
        else:
            parts3 = [act_n, body_n]
            if inj_n:
                parts3.append(inj_n)
            parts3.append(sev_n)
            chain_local_3 = _chain_str(parts3)

        if hm_n:
            chain_local_4 = _chain_str([job_n, loc_n, act_n, hm_n, sev_n])
        else:
            chain_local_4 = _chain_str([job_n, loc_n, act_n, sev_n])

        if hm_n:
            chain_local_5 = _chain_str([shift_n, loc_n, act_n, hm_n, sev_n])
        else:
            chain_local_5 = _chain_str([shift_n, loc_n, act_n, sev_n])

        chain_rows.append(
            {
                "record_id": record_id,
                "chain_full": _chain_str(main_nodes),
                "chain_local_1": chain_local_1,
                "chain_local_2": chain_local_2,
                "chain_local_3": chain_local_3,
                "chain_local_4": chain_local_4,
                "chain_local_5": chain_local_5,
                "severe_binary": sb,
                "severity_score": sc,
                "Sev": sev_raw,
            }
        )

    node_ids = sorted(node_freq.keys())
    nodes_out = []
    for nid in node_ids:
        nt = _node_type_from_id(nid)
        freq = node_freq[nid]
        sv = node_severe[nid]
        sev_rate = sv / freq if freq else np.nan
        n_sc = node_sev_n[nid]
        avg_sc = node_sev_sum[nid] / n_sc if n_sc else np.nan
        nodes_out.append(
            {
                "node_id": nid,
                "node_label": nid,
                "node_type": nt,
                "raw_value": _raw_display(nid),
                "frequency": freq,
                "severe_count": sv,
                "severity_rate": sev_rate,
                "average_severity_score": avg_sc,
            }
        )
    nodes_df = pd.DataFrame(nodes_out)

    edge_keys = sorted(edge_w.keys())
    edges_out = []
    for u, v in edge_keys:
        w = edge_w[(u, v)]
        es = edge_severe[(u, v)]
        er = es / w if w else np.nan
        n_e = edge_sev_n[(u, v)]
        avg_e = edge_sev_sum[(u, v)] / n_e if n_e else np.nan
        edges_out.append(
            {
                "source": u,
                "target": v,
                "source_type": _node_type_from_id(u),
                "target_type": _node_type_from_id(v),
                "weight": w,
                "severe_count": es,
                "edge_severity_rate": er,
                "average_severity_score": avg_e,
            }
        )
    edges_df = pd.DataFrame(edges_out)

    chains_df = pd.DataFrame(chain_rows)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(NODES_CSV, index=False, encoding="utf-8-sig")
    edges_df.to_csv(EDGES_CSV, index=False, encoding="utf-8-sig")
    chains_df.to_csv(CHAINS_CSV, index=False, encoding="utf-8-sig")

    print(f"OK: Nodes={len(nodes_df)} edges={len(edges_df)} accident chain rows={len(chains_df)}")
    print(f"OK: Wrote {NODES_CSV.resolve()}")
    print(f"OK: Wrote {EDGES_CSV.resolve()}")
    print(f"OK: Wrote {CHAINS_CSV.resolve()}")


if __name__ == "__main__":
    run()
