# -*- coding: utf-8 -*-
"""阶段 03：由 nodes.csv / edges.csv 构建有向加权图并计算节点/边网络指标。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

NODES_CSV = config.OUTPUT_DIR / "nodes.csv"
EDGES_CSV = config.OUTPUT_DIR / "edges.csv"
CHAINS_CSV = config.OUTPUT_DIR / "accident_chains.csv"

NODE_METRICS_CSV = config.OUTPUT_DIR / "node_metrics.csv"
EDGE_METRICS_CSV = config.OUTPUT_DIR / "edge_metrics.csv"
TOP_NODES_XLSX = config.OUTPUT_DIR / "top_30_risk_nodes.xlsx"
TOP_NODES_NO_SEV_XLSX = config.OUTPUT_DIR / "top_30_risk_nodes_excluding_sev.xlsx"
TOP_EDGES_XLSX = config.OUTPUT_DIR / "top_30_risk_edges.xlsx"
GRAPH_GEXF = config.OUTPUT_DIR / "graph.gexf"


def _minmax01(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    mn = float(v.min(skipna=True))
    mx = float(v.max(skipna=True))
    if not np.isfinite(mn) or not np.isfinite(mx):
        return pd.Series(0.0, index=s.index, dtype="float64")
    if mx == mn:
        # 无变异：全零保持 0；同取正常数视为并列“满分”以免整列被压成 0
        val = 0.0 if mx == 0.0 else 1.0
        return pd.Series(val, index=s.index, dtype="float64")
    out = (v - mn) / (mx - mn)
    return out.astype("float64").clip(0.0, 1.0)


def _global_severe_rate_from_chains(chains_path: Path) -> float:
    if not chains_path.is_file():
        return 0.0
    ch = pd.read_csv(chains_path, encoding="utf-8-sig")
    if ch.empty or "severe_binary" not in ch.columns:
        return 0.0
    sb = pd.to_numeric(ch["severe_binary"], errors="coerce").fillna(0)
    return float((sb == 1).sum()) / float(len(ch))


def _sanitize_graph_for_gexf(G: nx.DiGraph) -> nx.DiGraph:
    H = G.copy()
    for _, data in H.nodes(data=True):
        for k, val in list(data.items()):
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                data[k] = 0.0
            elif val is None:
                data[k] = ""
    for _, _, data in H.edges(data=True):
        for k, val in list(data.items()):
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                data[k] = 0.0
            elif val is None:
                data[k] = ""
    return H


def run() -> None:
    if not NODES_CSV.is_file():
        raise FileNotFoundError(f"未找到节点表: {NODES_CSV.resolve()} — 请先运行阶段 02。")
    if not EDGES_CSV.is_file():
        raise FileNotFoundError(f"未找到边表: {EDGES_CSV.resolve()} — 请先运行阶段 02。")

    nodes_df = pd.read_csv(NODES_CSV, encoding="utf-8-sig")
    edges_df = pd.read_csv(EDGES_CSV, encoding="utf-8-sig")

    required_node_cols = {"node_id", "frequency", "severity_rate", "average_severity_score"}
    missing_n = required_node_cols - set(nodes_df.columns)
    if missing_n:
        raise ValueError(f"nodes.csv 缺少列: {sorted(missing_n)}")

    required_edge_cols = {"source", "target", "weight", "edge_severity_rate", "average_severity_score"}
    missing_e = required_edge_cols - set(edges_df.columns)
    if missing_e:
        raise ValueError(f"edges.csv 缺少列: {sorted(missing_e)}")

    global_severe_rate = _global_severe_rate_from_chains(CHAINS_CSV)

    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        u, v = str(row["source"]), str(row["target"])
        w = float(row["weight"])
        if w < 0 or not np.isfinite(w):
            w = 0.0
        G.add_edge(u, v, weight=w)

    for nid in nodes_df["node_id"].astype(str):
        if nid not in G:
            G.add_node(nid)

    node_rows: list[dict] = []
    n_nodes = G.number_of_nodes()

    pr = nx.pagerank(G, alpha=0.85, weight="weight")

    try:
        bet = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        bet = {n: float("nan") for n in G.nodes()}

    try:
        try:
            clo = nx.closeness_centrality(G, wf_improved=True)
        except TypeError:
            clo = nx.closeness_centrality(G)
    except Exception:
        clo = {n: float("nan") for n in G.nodes()}

    nodes_by_id = nodes_df.set_index(nodes_df["node_id"].astype(str))

    for nid in G.nodes():
        row_n = nodes_by_id.loc[nid] if nid in nodes_by_id.index else None
        freq = int(row_n["frequency"]) if row_n is not None else int(G.degree(nid))
        sev_rate = float(row_n["severity_rate"]) if row_n is not None else 0.0
        avg_sc = float(row_n["average_severity_score"]) if row_n is not None else np.nan
        if row_n is not None and (pd.isna(avg_sc) or not np.isfinite(avg_sc)):
            avg_sc = np.nan

        w_in = float(G.in_degree(nid, weight="weight"))
        w_out = float(G.out_degree(nid, weight="weight"))

        node_rows.append(
            {
                "node_id": nid,
                "node_label": str(row_n["node_label"]) if row_n is not None and "node_label" in nodes_df.columns else nid,
                "node_type": str(row_n["node_type"]) if row_n is not None and "node_type" in nodes_df.columns else "",
                "frequency": freq,
                "degree": int(G.degree(nid)),
                "in_degree": int(G.in_degree(nid)),
                "out_degree": int(G.out_degree(nid)),
                "weighted_in_degree": w_in,
                "weighted_out_degree": w_out,
                "pagerank": float(pr.get(nid, 0.0)),
                "betweenness": float(bet.get(nid, float("nan"))),
                "closeness": float(clo.get(nid, float("nan"))),
                "severity_rate": sev_rate,
                "average_severity_score": avg_sc,
            }
        )

    nm = pd.DataFrame(node_rows)

    nm["frequency_norm"] = _minmax01(nm["frequency"])
    nm["pagerank_norm"] = _minmax01(nm["pagerank"])
    nm["betweenness_norm"] = _minmax01(
        nm["betweenness"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    nm["severity_rate_norm"] = _minmax01(nm["severity_rate"])
    avg_fill = pd.to_numeric(nm["average_severity_score"], errors="coerce").fillna(0.0)
    nm["average_severity_score_norm"] = _minmax01(avg_fill)

    nm["risk_score"] = (
        0.25 * nm["frequency_norm"]
        + 0.25 * nm["severity_rate_norm"]
        + 0.20 * nm["pagerank_norm"]
        + 0.15 * nm["betweenness_norm"]
        + 0.15 * nm["average_severity_score_norm"]
    )

    edge_recs: list[dict] = []
    for _, er in edges_df.iterrows():
        u, v = str(er["source"]), str(er["target"])
        w = float(er["weight"])
        esr = float(er["edge_severity_rate"])
        avg_e = er["average_severity_score"]
        try:
            avg_e_f = float(avg_e) if pd.notna(avg_e) and np.isfinite(float(avg_e)) else np.nan
        except (TypeError, ValueError):
            avg_e_f = np.nan

        conf = esr
        if global_severe_rate > 0:
            lift = esr / global_severe_rate
        else:
            lift = float("nan")

        edge_recs.append(
            {
                "source": u,
                "target": v,
                "source_type": er.get("source_type", ""),
                "target_type": er.get("target_type", ""),
                "weight": w,
                "edge_severity_rate": esr,
                "average_severity_score": avg_e_f,
                "confidence_to_severe": conf,
                "lift_to_severe": lift,
                "edge_risk_score": w * esr,
            }
        )

    em = pd.DataFrame(edge_recs)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_node_cols = [
        "node_id",
        "node_label",
        "node_type",
        "frequency",
        "degree",
        "in_degree",
        "out_degree",
        "weighted_in_degree",
        "weighted_out_degree",
        "pagerank",
        "betweenness",
        "closeness",
        "severity_rate",
        "average_severity_score",
        "frequency_norm",
        "severity_rate_norm",
        "pagerank_norm",
        "betweenness_norm",
        "average_severity_score_norm",
        "risk_score",
    ]
    nm[out_node_cols].to_csv(NODE_METRICS_CSV, index=False, encoding="utf-8-sig")

    out_edge_cols = [
        "source",
        "target",
        "source_type",
        "target_type",
        "weight",
        "edge_severity_rate",
        "average_severity_score",
        "confidence_to_severe",
        "lift_to_severe",
    ]
    em[out_edge_cols].to_csv(EDGE_METRICS_CSV, index=False, encoding="utf-8-sig")

    top_n = nm.sort_values("risk_score", ascending=False).head(30)
    top_e = em.sort_values("edge_risk_score", ascending=False).head(30)
    with pd.ExcelWriter(TOP_NODES_XLSX, engine="openpyxl") as wn:
        top_n.to_excel(wn, index=False, sheet_name="top_30")

    no_sev = nm[nm["node_type"].astype(str).str.strip() != "Sev"]
    top_n_no_sev = no_sev.sort_values("risk_score", ascending=False).head(30)
    with pd.ExcelWriter(TOP_NODES_NO_SEV_XLSX, engine="openpyxl") as wns:
        top_n_no_sev.to_excel(wns, index=False, sheet_name="top_30")

    with pd.ExcelWriter(TOP_EDGES_XLSX, engine="openpyxl") as we:
        top_e.to_excel(we, index=False, sheet_name="top_30")

    for _, r in nm.iterrows():
        nid = r["node_id"]
        if nid not in G:
            continue
        G.nodes[nid]["label"] = str(r.get("node_label", nid))
        G.nodes[nid]["node_type"] = str(r.get("node_type", ""))
        G.nodes[nid]["frequency"] = float(r["frequency"])
        G.nodes[nid]["severity_rate"] = float(r["severity_rate"])
        G.nodes[nid]["risk_score"] = float(r["risk_score"])
        G.nodes[nid]["pagerank"] = float(r["pagerank"])
        G.nodes[nid]["betweenness"] = float(r["betweenness"]) if np.isfinite(r["betweenness"]) else 0.0
        G.nodes[nid]["closeness"] = float(r["closeness"]) if np.isfinite(r["closeness"]) else 0.0
        av = r["average_severity_score"]
        G.nodes[nid]["average_severity_score"] = float(av) if pd.notna(av) and np.isfinite(float(av)) else 0.0

    H = _sanitize_graph_for_gexf(G)
    nx.write_gexf(H, str(GRAPH_GEXF), encoding="utf-8")

    print(
        f"OK: Nodes={n_nodes} edges={G.number_of_edges()} "
        f"global_severe_rate={global_severe_rate:.6g}"
    )
    print(f"OK: Wrote {NODE_METRICS_CSV.resolve()}")
    print(f"OK: Wrote {EDGE_METRICS_CSV.resolve()}")
    print(
        "OK: Wrote "
        f"{TOP_NODES_XLSX.resolve()} {TOP_NODES_NO_SEV_XLSX.resolve()} {TOP_EDGES_XLSX.resolve()}"
    )
    print(f"OK: Wrote {GRAPH_GEXF.resolve()}")


if __name__ == "__main__":
    run()
