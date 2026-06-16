# -*- coding: utf-8 -*-
"""Directed weighted graph, node and edge metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from . import config


def global_severe_rate(is_severe: pd.Series) -> float:
    n = len(is_severe)
    if n == 0:
        return 0.0
    return float(is_severe.sum()) / n


def build_digraph_from_chains(
    chain_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, Dict[str, int], Dict[Tuple[str, str], int], Dict[str, int], Dict[Tuple[str, str], int]]:
    """
    Returns:
      G with edge attribute 'weight' (count)
      node_freq: visits per node (each row counts a node at most once per appearance in chain)
      edge_freq: directed edge counts
      node_severe: count of severe rows that include this node in chain
      edge_severe: count of severe rows where edge appears as consecutive pair in chain
    """
    G = nx.DiGraph()
    node_freq: Counter = Counter()
    edge_freq: Counter = Counter()
    node_severe: Counter = Counter()
    edge_severe: Counter = Counter()

    for _, r in chain_df.iterrows():
        ch: List[str] = r["chain_list"]
        severe = bool(r["is_severe"])
        seen_nodes = set(ch)
        for n in seen_nodes:
            node_freq[n] += 1
            if severe:
                node_severe[n] += 1
        for u, v in zip(ch[:-1], ch[1:]):
            edge_freq[(u, v)] += 1
            if severe:
                edge_severe[(u, v)] += 1

    p_severe = global_severe_rate(chain_df["is_severe"])

    for (u, v), w in edge_freq.items():
        G.add_edge(u, v, weight=int(w))

    nx.set_node_attributes(G, dict(node_freq), "frequency")
    nx.set_node_attributes(G, dict(node_severe), "severe_count")

    for u, v, data in G.edges(data=True):
        w = data["weight"]
        sc = edge_severe.get((u, v), 0)
        data["severe_count"] = int(sc)
        data["edge_severity_rate"] = float(sc) / w if w else 0.0
        data["confidence_to_severe"] = data["edge_severity_rate"]
        denom = p_severe if p_severe > 0 else 1e-9
        data["lift_to_severe"] = (data["confidence_to_severe"] / denom) if denom else 0.0

    return G, dict(node_freq), dict(edge_freq), dict(node_severe), dict(edge_severe)


def compute_node_metrics(
    G: nx.DiGraph,
    chain_df: pd.DataFrame,
    node_freq: Dict[str, int],
    node_severe: Dict[str, int],
) -> pd.DataFrame:
    p_severe = global_severe_rate(chain_df["is_severe"])
    rows = []

    # Centrality (betweenness is expensive on large graphs)
    n_nodes = G.number_of_nodes()
    pr = nx.pagerank(G, alpha=config.PAGERANK_ALPHA, weight="weight")
    max_b = config.BETWEENNESS_MAX_NODES
    try:
        if max_b and n_nodes > max_b:
            bet = {n: float("nan") for n in G.nodes()}
        else:
            bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception:
        bet = {n: float("nan") for n in G.nodes()}
    try:
        clo = nx.closeness_centrality(G)
    except Exception:
        clo = {n: float("nan") for n in G.nodes()}

    for n in G.nodes():
        freq = int(node_freq.get(n, 0))
        sv = int(node_severe.get(n, 0))
        sev_rate = float(sv) / freq if freq else 0.0
        if config.RISK_SCORE_NODE == "freq_times_severity":
            risk = float(freq) * sev_rate
        else:
            risk = float(np.sqrt(freq + 1e-9)) * sev_rate

        rows.append(
            {
                "node": n,
                "frequency": freq,
                "degree": int(G.degree(n)),
                "in_degree": int(G.in_degree(n)),
                "out_degree": int(G.out_degree(n)),
                "pagerank": float(pr.get(n, 0.0)),
                "betweenness": float(bet.get(n, float("nan"))),
                "closeness": float(clo.get(n, float("nan"))),
                "severe_count": sv,
                "severity_rate": sev_rate,
                "global_severe_rate": p_severe,
                "risk_score": float(risk),
            }
        )

    return pd.DataFrame(rows)


def compute_edge_table(G: nx.DiGraph, chain_df: pd.DataFrame) -> pd.DataFrame:
    p_severe = global_severe_rate(chain_df["is_severe"])
    recs = []
    for u, v, data in G.edges(data=True):
        w = int(data["weight"])
        sc = int(data.get("severe_count", 0))
        esr = float(data.get("edge_severity_rate", 0.0))
        recs.append(
            {
                "source": u,
                "target": v,
                "weight": w,
                "severe_count": sc,
                "edge_severity_rate": esr,
                "confidence_to_severe": float(data.get("confidence_to_severe", 0.0)),
                "lift_to_severe": float(data.get("lift_to_severe", 0.0)),
                "global_severe_rate": p_severe,
                "edge_risk_score": float(w) * esr,
            }
        )
    return pd.DataFrame(recs)


def aggregate_chains(chain_df: pd.DataFrame) -> pd.DataFrame:
    g = chain_df.groupby("chain", as_index=False).agg(
        count=("chain", "size"),
        severe_count=("is_severe", "sum"),
    )
    g["severity_rate"] = g["severe_count"] / g["count"].replace(0, np.nan)
    g["risk_score"] = g["count"] * g["severity_rate"].fillna(0.0)
    g = g.sort_values(["risk_score", "count"], ascending=False)
    return g
