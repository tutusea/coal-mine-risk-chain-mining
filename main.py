# -*- coding: utf-8 -*-
"""
Entry point: load Excel, build risk chains, graph metrics, rules, exports.

Single-line run (CMD or Cursor terminal, from project root):
python main.py --excel "合成事故统计_含真实数据.xlsx"

Use all rows including synthetic:
python main.py --excel "合成事故统计_含真实数据.xlsx" --all

Heavy injury and death only as severe:
python main.py --excel "合成事故统计_含真实数据.xlsx" --severity-mode heavy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from risk_chain_kg import config
from risk_chain_kg.chains import build_chains_dataframe
from risk_chain_kg.gexf_export import export_gexf
from risk_chain_kg.graph_model import (
    aggregate_chains,
    build_digraph_from_chains,
    compute_edge_table,
    compute_node_metrics,
    global_severe_rate,
)
from risk_chain_kg.io_excel import pick_chain_columns, read_excel, resolve_columns
from risk_chain_kg.plots import (
    plot_association_rules,
    plot_network,
    plot_sankey,
    plot_top_edges,
    plot_top_nodes,
)
from risk_chain_kg.preprocess import clean_frame, filter_real_only, mark_severe
from risk_chain_kg.rules_mining import mine_severe_rules


def log(msg: str) -> None:
    # Plain English, ASCII-safe for Windows consoles
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def enrich_graph(G, node_df: pd.DataFrame, edge_df: pd.DataFrame) -> None:
    nd = node_df.set_index("node")
    for n in G.nodes():
        if n in nd.index:
            row = nd.loc[n]
            G.nodes[n]["frequency"] = int(row.get("frequency", 0))
            G.nodes[n]["in_degree"] = int(row.get("in_degree", 0))
            G.nodes[n]["out_degree"] = int(row.get("out_degree", 0))
            G.nodes[n]["severity_rate"] = float(row.get("severity_rate", 0.0))
            G.nodes[n]["risk_score"] = float(row.get("risk_score", 0.0))
            G.nodes[n]["pagerank"] = float(row.get("pagerank", 0.0))
            G.nodes[n]["betweenness"] = float(row.get("betweenness", 0.0))
            G.nodes[n]["closeness"] = float(row.get("closeness", 0.0))
    for u, v, d in G.edges(data=True):
        sub = edge_df[(edge_df["source"] == u) & (edge_df["target"] == v)]
        if not sub.empty:
            r0 = sub.iloc[0]
            d["edge_severity_rate"] = float(r0.get("edge_severity_rate", 0.0))
            d["lift_to_severe"] = float(r0.get("lift_to_severe", 0.0))
            d["confidence_to_severe"] = float(r0.get("confidence_to_severe", 0.0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Risk knowledge graph and chain mining")
    parser.add_argument(
        "--excel",
        type=str,
        default=str(config.DEFAULT_EXCEL),
        help="Path to input xlsx file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all rows including synthetic; default keeps only real rows if is_synthetic exists",
    )
    parser.add_argument(
        "--severity-mode",
        type=str,
        default=config.SEVERITY_MODE,
        choices=["minor_plus", "heavy"],
        help="How to label severe outcomes from harm degree text",
    )
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = Path(args.excel)
    if not excel_path.is_file():
        log("ERROR: Excel file not found at path: %s" % excel_path)
        return 1

    log("Loading Excel...")
    df = read_excel(excel_path)
    log("Rows loaded: %d, columns: %d" % (len(df), len(df.columns)))

    resolved = resolve_columns(df)
    chain_cols, missing_keys = pick_chain_columns(resolved)
    if missing_keys:
        log("WARNING: missing chain fields (edit CHAIN_FIELD_KEYS or COLUMN_ALIASES): %s" % (", ".join(missing_keys)))

    harm_col = resolved.get("harm_degree")
    if not harm_col:
        log("ERROR: harm degree column not found")
        return 1

    text_cols = [c for c in chain_cols if c in df.columns]
    df = clean_frame(df, text_cols)

    syn_key = "is_synthetic"
    syn_col = resolved.get(syn_key)
    if not args.all and syn_col:
        before = len(df)
        df = filter_real_only(df, syn_col)
        log("Real-only filter applied. Rows: %d -> %d" % (before, len(df)))
    else:
        log("Using all rows (no synthetic filter or --all set). Rows: %d" % len(df))

    df["__severe__"] = mark_severe(df, harm_col, mode=args.severity_mode)
    p_sev = global_severe_rate(df["__severe__"])
    log("Global severe rate: %.4f (mode=%s)" % (p_sev, args.severity_mode))

    chain_df = build_chains_dataframe(df, resolved, config.CHAIN_FIELD_KEYS)
    log("Built chains for %d records" % len(chain_df))

    G, node_freq, edge_freq, node_severe, edge_severe = build_digraph_from_chains(chain_df)
    log("Graph nodes: %d, edges: %d" % (G.number_of_nodes(), G.number_of_edges()))

    node_df = compute_node_metrics(G, chain_df, node_freq, node_severe)
    if G.number_of_nodes() > config.BETWEENNESS_MAX_NODES > 0:
        log(
            "Note: betweenness skipped because node count exceeds BETWEENNESS_MAX_NODES "
            "(edit config.py to raise the limit)."
        )
    edge_df = compute_edge_table(G, chain_df)
    chain_stats = aggregate_chains(chain_df)

    enrich_graph(G, node_df, edge_df)

    log("Mining association rules (may take time on large data)...")
    rules_raw = mine_severe_rules(chain_df, max_len=6)
    rules_out = pd.DataFrame()
    if rules_raw is not None and not rules_raw.empty:
        tag = config.SEVERE_ITEM_TAG
        r1 = rules_raw[rules_raw["consequents"].apply(lambda x: x == frozenset([tag]))]
        r1 = r1[~r1["antecedents"].apply(lambda x: tag in set(x))]
        if not r1.empty:
            rules_out = r1.copy()
            rules_out["antecedents_str"] = rules_out["antecedents"].astype(str)
            rules_out["consequents_str"] = rules_out["consequents"].astype(str)
            rules_out.to_csv(config.OUTPUT_DIR / "association_rules.csv", index=False)

    out_dir = config.OUTPUT_DIR
    per_rec = chain_df[["row_index", "chain", "is_severe"]].copy()
    per_rec.to_csv(out_dir / "chains_per_record.csv", index=False)

    node_df.sort_values("risk_score", ascending=False).to_csv(out_dir / "node_metrics.csv", index=False)
    edge_df.sort_values("edge_risk_score", ascending=False).to_csv(out_dir / "edge_metrics.csv", index=False)
    chain_stats.to_csv(out_dir / "chain_aggregates.csv", index=False)

    top_nodes = node_df.sort_values("risk_score", ascending=False).head(config.TOP_K_NODES)
    top_edges = edge_df.sort_values(
        ["edge_risk_score", "lift_to_severe", "weight"],
        ascending=False,
    ).head(config.TOP_K_EDGES)
    top_chains = chain_stats.head(config.TOP_K_CHAINS)
    top_nodes.to_csv(out_dir / "top_risk_nodes.csv", index=False)
    top_edges.to_csv(out_dir / "top_risk_edges.csv", index=False)
    top_chains.to_csv(out_dir / "top_severe_chains.csv", index=False)

    if rules_out.empty:
        log("WARNING: no association rules passed filters; try lower RULE_MIN_SUPPORT in config.py")

    export_gexf(G, out_dir / "graph.gexf")
    log("Exported GEXF: outputs/graph.gexf")

    log("Generating figures...")
    plot_top_nodes(node_df, config.TOP_K_NODES, out_dir / "fig_top_risk_nodes.png")
    plot_top_edges(edge_df, config.TOP_K_EDGES, out_dir / "fig_top_risk_edges.png")
    plot_sankey(chain_df, out_dir / "fig_risk_chain_sankey.html", out_dir / "fig_risk_chain_sankey.png")
    plot_network(G, node_df, min(80, max(20, config.TOP_K_NODES * 2)), out_dir / "fig_risk_network.png")
    plot_association_rules(rules_out, out_dir / "fig_association_rules.png")

    log("Done. All outputs saved under outputs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
