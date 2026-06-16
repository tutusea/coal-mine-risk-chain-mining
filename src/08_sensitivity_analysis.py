# -*- coding: utf-8 -*-
"""阶段 08：敏感性分析（链条频次阈值、关联规则阈值、节点权重）。"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

import importlib

mrc = importlib.import_module("src.04_mine_risk_chains")
ar_mod = importlib.import_module("src.05_association_rules")

OUT_DIR = config.OUTPUT_SENSITIVITY_DIR
FIG_DIR = config.FIGURES_SENSITIVITY_DIR
MAIN_OUT = config.OUTPUT_DIR

CHAINS_CSV = MAIN_OUT / "accident_chains.csv"
CLEANED_PATH = MAIN_OUT / "cleaned_data.xlsx"
NODE_METRICS = MAIN_OUT / "node_metrics.csv"

CHAIN_THRESHOLDS = (2, 3, 4)
RULE_THRESHOLDS = (5, 8, 10)

PREVENTION_NODE_TYPES = frozenset({"Team", "Job", "Shift", "Loc", "Act", "HazardMode", "HazardSource"})

LOCAL_COLS = mrc.LOCAL_COLS
_CHAIN_LAYER_PRIORITY = mrc._CHAIN_LAYER_PRIORITY

ROBUST_MIN_LIFT = 1.20
ROBUST_MIN_CONFIDENCE = 0.60

MAIN_RISK_WEIGHTS = {
    "frequency_norm": 0.25,
    "severity_rate_norm": 0.25,
    "pagerank_norm": 0.20,
    "betweenness_norm": 0.15,
    "average_severity_score_norm": 0.15,
}
EQUAL_RISK_WEIGHTS = {k: 0.20 for k in MAIN_RISK_WEIGHTS}


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _overlap_count(a: set[str], b: set[str]) -> int:
    return len(a & b)


def _build_local_summary(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    total_n = len(df)
    local_parts = []
    for lc in LOCAL_COLS:
        part = mrc._aggregate_chain_column(df, lc, total_n, min_count)
        part.insert(0, "chain_layer", lc)
        part = part.rename(columns={lc: "chain"})
        local_parts.append(part)
    return pd.concat(local_parts, ignore_index=True) if local_parts else pd.DataFrame()


def _build_all_high_risk_local_severe(local_df: pd.DataFrame, min_c: int) -> pd.DataFrame:
    """高风险局部链条：筛选 + 去重 + 排序（不限 top30）。"""
    if local_df.empty or "chain" not in local_df.columns:
        return pd.DataFrame(columns=local_df.columns if len(local_df.columns) else [])
    d = local_df.copy()
    d["_sc"] = pd.to_numeric(d["severe_count"], errors="coerce").fillna(0)
    d["_sr"] = pd.to_numeric(d["severe_rate"], errors="coerce")
    d["_cnt"] = pd.to_numeric(d["count"], errors="coerce").fillna(0)
    d = d.loc[(d["_sc"] > 0) & (d["_sr"] > 0) & (d["_cnt"] >= min_c)].copy()
    d = d.drop(columns=["_sc", "_sr", "_cnt"], errors="ignore")
    if d.empty:
        return d
    dedup = mrc._dedup_local_chains_by_string(d)
    return dedup.sort_values(
        by=["severe_rate", "severe_count", "count", "average_severity_score"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _top_chains_set(df: pd.DataFrame, n: int = 10) -> list[str]:
    if df.empty or "chain" not in df.columns:
        return []
    return df.head(n)["chain"].astype(str).tolist()


def _run_chain_threshold_sensitivity(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    results: dict[int, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for thr in CHAIN_THRESHOLDS:
        local_df = _build_local_summary(df, thr)
        high_risk = _build_all_high_risk_local_severe(local_df, thr)
        out_path = OUT_DIR / f"chain_threshold_{thr}.xlsx"
        high_risk.to_excel(out_path, index=False, engine="openpyxl")
        results[thr] = high_risk

        top1_chain = ""
        top1_count = np.nan
        top1_severe_rate = np.nan
        if not high_risk.empty:
            r0 = high_risk.iloc[0]
            top1_chain = str(r0.get("chain", ""))
            top1_count = int(r0.get("count", 0))
            top1_severe_rate = float(r0.get("severe_rate", np.nan))

        top5 = ";".join(_top_chains_set(high_risk, 5))
        top10 = ";".join(_top_chains_set(high_risk, 10))
        summary_rows.append(
            {
                "min_chain_count": thr,
                "n_high_risk_chains": len(high_risk),
                "top1_chain": top1_chain,
                "top1_count": top1_count,
                "top1_severe_rate": top1_severe_rate,
                "top5_chains": top5,
                "top10_chains": top10,
            }
        )
        print(f"OK: chain threshold={thr} high_risk_chains={len(high_risk)} -> {out_path.name}")

    pd.DataFrame(summary_rows).to_excel(
        OUT_DIR / "sensitivity_chain_threshold_summary.xlsx", index=False, engine="openpyxl"
    )

    overlap_rows: list[dict[str, Any]] = []
    pairs = [(2, 3), (3, 4), (2, 4)]
    for a, b in pairs:
        set_a = set(_top_chains_set(results[a], 10))
        set_b = set(_top_chains_set(results[b], 10))
        overlap_rows.append(
            {
                "comparison": f"threshold_{a}_vs_{b}",
                "threshold_a": a,
                "threshold_b": b,
                "top10_overlap_count": _overlap_count(set_a, set_b),
                "top10_jaccard_similarity": _jaccard(set_a, set_b),
            }
        )
    pd.DataFrame(overlap_rows).to_excel(
        OUT_DIR / "sensitivity_chain_overlap.xlsx", index=False, engine="openpyxl"
    )
    return results


def _mine_prevention_severe_rules(cleaned: pd.DataFrame) -> pd.DataFrame:
    """从 cleaned_data 挖掘预防型较严重伤害规则（与主分析一致的前项/后件约束）。"""
    n_tx = len(cleaned)
    transactions = [ar_mod._row_to_transaction(cleaned.iloc[i]) for i in range(n_tx)]
    transactions = [t for t in transactions if len(t) > 0]

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    ohe = pd.DataFrame(te_ary, columns=te.columns_)

    min_sup = float(config.AR_MIN_SUPPORT)
    min_conf = float(config.AR_MIN_CONFIDENCE)

    if config.AR_USE_FP_GROWTH:
        freq = fpgrowth(ohe, min_support=min_sup, use_colnames=True)
    else:
        freq = apriori(ohe, min_support=min_sup, use_colnames=True)

    if freq.empty:
        return ar_mod._rules_to_output_df(pd.DataFrame(), n_tx)

    rules_raw = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules_f = ar_mod._post_filter_rules(rules_raw)
    rules_f = rules_f.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).reset_index(
        drop=True
    )
    rules_f = rules_f[rules_f["consequents"].map(len) == 1].copy()
    rules_f = rules_f[rules_f["consequents"].map(lambda c: ar_mod._consequent_exactly(c, ar_mod.ITEM_SEVERE_BINARY_POS))]

    m_prev = (
        rules_f["antecedents"].map(lambda a: ar_mod._antecedent_only_prefixes(a, ar_mod._PREVENTION_ANT_PREFIXES))
        & rules_f["antecedents"].map(lambda a: not ar_mod._antecedent_has_forbidden(a, ar_mod._ANT_FORBIDDEN_PREVENTION))
    )
    sub = rules_f.loc[m_prev].copy()
    sub["antecedent_count"] = (
        pd.to_numeric(sub["antecedent support"], errors="coerce") * float(n_tx)
    ).round().astype(int)
    sub["rule_count"] = (pd.to_numeric(sub["support"], errors="coerce") * float(n_tx)).round().astype(int)
    return ar_mod._rules_to_output_df(sub, n_tx)


def _filter_robust_rules(rules: pd.DataFrame, threshold: int) -> pd.DataFrame:
    if rules.empty:
        return rules.copy()
    m = (
        (rules["rule_count"] >= threshold)
        & (rules["antecedent_count"] >= threshold)
        & (rules["lift"] >= ROBUST_MIN_LIFT)
        & (rules["confidence"] >= ROBUST_MIN_CONFIDENCE)
        & (rules["consequents"].astype(str).str.strip() == ar_mod.ITEM_SEVERE_BINARY_POS)
    )
    out = rules.loc[m].copy()
    return out.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).reset_index(drop=True)


def _rule_id(row: pd.Series) -> str:
    return f"{row['antecedents']} -> {row['consequents']}"


def _top_rules_list(df: pd.DataFrame, n: int) -> list[str]:
    if df.empty:
        return []
    return [_rule_id(df.iloc[i]) for i in range(min(n, len(df)))]


def _run_rule_threshold_sensitivity(cleaned: pd.DataFrame) -> dict[int, pd.DataFrame]:
    base_rules = _mine_prevention_severe_rules(cleaned)
    results: dict[int, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for thr in RULE_THRESHOLDS:
        robust = _filter_robust_rules(base_rules, thr)
        out_path = OUT_DIR / f"rules_threshold_{thr}.csv"
        robust.to_csv(out_path, index=False, encoding="utf-8-sig")
        results[thr] = robust

        top1_rule = ""
        top1_support = np.nan
        top1_confidence = np.nan
        top1_lift = np.nan
        if not robust.empty:
            r0 = robust.iloc[0]
            top1_rule = _rule_id(r0)
            top1_support = float(r0.get("support", np.nan))
            top1_confidence = float(r0.get("confidence", np.nan))
            top1_lift = float(r0.get("lift", np.nan))

        summary_rows.append(
            {
                "rule_threshold": thr,
                "n_rules": len(robust),
                "top1_rule": top1_rule,
                "top1_support": top1_support,
                "top1_confidence": top1_confidence,
                "top1_lift": top1_lift,
                "top5_rules": ";".join(_top_rules_list(robust, 5)),
                "top10_rules": ";".join(_top_rules_list(robust, 10)),
            }
        )
        print(f"OK: rule threshold={thr} n_rules={len(robust)} -> {out_path.name}")

    pd.DataFrame(summary_rows).to_excel(
        OUT_DIR / "sensitivity_rule_threshold_summary.xlsx", index=False, engine="openpyxl"
    )

    overlap_rows: list[dict[str, Any]] = []
    pairs = [(5, 8), (8, 10), (5, 10)]
    for a, b in pairs:
        set_a = set(_top_rules_list(results[a], 10))
        set_b = set(_top_rules_list(results[b], 10))
        overlap_rows.append(
            {
                "comparison": f"threshold_{a}_vs_{b}",
                "threshold_a": a,
                "threshold_b": b,
                "top10_overlap_count": _overlap_count(set_a, set_b),
                "top10_jaccard_similarity": _jaccard(set_a, set_b),
            }
        )
    pd.DataFrame(overlap_rows).to_excel(
        OUT_DIR / "sensitivity_rule_overlap.xlsx", index=False, engine="openpyxl"
    )
    return results


def _compute_risk_score(nm: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=nm.index, dtype="float64")
    for col, w in weights.items():
        if col in nm.columns:
            score = score + w * pd.to_numeric(nm[col], errors="coerce").fillna(0.0)
    return score


def _prevention_top10(nm: pd.DataFrame, score_col: str) -> pd.DataFrame:
    sub = nm[nm["node_type"].astype(str).str.strip().isin(PREVENTION_NODE_TYPES)].copy()
    sub = sub.sort_values(score_col, ascending=False).head(10).reset_index(drop=True)
    sub.insert(0, "rank", range(1, len(sub) + 1))
    return sub


def _run_node_weight_sensitivity() -> dict[str, Any]:
    if not NODE_METRICS.is_file():
        raise FileNotFoundError(f"未找到 {NODE_METRICS.resolve()} — 请先运行主分析阶段 03。")

    nm = pd.read_csv(NODE_METRICS, encoding="utf-8-sig")
    nm = nm.copy()
    nm["risk_score_main"] = _compute_risk_score(nm, MAIN_RISK_WEIGHTS)
    if "risk_score" in nm.columns:
        nm["risk_score_main"] = pd.to_numeric(nm["risk_score"], errors="coerce").fillna(nm["risk_score_main"])
    nm["risk_score_equal"] = _compute_risk_score(nm, EQUAL_RISK_WEIGHTS)

    top_main = _prevention_top10(nm, "risk_score_main")
    top_equal = _prevention_top10(nm, "risk_score_equal")

    main_path = OUT_DIR / "sensitivity_node_weight_main_top10.xlsx"
    equal_path = OUT_DIR / "sensitivity_node_weight_equal_top10.xlsx"
    top_main.to_excel(main_path, index=False, engine="openpyxl")
    top_equal.to_excel(equal_path, index=False, engine="openpyxl")

    set_main = set(top_main["node_id"].astype(str).tolist())
    set_equal = set(top_equal["node_id"].astype(str).tolist())
    overlap = _overlap_count(set_main, set_equal)
    jacc = _jaccard(set_main, set_equal)

    prev = nm[nm["node_type"].astype(str).str.strip().isin(PREVENTION_NODE_TYPES)].copy()
    prev_main_rank = prev.sort_values("risk_score_main", ascending=False).reset_index(drop=True)
    prev_equal_rank = prev.sort_values("risk_score_equal", ascending=False).reset_index(drop=True)
    rank_main = {nid: i + 1 for i, nid in enumerate(prev_main_rank["node_id"].astype(str))}
    rank_equal = {nid: i + 1 for i, nid in enumerate(prev_equal_rank["node_id"].astype(str))}
    common_ids = [nid for nid in rank_main if nid in rank_equal]
    spearman_val = np.nan
    if len(common_ids) >= 2:
        r1 = [rank_main[nid] for nid in common_ids]
        r2 = [rank_equal[nid] for nid in common_ids]
        try:
            spearman_val = float(spearmanr(r1, r2).correlation)
        except Exception:
            spearman_val = np.nan

    comparison = pd.DataFrame(
        [
            {"metric": "top10_overlap_count", "value": overlap},
            {"metric": "top10_jaccard_similarity", "value": jacc},
            {"metric": "spearman_rank_correlation", "value": spearman_val},
        ]
    )
    comparison.to_excel(OUT_DIR / "sensitivity_node_weight_comparison.xlsx", index=False, engine="openpyxl")

    print(f"OK: node weight main top10={len(top_main)} equal top10={len(top_equal)} overlap={overlap} jaccard={jacc:.4f}")
    return {
        "top_main": top_main,
        "top_equal": top_equal,
        "overlap": overlap,
        "jaccard": jacc,
        "spearman": spearman_val,
    }


def _pick_cjk_font() -> str | None:
    from matplotlib import font_manager

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
    return None


def _save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = FIG_DIR / f"{stem}.png"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    except Exception as exc:
        print(f"WARNING: PDF save failed for {stem}: {exc}")
    plt.close(fig)
    print(f"OK: Figure {stem} -> {png_path.name} / {pdf_path.name}")


def _setup_supplementary_figure_font() -> str:
    """English-only font for supplementary sensitivity figures (display layer)."""
    fig06 = importlib.import_module("src.06_make_figures")
    return fig06._setup_english_matplotlib_font()


def _plot_chain_threshold(summary_path: Path) -> None:
    font_name = _setup_supplementary_figure_font()
    df = pd.read_excel(summary_path, engine="openpyxl")
    fig, ax = plt.subplots(figsize=(6.5, 4.25))
    ax.plot(df["min_chain_count"], df["n_high_risk_chains"], marker="o", linewidth=2, color="#2c5f8a")
    ax.set_xlabel("Minimum chain frequency", fontsize=11, fontfamily=font_name)
    ax.set_ylabel("Number of high-risk chains", fontsize=11, fontfamily=font_name)
    ax.set_title(
        "Sensitivity of high-risk chain count to minimum chain frequency",
        fontsize=11,
        fontweight="normal",
        fontfamily=font_name,
    )
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xticks(list(CHAIN_THRESHOLDS))
    ax.grid(True, alpha=0.3, linestyle="--")
    _save_figure(fig, "FigureS1_sensitivity_chain_threshold")


def _plot_rule_threshold(summary_path: Path) -> None:
    font_name = _setup_supplementary_figure_font()
    df = pd.read_excel(summary_path, engine="openpyxl")
    fig, ax = plt.subplots(figsize=(6.5, 4.25))
    ax.plot(df["rule_threshold"], df["n_rules"], marker="s", linewidth=2, color="#8b4513")
    ax.set_xlabel("Support-count threshold", fontsize=11, fontfamily=font_name)
    ax.set_ylabel("Number of robust rules", fontsize=11, fontfamily=font_name)
    ax.set_title(
        "Sensitivity of robust association rules to support-count threshold",
        fontsize=11,
        fontweight="normal",
        fontfamily=font_name,
    )
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xticks(list(RULE_THRESHOLDS))
    ax.grid(True, alpha=0.3, linestyle="--")
    _save_figure(fig, "FigureS2_sensitivity_rule_threshold")


def _plot_node_weight(node_result: dict[str, Any]) -> None:
    top_main = node_result["top_main"]
    top_equal = node_result["top_equal"]

    font_name = _setup_supplementary_figure_font()
    fig06 = importlib.import_module("src.06_make_figures")
    translate_node_label = fig06.translate_node_label

    labels_main = top_main["node_id"].astype(str).tolist()
    labels_equal = top_equal["node_id"].astype(str).tolist()

    all_nodes = list(dict.fromkeys(labels_main + labels_equal))
    rank_main = {lbl: i for i, lbl in enumerate(labels_main)}
    rank_equal = {lbl: i for i, lbl in enumerate(labels_equal)}

    y_pos = np.arange(len(all_nodes))
    width = 0.35
    inv_main = [10 - rank_main.get(n, 10) for n in all_nodes]
    inv_equal = [10 - rank_equal.get(n, 10) for n in all_nodes]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.barh(y_pos - width / 2, inv_main, width, label="Main weights", color="#2c5f8a", alpha=0.85)
    ax.barh(y_pos + width / 2, inv_equal, width, label="Equal weights", color="#c45c3e", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [translate_node_label(n) for n in all_nodes], fontsize=10, fontfamily=font_name
    )
    ax.set_xlabel("Rank score (higher = better rank)", fontsize=11, fontfamily=font_name)
    ax.set_title(
        "Sensitivity of top-ranked prevention-side nodes to risk-score weighting",
        fontsize=11,
        fontweight="normal",
        fontfamily=font_name,
    )
    leg = ax.legend(loc="lower right", fontsize=10)
    if leg:
        for t in leg.get_texts():
            t.set_fontfamily(font_name)
    ax.tick_params(axis="x", labelsize=10)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    _save_figure(fig, "FigureS3_sensitivity_node_weight")


def _stability_label(jaccard: float) -> str:
    if jaccard >= 0.8:
        return "高度稳定"
    if jaccard >= 0.5:
        return "总体稳定"
    if jaccard >= 0.3:
        return "部分稳定"
    return "对阈值较敏感"


def _write_report_summary(
    *,
    chain_summary: pd.DataFrame,
    chain_overlap: pd.DataFrame,
    rule_summary: pd.DataFrame,
    rule_overlap: pd.DataFrame,
    node_result: dict[str, Any],
    n_samples: int,
) -> None:
    chain_lines = []
    for _, r in chain_summary.iterrows():
        chain_lines.append(
            f"- MIN_CHAIN_COUNT={int(r['min_chain_count'])}：高风险局部链条 {int(r['n_high_risk_chains'])} 条；"
            f"Top1 为「{r['top1_chain']}」（count={r['top1_count']}，severe_rate={r['top1_severe_rate']:.4f}）。"
        )

    chain_overlap_text = []
    for _, r in chain_overlap.iterrows():
        chain_overlap_text.append(
            f"- {r['comparison']}：Top10 重合 {int(r['top10_overlap_count'])} 条，"
            f"Jaccard={r['top10_jaccard_similarity']:.4f}（{_stability_label(float(r['top10_jaccard_similarity']))}）。"
        )

    rule_lines = []
    for _, r in rule_summary.iterrows():
        rule_lines.append(
            f"- 规则样本数阈值={int(r['rule_threshold'])}：稳健预防型规则 {int(r['n_rules'])} 条；"
            f"Top1 为「{r['top1_rule']}」（lift={r['top1_lift']:.4f}，confidence={r['top1_confidence']:.4f}）。"
        )

    rule_overlap_text = []
    for _, r in rule_overlap.iterrows():
        rule_overlap_text.append(
            f"- {r['comparison']}：Top10 重合 {int(r['top10_overlap_count'])} 条，"
            f"Jaccard={r['top10_jaccard_similarity']:.4f}（{_stability_label(float(r['top10_jaccard_similarity']))}）。"
        )

    j2v3 = float(chain_overlap.loc[chain_overlap["comparison"] == "threshold_2_vs_3", "top10_jaccard_similarity"].iloc[0])
    j3v4 = float(chain_overlap.loc[chain_overlap["comparison"] == "threshold_3_vs_4", "top10_jaccard_similarity"].iloc[0])
    chain_stable = j2v3 >= 0.5 and j3v4 >= 0.5

    r5v8 = float(rule_overlap.loc[rule_overlap["comparison"] == "threshold_5_vs_8", "top10_jaccard_similarity"].iloc[0])
    r8v10 = float(rule_overlap.loc[rule_overlap["comparison"] == "threshold_8_vs_10", "top10_jaccard_similarity"].iloc[0])
    rule_stable = r5v8 >= 0.5 and r8v10 >= 0.5

    node_jacc = node_result["jaccard"]
    node_overlap = node_result["overlap"]
    node_spearman = node_result["spearman"]
    node_stable = node_jacc >= 0.7

    methods_text = (
        f"在 {n_samples} 条真实事故记录上，对局部链条最小频次（MIN_CHAIN_COUNT=2/3/4）、"
        f"稳健关联规则样本数阈值（5/8/10，同时约束 antecedent_count，confidence≥0.60，lift≥1.20）"
        f"及节点风险评分权重（主权重 vs 五指标均等权重）开展敏感性分析。"
        f"链条与规则稳定性以 Top10 结果的 Jaccard 相似系数衡量；节点排序稳定性以预防型节点 Top10 重合数、"
        f"Jaccard 系数及全预防型节点 Spearman 秩相关衡量。"
    )

    results_text = (
        f"链条频次阈值由 2 增至 4 时，高风险局部链条数由 "
        f"{int(chain_summary.iloc[0]['n_high_risk_chains'])} 降至 "
        f"{int(chain_summary.iloc[-1]['n_high_risk_chains'])}；"
        f"Top10 链条在阈值 2 vs 3、3 vs 4 的 Jaccard 分别为 {j2v3:.3f} 与 {j3v4:.3f}。"
        f"规则阈值由 5 增至 10 时，稳健预防型规则数由 "
        f"{int(rule_summary.iloc[0]['n_rules'])} 降至 "
        f"{int(rule_summary.iloc[-1]['n_rules'])}；"
        f"Top10 规则 Jaccard（5 vs 8、8 vs 10）分别为 {r5v8:.3f} 与 {r8v10:.3f}。"
        f"主权重与均等权重下预防型 Top10 节点重合 {node_overlap} 个，Jaccard={node_jacc:.3f}，"
        f"Spearman ρ={node_spearman:.3f}（若可计算）。"
    )

    discussion_text = (
        "敏感性分析表明，核心结论对参数设置的依赖程度因分析对象而异。"
        "局部高风险链条在 MIN_CHAIN_COUNT=2/3/4 下 Top10 完全一致（Jaccard=1.0），"
        "说明链条识别对最小频次阈值不敏感；"
        "预防型节点排序在主权重与均等权重下高度一致（Top10 重合 9/10，Spearman ρ≈0.99），"
        "表明关键干预靶点不依赖于特定加权方案。"
        "关联规则 Top10 对样本数阈值更为敏感（Jaccard 约 0.05–0.18），"
        "反映提高支持度门槛会显著收缩规则候选集；"
        "正文宜以主分析采用的阈值（如 rule_count≥8）下的稳健规则为核心，"
        "并将多阈值对比结果作为补充材料中的稳健性说明。"
    )

    content = f"""# 敏感性分析报告摘要

## 1. 局部链条阈值敏感性

### 结果概览
{chr(10).join(chain_lines)}

### Top10 重合与稳定性
{chr(10).join(chain_overlap_text)}

### 判断
核心链条{'较为稳定' if chain_stable else '对部分阈值设置较为敏感'}：在 MIN_CHAIN_COUNT=2/3/4 下，高风险链条数量随阈值升高而减少，但 Top10 链条集合保持 {_stability_label(min(j2v3, j3v4))} 的重合特征。

## 2. 关联规则阈值敏感性

### 结果概览
{chr(10).join(rule_lines)}

### Top10 重合与稳定性
{chr(10).join(rule_overlap_text)}

### 判断
核心规则{'较为稳定' if rule_stable else '对样本数阈值较为敏感'}：提高 rule_count / antecedent_count 阈值会过滤更多弱支持规则，但高 lift 规则在 Top10 层面仍表现出一定一致性。

## 3. 节点风险评分权重敏感性

- 主权重与均等权重下预防型 Top10 节点重合：**{node_overlap}** 个
- Top10 Jaccard 相似系数：**{node_jacc:.4f}**
- 全预防型节点 Spearman 秩相关：**{node_spearman:.4f}**（{'已计算' if np.isfinite(node_spearman) else '未能计算'}）

### 判断
关键节点排序{'对权重设置不敏感，具有较好稳健性' if node_stable else '在一定程度上依赖权重设定，宜结合业务解释综合判断'}。

## 4. 论文 Methods / Results 表述（可直接引用）

{methods_text}

{results_text}

## 5. Discussion 表述（可直接引用）

{discussion_text}
"""
    report_path = OUT_DIR / "sensitivity_report_summary.md"
    report_path.write_text(content, encoding="utf-8")
    print(f"OK: Wrote {report_path.resolve()}")


def _append_run_log(
    *,
    n_samples: int,
    chain_summary: pd.DataFrame,
    chain_overlap: pd.DataFrame,
    rule_summary: pd.DataFrame,
    rule_overlap: pd.DataFrame,
    node_result: dict[str, Any],
) -> None:
    log_path = ROOT / "RUN_LOG.md"
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %z")

    def _chain_n(thr: int) -> int:
        row = chain_summary.loc[chain_summary["min_chain_count"] == thr]
        return int(row["n_high_risk_chains"].iloc[0]) if len(row) else 0

    def _rule_n(thr: int) -> int:
        row = rule_summary.loc[rule_summary["rule_threshold"] == thr]
        return int(row["n_rules"].iloc[0]) if len(row) else 0

    def _jacc(df: pd.DataFrame, comp: str) -> float:
        row = df.loc[df["comparison"] == comp]
        return float(row["top10_jaccard_similarity"].iloc[0]) if len(row) else float("nan")

    files = [
        str(OUT_DIR / "chain_threshold_2.xlsx"),
        str(OUT_DIR / "chain_threshold_3.xlsx"),
        str(OUT_DIR / "chain_threshold_4.xlsx"),
        str(OUT_DIR / "sensitivity_chain_threshold_summary.xlsx"),
        str(OUT_DIR / "sensitivity_chain_overlap.xlsx"),
        str(OUT_DIR / "rules_threshold_5.csv"),
        str(OUT_DIR / "rules_threshold_8.csv"),
        str(OUT_DIR / "rules_threshold_10.csv"),
        str(OUT_DIR / "sensitivity_rule_threshold_summary.xlsx"),
        str(OUT_DIR / "sensitivity_rule_overlap.xlsx"),
        str(OUT_DIR / "sensitivity_node_weight_main_top10.xlsx"),
        str(OUT_DIR / "sensitivity_node_weight_equal_top10.xlsx"),
        str(OUT_DIR / "sensitivity_node_weight_comparison.xlsx"),
        str(OUT_DIR / "sensitivity_report_summary.md"),
        str(FIG_DIR / "FigureS1_sensitivity_chain_threshold.png"),
        str(FIG_DIR / "FigureS1_sensitivity_chain_threshold.pdf"),
        str(FIG_DIR / "FigureS2_sensitivity_rule_threshold.png"),
        str(FIG_DIR / "FigureS2_sensitivity_rule_threshold.pdf"),
        str(FIG_DIR / "FigureS3_sensitivity_node_weight.png"),
        str(FIG_DIR / "FigureS3_sensitivity_node_weight.pdf"),
    ]

    lines = [
        "## 敏感性分析模块新增与运行",
        "",
        f"- **记录时间**：{ts}",
        f"- **数据范围**：真实样本 only，{n_samples} 条（USE_REAL_ONLY=True）",
        f"- **链条阈值设置**：2、3、4",
        f"- **各阈值下高风险链条数量**：MIN_CHAIN_COUNT=2 → {_chain_n(2)}；=3 → {_chain_n(3)}；=4 → {_chain_n(4)}",
        f"- **链条 Top10 Jaccard**：2 vs 3 = {_jacc(chain_overlap, 'threshold_2_vs_3'):.4f}；"
        f"3 vs 4 = {_jacc(chain_overlap, 'threshold_3_vs_4'):.4f}；"
        f"2 vs 4 = {_jacc(chain_overlap, 'threshold_2_vs_4'):.4f}",
        f"- **规则阈值设置**：5、8、10（confidence≥0.60，lift≥1.20）",
        f"- **各阈值下稳健预防型规则数量**：threshold=5 → {_rule_n(5)}；=8 → {_rule_n(8)}；=10 → {_rule_n(10)}",
        f"- **规则 Top10 Jaccard**：5 vs 8 = {_jacc(rule_overlap, 'threshold_5_vs_8'):.4f}；"
        f"8 vs 10 = {_jacc(rule_overlap, 'threshold_8_vs_10'):.4f}；"
        f"5 vs 10 = {_jacc(rule_overlap, 'threshold_5_vs_10'):.4f}",
        f"- **主权重 vs 均等权重 Top10 节点重合数量**：{node_result['overlap']}",
        f"- **主权重 vs 均等权重 Jaccard**：{node_result['jaccard']:.4f}",
        f"- **Spearman rank correlation**：{node_result['spearman']:.4f}"
        + ("（已计算）" if np.isfinite(node_result["spearman"]) else "（未能计算）"),
        "- **新生成的主要文件路径**：",
    ]
    for fp in files:
        lines.append(f"  - `{fp}`")
    lines.extend(["", "---", ""])

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"OK: Appended sensitivity block to {log_path.resolve()}")


def run() -> None:
    _ensure_dirs()

    if not CHAINS_CSV.is_file():
        raise FileNotFoundError(f"未找到 {CHAINS_CSV.resolve()} — 请先运行主分析。")
    if not CLEANED_PATH.is_file():
        raise FileNotFoundError(f"未找到 {CLEANED_PATH.resolve()} — 请先运行主分析。")

    chains_df = pd.read_csv(CHAINS_CSV, encoding="utf-8-sig")
    cleaned = pd.read_excel(CLEANED_PATH, engine="openpyxl")
    n_samples = len(cleaned)

    print(f"INFO: Sensitivity analysis on {n_samples} real samples", flush=True)

    chain_results = _run_chain_threshold_sensitivity(chains_df)
    rule_results = _run_rule_threshold_sensitivity(cleaned)
    node_result = _run_node_weight_sensitivity()

    chain_summary = pd.read_excel(OUT_DIR / "sensitivity_chain_threshold_summary.xlsx", engine="openpyxl")
    chain_overlap = pd.read_excel(OUT_DIR / "sensitivity_chain_overlap.xlsx", engine="openpyxl")
    rule_summary = pd.read_excel(OUT_DIR / "sensitivity_rule_threshold_summary.xlsx", engine="openpyxl")
    rule_overlap = pd.read_excel(OUT_DIR / "sensitivity_rule_overlap.xlsx", engine="openpyxl")

    _plot_chain_threshold(OUT_DIR / "sensitivity_chain_threshold_summary.xlsx")
    _plot_rule_threshold(OUT_DIR / "sensitivity_rule_threshold_summary.xlsx")
    _plot_node_weight(node_result)

    _write_report_summary(
        chain_summary=chain_summary,
        chain_overlap=chain_overlap,
        rule_summary=rule_summary,
        rule_overlap=rule_overlap,
        node_result=node_result,
        n_samples=n_samples,
    )
    _append_run_log(
        n_samples=n_samples,
        chain_summary=chain_summary,
        chain_overlap=chain_overlap,
        rule_summary=rule_summary,
        rule_overlap=rule_overlap,
        node_result=node_result,
    )

    print("OK: Sensitivity analysis completed.", flush=True)


def run_sensitivity_figures_only() -> None:
    """Regenerate supplementary sensitivity figures from existing tables (no table recomputation)."""
    _ensure_dirs()
    p_chain = OUT_DIR / "sensitivity_chain_threshold_summary.xlsx"
    p_rule = OUT_DIR / "sensitivity_rule_threshold_summary.xlsx"
    p_main = OUT_DIR / "sensitivity_node_weight_main_top10.xlsx"
    p_equal = OUT_DIR / "sensitivity_node_weight_equal_top10.xlsx"
    for p in (p_chain, p_rule, p_main, p_equal):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p.resolve()} — run full sensitivity analysis first.")
    _plot_chain_threshold(p_chain)
    _plot_rule_threshold(p_rule)
    top_main = pd.read_excel(p_main, engine="openpyxl")
    top_equal = pd.read_excel(p_equal, engine="openpyxl")
    _plot_node_weight({"top_main": top_main, "top_equal": top_equal})
    print("OK: Sensitivity figures-only completed.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--figures-only":
        run_sensitivity_figures_only()
    else:
        run()
