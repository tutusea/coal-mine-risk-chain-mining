# -*- coding: utf-8 -*-
"""阶段 04：读取事故链条 CSV，聚合统计、排序、导出汇总表与 Sankey 边表。"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

CHAINS_CSV = config.OUTPUT_DIR / "accident_chains.csv"
OUT_FULL_SUMMARY_CSV = config.OUTPUT_DIR / "chain_full_summary.csv"
OUT_FULL_SUMMARY_XLSX = config.OUTPUT_DIR / "full_chain_summary.xlsx"
OUT_LOCAL_SUMMARY_CSV = config.OUTPUT_DIR / "chain_local_summary.csv"
OUT_LOCAL_SUMMARY_XLSX = config.OUTPUT_DIR / "local_chain_summary.xlsx"
OUT_TOP_SEVERE_FULL = config.OUTPUT_DIR / "top_30_severe_chains.xlsx"
OUT_TOP_SERIOUS_FULL = config.OUTPUT_DIR / "top_30_serious_chains.xlsx"
OUT_TOP_LOCAL_SEVERE = config.OUTPUT_DIR / "top_30_local_severe_chains.xlsx"
OUT_TOP_LOCAL_SERIOUS = config.OUTPUT_DIR / "top_30_local_serious_chains.xlsx"
OUT_TOP_LOCAL_SEVERE_DEDUP = config.OUTPUT_DIR / "top_30_local_severe_chains_dedup.xlsx"
OUT_TOP_LOCAL_SERIOUS_DEDUP = config.OUTPUT_DIR / "top_30_local_serious_chains_dedup.xlsx"
OUT_RANKING = config.OUTPUT_DIR / "chain_ranking.xlsx"
OUT_SANKEY = config.OUTPUT_DIR / "sankey_edges.csv"

CHAIN_FULL_COL = "chain_full"
LOCAL_COLS = ("chain_local_1", "chain_local_2", "chain_local_3", "chain_local_4", "chain_local_5")

_CHAIN_LAYER_PRIORITY: dict[str, int] = {
    "chain_local_1": 0,
    "chain_local_2": 1,
    "chain_local_3": 2,
    "chain_local_4": 3,
    "chain_local_5": 4,
}


def _is_nonempty_chain(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    s = str(v).strip()
    return s != "" and s.lower() not in {"nan", "none"}


def _severe_mask(sb) -> np.ndarray:
    s = pd.to_numeric(sb, errors="coerce")
    return (s == 1) & s.notna()


def _serious_mask(sev: pd.Series) -> np.ndarray:
    return sev.astype(str).str.strip() == "重伤"


def _aggregate_chain_column(
    df: pd.DataFrame,
    col: str,
    total_n: int,
    min_count: int,
) -> pd.DataFrame:
    sub = df.loc[df[col].map(_is_nonempty_chain)].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                col,
                "count",
                "proportion",
                "severe_count",
                "severe_rate",
                "serious_count",
                "serious_rate",
                "average_severity_score",
            ]
        )

    sub["_severe_i"] = _severe_mask(sub["severe_binary"]).astype(np.int64)
    sub["_serious_i"] = _serious_mask(sub["Sev"]).astype(np.int64)
    sub["_sc_num"] = pd.to_numeric(sub["severity_score"], errors="coerce")

    g = sub.groupby(col, dropna=False)
    cnt = g.size().rename("count")
    severe_sum = g["_severe_i"].sum().rename("severe_count")
    serious_sum = g["_serious_i"].sum().rename("serious_count")
    avg_sc = g["_sc_num"].mean().rename("average_severity_score")

    out = pd.concat([cnt, severe_sum, serious_sum, avg_sc], axis=1).reset_index()
    out["proportion"] = out["count"] / total_n if total_n else np.nan
    out["severe_rate"] = np.where(out["count"] > 0, out["severe_count"] / out["count"], np.nan)
    out["serious_rate"] = np.where(out["count"] > 0, out["serious_count"] / out["count"], np.nan)

    out = out.loc[out["count"] >= min_count].copy()
    cols_order = [
        col,
        "count",
        "proportion",
        "severe_count",
        "severe_rate",
        "serious_count",
        "serious_rate",
        "average_severity_score",
    ]
    out = out[cols_order]
    out = out.sort_values(
        by=["severe_rate", "count", "average_severity_score"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return out


def _dedup_local_chains_by_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    对局部链汇总按 chain 字符串去重：保留 count 最大者；count 相同则 chain_layer 更靠前者优先
    （chain_local_1 优先于 chain_local_2，依此类推）。
    """
    if df.empty or "chain" not in df.columns:
        return df
    d = df.copy()
    d["_cnt"] = pd.to_numeric(d["count"], errors="coerce").fillna(0).astype(int)
    d["_layer_rank"] = d["chain_layer"].map(_CHAIN_LAYER_PRIORITY).fillna(99).astype(int)
    d = d.sort_values(
        by=["chain", "_cnt", "_layer_rank"],
        ascending=[True, False, True],
        na_position="last",
    )
    d = d.drop_duplicates(subset=["chain"], keep="first")
    return d.drop(columns=["_cnt", "_layer_rank"], errors="ignore").reset_index(drop=True)


def _build_high_risk_local_severe_dedup(local_df: pd.DataFrame, min_c: int) -> pd.DataFrame:
    """
    论文用「高风险局部链条」表：仅含 severe_count>0 且 severe_rate>0 且 count>=min_c 的链，
    按 chain 去重后按 severe_rate、severe_count、count、average_severity_score 降序，至多 30 条。
    """
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
    dedup = _dedup_local_chains_by_string(d)
    dedup = dedup.sort_values(
        by=["severe_rate", "severe_count", "count", "average_severity_score"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return dedup.head(30)


def _build_high_risk_local_serious_dedup(local_df: pd.DataFrame, min_c: int) -> pd.DataFrame:
    """
    重伤侧 dedup 表：仅含 serious_count>0 且 serious_rate>0 且 count>=min_c，去重后排序，至多 30 条。
    """
    if local_df.empty or "chain" not in local_df.columns:
        return pd.DataFrame(columns=local_df.columns if len(local_df.columns) else [])
    d = local_df.copy()
    d["_jc"] = pd.to_numeric(d["serious_count"], errors="coerce").fillna(0)
    d["_jr"] = pd.to_numeric(d["serious_rate"], errors="coerce")
    d["_cnt"] = pd.to_numeric(d["count"], errors="coerce").fillna(0)
    d = d.loc[(d["_jc"] > 0) & (d["_jr"] > 0) & (d["_cnt"] >= min_c)].copy()
    d = d.drop(columns=["_jc", "_jr", "_cnt"], errors="ignore")
    if d.empty:
        return d
    dedup = _dedup_local_chains_by_string(d)
    dedup = dedup.sort_values(
        by=["serious_rate", "serious_count", "count", "average_severity_score"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return dedup.head(30)


def _qc_and_log_local_chain_dedup(
    *,
    top_sev: pd.DataFrame,
    top_ser: pd.DataFrame,
    local_summary_rows: int,
    total_chain_records: int,
    min_c: int,
) -> None:
    """控制台质量检查；必要时将摘要追加到项目根 RUN_LOG.md。"""
    log_path = ROOT / "RUN_LOG.md"
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %z")

    n_zero_sev = 0
    n_zero_ser = 0
    if not top_sev.empty and "severe_rate" in top_sev.columns:
        sr = pd.to_numeric(top_sev["severe_rate"], errors="coerce").fillna(0)
        n_zero_sev = int((sr <= 0).sum())
    if not top_ser.empty and "serious_rate" in top_ser.columns:
        jr = pd.to_numeric(top_ser["serious_rate"], errors="coerce").fillna(0)
        n_zero_ser = int((jr <= 0).sum())

    if n_zero_sev > 0:
        print(
            f"ERROR: top_30_local_severe_chains_dedup.xlsx 逻辑异常：存在 {n_zero_sev} 行 severe_rate<=0。",
            flush=True,
        )
    if n_zero_ser > 0:
        print(
            f"ERROR: top_30_local_serious_chains_dedup.xlsx 逻辑异常：存在 {n_zero_ser} 行 serious_rate<=0。",
            flush=True,
        )
    if top_sev.empty and top_ser.empty:
        print(
            "WARNING: 在当前 MIN_CHAIN_COUNT 与「高风险」筛选下，"
            "top_30_local_severe_chains_dedup 与 top_30_local_serious_chains_dedup 均为空；"
            "正文高风险局部链条表可能无可用行。",
            flush=True,
        )

    lines = [
        f"### AUTO — 局部链条 dedup 质量检查 / {ts}",
        "",
        f"- **事故链条记录数（accident_chains.csv 行数）**：{total_chain_records}",
        f"- **MIN_CHAIN_COUNT**：{min_c}",
        f"- **local_chain_summary 行数（未过滤原始汇总）**：{local_summary_rows}",
        f"- **top_30_local_severe_chains_dedup 行数**：{len(top_sev)}",
        f"- **其中 severe_rate<=0 行数（应为 0）**：{n_zero_sev}",
        f"- **top_30_local_serious_chains_dedup 行数**：{len(top_ser)}",
        f"- **其中 serious_rate<=0 行数（应为 0）**：{n_zero_ser}",
        "",
        "---",
        "",
    ]
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"OK: Appended local-chain dedup QC block to {log_path.resolve()}", flush=True)
    except Exception as exc:
        print(f"WARNING: Could not append dedup QC to RUN_LOG.md: {exc}", flush=True)


def _build_sankey_edges(df: pd.DataFrame) -> pd.DataFrame:
    weight: dict[tuple[str, str], int] = defaultdict(int)
    severe_w: dict[tuple[str, str], int] = defaultdict(int)
    severe_m = _severe_mask(df["severe_binary"])

    for idx, row in df.iterrows():
        cf = row.get("chain_full")
        if not _is_nonempty_chain(cf):
            continue
        parts = [p.strip() for p in str(cf).split("->")]
        if len(parts) < 2:
            continue
        is_sev = bool(severe_m.loc[idx]) if idx in severe_m.index else False
        for a, b in zip(parts[:-1], parts[1:]):
            if not a or not b:
                continue
            key = (a, b)
            weight[key] += 1
            if is_sev:
                severe_w[key] += 1

    rows = []
    for (src, tgt), w in sorted(weight.items()):
        sv = severe_w.get((src, tgt), 0)
        rows.append(
            {
                "source": src,
                "target": tgt,
                "weight": w,
                "severe_count": sv,
                "edge_severity_rate": sv / w if w else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    if not CHAINS_CSV.is_file():
        raise FileNotFoundError(f"未找到链条表: {CHAINS_CSV.resolve()} — 请先运行阶段 02。")

    df = pd.read_csv(CHAINS_CSV, encoding="utf-8-sig")
    for c in [CHAIN_FULL_COL, *LOCAL_COLS, "severe_binary", "severity_score", "Sev"]:
        if c not in df.columns:
            raise ValueError(f"accident_chains.csv 缺少列: {c}")

    total_n = len(df)
    min_c = int(getattr(config, "MIN_CHAIN_COUNT", 3))

    full_df = _aggregate_chain_column(df, CHAIN_FULL_COL, total_n, min_c)
    local_parts = []
    for lc in LOCAL_COLS:
        part = _aggregate_chain_column(df, lc, total_n, min_c)
        part.insert(0, "chain_layer", lc)
        part = part.rename(columns={lc: "chain"})
        local_parts.append(part)
    local_df = pd.concat(local_parts, ignore_index=True) if local_parts else pd.DataFrame()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUT_FULL_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    full_df.to_excel(OUT_FULL_SUMMARY_XLSX, index=False, engine="openpyxl")

    local_df.to_csv(OUT_LOCAL_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    local_df.to_excel(OUT_LOCAL_SUMMARY_XLSX, index=False, engine="openpyxl")

    top_severe_full = full_df.head(30)
    top_severe_full.to_excel(OUT_TOP_SEVERE_FULL, index=False, engine="openpyxl")

    serious_sorted_full = full_df.sort_values(
        by=["serious_rate", "count", "average_severity_score"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    serious_sorted_full.head(30).to_excel(OUT_TOP_SERIOUS_FULL, index=False, engine="openpyxl")

    # 局部链：跨层合并后取 Top30（主分析）；另输出按 chain 去重后的论文用表
    if not local_df.empty and "chain" in local_df.columns:
        loc_sev = local_df.sort_values(
            by=["severe_rate", "count", "average_severity_score"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
        top_loc_sev = loc_sev.head(30)
        top_loc_sev_dedup = _build_high_risk_local_severe_dedup(local_df, min_c)

        loc_ser = local_df.sort_values(
            by=["serious_rate", "count", "average_severity_score"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
        top_loc_ser = loc_ser.head(30)
        top_loc_ser_dedup = _build_high_risk_local_serious_dedup(local_df, min_c)
    else:
        top_loc_sev = pd.DataFrame(columns=local_df.columns if not local_df.empty else [])
        top_loc_ser = top_loc_sev.copy()
        top_loc_sev_dedup = top_loc_sev.copy()
        top_loc_ser_dedup = top_loc_ser.copy()

    top_loc_sev.to_excel(OUT_TOP_LOCAL_SEVERE, index=False, engine="openpyxl")
    top_loc_ser.to_excel(OUT_TOP_LOCAL_SERIOUS, index=False, engine="openpyxl")
    top_loc_sev_dedup.to_excel(OUT_TOP_LOCAL_SEVERE_DEDUP, index=False, engine="openpyxl")
    top_loc_ser_dedup.to_excel(OUT_TOP_LOCAL_SERIOUS_DEDUP, index=False, engine="openpyxl")

    with pd.ExcelWriter(OUT_RANKING, engine="openpyxl") as xw:
        full_ranked = full_df.copy()
        full_ranked.insert(0, "rank", range(1, len(full_ranked) + 1))
        full_ranked.to_excel(xw, sheet_name="chain_full", index=False)

        local_ranked = local_df.copy()
        local_ranked = local_ranked.sort_values(
            by=["chain_layer", "severe_rate", "count", "average_severity_score"],
            ascending=[True, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
        local_ranked.insert(
            0,
            "rank_within_layer",
            local_ranked.groupby("chain_layer").cumcount() + 1,
        )
        local_ranked.to_excel(xw, sheet_name="chain_local", index=False)

    sankey_df = _build_sankey_edges(df)
    sankey_df.to_csv(OUT_SANKEY, index=False, encoding="utf-8-sig")

    print(f"OK: Total records={total_n} MIN_CHAIN_COUNT={min_c}")
    print(f"OK: chain_full summary rows={len(full_df)} local_chain summary rows={len(local_df)}")
    print(f"OK: Sankey edge rows={len(sankey_df)}")
    print(f"OK: Wrote {OUT_FULL_SUMMARY_CSV.resolve()} and {OUT_FULL_SUMMARY_XLSX.resolve()}")
    print(f"OK: Wrote {OUT_LOCAL_SUMMARY_CSV.resolve()} and {OUT_LOCAL_SUMMARY_XLSX.resolve()}")
    print(f"OK: Wrote {OUT_TOP_LOCAL_SEVERE.resolve()} (rows={len(top_loc_sev)})")
    print(f"OK: Wrote {OUT_TOP_LOCAL_SERIOUS.resolve()} (rows={len(top_loc_ser)})")
    print(f"OK: Wrote {OUT_TOP_LOCAL_SEVERE_DEDUP.resolve()} (rows={len(top_loc_sev_dedup)})")
    print(f"OK: Wrote {OUT_TOP_LOCAL_SERIOUS_DEDUP.resolve()} (rows={len(top_loc_ser_dedup)})")
    print(f"OK: Wrote {OUT_RANKING.resolve()}")
    print(f"OK: Wrote {OUT_SANKEY.resolve()}")

    _qc_and_log_local_chain_dedup(
        top_sev=top_loc_sev_dedup,
        top_ser=top_loc_ser_dedup,
        local_summary_rows=len(local_df),
        total_chain_records=total_n,
        min_c=min_c,
    )


if __name__ == "__main__":
    run()
