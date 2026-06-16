# -*- coding: utf-8 -*-
"""阶段 01：读取 Excel、原始数据审计、字段校验、清洗、样本筛选与中间表写出。"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

# 逻辑字段键（与 config.COLUMN_ALIASES 一致），用于校验与解析
REQUIRED_CANONICAL_KEYS = [
    "team",
    "job",
    "education",
    "shift",
    "work_place",
    "injury_part",
    "activity",
    "hazard_agent",
    "harm_degree",
    "accident_place",
    "accident_cause",
    "is_synthetic",
]

REQUIRED_LABEL_ZH = {
    "team": "队别",
    "job": "工种",
    "education": "学历",
    "shift": "班次",
    "work_place": "工作地点",
    "injury_part": "受伤部位",
    "activity": "作业活动",
    "hazard_agent": "致害物",
    "harm_degree": "伤害程度",
    "accident_place": "事故地点",
    "accident_cause": "事故原因",
    "is_synthetic": "is_synthetic",
}

OUTPUT_COLUMNS = [
    "Team",
    "Job",
    "Edu",
    "Shift",
    "Loc",
    "Act",
    "Haz",
    "HazardMode",
    "InjuryForm",
    "HazardSource",
    "Body",
    "Cause",
    "Sev",
    "is_synthetic",
    "is_synthetic_norm",
    "severe_binary",
    "severity_score",
]

PROFILE_FIELDS = [
    "Team",
    "Job",
    "Edu",
    "Shift",
    "Loc",
    "Act",
    "Haz",
    "HazardMode",
    "InjuryForm",
    "HazardSource",
    "Body",
    "Cause",
    "Sev",
    "is_synthetic",
    "is_synthetic_norm",
    "severe_binary",
    "severity_score",
]

# 伤害程度：非严重 / 较严重（二值）
_NON_SEVERE_HARM_LABELS = frozenset(
    {
        "皮伤",
        "轻微伤",
        "微伤",
        "擦伤",
        "轻度伤",
        "小伤",
        "非严重伤害",
        "minor",
        "non-severe",
    }
)
_SEVERE_HARM_LABELS = frozenset(
    {
        "轻伤",
        "重伤",
        "死亡",
        "工亡",
        "致残",
        "重大伤害",
        "严重伤害",
    }
)
# 评分档
_SCORE_1_LABELS = frozenset(_NON_SEVERE_HARM_LABELS)
_SCORE_2_LABELS = frozenset({"轻伤"})
_SCORE_3_LABELS = frozenset({"重伤", "死亡", "工亡", "致残", "重大伤害", "严重伤害"})

# 原始「致害物」字段语义拆分：致害方式 / 伤害形态 / 真正致害源（明确列表 + 子串兜底）
UNKNOWN = "Unknown"
NOT_AVAILABLE = "NotAvailable"

_HAZARD_MODE_CHARS = frozenset("砸碰挤摔刮崴压夹撞滑跌滚扭刺割划")
_INJURY_FORM_EXACT = frozenset(
    {
        "骨折",
        "离断",
        "内脏破裂",
        "破裂",
        "出血",
        "挫伤",
        "扭伤",
        "擦伤",
        "烧伤",
        "烫伤",
        "中毒",
        "窒息",
        "聋",
        "听力损伤",
    }
)
_INJURY_FORM_SUBSTR = sorted(_INJURY_FORM_EXACT, key=len, reverse=True)

_HAZARD_SOURCE_EXACT = frozenset(
    {
        "顶板",
        "设备",
        "车辆",
        "机械",
        "工具",
        "煤岩",
        "矸石",
        "巷道",
        "皮带",
        "运输设备",
        "电气",
        "支护",
        "冒顶",
        "片帮",
    }
)
_HAZARD_SOURCE_SUBSTR = sorted(_HAZARD_SOURCE_EXACT, key=len, reverse=True)


def _haz_chain_skip_value(s: str) -> bool:
    t = str(s).strip()
    if t == "":
        return True
    low = t.lower()
    return low in {"unknown", "notapplicable", "nan", "none", "-"}


def _first_substring_match(raw: str, keywords: tuple[str, ...]) -> tuple[int, str] | None:
    """在 raw 中取「起始位置最靠前；同位置取最长词」的首次命中，否则 None。"""
    best: tuple[int, int, str] | None = None  # (pos, -len, kw)
    for kw in keywords:
        p = raw.find(kw)
        if p < 0:
            continue
        cand = (p, -len(kw), kw)
        if best is None or cand < best:
            best = cand
    if best is None:
        return None
    return best[0], best[2]


def _span_from_match(start: int, kw: str) -> tuple[int, int]:
    return start, start + len(kw)


def _index_in_any_span(i: int, spans: list[tuple[int, int]]) -> bool:
    for a, b in spans:
        if a <= i < b:
            return True
    return False


def classify_hazard_agent(haz_display: str) -> tuple[str, str, str, str]:
    """
    将原始 hazard_agent（Haz）拆分为 HazardMode、InjuryForm、HazardSource（可非互斥填充），
    并给出用于统计主标签的 mapped_type。

    返回 (HazardMode, InjuryForm, HazardSource, mapped_type)，
    mapped_type ∈ {injury_form, hazard_source, hazard_mode, uncertain, empty}
    """
    raw = (
        str(haz_display).strip()
        if haz_display is not None and not (isinstance(haz_display, float) and np.isnan(haz_display))
        else ""
    )
    if _haz_chain_skip_value(raw):
        return UNKNOWN, UNKNOWN, NOT_AVAILABLE, "empty"

    spans: list[tuple[int, int]] = []

    injury_form = UNKNOWN
    inj_m = _first_substring_match(raw, tuple(_INJURY_FORM_SUBSTR))
    if inj_m is not None:
        p0, kw = inj_m
        injury_form = kw
        spans.append(_span_from_match(p0, kw))

    hazard_source = NOT_AVAILABLE
    src_m = _first_substring_match(raw, tuple(_HAZARD_SOURCE_SUBSTR))
    if src_m is not None:
        p0, kw = src_m
        hazard_source = kw
        spans.append(_span_from_match(p0, kw))

    hazard_mode = UNKNOWN
    mode_positions: list[tuple[int, str]] = []
    for i, ch in enumerate(raw):
        if ch in _HAZARD_MODE_CHARS and not _index_in_any_span(i, spans):
            mode_positions.append((i, ch))
    if mode_positions:
        mode_positions.sort(key=lambda x: (x[0], x[1]))
        hazard_mode = mode_positions[0][1]

    # 主类型（审计/频次）：伤害形态 > 致害源 > 致害方式 > 无法归类
    if injury_form != UNKNOWN:
        primary = "injury_form"
    elif hazard_source != NOT_AVAILABLE and str(hazard_source).strip() != "":
        primary = "hazard_source"
    elif hazard_mode != UNKNOWN:
        primary = "hazard_mode"
    else:
        primary = "uncertain"

    return hazard_mode, injury_form, hazard_source, primary


def _haz_audit_contributions_from_row(
    haz_raw: str,
    severe_i: int,
    hazard_mode: str,
    injury_form: str,
    hazard_source: str,
) -> list[tuple[str, str, str, int]]:
    """
    由单行语义拆分为 0–3 条审计贡献 + 全无法归类时一条 uncertain。
    每项：(mapped_field, mapped_value, mapped_type, severe_i)
    """
    rows: list[tuple[str, str, str, int]] = []
    if injury_form != UNKNOWN:
        rows.append(("InjuryForm", injury_form, "injury_form", severe_i))
    if hazard_source != NOT_AVAILABLE and str(hazard_source).strip() != "":
        rows.append(("HazardSource", str(hazard_source).strip(), "hazard_source", severe_i))
    if hazard_mode != UNKNOWN:
        rows.append(("HazardMode", str(hazard_mode).strip(), "hazard_mode", severe_i))
    if not rows:
        rows.append(("Unknown", str(haz_raw), "uncertain", severe_i))
    return rows


def _build_hazard_value_audit_xlsx(df: pd.DataFrame) -> pd.DataFrame:
    """
    original_hazard, mapped_field, mapped_value, mapped_type,
    count, severe_count, severity_rate
    """
    tmp = df.copy()
    tmp["_orig"] = tmp["Haz"].astype(str)
    tmp["_sev_i"] = (pd.to_numeric(tmp["severe_binary"], errors="coerce") == 1).astype(int)

    exp_rows: list[dict] = []
    for _, r in tmp.iterrows():
        for mf, mv, mt, sv in _haz_audit_contributions_from_row(
            str(r["_orig"]),
            int(r["_sev_i"]),
            str(r["HazardMode"]),
            str(r["InjuryForm"]),
            str(r["HazardSource"]),
        ):
            exp_rows.append(
                {
                    "original_hazard": str(r["_orig"]),
                    "mapped_field": mf,
                    "mapped_value": mv,
                    "mapped_type": mt,
                    "_sev_i": sv,
                }
            )
    if not exp_rows:
        return pd.DataFrame(
            columns=[
                "original_hazard",
                "mapped_field",
                "mapped_value",
                "mapped_type",
                "count",
                "severe_count",
                "severity_rate",
            ]
        )

    ex = pd.DataFrame(exp_rows)
    g = ex.groupby(["original_hazard", "mapped_field", "mapped_value", "mapped_type"], dropna=False)
    out = g.agg(count=("_sev_i", "size"), severe_count=("_sev_i", "sum")).reset_index()
    out["severity_rate"] = np.where(out["count"] > 0, out["severe_count"] / out["count"], np.nan)
    return out.sort_values(["mapped_type", "mapped_field", "count"], ascending=[True, True, False]).reset_index(
        drop=True
    )


def normalize_is_synthetic(value) -> str:
    """
    将原始 is_synthetic 单元格统一为 real / synthetic / unknown。
    不丢弃 unknown 行；筛选在 USE_REAL_ONLY 时仅保留 real。
    """
    if value is None:
        return "unknown"
    try:
        if pd.isna(value):
            return "unknown"
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        return "real" if not bool(value) else "synthetic"

    if isinstance(value, (int, np.integer)):
        iv = int(value)
        if iv == 0:
            return "real"
        if iv == 1:
            return "synthetic"
        return "unknown"

    if isinstance(value, (float, np.floating)):
        try:
            if np.isnan(float(value)):
                return "unknown"
        except (TypeError, ValueError):
            return "unknown"
        fv = float(value)
        if fv == 0.0:
            return "real"
        if fv == 1.0:
            return "synthetic"
        return "unknown"

    s = str(value).strip()
    if s == "":
        return "unknown"
    low = s.lower()
    if low in {"nan", "none", "-", "na", "<na>"}:
        return "unknown"

    real_tokens = {
        "0",
        "0.0",
        "false",
        "否",
        "真实",
        "real",
        "非合成",
        "原始",
        "人工",
        "actual",
    }
    syn_tokens = {
        "1",
        "1.0",
        "true",
        "是",
        "合成",
        "synthetic",
        "生成",
        "ai生成",
        "模拟",
    }
    if low in real_tokens:
        return "real"
    if low in syn_tokens:
        return "synthetic"

    try:
        fv = float(s)
        if fv == 0.0:
            return "real"
        if fv == 1.0:
            return "synthetic"
    except (TypeError, ValueError):
        pass

    return "unknown"


def _find_column(df: pd.DataFrame, canonical_key: str) -> str | None:
    candidates = list(
        dict.fromkeys(list(config.COLUMN_ALIASES.get(canonical_key, [])) + [canonical_key])
    )
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _is_empty_like(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"nan", "none", "-", "na", "<na>"}:
            return True
        return False
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return True
    return False


def _strip_cell(v):
    if _is_empty_like(v):
        return np.nan
    if isinstance(v, (int, np.integer, bool, np.bool_)):
        return v
    if isinstance(v, (float, np.floating)) and not np.isnan(v):
        if float(v).is_integer():
            return str(int(v))
        return str(v).strip()
    return str(v).strip()


def _series_stripped_or_na(s: pd.Series) -> pd.Series:
    return s.map(_strip_cell)


def _series_to_unknown_str(s: pd.Series) -> pd.Series:
    def cell(v):
        if _is_empty_like(v):
            return "Unknown"
        if isinstance(v, (int, np.integer, bool, np.bool_)):
            return str(int(v)) if isinstance(v, bool) or isinstance(v, np.bool_) else str(int(v))
        if isinstance(v, (float, np.floating)) and not np.isnan(v) and float(v).is_integer():
            return str(int(v))
        t = str(v).strip()
        if t == "" or t.lower() in {"nan", "none", "-", "na", "<na>"}:
            return "Unknown"
        return t

    return s.map(cell)


def _norm_harm_label(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    if t.isascii():
        return t.lower()
    return t


def _map_severity_vectors(sev_display: pd.Series, raw_harm_empty: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """返回 severe_binary, severity_score, is_unmapped（原始伤害非空且标签不在映射表）。"""
    sb: list[int | None] = []
    sc: list[int | None] = []
    unmapped: list[bool] = []
    for sev, empty_raw in zip(sev_display, raw_harm_empty):
        if bool(empty_raw) or sev == "Unknown" or _is_empty_like(sev):
            sb.append(None)
            sc.append(None)
            unmapped.append(False)
            continue
        key = _norm_harm_label(str(sev))
        if key in _NON_SEVERE_HARM_LABELS:
            sb.append(0)
            sc.append(1)
            unmapped.append(False)
            continue
        if key in _SEVERE_HARM_LABELS:
            sb.append(1)
            if key in _SCORE_2_LABELS:
                sc.append(2)
            elif key in _SCORE_3_LABELS:
                sc.append(3)
            else:
                sc.append(2)
            unmapped.append(False)
            continue
        sb.append(0)
        sc.append(1)
        unmapped.append(True)
    idx = sev_display.index
    return (
        pd.Series(sb, index=idx, dtype=pd.Int64Dtype()),
        pd.Series(sc, index=idx, dtype=pd.Int64Dtype()),
        pd.Series(unmapped, index=idx, dtype=bool),
    )


def _missing_count(col: pd.Series) -> int:
    if col.dtype == object:
        return int((col == "Unknown").sum() + col.isna().sum())
    return int(col.isna().sum())


def _unique_count(col: pd.Series) -> int:
    return int(col.nunique(dropna=False))


def _check_required_columns(df: pd.DataFrame) -> tuple[dict[str, bool], list[str]]:
    present: dict[str, bool] = {}
    missing: list[str] = []
    for key in REQUIRED_CANONICAL_KEYS:
        col = _find_column(df, key)
        ok = col is not None
        present[key] = ok
        if not ok:
            missing.append(f"{key}（{REQUIRED_LABEL_ZH.get(key, key)}）")
    return present, missing


def _build_loc(acc: pd.Series, wp: pd.Series) -> pd.Series:
    acc_s = _series_stripped_or_na(acc)
    wp_s = _series_stripped_or_na(wp)
    loc = acc_s.combine_first(wp_s)
    return _series_to_unknown_str(loc)


def _python_type_name(v) -> str:
    return type(v).__name__


def _is_synthetic_raw_audit_rows(series: pd.Series) -> pd.DataFrame:
    rows = []
    for v in series:
        tname = _python_type_name(v)
        try:
            is_na = bool(pd.isna(v))
        except (TypeError, ValueError):
            is_na = False
        if is_na:
            rows.append(
                {
                    "原始值": "<NA>",
                    "Python类型": tname,
                    "去空格后的值": "",
                    "小写后的值": "",
                }
            )
            continue
        raw_disp = str(v)
        if isinstance(v, str):
            stripped = v.strip()
        else:
            stripped = raw_disp.strip()
        lowered = stripped.lower()
        rows.append(
            {
                "原始值": raw_disp,
                "Python类型": tname,
                "去空格后的值": stripped,
                "小写后的值": lowered,
            }
        )
    base = pd.DataFrame(rows)
    n = len(base)
    if n == 0:
        return pd.DataFrame(
            columns=["原始值", "Python类型", "去空格后的值", "小写后的值", "数量", "占比"]
        )
    g = base.groupby(["原始值", "Python类型", "去空格后的值", "小写后的值"], dropna=False).size().reset_index(name="数量")
    g["占比"] = g["数量"] / float(n)
    return g


def _write_original_audit(xls: pd.ExcelFile, audit_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """写出 sheet 汇总与列明细 CSV；返回 (sheet_summary_df, columns_long_df)。"""
    audit_dir.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    col_rows = []
    for name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=name, header=0, engine="openpyxl")
        cols = [str(c) for c in df.columns]
        sheet_rows.append(
            {
                "sheet_name": name,
                "n_rows": len(df),
                "n_columns": len(cols),
                "column_names_joined": " | ".join(cols),
            }
        )
        for j, c in enumerate(cols):
            col_rows.append({"sheet_name": name, "column_index": j, "column_name": c})
    summary_df = pd.DataFrame(sheet_rows)
    columns_df = pd.DataFrame(col_rows)
    with pd.ExcelWriter(audit_dir / "original_sheet_summary.xlsx", engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="sheet_summary", index=False)
    columns_df.to_csv(audit_dir / "original_columns.csv", index=False, encoding="utf-8-sig")
    return summary_df, columns_df


def _read_selected_sheet(excel_path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    names = list(xls.sheet_names)
    if config.SHEET_NAME is None:
        sheet = names[0]
    else:
        if config.SHEET_NAME not in names:
            raise ValueError(
                f"config.SHEET_NAME={config.SHEET_NAME!r} 不在工作簿中。"
                f" 可用 sheet：{names}"
            )
        sheet = config.SHEET_NAME
    raw = pd.read_excel(xls, sheet_name=sheet, header=0, engine="openpyxl")
    return raw, sheet, names


def _append_run_log_block(lines: list[str]) -> None:
    path = config.PROJECT_ROOT / "RUN_LOG.md"
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %z")
    block = "\n".join(["", f"### AUTO — 阶段01数据审计 / {ts}", ""] + lines + ["", "---", ""])
    with open(path, "a", encoding="utf-8") as f:
        f.write(block)


def _emit_warnings_and_log(
    *,
    sheet_used: str,
    all_sheet_names: list[str],
    n_total_raw: int,
    n_real: int,
    n_synth: int,
    n_unknown: int,
    sev_dist: pd.Series,
    sb_dist: pd.Series,
    warnings: list[str],
) -> None:
    for w in warnings:
        print(f"WARNING: {w}", flush=True)
    log_lines = [
        f"- **读取的 sheet 名称**：`{sheet_used}`（工作簿全部 sheet：{all_sheet_names}）",
        f"- **原始总样本数（所选 sheet 行数）**：{n_total_raw}",
        f"- **识别为真实样本数**：{n_real}",
        f"- **识别为合成样本数**：{n_synth}",
        f"- **unknown 样本数**：{n_unknown}",
        "- **伤害程度分布（Sev，当前写出之 cleaned 子集）**："
        + "；".join(
            f"{'<NA>' if pd.isna(k) else k}={int(v)}"
            for k, v in sev_dist.items()
            if pd.notna(v) and int(v) > 0
        ),
        "- **severe_binary 分布（当前写出之 cleaned 子集）**："
        + "；".join(
            f"{'<NA>' if pd.isna(k) else k}={int(v)}"
            for k, v in sb_dist.items()
            if pd.notna(v) and int(v) > 0
        ),
        f"- **输出目录**：`{config.OUTPUT_DIR}`；**图表目录**：`{config.FIGURES_DIR}`；**数据审计目录**：`{config.DATA_AUDIT_DIR}`",
    ]
    for w in warnings:
        log_lines.append(f"- **WARNING**：{w}")
    _append_run_log_block(log_lines)


def _write_data_profile(
    path: Path,
    df: pd.DataFrame,
    raw_nrows: int,
    col_present: dict[str, bool],
    sheet_used: str,
) -> None:
    n_total = len(df)
    norm = df["is_synthetic_norm"] if "is_synthetic_norm" in df.columns else pd.Series(["unknown"] * n_total)
    n_real = int((norm == "real").sum())
    n_synth = int((norm == "synthetic").sum())
    n_unk = int((norm == "unknown").sum())

    summary_rows = [
        ("总样本数", n_total),
        ("真实样本数", n_real),
        ("合成样本数", n_synth),
        ("unknown样本数", n_unk),
        ("原始行数（所选 sheet，筛选前）", raw_nrows),
        ("读取的sheet", sheet_used),
        ("USE_REAL_ONLY", config.USE_REAL_ONLY),
        ("EXCEL_PATH", str(config.EXCEL_PATH)),
        ("SHEET_NAME配置", str(config.SHEET_NAME)),
        ("OUTPUT_DIR", str(config.OUTPUT_DIR)),
        ("FIGURES_DIR", str(config.FIGURES_DIR)),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["指标", "值"])

    sev_dist = df["Sev"].value_counts(dropna=False).reset_index()
    sev_dist.columns = ["伤害程度(Sev)", "数量"]

    sb_dist = df["severe_binary"].value_counts(dropna=False).reset_index()
    sb_dist.columns = ["severe_binary", "数量"]

    field_stats = []
    for field in PROFILE_FIELDS:
        if field not in df.columns:
            continue
        field_stats.append(
            {
                "字段": field,
                "缺失数量": _missing_count(df[field]),
                "唯一值数量": _unique_count(df[field]),
            }
        )
    field_stats_df = pd.DataFrame(field_stats)

    req_df = pd.DataFrame(
        [
            {"要求字段": REQUIRED_LABEL_ZH.get(k, k), "逻辑键": k, "是否存在于原始表": "是" if col_present.get(k) else "否"}
            for k in REQUIRED_CANONICAL_KEYS
        ]
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        sev_dist.to_excel(writer, sheet_name="伤害程度分布", index=False)
        sb_dist.to_excel(writer, sheet_name="severe_binary分布", index=False)
        field_stats_df.to_excel(writer, sheet_name="字段缺失与唯一值", index=False)
        req_df.to_excel(writer, sheet_name="原始表字段检查", index=False)


def run() -> None:
    excel_path = Path(config.EXCEL_PATH)
    if not excel_path.is_file():
        raise FileNotFoundError(f"未找到 Excel 文件: {excel_path.resolve()} — 请在 config.EXCEL_PATH 中配置正确路径。")

    audit_dir = config.DATA_AUDIT_DIR
    audit_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    all_sheet_names = list(xls.sheet_names)
    print("INFO: Excel workbook sheets (in order):", flush=True)
    for i, sn in enumerate(all_sheet_names):
        print(f"  [{i}] {sn!r}", flush=True)

    _write_original_audit(xls, audit_dir)

    raw, sheet_used, _ = _read_selected_sheet(excel_path)
    raw_nrows = len(raw)
    print(f"INFO: Using sheet {sheet_used!r} (config.SHEET_NAME={config.SHEET_NAME!r}), rows={raw_nrows}.", flush=True)

    syn_col_name = _find_column(raw, "is_synthetic")
    if syn_col_name is None:
        syn_series = pd.Series([np.nan] * len(raw), index=raw.index)
    else:
        syn_series = raw[syn_col_name]

    raw_audit_df = _is_synthetic_raw_audit_rows(syn_series)
    raw_audit_df.to_csv(audit_dir / "is_synthetic_raw_distribution.csv", index=False, encoding="utf-8-sig")

    norm_series = syn_series.map(normalize_is_synthetic)
    norm_counts = norm_series.value_counts(dropna=False)
    norm_prop = norm_counts.rename_axis("is_synthetic_normalized").reset_index(name="count")
    norm_prop["proportion"] = norm_prop["count"] / float(len(raw)) if len(raw) else 0.0
    norm_prop.to_csv(audit_dir / "is_synthetic_normalized_distribution.csv", index=False, encoding="utf-8-sig")

    unk_mask = norm_series == "unknown"
    if unk_mask.any():
        raw.loc[unk_mask].to_excel(audit_dir / "unknown_is_synthetic_rows.xlsx", index=False, engine="openpyxl")

    col_present, missing = _check_required_columns(raw)
    if missing:
        print(
            "WARNING: The following logical fields were not matched to Excel columns "
            "(empty values and Unknown will be used where applicable):",
            flush=True,
        )
        for m in missing:
            print(f"  missing: {m}", flush=True)

    def col_series(key: str) -> pd.Series:
        c = _find_column(raw, key)
        if c is None:
            return pd.Series([np.nan] * len(raw), index=raw.index)
        return raw[c]

    team = _series_to_unknown_str(col_series("team"))
    job = _series_to_unknown_str(col_series("job"))
    edu = _series_to_unknown_str(col_series("education"))
    shift = _series_to_unknown_str(col_series("shift"))
    act = _series_to_unknown_str(col_series("activity"))
    haz = _series_to_unknown_str(col_series("hazard_agent"))
    haz_tpls = [classify_hazard_agent(v) for v in haz.tolist()]
    hazard_mode = pd.Series([t[0] for t in haz_tpls], index=haz.index, dtype=object)
    injury_form = pd.Series([t[1] for t in haz_tpls], index=haz.index, dtype=object)
    hazard_source = pd.Series([t[2] for t in haz_tpls], index=haz.index, dtype=object)
    haz_mapped_type = pd.Series([t[3] for t in haz_tpls], index=haz.index, dtype=object)

    body = _series_to_unknown_str(col_series("injury_part"))
    cause = _series_to_unknown_str(col_series("accident_cause"))

    harm_raw = col_series("harm_degree")
    raw_harm_empty = harm_raw.map(_is_empty_like)
    sev = _series_to_unknown_str(harm_raw)

    acc_raw = col_series("accident_place")
    wp_raw = col_series("work_place")
    loc = _build_loc(acc_raw, wp_raw)

    severe_binary, severity_score, unmapped_flag = _map_severity_vectors(sev, raw_harm_empty)

    is_syn_int: list[int | None] = []
    for lab in norm_series:
        if lab == "real":
            is_syn_int.append(0)
        elif lab == "synthetic":
            is_syn_int.append(1)
        else:
            is_syn_int.append(None)
    is_synthetic_int = pd.Series(is_syn_int, index=raw.index, dtype=pd.Int64Dtype())

    out = pd.DataFrame(
        {
            "Team": team,
            "Job": job,
            "Edu": edu,
            "Shift": shift,
            "Loc": loc,
            "Act": act,
            "Haz": haz,
            "HazardMode": hazard_mode,
            "InjuryForm": injury_form,
            "HazardSource": hazard_source,
            "Body": body,
            "Cause": cause,
            "Sev": sev,
            "is_synthetic": is_synthetic_int,
            "is_synthetic_norm": norm_series,
            "severe_binary": severe_binary,
            "severity_score": severity_score,
            "_haz_mapped_type": haz_mapped_type,
        }
    )

    if unmapped_flag.any():
        bad = out.loc[unmapped_flag, "Sev"].value_counts(dropna=False).reset_index()
        bad.columns = ["Sev_label", "count"]
        bad.to_csv(audit_dir / "unmapped_severity_labels.csv", index=False, encoding="utf-8-sig")
        out.loc[unmapped_flag].to_excel(audit_dir / "unmapped_severity_rows.xlsx", index=False, engine="openpyxl")

    n_real_all = int((norm_series == "real").sum())
    n_synth_all = int((norm_series == "synthetic").sum())
    n_unknown_all = int((norm_series == "unknown").sum())

    print(f"INFO: Raw total rows (selected sheet): {raw_nrows}", flush=True)
    print(f"INFO: Classified real={n_real_all}, synthetic={n_synth_all}, unknown={n_unknown_all}", flush=True)

    warn_msgs: list[str] = []
    if config.EXPECTED_REAL_COUNT is not None and n_real_all != int(config.EXPECTED_REAL_COUNT):
        warn_msgs.append(
            f"真实样本数 {n_real_all} 与 config.EXPECTED_REAL_COUNT={config.EXPECTED_REAL_COUNT} 不一致，请检查 sheet 或 is_synthetic 列。"
        )
    if config.EXPECTED_TOTAL_COUNT is not None and raw_nrows != int(config.EXPECTED_TOTAL_COUNT):
        warn_msgs.append(
            f"原始总行数 {raw_nrows} 与 config.EXPECTED_TOTAL_COUNT={config.EXPECTED_TOTAL_COUNT} 不一致。"
        )

    if config.USE_REAL_ONLY:
        before = len(out)
        mask = out["is_synthetic_norm"] == "real"
        out = out.loc[mask].copy()
        dropped = before - len(out)
        print(
            f"INFO: USE_REAL_ONLY=True, kept is_synthetic_norm=='real' only, dropped {dropped} rows, "
            f"remaining rows: {len(out)}.",
            flush=True,
        )
    else:
        print(f"INFO: USE_REAL_ONLY=False, using all samples, rows: {len(out)}.", flush=True)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Haz 审计：不确定取值列表 + 全量映射汇总表
    unc_mask = out["_haz_mapped_type"] == "uncertain"
    if unc_mask.any():
        uvc = out.loc[unc_mask, "Haz"].astype(str).value_counts(dropna=False).reset_index()
        uvc.columns = ["original_hazard", "count"]
        uvc.to_csv(audit_dir / "hazard_uncertain_values.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["original_hazard", "count"]).to_csv(
            audit_dir / "hazard_uncertain_values.csv", index=False, encoding="utf-8-sig"
        )

    audit_for_xlsx = out.assign(mapped_type=out["_haz_mapped_type"]).drop(columns=["_haz_mapped_type"])
    haz_audit_df = _build_hazard_value_audit_xlsx(audit_for_xlsx)
    haz_audit_path = config.OUTPUT_DIR / "hazard_value_audit.xlsx"
    haz_audit_df.to_excel(haz_audit_path, index=False, engine="openpyxl")

    out = out.drop(columns=["_haz_mapped_type"], errors="ignore")
    cleaned_path = config.OUTPUT_DIR / "cleaned_data.xlsx"
    profile_path = config.OUTPUT_DIR / "data_profile.xlsx"

    out_export = out[OUTPUT_COLUMNS]
    out_export.to_excel(cleaned_path, index=False, engine="openpyxl")
    _write_data_profile(profile_path, out_export, raw_nrows, col_present, sheet_used)

    print(
        "INFO: Haz semantic remap — hazard_mode={}, injury_form={}, hazard_source={}, uncertain={}, empty={}.".format(
            int((audit_for_xlsx["mapped_type"] == "hazard_mode").sum()),
            int((audit_for_xlsx["mapped_type"] == "injury_form").sum()),
            int((audit_for_xlsx["mapped_type"] == "hazard_source").sum()),
            int((audit_for_xlsx["mapped_type"] == "uncertain").sum()),
            int((audit_for_xlsx["mapped_type"] == "empty").sum()),
        ),
        flush=True,
    )

    hm_s = out_export["HazardMode"].astype(str).str.strip()
    if_s = out_export["InjuryForm"].astype(str).str.strip()
    hs_s = out_export["HazardSource"].astype(str).str.strip()
    stage01_stats = {
        "n_rows_exported": int(len(out_export)),
        "n_haz_mapped_hazard_mode": int((audit_for_xlsx["mapped_type"] == "hazard_mode").sum()),
        "n_haz_mapped_injury_form": int((audit_for_xlsx["mapped_type"] == "injury_form").sum()),
        "n_haz_mapped_hazard_source": int((audit_for_xlsx["mapped_type"] == "hazard_source").sum()),
        "n_haz_uncertain": int((audit_for_xlsx["mapped_type"] == "uncertain").sum()),
        "n_hazard_mode_nonempty": int((hm_s.str.lower() != "unknown").sum()),
        "n_injury_form_nonempty": int((if_s.str.lower() != "unknown").sum()),
        "n_hazard_source_nonempty": int(
            ((hs_s.str.len() > 0) & (hs_s.str.lower() != "unknown") & (hs_s.str.lower() != "notavailable")).sum()
        ),
        "n_hazard_mode_unique": int(out_export.loc[hm_s.str.lower() != "unknown", "HazardMode"].astype(str).nunique(dropna=False)),
        "n_injury_form_unique": int(out_export.loc[if_s.str.lower() != "unknown", "InjuryForm"].astype(str).nunique(dropna=False)),
        "n_hazard_source_unique": int(
            out_export.loc[
                (hs_s.str.len() > 0) & (hs_s.str.lower() != "unknown") & (hs_s.str.lower() != "notavailable"),
                "HazardSource",
            ]
            .astype(str)
            .nunique(dropna=False)
        ),
        "n_injury_form_unique_all": int(out_export["InjuryForm"].astype(str).nunique(dropna=False)),
    }
    with open(config.OUTPUT_DIR / "pipeline_stage01_stats.json", "w", encoding="utf-8") as jf:
        json.dump(stage01_stats, jf, ensure_ascii=False, indent=2)

    sev_vc = out_export["Sev"].value_counts(dropna=False)
    sb_vc = out_export["severe_binary"].value_counts(dropna=False)
    _emit_warnings_and_log(
        sheet_used=sheet_used,
        all_sheet_names=all_sheet_names,
        n_total_raw=raw_nrows,
        n_real=n_real_all,
        n_synth=n_synth_all,
        n_unknown=n_unknown_all,
        sev_dist=sev_vc,
        sb_dist=sb_vc,
        warnings=warn_msgs,
    )

    print(f"OK: Wrote {cleaned_path.resolve()}", flush=True)
    print(f"OK: Wrote {profile_path.resolve()}", flush=True)
    print(f"OK: Wrote {haz_audit_path.resolve()}", flush=True)
    print(f"OK: Data audit under {audit_dir.resolve()}", flush=True)


if __name__ == "__main__":
    run()
