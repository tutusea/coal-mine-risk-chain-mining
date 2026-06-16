# -*- coding: utf-8 -*-
"""阶段 05：以每条事故为 transaction，用 mlxtend 挖掘频繁项集与关联规则。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

CLEANED_PATH = config.OUTPUT_DIR / "cleaned_data.xlsx"

# 稳健规则（较严重伤害预防/后果）：小样本下用于正文展示；重伤规则不做稳健过滤
ROBUST_MIN_ANTECEDENT_COUNT = 8
ROBUST_MIN_RULE_COUNT = 8
ROBUST_MIN_LIFT = 1.20
ROBUST_MIN_CONFIDENCE = 0.60

SEVERITY_ANTECEDENT_PREFIXES: tuple[str, ...] = ("Sev:", "SevereBinary:")

PREFIX_SEVERE_BINARY = "SevereBinary"
ITEM_SEVERE_BINARY_POS = f"{PREFIX_SEVERE_BINARY}:较严重伤害"
ITEM_SEVERE_BINARY_NEG = f"{PREFIX_SEVERE_BINARY}:非严重伤害"
ITEM_SEV_SERIOUS = "Sev:重伤"

# 预防型前项允许的前缀（不含 Cause、Body、InjuryForm）
_PREVENTION_ANT_PREFIXES: tuple[str, ...] = (
    "Team:",
    "Job:",
    "Shift:",
    "Loc:",
    "Act:",
    "HazardMode:",
    "HazardSource:",
)
# 前项禁止（预防型）
_ANT_FORBIDDEN_PREVENTION: tuple[str, ...] = (
    "Body:",
    "InjuryForm:",
    "Sev:",
    "SevereBinary:",
    "Haz:",
    "Cause:",
)
# 后果型：禁止结局项、组织语境项、致害源项等
_ANT_FORBIDDEN_CONSEQUENCE: tuple[str, ...] = (
    "Sev:",
    "SevereBinary:",
    "Haz:",
    "Team:",
    "Job:",
    "Shift:",
    "HazardSource:",
    "Cause:",
)

# 后果型严重伤害：前项仅允许
_CONSEQ_SEVERE_ANT_PREFIXES: tuple[str, ...] = (
    "Body:",
    "InjuryForm:",
    "HazardMode:",
    "Loc:",
    "Act:",
)


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
    return False


def _str_cell(v) -> str:
    if _is_empty_like(v):
        return "Unknown"
    if isinstance(v, (float, np.floating)) and not np.isnan(v) and float(v).is_integer():
        return str(int(v))
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    return str(v).strip()


def _skip_unknown_dim(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    s = str(v).strip()
    if s == "":
        return True
    low = s.lower()
    return low in {"unknown", "notapplicable", "nan", "none", "-"}


def _skip_hazard_source_item(v) -> bool:
    if _is_empty_like(v):
        return True
    s = str(v).strip()
    low = s.lower()
    return low in {"unknown", "notapplicable", "nan", "none", "-", "notavailable"}


def _row_to_transaction(row: pd.Series) -> list[str]:
    """每条记录为一个 transaction；不含原 Haz 列；不含 Cause（预防型前项不允许）。"""
    items: list[str] = [
        f"Team:{_str_cell(row.get('Team'))}",
        f"Job:{_str_cell(row.get('Job'))}",
        f"Shift:{_str_cell(row.get('Shift'))}",
        f"Loc:{_str_cell(row.get('Loc'))}",
        f"Act:{_str_cell(row.get('Act'))}",
        f"Body:{_str_cell(row.get('Body'))}",
    ]
    if not _skip_unknown_dim(row.get("HazardMode")):
        items.append(f"HazardMode:{_str_cell(row.get('HazardMode'))}")
    if not _skip_hazard_source_item(row.get("HazardSource")):
        items.append(f"HazardSource:{_str_cell(row.get('HazardSource'))}")
    if not _skip_unknown_dim(row.get("InjuryForm")):
        items.append(f"InjuryForm:{_str_cell(row.get('InjuryForm'))}")
    items.append(f"Sev:{_str_cell(row.get('Sev'))}")
    sb = row.get("severe_binary")
    if pd.notna(sb):
        try:
            b = int(sb)
            if b == 1:
                items.append(ITEM_SEVERE_BINARY_POS)
            elif b == 0:
                items.append(ITEM_SEVERE_BINARY_NEG)
        except (TypeError, ValueError):
            pass
    return items


def _frozenset_join(fs: frozenset) -> str:
    return ";".join(sorted(fs, key=str))


def _item_is_severity_dimension(s: str) -> bool:
    t = s.strip()
    return any(t.startswith(p) for p in SEVERITY_ANTECEDENT_PREFIXES)


def _antecedent_contains_severity(ant: frozenset) -> bool:
    for it in ant:
        if _item_is_severity_dimension(str(it)):
            return True
    return False


def _antecedent_has_forbidden(ant: frozenset, forbidden: tuple[str, ...]) -> bool:
    for it in ant:
        s = str(it).strip()
        for p in forbidden:
            if s.startswith(p):
                return True
    return False


def _antecedent_only_prefixes(ant: frozenset, allowed: tuple[str, ...]) -> bool:
    for it in ant:
        s = str(it).strip()
        if not any(s.startswith(p) for p in allowed):
            return False
    return True


def _consequent_exactly(conseq: frozenset, item: str) -> bool:
    return conseq == frozenset({item})


def _post_filter_rules(rules: pd.DataFrame) -> pd.DataFrame:
    if rules.empty:
        return rules
    return rules[
        (rules["support"] >= config.AR_MIN_SUPPORT)
        & (rules["lift"] >= config.AR_MIN_LIFT)
        & (~rules["antecedents"].map(_antecedent_contains_severity))
    ].copy()


def _rules_to_output_df(rules: pd.DataFrame, n_transactions: int) -> pd.DataFrame:
    if rules.empty:
        return pd.DataFrame(
            columns=[
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
                "antecedent_count",
                "rule_count",
            ]
        )
    if "antecedent_count" in rules.columns and "rule_count" in rules.columns:
        ant_cnt = rules["antecedent_count"].astype(int)
        rule_cnt = rules["rule_count"].astype(int)
    else:
        ant_sup = pd.to_numeric(rules["antecedent support"], errors="coerce")
        sup = pd.to_numeric(rules["support"], errors="coerce")
        ant_cnt = (ant_sup * float(n_transactions)).round().astype(int)
        rule_cnt = (sup * float(n_transactions)).round().astype(int)
    out = pd.DataFrame(
        {
            "antecedents": rules["antecedents"].map(_frozenset_join),
            "consequents": rules["consequents"].map(_frozenset_join),
            "support": rules["support"].astype(float),
            "confidence": rules["confidence"].astype(float),
            "lift": rules["lift"].astype(float),
            "leverage": rules["leverage"].astype(float),
            "conviction": rules["conviction"].replace([np.inf, -np.inf], np.nan).astype(float),
            "antecedent_count": ant_cnt,
            "rule_count": rule_cnt,
        }
    )
    return out


def _apply_min_counts(df: pd.DataFrame, *, serious: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    rmin = int(getattr(config, "AR_SERIOUS_MIN_RULE_COUNT", 3)) if serious else int(
        getattr(config, "AR_MIN_RULE_COUNT", 5)
    )
    amin = int(getattr(config, "AR_MIN_ANTECEDENT_COUNT", 5))
    m = (df["rule_count"] >= rmin) & (df["antecedent_count"] >= amin)
    return df.loc[m].copy()


def _filter_robust_severe(df: pd.DataFrame) -> pd.DataFrame:
    """severe 预防/后果稳健子集（重伤规则不使用）。"""
    if df.empty:
        return df.copy()
    m = (
        (df["antecedent_count"] >= ROBUST_MIN_ANTECEDENT_COUNT)
        & (df["rule_count"] >= ROBUST_MIN_RULE_COUNT)
        & (df["lift"] >= ROBUST_MIN_LIFT)
        & (df["confidence"] >= ROBUST_MIN_CONFIDENCE)
    )
    return df.loc[m].copy()


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
        "HazardSource",
        "Body",
        "InjuryForm",
        "Sev",
        "severe_binary",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"cleaned_data 缺少列: {missing}")

    n_tx = len(df)
    transactions = [_row_to_transaction(df.iloc[i]) for i in range(n_tx)]
    transactions = [t for t in transactions if len(t) > 0]
    print(f"INFO: Association rules transactions={len(transactions)} from {CLEANED_PATH.name}")

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    ohe = pd.DataFrame(te_ary, columns=te.columns_)

    min_sup = float(config.AR_MIN_SUPPORT)
    min_conf = float(config.AR_MIN_CONFIDENCE)

    if config.AR_USE_FP_GROWTH:
        freq = fpgrowth(ohe, min_support=min_sup, use_colnames=True)
        print("INFO: Frequent itemsets: fpgrowth")
    else:
        freq = apriori(ohe, min_support=min_sup, use_colnames=True)
        print("INFO: Frequent itemsets: apriori")

    empty_out = _rules_to_output_df(pd.DataFrame(), n_tx)

    def _write_all_empty() -> None:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name in (
            "association_rules_all.csv",
            "association_rules_severe_prevention.csv",
            "association_rules_severe_consequence.csv",
            "association_rules_serious_prevention.csv",
            "association_rules_severe.csv",
            "association_rules_serious.csv",
            "association_rules_severe_prevention_robust.csv",
            "association_rules_severe_consequence_robust.csv",
        ):
            empty_out.to_csv(config.OUTPUT_DIR / name, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(config.OUTPUT_DIR / "top_30_association_rules.xlsx", engine="openpyxl") as w:
            empty_out.to_excel(w, sheet_name="top_30", index=False)

    if freq.empty:
        print("WARNING: Frequent itemsets are empty. Lower AR_MIN_SUPPORT or check input data.")
        _write_all_empty()
        return

    rules_raw = association_rules(
        freq,
        metric="confidence",
        min_threshold=min_conf,
    )
    rules_f = _post_filter_rules(rules_raw)
    rules_f = rules_f.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # 仅保留后件为单一项的规则（论文可解释）
    rules_f = rules_f[rules_f["consequents"].map(len) == 1].copy()

    # 后件不得含 Haz:（本工程 transaction 不含 Haz，此条为防御性）
    def _consequent_ok(fs: frozenset) -> bool:
        return not any(str(x).strip().startswith("Haz:") for x in fs)

    rules_f = rules_f[rules_f["consequents"].map(_consequent_ok)].copy()

    rules_f["antecedent_count"] = (pd.to_numeric(rules_f["antecedent support"], errors="coerce") * float(n_tx)).round().astype(int)
    rules_f["rule_count"] = (pd.to_numeric(rules_f["support"], errors="coerce") * float(n_tx)).round().astype(int)

    # A 预防型严重伤害
    m_prev_sev = (
        rules_f["consequents"].map(lambda c: _consequent_exactly(c, ITEM_SEVERE_BINARY_POS))
        & rules_f["antecedents"].map(lambda a: _antecedent_only_prefixes(a, _PREVENTION_ANT_PREFIXES))
        & rules_f["antecedents"].map(lambda a: not _antecedent_has_forbidden(a, _ANT_FORBIDDEN_PREVENTION))
    )
    # B 后果型严重伤害
    m_cons_sev = (
        rules_f["consequents"].map(lambda c: _consequent_exactly(c, ITEM_SEVERE_BINARY_POS))
        & rules_f["antecedents"].map(lambda a: _antecedent_only_prefixes(a, _CONSEQ_SEVERE_ANT_PREFIXES))
        & rules_f["antecedents"].map(lambda a: not _antecedent_has_forbidden(a, _ANT_FORBIDDEN_CONSEQUENCE))
    )
    # C 预防型重伤
    m_prev_ser = (
        rules_f["consequents"].map(lambda c: _consequent_exactly(c, ITEM_SEV_SERIOUS))
        & rules_f["antecedents"].map(lambda a: _antecedent_only_prefixes(a, _PREVENTION_ANT_PREFIXES))
        & rules_f["antecedents"].map(lambda a: not _antecedent_has_forbidden(a, _ANT_FORBIDDEN_PREVENTION))
    )

    sub_prev_sev = _apply_min_counts(rules_f.loc[m_prev_sev].copy(), serious=False)
    sub_cons_sev = _apply_min_counts(rules_f.loc[m_cons_sev].copy(), serious=False)
    sub_prev_ser = _apply_min_counts(rules_f.loc[m_prev_ser].copy(), serious=True)

    out_prev_sev = _rules_to_output_df(sub_prev_sev, n_tx)
    out_cons_sev = _rules_to_output_df(sub_cons_sev, n_tx)
    out_prev_ser = _rules_to_output_df(sub_prev_ser, n_tx)

    # 兼容旧文件名：severe / serious 与 prevention 主文件一致
    out_all = _rules_to_output_df(rules_f, n_tx)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_all.to_csv(config.OUTPUT_DIR / "association_rules_all.csv", index=False, encoding="utf-8-sig")
    out_prev_sev.to_csv(config.OUTPUT_DIR / "association_rules_severe_prevention.csv", index=False, encoding="utf-8-sig")
    out_cons_sev.to_csv(config.OUTPUT_DIR / "association_rules_severe_consequence.csv", index=False, encoding="utf-8-sig")
    out_prev_ser.to_csv(config.OUTPUT_DIR / "association_rules_serious_prevention.csv", index=False, encoding="utf-8-sig")
    out_prev_sev.to_csv(config.OUTPUT_DIR / "association_rules_severe.csv", index=False, encoding="utf-8-sig")
    out_prev_ser.to_csv(config.OUTPUT_DIR / "association_rules_serious.csv", index=False, encoding="utf-8-sig")

    out_prev_sev_rob = _filter_robust_severe(out_prev_sev)
    out_cons_sev_rob = _filter_robust_severe(out_cons_sev)
    out_prev_sev_rob.to_csv(
        config.OUTPUT_DIR / "association_rules_severe_prevention_robust.csv", index=False, encoding="utf-8-sig"
    )
    out_cons_sev_rob.to_csv(
        config.OUTPUT_DIR / "association_rules_severe_consequence_robust.csv", index=False, encoding="utf-8-sig"
    )

    top_base = out_prev_sev.head(30).copy() if not out_prev_sev.empty else out_cons_sev.head(30).copy()
    top_base = top_base.replace([np.inf, -np.inf], np.nan)
    with pd.ExcelWriter(config.OUTPUT_DIR / "top_30_association_rules.xlsx", engine="openpyxl") as writer:
        top_base.to_excel(writer, sheet_name="top_30", index=False)

    print(
        "OK: Rule counts: "
        f"all={len(out_all)} severe_prevention={len(out_prev_sev)} severe_consequence={len(out_cons_sev)} "
        f"serious_prevention={len(out_prev_ser)} "
        f"severe_prevention_robust={len(out_prev_sev_rob)} severe_consequence_robust={len(out_cons_sev_rob)}"
    )


if __name__ == "__main__":
    run()
