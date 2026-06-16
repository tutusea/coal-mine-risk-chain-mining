# -*- coding: utf-8 -*-
"""Clean text fields and build binary severe flag."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import config


def _clean_cell(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", "null", "-", "--", ""):
        return ""
    return s


def clean_frame(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in text_cols:
        if c in out.columns:
            out[c] = out[c].map(_clean_cell)
    return out


def severe_labels_for_mode(mode: str):
    if mode == "heavy":
        return set(config.SEVERE_LABELS_HEAVY)
    return set(config.SEVERE_LABELS_MINOR_PLUS)


def mark_severe(df: pd.DataFrame, harm_col: str, mode: Optional[str] = None) -> pd.Series:
    mode = mode or config.SEVERITY_MODE
    labels = severe_labels_for_mode(mode)
    if harm_col not in df.columns:
        return pd.Series(False, index=df.index)

    def is_severe(v: str) -> bool:
        v = _clean_cell(v)
        if not v:
            return False
        return v in labels

    return df[harm_col].map(is_severe)


def filter_real_only(df: pd.DataFrame, syn_col: str | None) -> pd.DataFrame:
    if not syn_col or syn_col not in df.columns:
        return df
    markers = config.REAL_MARKERS

    def is_real(x):
        if pd.isna(x):
            return True
        if x in markers:
            return True
        try:
            return int(x) == 0
        except Exception:
            return str(x).strip() in {str(m) for m in markers}

    mask = df[syn_col].map(is_real)
    return df.loc[mask].copy()
