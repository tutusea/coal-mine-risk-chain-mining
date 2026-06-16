# -*- coding: utf-8 -*-
"""Load Excel and resolve column names from aliases."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import config


def _norm_header(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    return s


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Map canonical keys in COLUMN_ALIASES to actual column names in df."""
    inv: Dict[str, str] = {}
    normalized = {_norm_header(c): c for c in df.columns}
    for key, aliases in config.COLUMN_ALIASES.items():
        found: Optional[str] = None
        for a in aliases:
            na = _norm_header(a)
            if na in normalized:
                found = normalized[na]
                break
        if found is not None:
            inv[key] = found
    return inv


def read_excel(path) -> pd.DataFrame:
    p = str(path)
    return pd.read_excel(p, engine="openpyxl")


def pick_chain_columns(resolved: Dict[str, str]) -> Tuple[List[str], List[str]]:
    cols: List[str] = []
    missing: List[str] = []
    for k in config.CHAIN_FIELD_KEYS:
        if k in resolved:
            cols.append(resolved[k])
        else:
            missing.append(k)
    return cols, missing
