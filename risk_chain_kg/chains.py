# -*- coding: utf-8 -*-
"""Turn each accident row into an ordered risk chain (typed node ids)."""

from __future__ import annotations

from typing import List

import pandas as pd


def node_id(field_key: str, value: str) -> str:
    """Stable node label: FIELD_KEY|value (avoids collisions across fields)."""
    v = (value or "").strip()
    if not v:
        v = "_EMPTY_"
    return f"{field_key}|{v}"


def row_chain(
    row: pd.Series,
    field_keys: List[str],
    resolved: dict,
) -> List[str]:
    """Ordered list of node ids for one record."""
    chain = []
    for fk in field_keys:
        col = resolved.get(fk)
        if not col:
            continue
        val = row.get(col, "")
        chain.append(node_id(fk, str(val)))
    return chain


def build_chains_dataframe(df: pd.DataFrame, resolved: dict, field_keys: List[str]) -> pd.DataFrame:
    """One row per accident: string chain plus list of typed node ids."""
    chains = []
    for idx, row in df.iterrows():
        ch = row_chain(row, field_keys, resolved)
        chains.append({"row_index": idx, "chain": " -> ".join(ch), "chain_list": ch})
    out = pd.DataFrame(
        {
            "row_index": [c["row_index"] for c in chains],
            "chain": [c["chain"] for c in chains],
            "chain_list": [c["chain_list"] for c in chains],
        }
    )
    out["is_severe"] = df["__severe__"].values
    return out
