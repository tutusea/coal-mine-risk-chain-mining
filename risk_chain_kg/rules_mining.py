# -*- coding: utf-8 -*-
"""
Association rules toward severe outcome.
Primary: mlxtend Apriori (install with: pip install mlxtend).
Fallback: subset co-occurrence miner (no extra dependency) if mlxtend is missing.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from . import config

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    _HAS_MLXTEND = True
except ImportError:
    _HAS_MLXTEND = False


def chains_to_transactions(chain_lists: List[list], is_severe: pd.Series) -> List[List[str]]:
    txs = []
    for ch, sev in zip(chain_lists, is_severe.values):
        items = list(dict.fromkeys(ch))
        if sev:
            items = items + [config.SEVERE_ITEM_TAG]
        txs.append(items)
    return txs


def _mine_rules_fallback(chain_df: pd.DataFrame, max_len: int) -> pd.DataFrame:
    """
    Rules antecedent -> severe using contiguous subsequences of each chain (ordered),
    so complexity stays linear in chain length. Same support/confidence/lift as
    standard rules toward a binary outcome.
    """
    n = len(chain_df)
    if n == 0:
        return pd.DataFrame()
    p_sev = float(chain_df["is_severe"].sum()) / n
    if p_sev <= 0:
        return pd.DataFrame()

    from collections import Counter

    count_ant: Counter = Counter()
    count_ant_sev: Counter = Counter()

    for _, r in chain_df.iterrows():
        items: List[str] = list(r["chain_list"])
        sev = bool(r["is_severe"])
        L = len(items)
        for start in range(L):
            for ln in range(1, min(max_len, L - start) + 1):
                end = start + ln
                window = items[start:end]
                ant = frozenset(window)
                count_ant[ant] += 1
                if sev:
                    count_ant_sev[ant] += 1

    tag = config.SEVERE_ITEM_TAG
    rows = []
    for ant, c_all in count_ant.items():
        sup = c_all / n
        if sup < config.RULE_MIN_SUPPORT:
            continue
        c_sev = count_ant_sev.get(ant, 0)
        conf = c_sev / c_all if c_all else 0.0
        if conf < config.RULE_MIN_CONFIDENCE:
            continue
        lift = conf / p_sev if p_sev > 0 else 0.0
        if lift < config.RULE_MIN_LIFT:
            continue
        rows.append(
            {
                "antecedents": ant,
                "consequents": frozenset([tag]),
                "support": c_sev / n,
                "confidence": conf,
                "lift": lift,
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values(["lift", "confidence"], ascending=False)
    out = out.head(config.TOP_K_RULES * 3)
    return out


def mine_severe_rules(chain_df: pd.DataFrame, max_len: int = 5) -> pd.DataFrame:
    if not _HAS_MLXTEND:
        return _mine_rules_fallback(chain_df, max_len=max_len)

    txs = chains_to_transactions(chain_df["chain_list"].tolist(), chain_df["is_severe"])
    te = TransactionEncoder()
    te_ary = te.fit(txs).transform(txs)
    df1 = pd.DataFrame(te_ary, columns=te.columns_)

    if config.SEVERE_ITEM_TAG not in df1.columns:
        return _mine_rules_fallback(chain_df, max_len=max_len)

    try:
        freq = apriori(
            df1,
            min_support=config.RULE_MIN_SUPPORT,
            use_colnames=True,
            max_len=max_len,
            verbose=0,
        )
    except Exception:
        return _mine_rules_fallback(chain_df, max_len=max_len)

    if freq.empty:
        return _mine_rules_fallback(chain_df, max_len=max_len)

    try:
        rules = association_rules(
            freq,
            metric="confidence",
            min_threshold=config.RULE_MIN_CONFIDENCE,
        )
    except Exception:
        return _mine_rules_fallback(chain_df, max_len=max_len)

    if rules.empty:
        return _mine_rules_fallback(chain_df, max_len=max_len)

    tag = config.SEVERE_ITEM_TAG
    mask_cons = rules["consequents"].apply(lambda x: tag in set(x))
    rules = rules.loc[mask_cons].copy()
    rules = rules[~rules["antecedents"].apply(lambda x: tag in set(x))]
    rules = rules[rules["lift"] >= config.RULE_MIN_LIFT]
    rules = rules.sort_values(["lift", "confidence"], ascending=False)
    rules = rules.head(config.TOP_K_RULES * 3)
    return rules
