# -*- coding: utf-8 -*-
"""Export NetworkX graph to Gephi-compatible GEXF."""

from __future__ import annotations

import math

import networkx as nx

from . import config


def export_gexf(G: nx.DiGraph, path) -> None:
    """Write GEXF for Gephi; replace NaN attributes with 0.0 for compatibility."""
    p = str(path)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    H = G.copy()
    for _, data in H.nodes(data=True):
        for k, v in list(data.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                data[k] = 0.0
    for _, _, data in H.edges(data=True):
        for k, v in list(data.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                data[k] = 0.0
    nx.write_gexf(H, p)
