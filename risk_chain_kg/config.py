# -*- coding: utf-8 -*-
"""
Project configuration: paths, column names, severity labels, chain order.
Edit CHAIN_FIELD_KEYS and COLUMN_ALIASES to match your Excel headers.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths (outputs under project root by default)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_EXCEL = PROJECT_ROOT / "合成事故统计_含真实数据.xlsx"

# -----------------------------------------------------------------------------
# Column detection: canonical name -> possible Excel header variants
# Edit values if your spreadsheet uses different spelling.
# -----------------------------------------------------------------------------
COLUMN_ALIASES = {
    "team": ["队别", "队伍", "区队"],
    "job": ["工种", "岗位"],
    "education": ["学历"],
    "shift": ["班次", "班制"],
    "work_place": ["工作地点", "作业地点"],
    "activity": ["作业活动", "作业工序", "活动"],
    "hazard_agent": ["致害物", "致病物", "致害因素"],
    "injury_part": ["受伤部位", "伤害部位"],
    "harm_degree": ["伤害程度", "损伤程度"],
    "accident_place": ["事故地点"],
    "accident_cause": ["事故原因", "原因"],
    "is_synthetic": ["is_synthetic", "synthetic", "是否合成", "合成"],
}

# Ordered risk chain: upstream context to outcome (paper-style path).
# Default matches: team -> job -> shift -> work_place -> activity ->
# hazard_agent -> injury_part -> harm_degree
# To include education (e.g. after job), insert "education" in this list.
CHAIN_FIELD_KEYS = [
    "team",
    "job",
    "shift",
    "work_place",
    "activity",
    "hazard_agent",
    "injury_part",
    "harm_degree",
]

# Optional: insert "education" after "job", or append place/cause:
# CHAIN_FIELD_KEYS = ["team", "job", "education", "shift", ...]
# CHAIN_FIELD_KEYS += ["accident_place", "accident_cause"]

# Short Chinese labels for Sankey / figures (edit freely).
FIELD_DISPLAY_NAME = {
    "team": "队别",
    "job": "工种",
    "education": "学历",
    "shift": "班次",
    "work_place": "地点",
    "activity": "活动",
    "hazard_agent": "致害物",
    "injury_part": "部位",
    "harm_degree": "程度",
    "accident_place": "事故地",
    "accident_cause": "原因",
}

# -----------------------------------------------------------------------------
# Severity: define which harm_degree text counts as "severe" for metrics/rules
# "minor_plus": minor injury and above; "heavy": heavy injury and death
# -----------------------------------------------------------------------------
SEVERITY_MODE = "minor_plus"  # or "heavy"

# Labels treated as severe when SEVERITY_MODE == "minor_plus"
SEVERE_LABELS_MINOR_PLUS = [
    "轻伤",
    "重伤",
    "死亡",
    "工亡",
    "致残",
    "重大伤害",
]

# Labels treated as severe when SEVERITY_MODE == "heavy"
SEVERE_LABELS_HEAVY = [
    "重伤",
    "死亡",
    "工亡",
]

# Synthetic filter: real rows when this column equals one of REAL_MARKERS
REAL_MARKERS = {0, "0", False, "否", "no", "No", "NO", "real", "真实"}

# Association rules
RULE_MIN_SUPPORT = 0.02
RULE_MIN_CONFIDENCE = 0.3
RULE_MIN_LIFT = 1.0
SEVERE_ITEM_TAG = "__OUTCOME_SEVERE__"  # synthetic item appended for severe rows

# Graph / metrics
PAGERANK_ALPHA = 0.85
RISK_SCORE_NODE = "freq_times_severity"  # frequency * severity_rate

# Top-K tables and figures
TOP_K_NODES = 25
TOP_K_EDGES = 25
TOP_K_CHAINS = 30
TOP_K_RULES = 30

# Plot style
FIG_DPI = 150
RANDOM_LAYOUT_SEED = 42

# Skip expensive betweenness if graph is huge (0 = never skip)
BETWEENNESS_MAX_NODES = 800
