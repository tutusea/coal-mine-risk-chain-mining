# -*- coding: utf-8 -*-
"""
项目级配置：路径、字段名、样本筛选、链条频次阈值、严重伤害定义。
各阶段脚本从项目根目录导入：`import config`（自 `run_all.py` 启动时 cwd 在根目录）。
"""

from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# 路径
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

# 数据审计目录（与 USE_REAL_ONLY 无关，固定路径）
DATA_AUDIT_DIR = PROJECT_ROOT / "outputs_data_audit"

# -----------------------------------------------------------------------------#
# 运行模式：输出目录随 USE_REAL_ONLY 切换，避免互相覆盖
# USE_REAL_ONLY = True  -> outputs_real/、figures_real/
# USE_REAL_ONLY = False -> outputs_all/、figures_all/
# -----------------------------------------------------------------------------#
USE_REAL_ONLY = True
USE_REAL_SAMPLES_ONLY = USE_REAL_ONLY

OUTPUT_DIR = PROJECT_ROOT / ("outputs_real" if USE_REAL_ONLY else "outputs_all")
FIGURES_DIR = PROJECT_ROOT / ("figures_real" if USE_REAL_ONLY else "figures_all")

# 敏感性分析独立输出目录（不覆盖主分析 outputs_real / figures_real）
OUTPUT_SENSITIVITY_DIR = PROJECT_ROOT / "outputs_sensitivity"
FIGURES_SENSITIVITY_DIR = PROJECT_ROOT / "figures_sensitivity"

# 默认 Excel（可将数据文件放在 data/ 下）
DEFAULT_EXCEL_PATH = DATA_DIR / "合成事故统计_含真实数据.xlsx"

# 实际读取路径：默认同目录下「1合成事故统计_含真实数据.xlsx」，若不存在则回退无「1-」前缀文件名
_EXCEL_PRIMARY = PROJECT_ROOT / "1合成事故统计_含真实数据.xlsx"
_EXCEL_FALLBACK = PROJECT_ROOT / "合成事故统计_含真实数据.xlsx"
EXCEL_PATH = _EXCEL_PRIMARY if _EXCEL_PRIMARY.is_file() else _EXCEL_FALLBACK

# 读取的工作表：None 表示使用 Excel 中第一个 sheet；否则填确切 sheet 名
SHEET_NAME: str | None = None

# 样本量预期（用于告警，不中断运行）：EXPECTED_TOTAL_COUNT=None 时不检查总行数
EXPECTED_REAL_COUNT = 234
EXPECTED_TOTAL_COUNT: int | None = None

# -----------------------------------------------------------------------------
# 字段：表头别名 -> 统一字段键（canonical keys）
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

# 风险链字段顺序（上游语境 -> 结局）
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

# 兼容旧逻辑：真实标记集合（实际判定以 src/01_clean_data.normalize_is_synthetic 为准）
REAL_MARKERS = {0, "0", False, "否", "no", "No", "NO", "real", "真实"}

# -----------------------------------------------------------------------------
# 链条挖掘：最小链条频次（完整链字符串或聚合键出现次数下限，占位）
# MIN_CHAIN_COUNT：阶段 04 链条汇总中保留链条的最小出现次数（count 下限）
# -----------------------------------------------------------------------------
MIN_CHAIN_FREQUENCY = 1
MIN_CHAIN_COUNT = 3

# -----------------------------------------------------------------------------
# 关联规则挖掘（阶段 05）：支持度、置信度、提升度阈值（写入此处置便于调参）
# AR_MIN_SUPPORT：频繁项集最小支持度（传给 mlxtend fpgrowth/apriori）
# AR_MIN_CONFIDENCE：association_rules 的 metric 阈值（置信度下限）
# AR_MIN_LIFT：规则生成后再筛，lift 下限
# AR_USE_FP_GROWTH：True 用 fpgrowth，False 用 apriori
# -----------------------------------------------------------------------------
AR_MIN_SUPPORT = 0.01
AR_MIN_CONFIDENCE = 0.30
AR_MIN_LIFT = 1.20
AR_USE_FP_GROWTH = True

# 关联规则最小样本计数（由 support × 样本量 四舍五入得到 rule_count / antecedent_count 后过滤）
AR_MIN_RULE_COUNT = 5
AR_MIN_ANTECEDENT_COUNT = 5
# 重伤结局样本较少时，仅对「预防型重伤」规则放宽 rule_count 下限
AR_SERIOUS_MIN_RULE_COUNT = 3

# -----------------------------------------------------------------------------
# 严重伤害定义
# SEVERITY_MODE: "minor_plus" 轻伤及以上；"heavy" 重伤及以上
# -----------------------------------------------------------------------------
SEVERITY_MODE = "minor_plus"

SEVERE_LABELS_MINOR_PLUS = [
    "轻伤",
    "重伤",
    "死亡",
    "工亡",
    "致残",
    "重大伤害",
    "严重伤害",
]

SEVERE_LABELS_HEAVY = [
    "重伤",
    "死亡",
    "工亡",
]


def severe_labels_for_mode(mode: str | None = None) -> list[str]:
    """返回当前模式下视为「严重」的伤害程度文本列表。"""
    m = mode or SEVERITY_MODE
    if m == "heavy":
        return list(SEVERE_LABELS_HEAVY)
    return list(SEVERE_LABELS_MINOR_PLUS)
