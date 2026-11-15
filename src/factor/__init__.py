"""A红小策 · 统一因子研究引擎。

把「假设 → 数据 → 加工 → 检验 → 证伪 → 合成 → 组合 → 结论」固化成可复用管线，
替代每做一个因子就写一份一次性脚本的做法。

最小用法::

    from src.factor import build_panel, synthetic_bars, StudyConfig, run_study

    bars = synthetic_bars()
    report = run_study(build_panel(bars), StudyConfig(factor_ids=("rev_5",)))
    print(report["summary"]["headline"])
"""

from __future__ import annotations

from .combine import (
    build_combination,
    combine_scores,
    combine_weights,
    correlation_matrix,
    select_decorrelated,
)
from .dataset import load_bars, synthetic_bars
from .evaluate import (
    EvalConfig,
    evaluate_factor,
    group_turnover,
    ic_decay,
    ic_series,
    ic_summary,
    layered_stats,
    quantile_returns,
)
from .gates import GateConfig, judge
from .panel import (
    PanelBuild,
    attach_labels,
    build_panel,
    compute_factor,
    forward_returns,
    industry_series,
    price_limit_pct,
    rebalance_dates,
    size_proxy,
    tradability,
)
from .pipeline import SCHEMA, StudyConfig, run_study
from .preprocess import (
    PreprocessConfig,
    neutralize,
    preprocess_cross_section,
    standardize,
    winsorize_mad,
)
from .spec import (
    BUILTIN_FACTORS,
    FactorRegistryError,
    FactorSpec,
    catalog,
    get_factor,
    list_factors,
    register,
)

__all__ = [
    "BUILTIN_FACTORS",
    "EvalConfig",
    "FactorRegistryError",
    "FactorSpec",
    "GateConfig",
    "PanelBuild",
    "PreprocessConfig",
    "SCHEMA",
    "StudyConfig",
    "attach_labels",
    "build_combination",
    "build_panel",
    "catalog",
    "combine_scores",
    "combine_weights",
    "compute_factor",
    "correlation_matrix",
    "evaluate_factor",
    "forward_returns",
    "get_factor",
    "group_turnover",
    "ic_decay",
    "ic_series",
    "ic_summary",
    "industry_series",
    "judge",
    "layered_stats",
    "list_factors",
    "load_bars",
    "neutralize",
    "preprocess_cross_section",
    "price_limit_pct",
    "quantile_returns",
    "rebalance_dates",
    "register",
    "run_study",
    "select_decorrelated",
    "size_proxy",
    "standardize",
    "synthetic_bars",
    "tradability",
    "winsorize_mad",
]
