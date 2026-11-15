"""多因子降维与合成。

先看相关性再谈加权：两个 |ρ| > 0.7 的因子等权相加，等于给同一个信号双倍权重。
合成方法按复杂度递增，每加一层都要能在样本外证明有增量，否则退回上一层。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import _spearman


def correlation_matrix(
    scores: pd.DataFrame,
    min_names: int = 20,
    max_pairs: int = 2000,
) -> pd.DataFrame:
    """逐日期截面秩相关后取均值，避免混入日历效应。

    ``scores`` 需为 ``(dt, code)`` 双层索引，列为因子 id。
    """
    columns = list(scores.columns)
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    if len(columns) < 2:
        return matrix
    dates = scores.index.get_level_values(0).unique()
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            values: list[float] = []
            for dt in dates:
                block = scores.loc[dt, [left, right]].dropna()
                if len(block) < int(min_names):
                    continue
                value = _spearman(block[left], block[right])
                if np.isfinite(value):
                    values.append(value)
                if len(values) >= max_pairs:
                    break
            correlation = float(np.mean(values)) if values else float("nan")
            matrix.loc[left, right] = correlation
            matrix.loc[right, left] = correlation
    return matrix


def select_decorrelated(
    correlation: pd.DataFrame,
    max_abs_corr: float = 0.7,
    priority: list[str] | None = None,
) -> tuple[list[str], list[dict[str, object]]]:
    """按优先级贪心保留低相关因子，返回 (保留列表, 淘汰原因)。"""
    order = [c for c in (priority or list(correlation.columns)) if c in correlation.columns]
    kept: list[str] = []
    dropped: list[dict[str, object]] = []
    for column in order:
        conflict = None
        for existing in kept:
            value = correlation.loc[column, existing]
            if np.isfinite(value) and abs(value) > max_abs_corr:
                conflict = (existing, float(value))
                break
        if conflict is None:
            kept.append(column)
        else:
            dropped.append({
                "factor_id": column,
                "reason": f"与 {conflict[0]} 的相关系数 {conflict[1]:.2f} 超过阈值 {max_abs_corr:g}",
                "correlation": conflict[1],
                "against": conflict[0],
            })
    return kept, dropped


def combine_weights(
    metrics_map: dict[str, dict[str, object]],
    method: str = "icir",
    cap: float = 3.0,
) -> tuple[dict[str, float], list[str]]:
    """等权 / |IC| 加权 / |ICIR| 加权。返回 (权重, 警告)。"""
    warnings: list[str] = []
    ids = [f for f in metrics_map if f]
    if not ids:
        return {}, ["没有可用于合成的因子"]

    if method == "equal":
        weight = 1.0 / len(ids)
        return {f: weight for f in ids}, warnings

    raw: dict[str, float] = {}
    for factor_id in ids:
        ic = metrics_map[factor_id].get("ic") or {}
        if method == "ic":
            value = abs(float(ic.get("mean_adj") or 0.0))
        elif method == "icir":
            value = min(abs(float(ic.get("icir_adj") or 0.0)), cap)
        else:
            raise ValueError(f"不支持的合成方法: {method}")
        raw[factor_id] = value

    total = float(sum(raw.values()))
    if not np.isfinite(total) or total <= 0:
        warnings.append("权重全为 0，已退化为等权")
        weight = 1.0 / len(ids)
        return {f: weight for f in ids}, warnings
    return {f: v / total for f, v in raw.items()}, warnings


def combine_scores(scores: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """加权求和后按日期再标准化，保证截面可比。"""
    columns = [c for c in weights if c in scores.columns]
    if not columns:
        raise ValueError("权重与因子矩阵没有交集")
    weight_vector = pd.Series({c: float(weights[c]) for c in columns})
    combined = scores[columns].mul(weight_vector, axis=1).sum(axis=1, min_count=1)
    combined = combined.rename("combined")
    grouped = combined.groupby(level=0)
    standardized = grouped.transform(lambda s: _zscore(s))
    return standardized.rename("combined")


def _zscore(series: pd.Series) -> pd.Series:
    sd = float(series.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return series * 0.0
    return (series - float(series.mean())) / sd


def build_combination(
    scores: pd.DataFrame,
    metrics_map: dict[str, dict[str, object]],
    method: str = "icir",
    max_abs_corr: float = 0.7,
) -> dict[str, object]:
    """完整合成流程：相关性 → 去冗余 → 加权 → 合成得分。"""
    correlation = correlation_matrix(scores)
    priority = sorted(
        metrics_map,
        key=lambda f: abs(float((metrics_map[f].get("ic") or {}).get("icir_adj") or 0.0)),
        reverse=True,
    )
    kept, dropped = select_decorrelated(correlation, max_abs_corr=max_abs_corr, priority=priority)
    weights, warnings = combine_weights({f: metrics_map[f] for f in kept}, method=method)
    combined = combine_scores(scores[kept], weights) if kept else pd.Series(dtype=float, name="combined")
    return {
        "method": method,
        "max_abs_corr": max_abs_corr,
        "correlation": {
            row: {col: (None if not np.isfinite(correlation.loc[row, col]) else float(correlation.loc[row, col]))
                  for col in correlation.columns}
            for row in correlation.index
        },
        "kept": kept,
        "dropped": dropped,
        "weights": weights,
        "warnings": warnings,
        "scores": combined,
    }
