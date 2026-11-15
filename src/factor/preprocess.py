"""截面预处理链：去极值 → 标准化 → 中性化 → 再标准化。

顺序是硬约束，不是风格问题：

- 先去极值：原始因子常带市值/换手造成的厚尾，不压极值就标准化，标准差被极值抬高，
  大部分样本被压到 0 附近，分层退化成「极值组 vs 其余」。
- 先标准化再中性化：中性化是回归取残差，量纲差异过大时数值条件变差。
- 中性化后再标准化：残差尺度与原始因子不同，评分和加权前必须拉回同一量纲。

所有函数都保持输入索引，缺失值一律保持缺失，不用 0 或均值顶替。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class PreprocessConfig:
    """一次研究的预处理口径，写进报告便于复现。"""

    winsor_n: float = 3.0
    method: str = "zscore"
    neutralize: bool = True
    min_valid: int = 5

    def describe(self) -> dict[str, object]:
        return {
            "去极值": f"MAD × {self.winsor_n:g}",
            "标准化": "z-score" if self.method == "zscore" else "rank",
            "中性化": "行业 + 市值" if self.neutralize else "关闭",
            "最少有效样本": self.min_valid,
        }


def winsorize_mad(values: pd.Series, n: float = 3.0) -> pd.Series:
    """用中位数 ± n × 1.4826 × MAD 截断，抗极值优于 3σ。"""
    s = pd.to_numeric(values, errors="coerce").astype(float)
    valid = s.dropna()
    if valid.empty:
        return s
    median = float(valid.median())
    mad = float((valid - median).abs().median())
    if not np.isfinite(mad) or mad <= 0:
        # 超过一半样本同值时 MAD 为 0，此时不做截断，交由标准化处理。
        return s
    half_width = float(n) * MAD_TO_SIGMA * mad
    return s.clip(lower=median - half_width, upper=median + half_width)


def standardize(values: pd.Series, method: str = "zscore") -> pd.Series:
    """z-score 或 rank 标准化；零方差截面返回全 0，不抛异常。"""
    s = pd.to_numeric(values, errors="coerce").astype(float)
    if method == "zscore":
        sd = float(s.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            return s.where(s.isna(), 0.0)
        return (s - float(s.mean())) / sd
    if method == "rank":
        rank = s.rank(method="average")
        count = int(rank.notna().sum())
        if count <= 1:
            return s.where(s.isna(), 0.0)
        return (rank - (count + 1) / 2) / ((count - 1) / 2)
    raise ValueError(f"不支持的标准化方法: {method}")


def neutralize(
    values: pd.Series,
    industry: pd.Series | None = None,
    size: pd.Series | None = None,
) -> pd.Series:
    """对行业哑变量与对数市值回归取残差。

    ``industry`` / ``size`` 必须是与 ``values`` 同索引的 Series；
    由调用方负责对齐，避免在这里隐式猜索引层级。
    """
    y = pd.to_numeric(values, errors="coerce").astype(float)
    index = y.index
    columns: list[np.ndarray] = [np.ones(len(index), dtype=float)]
    labels: list[str] = ["const"]

    if size is not None:
        raw_size = pd.to_numeric(pd.Series(size).reindex(index), errors="coerce").astype(float)
        log_size = np.log(raw_size.where(raw_size > 0))
        columns.append(log_size.to_numpy(dtype=float))
        labels.append("log_size")

    if industry is not None:
        codes = pd.Series(industry).reindex(index)
        dummies = pd.get_dummies(codes.astype("object"), prefix="ind", dtype=float)
        if len(dummies.columns) > 1:
            # 丢弃一列，避免与常数项完全共线。
            dummies = dummies.iloc[:, 1:]
        columns.extend(dummies[c].to_numpy(dtype=float) for c in dummies.columns)
        labels.extend(dummies.columns)

    design = np.column_stack(columns)
    finite_rows = np.isfinite(y.to_numpy(dtype=float)) & np.isfinite(design).all(axis=1)
    if finite_rows.sum() <= design.shape[1]:
        # 有效样本不足以支撑回归，退化为去均值，避免拟合出无意义残差。
        return y - float(y.mean())

    beta, *_ = np.linalg.lstsq(design[finite_rows], y.to_numpy(dtype=float)[finite_rows], rcond=None)
    residual = np.full(len(index), np.nan, dtype=float)
    residual[finite_rows] = y.to_numpy(dtype=float)[finite_rows] - design[finite_rows] @ beta
    return pd.Series(residual, index=index, name=y.name)


def preprocess_cross_section(
    values: pd.Series,
    industry: pd.Series | None = None,
    size: pd.Series | None = None,
    config: PreprocessConfig | None = None,
) -> pd.Series:
    """完整预处理链，返回可直接比较的截面得分。"""
    cfg = config or PreprocessConfig()
    if int(values.notna().sum()) < cfg.min_valid:
        return pd.Series(np.nan, index=values.index, name=values.name)

    cleaned = winsorize_mad(values, n=cfg.winsor_n)
    scaled = standardize(cleaned, method=cfg.method)
    if cfg.neutralize and (industry is not None or size is not None):
        scaled = neutralize(scaled, industry=industry, size=size)
        scaled = standardize(scaled, method=cfg.method)
    return scaled.rename(values.name)
