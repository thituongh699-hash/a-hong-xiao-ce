"""单因子评估：IC / ICIR / 分层 / 衰减 / 换手。

全部统计按日期截面计算，不对「股票-日」做独立样本假设。默认可交易掩码已在前置
步骤生效，不可建仓或不可离场的样本不参与统计。

方向约定：``direction=1`` 表示因子值越大越好，``direction=-1`` 反之。
``*_adj`` 字段统一为「方向调整后」的指标，关卡判定只看这些字段。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .panel import attach_labels
from .spec import FactorSpec


@dataclass(frozen=True)
class EvalConfig:
    horizon: int = 10
    quantiles: int = 5
    entry_lag: int = 1
    min_names: int = 20
    min_names_per_group: int = 3
    round_trip_cost_bps: float = 20.0
    cost_multiple: float = 2.0


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """秩相关直接用 Pearson 算秩，不引入 scipy 依赖。"""
    if len(a) < 3:
        return float("nan")
    ra, rb = a.rank(), b.rank()
    if ra.nunique() < 2 or rb.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(ra.to_numpy(dtype=float), rb.to_numpy(dtype=float))[0, 1])


def _max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return float("nan")
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    return float(-drawdown.min())


def _safe_ratio(numerator: float, denominator: float) -> float:
    # 浮点误差会让「零离散度」变成 1e-17，直接用会算出天文数字的 ICIR。
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def ic_series(
    frame: pd.DataFrame,
    factor_col: str = "score",
    ret_col: str = "fwd_ret",
    min_names: int = 20,
) -> pd.Series:
    """逐日期截面秩相关序列。"""
    records: dict[pd.Timestamp, float] = {}
    for dt, sub in frame.groupby("dt", sort=True):
        values = sub[[factor_col, ret_col]].dropna()
        if len(values) < int(min_names):
            continue
        value = _spearman(values[factor_col], values[ret_col])
        if np.isfinite(value):
            records[pd.Timestamp(dt)] = value
    return pd.Series(records, dtype=float).sort_index()


def ic_summary(ic: pd.Series, direction: int = 1) -> dict[str, float]:
    clean = ic.dropna()
    count = int(clean.count())
    if count == 0:
        return {
            "mean": float("nan"), "std": float("nan"), "icir": float("nan"),
            "hit_rate": float("nan"), "t": float("nan"), "n": 0,
            "mean_adj": float("nan"), "icir_adj": float("nan"),
            "hit_rate_adj": float("nan"), "t_adj": float("nan"),
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if count > 1 else float("nan")
    icir = _safe_ratio(mean, std)
    t_value = _safe_ratio(mean, std / np.sqrt(count)) if np.isfinite(std) else float("nan")
    return {
        "mean": mean,
        "std": std,
        "icir": icir,
        "hit_rate": float((clean > 0).mean()),
        "t": t_value,
        "n": count,
        "mean_adj": mean * direction,
        "icir_adj": icir * direction,
        "hit_rate_adj": float((clean * direction > 0).mean()),
        "t_adj": t_value * direction,
    }


def quantile_returns(
    frame: pd.DataFrame,
    factor_col: str = "score",
    ret_col: str = "fwd_ret",
    quantiles: int = 5,
    min_names: int = 20,
    min_names_per_group: int = 3,
) -> pd.DataFrame:
    """按因子值分组，返回每期各组等权收益。

    用 ``rank(method="first")`` 分箱，重复值不会导致空组或组间人数失衡。
    """
    if quantiles < 2:
        raise ValueError("quantiles 至少为 2")
    rows: dict[pd.Timestamp, dict[int, float]] = {}
    for dt, sub in frame.groupby("dt", sort=True):
        values = sub[[factor_col, ret_col]].dropna()
        if len(values) < int(min_names):
            continue
        ranks = values[factor_col].rank(method="first")
        buckets = np.floor((ranks.to_numpy(dtype=float) - 1) * quantiles / len(values))
        groups = np.clip(buckets, 0, quantiles - 1).astype(int) + 1
        sizes = pd.Series(groups).value_counts()
        if len(sizes) < quantiles or int(sizes.min()) < int(min_names_per_group):
            continue
        means = values[ret_col].groupby(groups).mean()
        rows[pd.Timestamp(dt)] = {int(k): float(v) for k, v in means.items()}
    if not rows:
        return pd.DataFrame(columns=list(range(1, quantiles + 1)))
    result = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    result.columns = [int(c) for c in result.columns]
    return result[[c for c in range(1, quantiles + 1)]]


def layered_stats(group_returns: pd.DataFrame, direction: int = 1) -> dict[str, object]:
    """分层回测：单调性、多空价差、净值、回撤、多头超额。"""
    empty = {
        "group_mean": [], "nav": {}, "monotone": float("nan"), "monotone_adj": float("nan"),
        "reversals": 0, "spread_mean": float("nan"), "spread_ir": float("nan"),
        "spread_t": float("nan"), "spread_win": float("nan"),
        "long_excess": float("nan"), "long_short_mdd": float("nan"),
        "best_group": None, "periods": [],
    }
    if group_returns.empty:
        return empty

    groups = list(group_returns.columns)
    group_mean = group_returns.mean()
    nav = (1.0 + group_returns).cumprod()

    raw = _spearman(pd.Series(groups, dtype=float), pd.Series(group_mean.to_numpy(dtype=float)))
    best_group = groups[-1] if direction == 1 else groups[0]
    worst_group = groups[0] if direction == 1 else groups[-1]

    ordered = group_mean if direction == 1 else group_mean.iloc[::-1]
    diffs = ordered.diff().dropna()
    reversals = int((diffs < 0).sum())

    spread = group_returns[best_group] - group_returns[worst_group]
    spread_std = float(spread.std(ddof=1)) if len(spread) > 1 else float("nan")
    long_excess = group_returns[best_group] - group_returns.mean(axis=1)
    long_short_nav = (1.0 + spread).cumprod()

    return {
        "group_mean": [float(v) for v in group_mean.to_numpy(dtype=float)],
        "nav": {
            f"Q{g}": [float(v) for v in nav[g].to_numpy(dtype=float)] for g in groups
        },
        "monotone": raw,
        "monotone_adj": raw * direction,
        "reversals": reversals,
        "spread_mean": float(spread.mean()),
        "spread_ir": _safe_ratio(float(spread.mean()), spread_std),
        "spread_t": _safe_ratio(float(spread.mean()), spread_std / np.sqrt(len(spread)))
        if np.isfinite(spread_std)
        else float("nan"),
        "spread_win": float((spread > 0).mean()),
        "long_excess": float(long_excess.mean()),
        "long_short_mdd": _max_drawdown(long_short_nav),
        "best_group": f"Q{best_group}",
        "periods": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in group_returns.index],
    }


def group_turnover(
    frame: pd.DataFrame,
    factor_col: str = "score",
    ret_col: str = "fwd_ret",
    quantiles: int = 5,
    direction: int = 1,
    min_names: int = 20,
) -> float:
    """多头组合的期间换手率均值，用于成本与容量估算。"""
    previous: set[str] | None = None
    values: list[float] = []
    for _, sub in frame.groupby("dt", sort=True):
        valid = sub[[factor_col, ret_col, "code"]].dropna(subset=[factor_col, ret_col])
        if len(valid) < int(min_names):
            continue
        # direction=1 取分数最高的一批，direction=-1 取最低的一批。
        ranks = valid[factor_col].rank(method="first", ascending=direction < 0)
        keep_n = max(1, int(np.ceil(len(valid) / quantiles)))
        ordered = valid.assign(_rank=ranks).sort_values("_rank")
        current = set(ordered.head(keep_n)["code"].astype(str))
        if previous:
            values.append(1.0 - len(current & previous) / max(1, len(previous)))
        previous = current
    return float(np.mean(values)) if values else float("nan")


def ic_decay(
    frame: pd.DataFrame,
    factor_col: str = "score",
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    entry_lag: int = 1,
    min_names: int = 20,
) -> dict[str, float]:
    """不同持有期的 IC 均值，用来确定调仓频率而不是拍脑袋。"""
    result: dict[str, float] = {}
    for horizon in horizons:
        labelled = attach_labels(frame, horizon=horizon, entry_lag=entry_lag)
        ic = ic_series(labelled, factor_col=factor_col, ret_col="fwd_ret", min_names=min_names)
        result[str(horizon)] = float(ic.mean()) if not ic.empty else float("nan")
    return result


def evaluate_factor(
    labelled: pd.DataFrame,
    spec: FactorSpec,
    config: EvalConfig | None = None,
    factor_col: str = "score",
    decay_panel: pd.DataFrame | None = None,
) -> dict[str, object]:
    """输出单个因子的完整证据包（可直接 JSON 序列化）。

    ``decay_panel`` 需要是带 OHLCV 的日频面板，用于重算多持有期收益；
    不传则跳过衰减检验（报告中会标记为未检验）。
    """
    cfg = config or EvalConfig()
    ic = ic_series(labelled, factor_col=factor_col, ret_col="fwd_ret", min_names=cfg.min_names)
    summary = ic_summary(ic, direction=spec.direction)
    groups = quantile_returns(
        labelled,
        factor_col=factor_col,
        ret_col="fwd_ret",
        quantiles=cfg.quantiles,
        min_names=cfg.min_names,
        min_names_per_group=cfg.min_names_per_group,
    )
    layered = layered_stats(groups, direction=spec.direction)
    turnover = group_turnover(
        labelled,
        factor_col=factor_col,
        ret_col="fwd_ret",
        quantiles=cfg.quantiles,
        direction=spec.direction,
        min_names=cfg.min_names,
    )

    cost_per_period = turnover * cfg.round_trip_cost_bps / 1e4 * cfg.cost_multiple
    long_excess = layered["long_excess"]
    net_long_excess = long_excess - cost_per_period if np.isfinite(long_excess) else float("nan")

    dates = sorted(pd.to_datetime(labelled["dt"]).unique())
    coverage = float(labelled[factor_col].notna().mean()) if len(labelled) else float("nan")

    return {
        "factor_id": spec.factor_id,
        "name": spec.name,
        "category": spec.category,
        "direction": spec.direction,
        "hypothesis": spec.hypothesis,
        "ic": {**summary, "series": [
            {"dt": pd.Timestamp(dt).strftime("%Y-%m-%d"), "value": float(v)}
            for dt, v in ic.items()
        ]},
        "layered": layered,
        "turnover": turnover,
        "coverage": coverage,
        "decay": ic_decay(
            decay_panel, factor_col=factor_col, entry_lag=cfg.entry_lag,
            min_names=cfg.min_names,
        ) if decay_panel is not None else {},
        "cost": {
            "turnover": turnover,
            "round_trip_cost_bps": cfg.round_trip_cost_bps,
            "cost_multiple": cfg.cost_multiple,
            "cost_per_period": cost_per_period,
            "long_excess": long_excess,
            "net_long_excess": net_long_excess,
        },
        "sample": {
            "periods": int(len(ic)),
            "date_start": pd.Timestamp(dates[0]).strftime("%Y-%m-%d") if dates else None,
            "date_end": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d") if dates else None,
            "avg_names": float(
                labelled.dropna(subset=[factor_col, "fwd_ret"]).groupby("dt").size().mean()
            ) if len(labelled) else float("nan"),
        },
    }
