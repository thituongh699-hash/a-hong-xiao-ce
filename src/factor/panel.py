"""面板构建、前向收益与可交易掩码。

三个容易出错的地方在这里一次性钉死：

1. **前向收益从 T+1 收盘开始算**。信号在 T 日收盘后才成立，T→T+1 那段涨跌拿不到。
   ``entry_lag=1`` 是默认值，改 0 等于假设能以 T 日收盘价成交，属于偏乐观口径。
2. **可交易掩码**。次日涨停买不进、停牌没有成交、卖出日跌停卖不掉，
   这些样本的真实收益不可实现，默认按缺失处理而不是照算。
3. **重叠收益**。h 日持有期逐日计算 IC 会让相邻样本共享同一段未来收益，
   t 值被系统性高估。``rebalance_dates(step=h)`` 提供不重叠采样。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from .spec import FactorSpec

PRICE_COLUMNS = ("open", "high", "low", "close")
REQUIRED_COLUMNS = ("dt", "open", "high", "low", "close", "vol")
OPTIONAL_COLUMNS = ("amount",)

# 涨跌停幅度按板块代码前缀近似。ST 的历史逐日状态无法从当前名单还原，
# 这里不做猜测，相关限制在报告里作为已知缺陷披露。
_LIMIT_RULES: tuple[tuple[tuple[str, ...], float], ...] = (
    (("300", "301", "688", "689"), 0.20),
    (("8", "4", "92"), 0.30),
)
_DEFAULT_LIMIT = 0.10
LIMIT_BUFFER = 0.002


@dataclass
class PanelBuild:
    panel: pd.DataFrame
    quality: dict[str, object] = field(default_factory=dict)


def _normalize_code(code: object) -> str:
    text = str(code).strip().upper()
    for suffix in (".SZ", ".SH", ".BJ"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def price_limit_pct(code: str) -> float:
    text = _normalize_code(code)
    for prefixes, limit in _LIMIT_RULES:
        if text.startswith(prefixes):
            return limit
    return _DEFAULT_LIMIT


def _normalize_one(code: str, frame: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    if frame is None or frame.empty:
        return None, "空数据"
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        return None, f"缺少字段:{','.join(missing)}"

    work = frame.copy()
    work["dt"] = pd.to_datetime(work["dt"], errors="coerce")
    for column in REQUIRED_COLUMNS[1:] + OPTIONAL_COLUMNS:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    if "amount" not in work.columns:
        work["amount"] = np.nan

    work = work.loc[work["dt"].notna()].copy()
    if work.empty:
        return None, "日期全部无效"
    if work["dt"].duplicated().any():
        return None, "存在重复日期"
    work = work.sort_values("dt").reset_index(drop=True)

    prices = work[list(PRICE_COLUMNS)].to_numpy(dtype=float)
    finite = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    envelope_ok = (
        (work["high"].to_numpy(dtype=float) + 1e-8 >= work[["open", "close"]].max(axis=1).to_numpy(dtype=float))
        & (work["low"].to_numpy(dtype=float) - 1e-8 <= work[["open", "close"]].min(axis=1).to_numpy(dtype=float))
    )
    if not finite.all() or not envelope_ok.all():
        return None, "OHLC 非法（非正价格或包络越界）"

    work["code"] = _normalize_code(code)
    return work[["dt", "code", "open", "high", "low", "close", "vol", "amount"]], None


def build_panel(bars: Mapping[str, pd.DataFrame]) -> PanelBuild:
    """把 ``{code: 日线}`` 合并为长表；坏数据整只剔除并在质量里记录原因。"""
    frames: list[pd.DataFrame] = []
    rejected: dict[str, str] = {}
    for code, frame in bars.items():
        normalized, reason = _normalize_one(code, frame)
        if normalized is None:
            rejected[_normalize_code(code)] = reason or "未知原因"
            continue
        frames.append(normalized)
    if not frames:
        raise ValueError(f"没有可用标的：{rejected}")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["code", "dt"]).reset_index(drop=True)
    bar_counts = panel.groupby("code")["dt"].count()
    quality = {
        "codes_requested": len(bars),
        "codes_used": int(len(bar_counts)),
        "codes_rejected": rejected,
        "bar_count_min": int(bar_counts.min()),
        "bar_count_median": float(bar_counts.median()),
        "bar_count_max": int(bar_counts.max()),
        "date_start": panel["dt"].min().strftime("%Y-%m-%d"),
        "date_end": panel["dt"].max().strftime("%Y-%m-%d"),
        "row_count": int(len(panel)),
    }
    return PanelBuild(panel=panel, quality=quality)


def _grouped(panel: pd.DataFrame, column: str) -> pd.core.groupby.generic.SeriesGroupBy:
    return panel.groupby("code", sort=False)[column]


def forward_returns(panel: pd.DataFrame, horizon: int, entry_lag: int = 1) -> pd.Series:
    """T 日信号 → T+entry_lag 建仓 → 持有 horizon 个交易日的复权收盘收益。"""
    if horizon <= 0:
        raise ValueError("horizon 必须为正整数")
    if entry_lag < 0:
        raise ValueError("entry_lag 不能为负")
    close = pd.to_numeric(panel["close"], errors="coerce")
    grouped = panel.groupby("code", sort=False)["close"]
    entry = grouped.shift(-entry_lag)
    exit_ = grouped.shift(-(entry_lag + horizon))
    entry = entry.where(entry > 0)
    result = exit_ / entry - 1
    return result.rename("fwd_ret")


def tradability(panel: pd.DataFrame, horizon: int, entry_lag: int = 1) -> pd.DataFrame:
    """返回与面板对齐的建仓/离场可交易标记。"""
    close = pd.to_numeric(panel["close"], errors="coerce")
    vol = pd.to_numeric(panel["vol"], errors="coerce")
    grouped_close = panel.groupby("code", sort=False)["close"]
    grouped_vol = panel.groupby("code", sort=False)["vol"]
    daily_ret = grouped_close.pct_change(fill_method=None)

    entry_close = grouped_close.shift(-entry_lag)
    entry_vol = grouped_vol.shift(-entry_lag)
    entry_ret = daily_ret.groupby(panel["code"], sort=False).shift(-entry_lag)

    exit_offset = -(entry_lag + horizon)
    exit_close = grouped_close.shift(exit_offset)
    exit_vol = grouped_vol.shift(exit_offset)
    exit_ret = daily_ret.groupby(panel["code"], sort=False).shift(exit_offset)

    limit = panel["code"].map(price_limit_pct).astype(float).to_numpy()
    entry_ret_values = entry_ret.to_numpy(dtype=float)
    exit_ret_values = exit_ret.to_numpy(dtype=float)

    entry_ok = (
        np.isfinite(entry_close.to_numpy(dtype=float))
        & np.isfinite(entry_vol.to_numpy(dtype=float))
        & (entry_vol.to_numpy(dtype=float) > 0)
        & np.isfinite(entry_ret_values)
        & (entry_ret_values < limit - LIMIT_BUFFER)
    )
    exit_ok = (
        np.isfinite(exit_close.to_numpy(dtype=float))
        & np.isfinite(exit_vol.to_numpy(dtype=float))
        & (exit_vol.to_numpy(dtype=float) > 0)
        & np.isfinite(exit_ret_values)
        & (exit_ret_values > -(limit - LIMIT_BUFFER))
    )
    return pd.DataFrame({"entry_ok": entry_ok, "exit_ok": exit_ok}, index=panel.index)


def attach_labels(
    panel: pd.DataFrame,
    horizon: int,
    entry_lag: int = 1,
    mask_untradable: bool = True,
) -> pd.DataFrame:
    """把前向收益与可交易标记挂到面板上；不可交易的样本收益置为缺失。"""
    work = panel.copy()
    work["fwd_ret"] = forward_returns(work, horizon, entry_lag)
    flags = tradability(work, horizon, entry_lag)
    work["entry_ok"] = flags["entry_ok"]
    work["exit_ok"] = flags["exit_ok"]
    if mask_untradable:
        work.loc[~(work["entry_ok"] & work["exit_ok"]), "fwd_ret"] = np.nan
    return work


def compute_factor(
    panel: pd.DataFrame,
    spec: FactorSpec,
    params: Mapping[str, object] | None = None,
) -> pd.Series:
    """按标的计算因子值，返回与面板同索引的序列。"""
    missing = [c for c in spec.required_columns if c not in panel.columns]
    if missing:
        raise ValueError(f"{spec.factor_id}: 面板缺少字段 {','.join(missing)}")
    merged = {**dict(spec.params), **dict(params or {})}

    pieces: list[pd.Series] = []
    for _, sub in panel.groupby("code", sort=False):
        ordered = sub.sort_values("dt")
        values = spec.compute(ordered, merged)
        array = np.array(pd.Series(values).to_numpy(dtype=float), dtype=float, copy=True)
        if array.shape[0] != len(ordered):
            raise ValueError(f"{spec.factor_id}: 因子输出长度与行情不一致")
        array[~np.isfinite(array)] = np.nan
        pieces.append(pd.Series(array, index=ordered.index, name=spec.factor_id))
    return pd.concat(pieces).reindex(panel.index)


def size_proxy(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """市值代理：过去 window 日平均成交额（对数化在预处理里做）。"""
    amount = pd.to_numeric(panel["amount"], errors="coerce")
    smoothed = panel.assign(_amount=amount).groupby("code", sort=False)["_amount"].transform(
        lambda s: s.rolling(window, min_periods=max(5, window // 4)).mean()
    )
    return smoothed.rename("size_proxy")


def industry_series(panel: pd.DataFrame, mapping: Mapping[str, str] | None) -> pd.Series | None:
    if not mapping:
        return None
    normalized = {_normalize_code(k): v for k, v in mapping.items()}
    return panel["code"].map(normalized).rename("industry")


def rebalance_dates(
    panel: pd.DataFrame,
    step: int,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """按交易步长取不重叠的截面日期，降低重叠收益造成的 t 值虚高。"""
    if step <= 0:
        raise ValueError("step 必须为正整数")
    calendar = pd.Index(sorted(pd.to_datetime(panel["dt"]).unique()))
    if start is not None:
        calendar = calendar[calendar >= pd.Timestamp(start)]
    if end is not None:
        calendar = calendar[calendar <= pd.Timestamp(end)]
    return list(calendar[::step])
