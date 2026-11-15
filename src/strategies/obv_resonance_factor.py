"""Causal daily/weekly OBV resonance, including the current partial week.

Input is a single security's completed, verified raw daily bars. Adjust each
daily close before constructing weekly closes; volume always stays raw.
This module computes factors only and does not place orders.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from src.utils.price_adjustment import AdjustmentPlan, assert_raw_prices

DEFAULT_MA_PERIOD = 18
OBV_SCALE = 10000.0


def _daily_input(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"dt", "open", "high", "low", "close", "vol"}
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("nonempty daily bars required")
    if missing := required - set(raw.columns):
        raise ValueError(f"daily bars lack columns: {sorted(missing)}")
    assert_raw_prices(None, "default", raw)
    work = raw.copy(deep=True)
    work["dt"] = pd.to_datetime(work["dt"], errors="coerce")
    if work.dt.isna().any():
        raise ValueError("invalid daily dates")
    if work.dt.dt.tz is not None:
        work["dt"] = work.dt.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    work["dt"] = work.dt.dt.normalize()
    if work.dt.duplicated().any():
        raise ValueError("one bar per trading day required")
    if work.dt.dt.dayofweek.ge(5).any():
        raise ValueError("A-share daily bars must be Monday through Friday")
    if "code" in work and (work.code.isna().any() or work.code.nunique() != 1):
        raise ValueError("single security required; calculate each code separately")
    for column in ("open", "high", "low", "close", "vol"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    prices = work[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if (not np.isfinite(prices).all() or (prices <= 0).any()
            or work.high.lt(work[["open", "close"]].max(axis=1)).any()
            or work.low.gt(work[["open", "close"]].min(axis=1)).any()):
        raise ValueError("invalid OHLC")
    if not np.isfinite(work.vol).all() or work.vol.lt(0).any():
        raise ValueError("finite nonnegative volume required")
    return work.sort_values("dt").reset_index(drop=True)


def features(raw: pd.DataFrame, factors: pd.DataFrame,
             *, ma_period: int = DEFAULT_MA_PERIOD) -> pd.DataFrame:
    """Return per-day OBV diagnostics and the binary resonance factor.

    ``raw`` needs raw-price provenance accepted by ``assert_raw_prices``;
    ``factors`` needs the project's verified back-adjustment metadata.
    Both OBV moving averages use ``ma_period`` bars (days/weeks respectively).
    The week's close and volume are provisional values as of each row's close.
    Empty calendar weeks are skipped, as on a security's weekly K-line chart.
    The first observed daily/weekly OBV is initialized at zero.
    """
    if isinstance(ma_period, bool) or not isinstance(ma_period, Integral) or ma_period < 2:
        raise ValueError("ma_period must be an integer >= 2")
    ma_period = int(ma_period)
    work = _daily_input(raw)
    plan = AdjustmentPlan(work, factors, "hfq")
    work["adjustment_factor"] = plan.frame["__factor"].to_numpy()
    work["signal_multiplier"] = plan.frame["__signal_multiplier"].to_numpy()
    work["signal_close"] = work.close * work.signal_multiplier

    daily_signed_volume = np.sign(work.signal_close.diff()).fillna(0) * work.vol
    work["daily_obv"] = daily_signed_volume.cumsum() / OBV_SCALE
    work["daily_maobv"] = work.daily_obv.rolling(ma_period).mean()
    work["daily_above_maobv"] = work.daily_obv.gt(work.daily_maobv)

    week = work.dt.dt.to_period("W-FRI")
    work["week_start"] = week.dt.start_time
    work["weekly_close"] = work.signal_close
    work["weekly_volume"] = work.vol.groupby(week, sort=False).cumsum()
    completed = work.groupby(week, sort=True).agg(
        close=("signal_close", "last"), volume=("vol", "sum"))
    completed["obv"] = (
        np.sign(completed.close.diff()).fillna(0) * completed.volume
    ).cumsum() / OBV_SCALE

    # Every mapping below is shifted first: a day's row can only see final
    # values of PREVIOUS weeks. The full current week's final values are never
    # mapped back onto its earlier days.
    work["weekly_prev_close"] = week.map(completed.close.shift())
    work["weekly_prev_obv"] = week.map(completed.obv.shift())
    work["weekly_prev2_obv"] = week.map(completed.obv.shift(2))
    partial_signed_volume = (
        np.sign(work.weekly_close - work.weekly_prev_close).fillna(0)
        * work.weekly_volume
    )
    work["weekly_obv"] = work.weekly_prev_obv.fillna(0) + partial_signed_volume / OBV_SCALE
    prior_sum = completed.obv.shift().rolling(ma_period - 1).sum()
    work["weekly_maobv"] = (week.map(prior_sum) + work.weekly_obv) / ma_period
    work["weekly_above_maobv"] = work.weekly_obv.gt(work.weekly_maobv)
    work["weekly_rising_two"] = (
        work.weekly_obv.gt(work.weekly_prev_obv)
        & work.weekly_prev_obv.gt(work.weekly_prev2_obv)
    )
    work["history_ready"] = (
        work.daily_maobv.notna() & work.weekly_maobv.notna()
        & work.weekly_prev2_obv.notna()
    )
    active = (work.history_ready & work.weekly_rising_two
              & work.weekly_above_maobv & work.daily_above_maobv)
    work["obv_resonance_factor"] = active.astype("int8")
    work["resonance_trigger"] = active & ~active.shift(fill_value=False)
    work.attrs.update(signal_price_mode="hfq", obv_volume_mode="raw",
                      ma_period=ma_period, weekly_mode="current_week_as_of_daily_close")
    return work


def signal_mask(work: pd.DataFrame, *, first_only: bool = False) -> pd.Series:
    """Select all resonant days, or only transitions from 0 to 1."""
    if first_only:
        return work["resonance_trigger"].copy()
    return work["obv_resonance_factor"].eq(1)
