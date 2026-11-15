"""Causal daily factors for a high excursion / positive weak close near a low.

Signal prices use verified, causal back-adjustment factors. This module does
not place orders. All windows end on the signal day or the previous day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.price_adjustment import AdjustmentPlan

EPS = 1e-10
DEFAULT = dict(a=7.2, a2=4.8, a3=1., low_mode="pullback", window=60,
               range_max=.35, drawdown_min=8., near_low_max=18.)


def features(raw: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    raw = raw.sort_values("dt").reset_index(drop=True).copy()
    if raw.empty or raw.dt.duplicated().any():
        raise ValueError("nonempty unique daily bars required")
    prices = raw[["open", "high", "low", "close"]].to_numpy(float)
    if (not np.isfinite(prices).all() or (prices <= 0).any()
            or (raw.high + 1e-8 < raw[["open", "close"]].max(axis=1)).any()
            or (raw.low - 1e-8 > raw[["open", "close"]].min(axis=1)).any()):
        raise ValueError("invalid OHLC")
    plan = AdjustmentPlan(raw, factors, "hfq")
    work = raw.copy()
    work["factor"] = plan.frame["__factor"].to_numpy()
    multiplier = plan.frame["__signal_multiplier"].to_numpy()
    c, h, l = (raw[k] * multiplier for k in ("close", "high", "low"))
    work["high_pct"] = (h / c.shift() - 1) * 100
    work["close_pct"] = (c / c.shift() - 1) * 100
    work["fade_pct"] = work.high_pct - work.close_pct
    ma5 = c.rolling(5).mean()
    slope = ma5.diff()
    turn = slope.gt(EPS) & slope.shift().le(EPS)
    work["ma5_turn"] = turn.rolling(5, min_periods=5).max().fillna(0).astype(bool) & slope.gt(EPS)
    ma20 = c.rolling(20).mean()
    work["below_ma20_days"] = c.lt(ma20).shift().rolling(10).sum()
    work["near_low20_pct"] = (c / l.rolling(20).min() - 1) * 100
    for n in (20, 60):
        low, high = l.rolling(n).min(), h.rolling(n).max()
        work[f"range_pos{n}"] = (c - low) / (high - low).replace(0, np.nan)
    prior_peak = h.shift().rolling(60).max()
    work["drawdown60_pct"] = (1 - c / prior_peak) * 100
    # Ties use the most recent peak. Window is strictly before the signal bar.
    work["peak_age60"] = h.shift().rolling(60).apply(
        lambda x: 60 - int(np.flatnonzero(x == x.max())[-1]), raw=True)
    work["history_bars"] = np.arange(len(work))
    work["history_ready"] = work.history_bars.ge(120)
    return work


def validate_params(params: dict) -> dict:
    p = {**DEFAULT, **params}
    if not all(np.isfinite(p[k]) for k in ("a", "a2", "a3", "range_max", "drawdown_min", "near_low_max")):
        raise ValueError("finite parameters required")
    if not 0 <= p["a3"] < p["a2"] < 10 or not 0 < p["a"] < 10:
        raise ValueError("require 0 <= A3 < A2 < 10 and 0 < A < 10")
    if p["low_mode"] not in {"none", "range", "pullback", "turn"} or p["window"] not in (20, 60):
        raise ValueError("unsupported low mode/window")
    if not 0 < p["range_max"] <= 1 or not 0 <= p["drawdown_min"] < 100 or p["near_low_max"] < 0:
        raise ValueError("invalid low-position bounds")
    return p


def low_mask(work: pd.DataFrame, params: dict) -> pd.Series:
    p = validate_params(params)
    if p["low_mode"] == "none":
        low = pd.Series(True, index=work.index)
    elif p["low_mode"] == "range":
        low = work[f"range_pos{p['window']}"] <= p["range_max"]
    else:
        low = (work.drawdown60_pct.ge(p["drawdown_min"])
               & work.near_low20_pct.le(p["near_low_max"])
               & work.peak_age60.between(3, 55) & work.below_ma20_days.ge(3))
        if p["low_mode"] == "turn":
            low &= work.ma5_turn
    return low & work.history_ready


def signal_mask(work: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = validate_params(params or {})
    return (work.high_pct.gt(p["a"] + EPS) & work.close_pct.lt(p["a2"] - EPS)
            & work.close_pct.gt(p["a3"] + EPS) & low_mask(work, p))


def parameter_grid() -> list[dict]:
    lows = [dict(low_mode="none")]
    lows += [dict(low_mode="range", window=n, range_max=q)
             for n in (20, 60) for q in (.2, .35)]
    lows += [dict(low_mode=mode, drawdown_min=d, near_low_max=b)
             for mode in ("pullback", "turn") for d in (8., 15., 25.) for b in (10., 18.)]
    return [validate_params(dict(a=a, a2=a2, a3=a3, **low))
            for a in (6., 7.2, 8.5) for a2 in (3.5, 4.8, 6.)
            for a3 in (0., 1., 2.) for low in lows]
