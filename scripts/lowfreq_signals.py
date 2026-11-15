"""Fixed, interpretable low-frequency OHLCV signal candidates.

The functions in this module only calculate end-of-day signals.  They do not
choose a weekly observation day or simulate orders, fills, stops, or exits at
the next open.  Raw prices remain available for those execution decisions;
``s_*`` columns are the back-adjusted series used by the signal rules.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


SPECS = [
    dict(id="breakout20", family="breakout", lookback=20, stop=.08, trail=.12,
         take=None, max_hold=40, exit_ma=20, weekly_only=False),
    dict(id="breakout60", family="breakout", lookback=60, stop=.08, trail=.12,
         take=None, max_hold=60, exit_ma=20, weekly_only=False),
    dict(id="pullback3", family="pullback", lookback=3, stop=.06, trail=.10,
         take=None, max_hold=20, exit_ma=10, weekly_only=False),
    dict(id="pullback5", family="pullback", lookback=5, stop=.06, trail=.10,
         take=None, max_hold=20, exit_ma=10, weekly_only=False),
    dict(id="reversal3", family="reversal", lookback=3, stop=.06, trail=None,
         take=.12, max_hold=10, exit_ma=0, weekly_only=False),
    dict(id="reversal5", family="reversal", lookback=5, stop=.06, trail=None,
         take=.12, max_hold=10, exit_ma=0, weekly_only=False),
    dict(id="compression20", family="compression", lookback=20, stop=.08, trail=.12,
         take=None, max_hold=40, exit_ma=20, weekly_only=False),
    dict(id="compression40", family="compression", lookback=40, stop=.08, trail=.12,
         take=None, max_hold=40, exit_ma=20, weekly_only=False),
    dict(id="rotation20", family="rotation", lookback=20, stop=.10, trail=.15,
         take=None, max_hold=20, exit_ma=20, weekly_only=True),
    dict(id="rotation60", family="rotation", lookback=60, stop=.10, trail=.15,
         take=None, max_hold=40, exit_ma=20, weekly_only=True),
]

_SPECS_BY_ID = {spec["id"]: spec for spec in SPECS}
_SPEC_FIELDS = tuple(SPECS[0])
_RAW_COLUMNS = ("dt", "open", "high", "low", "close", "vol", "amount",
                "adjustment_factor")


def prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate daily bars and append causal features used by all candidates.

    ``adjustment_factor`` is the already-verified cumulative back-adjustment
    factor.  Signal OHLC is therefore raw OHLC multiplied by that factor.
    Rolling volume/amount means and prior ranges exclude the current bar.
    """
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("nonempty daily OHLCV frame required")
    if missing := set(_RAW_COLUMNS) - set(raw.columns):
        raise ValueError(f"daily bars lack columns: {sorted(missing)}")
    if not raw.index.is_unique:
        raise ValueError("unique row index required")

    work = raw.copy(deep=True)
    parsed_dt = pd.to_datetime(work["dt"], errors="coerce")
    if parsed_dt.isna().any():
        raise ValueError("invalid daily dates")
    if parsed_dt.dt.tz is not None:
        parsed_dt = parsed_dt.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    if not parsed_dt.is_monotonic_increasing or parsed_dt.duplicated().any():
        raise ValueError("daily dates must be strictly increasing and unique")
    work["dt"] = parsed_dt

    numeric = ("open", "high", "low", "close", "vol", "amount",
               "adjustment_factor")
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    values = work[list(numeric)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("finite OHLCV, amount, and adjustment factor required")
    prices = work[["open", "high", "low", "close"]]
    if ((prices <= 0).any().any()
            or work["high"].lt(prices[["open", "close"]].max(axis=1)).any()
            or work["low"].gt(prices[["open", "close"]].min(axis=1)).any()):
        raise ValueError("invalid OHLC")
    if work["vol"].lt(0).any() or work["amount"].lt(0).any():
        raise ValueError("nonnegative volume and amount required")
    if work["adjustment_factor"].le(0).any():
        raise ValueError("positive adjustment_factor required")

    for column in ("open", "high", "low", "close"):
        work[f"s_{column}"] = work[column] * work["adjustment_factor"]

    # Zero based: 120 means exactly 120 completed preceding bars are present.
    work["history_index"] = np.arange(len(work), dtype=np.int64)
    for period in (10, 20, 60, 120):
        work[f"ma{period}"] = work["s_close"].rolling(period).mean()
    for period in (3, 5, 20, 60):
        work[f"momentum{period}"] = work["s_close"].div(
            work["s_close"].shift(period)
        ).sub(1.0)

    prior_vol = work["vol"].shift(1)
    prior_amount = work["amount"].shift(1)
    for period in (3, 5, 20):
        work[f"vol{period}"] = prior_vol.rolling(period).mean()
    work["amount20"] = prior_amount.rolling(20).mean()
    work["prior_high"] = work["s_high"].shift(1)
    work["prior_low"] = work["s_low"].shift(1)
    for period in (20, 40, 60):
        work[f"prior_high{period}"] = work["s_high"].shift(1).rolling(period).max()
        work[f"prior_low{period}"] = work["s_low"].shift(1).rolling(period).min()
    return work


def _canonical_spec(spec: Mapping) -> dict:
    if not isinstance(spec, Mapping):
        raise ValueError("spec must be one of the predefined SPECS mappings")
    spec_id = spec.get("id")
    canonical = _SPECS_BY_ID.get(spec_id)
    if canonical is None or set(spec) != set(_SPEC_FIELDS):
        raise ValueError("unknown or modified candidate spec")
    if any(spec[field] != canonical[field] for field in _SPEC_FIELDS):
        raise ValueError("candidate parameters must match predefined SPECS")
    return canonical


def candidate_signals(prepared: pd.DataFrame, spec: Mapping) -> pd.DataFrame:
    """Return ``dt``, entry, decimal score, and moving-average exit signals."""
    canonical = _canonical_spec(spec)
    if not isinstance(prepared, pd.DataFrame) or prepared.empty:
        raise ValueError("nonempty prepared frame required")
    required = {
        "dt", "s_close", "s_high", "vol", "amount20", "vol20",
        "history_index", "ma10", "ma20", "ma60", "ma120",
        "momentum3", "momentum5", "momentum20", "momentum60",
        "prior_high", "prior_high20", "prior_high40", "prior_high60",
        "prior_low20", "prior_low40",
    }
    if missing := required - set(prepared.columns):
        raise ValueError(f"prepared frame lacks columns: {sorted(missing)}")

    family = canonical["family"]
    lookback = canonical["lookback"]
    common = (
        prepared["history_index"].ge(120)
        & prepared["vol"].gt(0)
        & prepared["amount20"].ge(50_000_000.0)
    )

    if family == "breakout":
        score = prepared["momentum60"]
        entry = (
            prepared["s_close"].gt(prepared[f"prior_high{lookback}"])
            & prepared["s_close"].gt(prepared["ma60"])
            & prepared["ma60"].gt(prepared["ma60"].shift(5))
            & prepared["vol"].div(prepared["vol20"]).ge(1.2)
        )
    elif family == "pullback":
        previous_short = prepared[f"momentum{lookback}"].shift(1)
        previous_mom60 = prepared["momentum60"].shift(1)
        daily_return = prepared["s_close"].div(prepared["s_close"].shift(1)).sub(1.0)
        score = previous_mom60
        entry = (
            previous_short.lt(-.03)
            & previous_mom60.gt(.10)
            & prepared["s_close"].shift(1).gt(prepared["ma60"].shift(1))
            & prepared["s_close"].gt(prepared["prior_high"])
            & daily_return.gt(0)
            & daily_return.lt(.07)
            & prepared[f"vol{lookback}"].lt(prepared["vol20"])
        )
    elif family == "reversal":
        previous_short = prepared[f"momentum{lookback}"].shift(1)
        previous_mom60 = prepared["momentum60"].shift(1)
        threshold = -.06 if lookback == 3 else -.08
        score = -previous_short
        entry = (
            previous_short.lt(threshold)
            & previous_mom60.gt(0)
            & prepared["s_close"].shift(1).gt(prepared["ma120"].shift(1))
            & prepared["s_close"].gt(prepared["prior_high"])
        )
    elif family == "compression":
        score = prepared["momentum60"]
        range_width = prepared[f"prior_high{lookback}"].div(
            prepared[f"prior_low{lookback}"]
        ).sub(1.0)
        entry = (
            prepared["s_close"].gt(prepared[f"prior_high{lookback}"])
            & range_width.lt(.20)
            & prepared["s_close"].gt(prepared["ma60"])
            & prepared["vol"].div(prepared["vol20"]).ge(1.5)
        )
    elif family == "rotation":
        score = prepared[f"momentum{lookback}"]
        threshold = .05 if lookback == 20 else .10
        entry = (
            prepared["s_close"].gt(prepared["ma60"])
            & score.gt(threshold)
        )
    else:  # Defensive even though predefined specs make this unreachable.
        raise ValueError(f"unsupported candidate family: {family}")

    entry = (common & entry).fillna(False).astype(bool)
    exit_ma = canonical["exit_ma"]
    if exit_ma:
        exit_signal = prepared["s_close"].lt(prepared[f"ma{exit_ma}"]).fillna(False)
    else:
        exit_signal = pd.Series(False, index=prepared.index)
    return pd.DataFrame(
        {"dt": prepared["dt"], "entry": entry,
         "score": pd.to_numeric(score, errors="coerce").astype(float),
         "exit": exit_signal.astype(bool)},
        index=prepared.index,
    )
