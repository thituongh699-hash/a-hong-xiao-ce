"""Strict validation for raw market-data bars.

The validator normalizes datetimes and numeric fields but never fills, drops,
or deduplicates bars.  Callers may request chronological sorting explicitly;
duplicate timestamps are always rejected before sorting.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class MarketDataValidationError(ValueError):
    """Raised when a market-data frame cannot be used without inventing data."""


_OHLC = ("open", "high", "low", "close")


def _parse_datetimes(values: Iterable[object]) -> pd.Series:
    parsed = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            timestamp = pd.NaT
        if not pd.isna(timestamp) and timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
        parsed.append(timestamp)
    index = values.index if isinstance(values, pd.Series) else None
    return pd.Series(parsed, index=index, dtype="datetime64[ns]")


def validate_market_data(
    frame: pd.DataFrame,
    *,
    require_volume: bool = True,
    require_amount: bool = True,
    sort: bool = False,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Return a normalized copy after strict OHLCV validation.

    ``dt`` and OHLC are always required. ``quantity`` is accepted as an alias
    for ``vol`` only when ``vol`` is absent.  Volume and amount may be made
    optional for price-only callers, but when either field is present it must
    be finite and non-negative.  Prices must be finite, positive, and obey the
    OHLC envelope.  Timestamps must be unique and strictly increasing after an
    optional stable sort.
    """
    if not isinstance(frame, pd.DataFrame):
        raise MarketDataValidationError("market data must be a pandas DataFrame")
    if frame.empty:
        if allow_empty:
            return frame.copy(deep=True)
        raise MarketDataValidationError("market data is empty")

    out = frame.copy(deep=True)
    if "vol" not in out.columns and "quantity" in out.columns:
        out = out.rename(columns={"quantity": "vol"})

    required = {"dt", *_OHLC}
    if require_volume:
        required.add("vol")
    if require_amount:
        required.add("amount")
    missing = sorted(required - set(out.columns))
    if missing:
        raise MarketDataValidationError(f"market data lacks required columns: {missing}")

    out["dt"] = _parse_datetimes(out["dt"])
    if out["dt"].isna().any():
        rows = out.index[out["dt"].isna()].tolist()[:5]
        raise MarketDataValidationError(f"market data contains invalid dt values at rows {rows}")
    if out["dt"].duplicated(keep=False).any():
        duplicates = out.loc[out["dt"].duplicated(keep=False), "dt"].astype(str).unique()[:5]
        raise MarketDataValidationError(f"market data contains duplicate dt values: {duplicates.tolist()}")
    if sort:
        out = out.sort_values("dt", kind="stable").reset_index(drop=True)
    elif not out["dt"].is_monotonic_increasing:
        raise MarketDataValidationError("market data dt values must be strictly increasing")

    for column in _OHLC:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        values = out[column].to_numpy(dtype=float)
        invalid = ~np.isfinite(values) | (values <= 0)
        if invalid.any():
            rows = out.index[invalid].tolist()[:5]
            raise MarketDataValidationError(
                f"market data {column} must be finite and positive at rows {rows}"
            )

    high_floor = out[["open", "close", "low"]].max(axis=1)
    low_ceiling = out[["open", "close", "high"]].min(axis=1)
    invalid_envelope = (out["high"] < high_floor) | (out["low"] > low_ceiling)
    if invalid_envelope.any():
        rows = out.index[invalid_envelope].tolist()[:5]
        raise MarketDataValidationError(f"market data violates the OHLC envelope at rows {rows}")

    for column, required_field in (("vol", require_volume), ("amount", require_amount)):
        if column not in out.columns:
            if required_field:
                raise MarketDataValidationError(f"market data lacks required column: {column}")
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce")
        values = out[column].to_numpy(dtype=float)
        invalid = ~np.isfinite(values) | (values < 0)
        if invalid.any():
            rows = out.index[invalid].tolist()[:5]
            raise MarketDataValidationError(
                f"market data {column} must be finite and non-negative at rows {rows}"
            )

    out.attrs.update(frame.attrs)
    return out
