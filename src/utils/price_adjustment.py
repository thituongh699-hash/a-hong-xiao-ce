"""Causal signal-price adjustment over independently verified raw prices.

Only OHLC sent to a strategy may be adjusted. Orders, cash and marks remain in
raw exchange-price units. BaoStock's cumulative *backAdjustFactor* is used;
its foreAdjustFactor must never be substituted for it.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import pandas as pd

from src.utils.market_data_validation import MarketDataValidationError, validate_market_data


class PriceAdjustmentError(RuntimeError):
    """Raw-price provenance or factor availability cannot be established."""


_OHLC = ("open", "high", "low", "close")
_CACHE_VERSION = 1
_FACTOR_SOURCE = "baostock.backAdjustFactor"
_BASE_DATE = pd.Timestamp("1990-01-01")


def _setting(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        if key in config:
            return config[key]
        value: Any = config
        for part in key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return default
            value = value[part]
        return value
    getter = getattr(config, "get", None)
    return getter(key, default) if callable(getter) else default


def _day(value: Any, label: str) -> pd.Timestamp:
    try:
        day = pd.Timestamp(value)
        if pd.isna(day):
            raise ValueError("missing date")
        if day.tz is not None:
            day = day.tz_convert("Asia/Shanghai").tz_localize(None)
        return day.normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise PriceAdjustmentError(f"invalid {label}: {value!r}") from exc


def _stock_code(value: str) -> str:
    code = str(value or "").strip().upper()
    match = re.fullmatch(r"(?:(SH|SZ)\.(\d{6})|(\d{6})\.(SH|SZ)|(\d{6}))", code)
    if not match:
        raise PriceAdjustmentError(f"unsupported mainland stock code: {value!r}")
    market = match.group(1) or match.group(4)
    number = match.group(2) or match.group(3) or match.group(5)
    market = market or ("SH" if number.startswith(("6", "9")) else "SZ")
    supported = (number.startswith(("600", "601", "603", "605", "688")) if market == "SH"
                 else number.startswith(("000", "001", "002", "003", "300", "301")))
    if not supported:
        raise PriceAdjustmentError("uniform adjustment supports Shanghai/Shenzhen A-share stocks only; indices, funds and B-shares are unsupported")
    return f"{market.lower()}.{number}"


def _positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PriceAdjustmentError(f"invalid {label}: {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise PriceAdjustmentError(f"{label} must be finite and positive")
    return number


def _factor_frame(rows: list[dict[str, Any]], coverage_end: Any,
                  *, source: str = _FACTOR_SOURCE) -> pd.DataFrame:
    end = _day(coverage_end, "coverage_end")
    if end < _BASE_DATE:
        raise PriceAdjustmentError("factor coverage predates BaoStock baseline")
    values: dict[pd.Timestamp, float] = {_BASE_DATE: 1.0}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"dt", "factor"}:
            raise PriceAdjustmentError("factor row has an unexpected schema")
        dt = _day(row["dt"], "factor date")
        if dt < _BASE_DATE or dt > end:
            raise PriceAdjustmentError("factor date outside verified query coverage")
        factor = _positive(row["factor"], "factor")
        if dt in values and not math.isclose(values[dt], factor, rel_tol=1e-12):
            raise PriceAdjustmentError(f"conflicting factors on {dt.date()}")
        values[dt] = factor
    frame = pd.DataFrame(sorted(values.items()), columns=["dt", "factor"])
    frame.attrs.update(source=source, coverage_start=_BASE_DATE.date().isoformat(),
                       coverage_end=end.date().isoformat(), verified=True)
    return frame


def _worker_payload(code: str, end: pd.Timestamp, timeout: float) -> dict[str, Any]:
    if not sys.executable or not Path(sys.executable).is_absolute():
        raise PriceAdjustmentError("absolute Python executable is required")
    worker = Path(__file__).with_name("price_factor_worker.py").resolve()
    try:
        process = subprocess.run(
            [sys.executable, str(worker), code, end.date().isoformat(), str(timeout)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills and reaps the child on timeout.
        raise PriceAdjustmentError(f"BaoStock factor query timed out after {timeout:g}s") from exc
    except OSError as exc:
        raise PriceAdjustmentError(f"factor worker could not start: {exc}") from exc
    if process.returncode == 124:
        raise PriceAdjustmentError(f"BaoStock factor query timed out after {timeout:g}s")
    try:
        payload = json.loads(process.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise PriceAdjustmentError("factor worker returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        reason = payload.get("error", "unknown error") if isinstance(payload, dict) else "invalid response"
        raise PriceAdjustmentError(f"BaoStock factor query failed: {reason}")
    if process.returncode != 0:
        raise PriceAdjustmentError(f"factor worker failed with exit code {process.returncode}")
    return payload


def _read_cache(path: Path, code: str, end: pd.Timestamp, recent_ttl: float) -> pd.DataFrame | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if (cached.get("version") != _CACHE_VERSION or cached.get("code") != code
                or cached.get("source") != _FACTOR_SOURCE):
            return None
        coverage = _day(cached["coverage_end"], "cached coverage_end")
        if coverage < end:
            return None
        # Historical queries are immutable for this cache policy. A request
        # touching the last 14 days must have a recent successful query.
        if end >= pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize() - timedelta(days=14):
            fetched = datetime.fromisoformat(cached["fetched_at"])
            if fetched.tzinfo is None or (datetime.now(timezone.utc) - fetched).total_seconds() > recent_ttl:
                return None
        frame = _factor_frame(cached["rows"], coverage)
        return frame
    except (OSError, KeyError, ValueError, TypeError, PriceAdjustmentError):
        return None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load_adjustment_factors(stock_code: str, start_date: Any, end_date: Any,
                            config: Any = None, timeout: float = 20) -> pd.DataFrame:
    """Load cumulative BaoStock back factors, including the pre-range anchor.

    A successful empty event query still establishes factor 1 from 1990. Cache
    coverage records the requested query end, not the last corporate event.
    """
    start, end = _day(start_date, "start_date"), _day(end_date, "end_date")
    if start < _BASE_DATE or end < start:
        raise PriceAdjustmentError("invalid factor date range")
    if end > pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize():
        raise PriceAdjustmentError("future factor coverage cannot be verified")
    timeout = _positive(timeout, "timeout")
    code = _stock_code(stock_code)
    root = Path(__file__).resolve().parents[2]
    cache_dir = Path(str(_setting(config, "data_provider.factor_cache_dir",
                                  root / "data" / "history" / "cache")))
    recent_ttl = _positive(_setting(config, "data_provider.factor_cache_recent_ttl_seconds", 21600),
                           "recent cache TTL")
    path = cache_dir / f"adjustment_factors_v{_CACHE_VERSION}_{code.replace('.', '_')}.json"
    cached = _read_cache(path, code, end, recent_ttl)
    if cached is not None:
        result = cached.loc[cached["dt"] <= end].reset_index(drop=True)
        result.attrs["coverage_end"] = end.date().isoformat()
        return result
    payload = _worker_payload(code, end, timeout)
    fields, raw_rows = payload.get("fields"), payload.get("rows")
    if (not isinstance(fields, list) or len(fields) != len(set(fields))
            or "dividOperateDate" not in fields or "backAdjustFactor" not in fields
            or not isinstance(raw_rows, list)):
        raise PriceAdjustmentError("BaoStock factor response has an unexpected schema")
    parsed = []
    for row in raw_rows:
        if not isinstance(row, list) or len(row) != len(fields):
            raise PriceAdjustmentError("BaoStock factor row has an unexpected schema")
        record = dict(zip(fields, row))
        if "code" in record and str(record["code"]).lower() != code:
            raise PriceAdjustmentError("BaoStock factor row belongs to another stock")
        parsed.append({"dt": record["dividOperateDate"],
                       "factor": record["backAdjustFactor"]})
    frame = _factor_frame(parsed, end)
    cache_payload = {
        "version": _CACHE_VERSION, "code": code, "source": _FACTOR_SOURCE,
        "coverage_end": end.date().isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "rows": [{"dt": dt.date().isoformat(), "factor": factor}
                 for dt, factor in zip(frame["dt"], frame["factor"])],
    }
    _write_cache(path, cache_payload)
    return frame


def assert_raw_prices(provider: Any, source: str, raw_df: pd.DataFrame,
                      config: Any = None) -> bool:
    """Require evidence that bars are in exchange-price units.

    Known in-repository providers explicitly request raw data. Stored or
    unknown sources need an explicit storage declaration or external check.
    """
    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        raise PriceAdjustmentError("raw price frame is empty")
    name = str(source or "").strip().lower()
    known = {"tdx", "tushare", "baostock", "akshare", "jqdata"}
    declared = getattr(provider, "price_mode", None)
    frame_mode = raw_df.attrs.get("price_mode")
    if declared not in (None, "none") or frame_mode not in (None, "none"):
        raise PriceAdjustmentError("source declares adjusted prices; raw prices are required")
    if name in known and declared == "none":
        return True
    stored = _setting(config, "data_provider.stored_price_mode")
    stored = stored if stored is not None else raw_df.attrs.get("stored_price_mode")
    externally_verified = raw_df.attrs.get("raw_price_verified_external") is True
    if stored == "none" or externally_verified:
        return True
    raise PriceAdjustmentError(f"raw price mode is unverified for source {name or 'unknown'}")


class AdjustmentPlan:
    """Attach causal factors to raw bars and translate strategy price units."""

    def __init__(self, raw_df: pd.DataFrame, factors: pd.DataFrame | None, mode: str):
        self.mode = str(mode or "").lower()
        if self.mode not in {"none", "qfq", "hfq"}:
            raise PriceAdjustmentError(f"unsupported adjustment mode: {mode!r}")
        try:
            # Adjustment callers sometimes operate on price-only anchors.  Keep
            # volume and amount optional, while validating them when supplied.
            self.raw_df = validate_market_data(
                raw_df, require_volume=False, require_amount=False, sort=False
            )
        except MarketDataValidationError as exc:
            raise PriceAdjustmentError(f"invalid raw bars: {exc}") from exc
        if self.mode == "none" and factors is None:
            self.factors = None
            self._dates: list[pd.Timestamp] = []
            self._values: list[float] = []
            self.factor_source = None
            self.raw_df["__factor"] = 1.0
        else:
            if not isinstance(factors, pd.DataFrame) or not factors.attrs.get("verified"):
                raise PriceAdjustmentError("adjustment factors are not verified")
            if set(("dt", "factor")) - set(factors.columns):
                raise PriceAdjustmentError("factor frame needs dt and factor")
            if _day(factors.attrs.get("coverage_start"), "coverage_start") != _BASE_DATE:
                raise PriceAdjustmentError("factor query must cover the 1990 baseline")
            coverage_end = _day(factors.attrs.get("coverage_end"), "coverage_end")
            if coverage_end < self.raw_df["dt"].max().normalize():
                raise PriceAdjustmentError("factor coverage ends before the final raw bar")
            source = factors.attrs.get("source")
            if not source:
                raise PriceAdjustmentError("factor source is missing")
            rows = [{"dt": dt, "factor": factor}
                    for dt, factor in zip(factors["dt"], factors["factor"])]
            self.factors = _factor_frame(rows, coverage_end, source=str(source))
            self._dates = self.factors["dt"].tolist()
            self._values = self.factors["factor"].tolist()
            self.factor_source = str(source)
            first = self.raw_df["dt"].min().normalize()
            if first < self._dates[0]:
                raise PriceAdjustmentError("factor series has no prior anchor")
            self.raw_df["__factor"] = [self.factor_at(dt) for dt in self.raw_df["dt"]]
        first_factor = float(self.raw_df["__factor"].iloc[0])
        self.raw_df["__signal_multiplier"] = (
            self.raw_df["__factor"] / first_factor if self.mode == "hfq" else 1.0
        )
        self.frame = self.raw_df

    def factor_at(self, dt: Any) -> float:
        if self.factors is None:
            return 1.0
        date_value = _day(dt, "bar date")
        if date_value > _day(self.factors.attrs["coverage_end"], "coverage_end"):
            raise PriceAdjustmentError("bar is outside verified factor coverage")
        index = bisect_right(self._dates, date_value) - 1
        if index < 0:
            raise PriceAdjustmentError("bar precedes first factor anchor")
        return self._values[index]

    @staticmethod
    def rebase_ratio(previous_factor: Any, current_factor: Any) -> float:
        return _positive(previous_factor, "previous factor") / _positive(current_factor, "current factor")

    def signal_bar(self, raw_bar: Mapping[str, Any]) -> dict[str, Any]:
        if "dt" not in raw_bar:
            raise PriceAdjustmentError("signal bar lacks dt")
        factor = self.factor_at(raw_bar["dt"])
        multiplier = factor / float(self.raw_df["__factor"].iloc[0]) if self.mode == "hfq" else 1.0
        result = dict(raw_bar)
        for column in _OHLC:
            if column not in raw_bar:
                raise PriceAdjustmentError(f"signal bar lacks raw {column}")
            raw_price = _positive(raw_bar[column], f"raw {column}")
            result[f"raw_{column}"] = raw_price
            result[column] = raw_price * multiplier
        result.update(__factor=factor, __signal_multiplier=multiplier,
                      factor=factor, signal_multiplier=multiplier,
                      execution_price_mode="none", signal_price_mode=self.mode)
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "requested_mode": self.mode, "signal_price_mode": self.mode,
            "execution_price_mode": "none", "factor_source": self.factor_source,
            "factor_available": self.factors is not None,
            "causal_adjustment": True, "corporate_action_accounting": "unsupported",
            "raw_data_verified": False,
            "hfq_anchor_date": self.raw_df['dt'].iloc[0].isoformat() if self.mode == 'hfq' else None,
            "hfq_anchor_factor": float(self.raw_df['__factor'].iloc[0]) if self.mode == 'hfq' else None,
            "limitations": ["统一因子仅支持沪深 A 股股票；后复权按起始日归一，前复权按当时日期定基。",
                            "分红、税费、配股及送转到账尚未实现。"],
        }


def signal_to_execution(signal: Mapping[str, Any], signal_bar: Mapping[str, Any],
                        raw_bar: Mapping[str, Any]) -> dict[str, Any]:
    """Invert signal price scaling for a raw execution engine; preserve quantity."""
    if _day(signal_bar.get("dt"), "signal date") != _day(raw_bar.get("dt"), "raw date"):
        raise PriceAdjustmentError("signal and execution bars have different dates")
    multiplier = _positive(signal_bar.get("signal_multiplier",
                                          signal_bar.get("__signal_multiplier")),
                           "signal multiplier")
    result = dict(signal)
    for field in ("price", "stop_loss", "take_profit"):
        if field in result and result[field] is not None:
            if field != "price" and result[field] == 0:
                continue  # Built-in strategies use zero as a disabled stop/target.
            result[field] = _positive(result[field], field) / multiplier
    result["execution_price_mode"] = "none"
    return result


def rebase_strategy_prices(strategy: Any, code: str, ratio: Any) -> None:
    """Rebase qfq state only when the strategy has an audited/declared hook."""
    value = _positive(ratio, "rebase ratio")
    hook = getattr(strategy, "rebase_signal_prices", None)
    if not callable(hook):
        raise PriceAdjustmentError("strategy lacks rebase_signal_prices; use hfq")
    function = getattr(hook, "__func__", hook)
    inherited_default = getattr(function, "__qualname__", "") == "BaseImplementedStrategy.rebase_signal_prices"
    builtin = type(strategy).__module__ == "src.strategies.implemented_strategies"
    if inherited_default and not builtin and getattr(strategy, "supports_dynamic_qfq", False) is not True:
        raise PriceAdjustmentError(
            "custom strategy must rebase its own price state or explicitly set "
            "supports_dynamic_qfq=True; use hfq otherwise"
        )
    hook(code, value)
