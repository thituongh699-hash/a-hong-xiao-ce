"""Causal shared-account replay for the fixed low-frequency signal matrices."""
from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.execution_engine import ExecutionEngine
from src.services.account_ledger import AccountLedger
from src.services.order_matcher import OrderMatcher
from src.services.risk_audit_log import RiskAuditLog
from src.utils.runtime_params import parameter_overrides


_STRATEGY = "lowfreq_portfolio"
_COSTS = {
    "trading_cost.commission_rate": .00025,
    "trading_cost.min_commission": 5.,
    "trading_cost.stamp_duty": .0005,
    "trading_cost.transfer_fee": .00001,
}


def _frame(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records, columns=columns)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _matrix(data: Mapping[str, Any], key: str, shape: tuple[int, int], dtype=None):
    if key not in data:
        raise ValueError(f"data lacks {key}")
    value = np.asarray(data[key], dtype=dtype)
    if value.shape != shape:
        raise ValueError(f"{key} must have shape {shape}")
    return value


def _signal_matrix(data: Mapping[str, Any], key: str, spec_id: str,
                   shape: tuple[int, int], dtype=None):
    group = data.get(key)
    if not isinstance(group, Mapping) or spec_id not in group:
        raise ValueError(f"data lacks {key}[{spec_id!r}]")
    value = np.asarray(group[spec_id], dtype=dtype)
    if value.shape != shape:
        raise ValueError(f"{key}[{spec_id!r}] must have shape {shape}")
    return value


def _affordable_qty(account: AccountLedger, price: float, budget: float) -> int:
    high = max(0, int(budget / price) // 100)
    low = 0
    while low < high:
        mid = (low + high + 1) // 2
        qty = mid * 100
        amount = price * qty
        fees = account.calculate_cost(amount, "BUY", price, qty)[0]
        if amount + fees <= budget + 1e-9:
            low = mid
        else:
            high = mid - 1
    return low * 100


def _opening_bar(code: str, day: pd.Timestamp, index: int, stock: int,
                 arrays: Mapping[str, np.ndarray]) -> dict[str, Any] | None:
    opening = arrays["open"][index, stock]
    upper = arrays["up_limit"][index, stock]
    lower = arrays["down_limit"][index, stock]
    if not (_finite(opening) and float(opening) > 0
            and _finite(upper) and float(upper) > 0
            and _finite(lower) and float(lower) > 0
            and bool(arrays["open_valid"][index, stock])):
        return None
    # Opening execution must not inspect the completed current-day H/L/C.
    return {
        "code": code, "dt": day, "open": float(opening),
        "high": float(opening), "low": float(opening), "close": float(opening),
        # The matcher only needs a positive tradability sentinel.  Actual daily
        # volume is completed-bar information and is validated after the open.
        "vol": 1., "volume": 1.,
        "up_limit": float(upper), "down_limit": float(lower),
    }


def _event_index(events: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    normalized: dict[str, list[dict[str, Any]]] = {}
    for code, rows in (events or {}).items():
        if not isinstance(rows, list):
            raise ValueError("events values must be lists")
        parsed = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("event rows must be mappings")
            announcement = pd.to_datetime(row.get("announcement_date"), errors="coerce")
            ex_date = pd.to_datetime(row.get("ex_date"), errors="coerce")
            if pd.isna(announcement) or pd.isna(ex_date):
                raise ValueError("events require valid announcement_date and ex_date")
            parsed.append({**dict(row), "announcement_date": announcement.normalize(),
                           "ex_date": ex_date.normalize()})
        normalized[str(code)] = parsed
    return normalized


def simulate_portfolio(
    data: Mapping[str, Any], spec: Mapping[str, Any], start: str, end: str,
    events: Mapping[str, Any] | None = None, capital: float = 50000.,
    slippage: float = .001, entry_delay: int = 0,
) -> dict[str, Any]:
    """Replay one signal specification in one cash account.

    Result keys are ``summary``, ``nav``, ``orders``, ``trades``, ``skips``,
    and ``blockers``.  Signals and completed-close exits are observed at close;
    every order is attempted only at a later market open.
    """
    if not isinstance(data, Mapping) or not isinstance(spec, Mapping):
        raise ValueError("data and spec must be mappings")
    spec_id = str(spec.get("id", ""))
    if not spec_id:
        raise ValueError("spec id required")
    if not _finite(capital) or float(capital) <= 0:
        raise ValueError("capital must be finite and positive")
    if not _finite(slippage) or not 0 <= float(slippage) < 1:
        raise ValueError("slippage must be in [0, 1)")
    if isinstance(entry_delay, bool) or not isinstance(entry_delay, int) or entry_delay < 0:
        raise ValueError("entry_delay must be a nonnegative integer")

    raw_dates = list(data.get("dates", []))
    codes = [str(x) for x in data.get("codes", [])]
    dates = pd.DatetimeIndex(pd.to_datetime(raw_dates, errors="coerce")).normalize()
    if dates.isna().any() or len(dates) == 0 or not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("dates must be unique ascending ISO dates")
    if raw_dates != [x.strftime("%Y-%m-%d") for x in dates]:
        raise ValueError("dates must be canonical ISO strings")
    if not codes or codes != sorted(set(codes)):
        raise ValueError("codes must be sorted unique strings")
    shape = (len(dates), len(codes))
    arrays = {key: _matrix(data, key, shape, bool if key in {
        "factor_changed", "open_valid", "day_valid"} else float)
        for key in ("open", "close", "vol", "up_limit", "down_limit",
                    "factor_changed", "open_valid", "day_valid")}
    market_ok = np.asarray(data.get("market_ok"), dtype=bool)
    week_last = np.asarray(data.get("week_last"), dtype=bool)
    if market_ok.shape != (len(dates),) or week_last.shape != (len(dates),):
        raise ValueError("market_ok and week_last must have length T")
    entry = _signal_matrix(data, "entry", spec_id, shape, bool)
    score = _signal_matrix(data, "score", spec_id, shape, float)
    exit_signal = _signal_matrix(data, "exit", spec_id, shape, bool)
    event_rows = _event_index(events)

    requested_start = pd.Timestamp(start).normalize()
    requested_end = pd.Timestamp(end).normalize()
    if pd.isna(requested_start) or pd.isna(requested_end) or requested_start > requested_end:
        raise ValueError("valid start <= end required")
    selected = np.flatnonzero((dates >= requested_start) & (dates <= requested_end))
    if len(selected) == 0:
        raise ValueError("requested period has no market sessions")

    account = AccountLedger(float(capital))
    engine = ExecutionEngine(account, OrderMatcher(), RiskAuditLog())
    positions: dict[str, dict[str, Any]] = {}
    last_marks: dict[str, float] = {}
    last_mark_dates: dict[str, pd.Timestamp] = {}
    nav_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    operations: dict[tuple[int, int], int] = {}
    halted = False
    actual_end: pd.Timestamp | None = None
    costs = {**_COSTS, "execution.slippage": float(slippage)}

    def skip(day, code, reason, **extra):
        skip_rows.append({"date": day.strftime("%Y-%m-%d"), "code": code,
                          "reason": reason, **extra})

    def block(day, code, reason):
        nonlocal halted
        blockers.append({"code": code, "year": int(day.year), "reason": reason,
                         "date": day.strftime("%Y-%m-%d")})
        halted = True

    def visible_events(code: str, previous_day: pd.Timestamp):
        return [row for row in event_rows.get(code, [])
                if row["announcement_date"] <= previous_day]

    with parameter_overrides(costs):
        for index in selected:
            day = dates[index]
            week = (int(day.isocalendar().year), int(day.isocalendar().week))
            operations.setdefault(week, 0)
            prior_nav = float(nav_rows[-1]["nav"]) if nav_rows else float(capital)
            cash_at_open = float(account.cash)
            previous_market_day = dates[index - 1] if index > 0 else day - pd.Timedelta(days=1)

            # A factor change on a carried holding makes the raw-price account
            # incomplete.  Stop before any order can obscure that fact.
            for code in sorted(positions):
                stock = codes.index(code)
                if arrays["factor_changed"][index, stock]:
                    block(day, code, "corporate_action")
                    break
            if halted:
                actual_end = day
                break

            # Corporate actions known by the previous market close force the
            # earliest possible pre-ex-date opening exit.
            for code, pos in positions.items():
                for event in visible_events(code, previous_market_day):
                    if day < event["ex_date"]:
                        # Keep the original due date and retry history once a
                        # corporate-action exit is pending.  Re-observing the
                        # same announcement on later days must not make an
                        # unfillable exit look like a fresh first attempt.
                        if pos.get("pending_reason") != "announced_corporate_action":
                            pos["pending_reason"] = "announced_corporate_action"
                            due = pos.get("pending_index")
                            pos["pending_index"] = index if due is None else min(due, index)
                        break

            # Maximum holding time is counted in market sessions and is due at
            # the open, independently of whether the stock had usable bars.
            for pos in positions.values():
                if (pos.get("pending_reason") is None
                        and index - pos["entry_index"] >= int(spec["max_hold"])):
                    pos["pending_reason"] = "max_hold"
                    pos["pending_index"] = index
                    pos["failed_exit_attempts"] = 0

            # Exits always run before entries.  Their proceeds are deliberately
            # absent from cash_at_open and cannot finance same-day buys.
            for code in sorted(list(positions)):
                pos = positions[code]
                if pos.get("pending_reason") is None or index < pos["pending_index"]:
                    continue
                stock = codes.index(code)
                bar = _opening_bar(code, day, index, stock, arrays)
                if operations[week] >= 4:
                    block(day, code, "weekly_operation_limit_incomplete_account")
                    break
                before = float(account.cash)
                sell = {"strategy_id": _STRATEGY, "code": code, "direction": "SELL",
                        "qty": pos["qty"], "execution_mode": "at_open", "type": "MARKET"}
                success = bar is not None and engine.execute_order(
                    _STRATEGY, sell, bar, account=account)
                if success:
                    tx = account.transactions[-1]
                    operations[week] += 1
                    order_rows.append({
                        "date": day.strftime("%Y-%m-%d"), "code": code,
                        "direction": "SELL", "qty": int(tx["quantity"]),
                        "price": float(tx["price"]), "fees": float(tx["cost"]),
                        "cash_before": before, "cash_after": float(account.cash),
                        "reason": pos["pending_reason"], "signal_date": None,
                    })
                    trade_rows.append({
                        "code": code, "entry_date": pos["entry_date"],
                        "exit_date": day.strftime("%Y-%m-%d"),
                        "entry_price": pos["entry_price"], "exit_price": float(tx["price"]),
                        "qty": int(tx["quantity"]), "entry_fee": pos["entry_fee"],
                        "exit_fee": float(tx["cost"]),
                        "fees": pos["entry_fee"] + float(tx["cost"]),
                        "pnl": float(tx["pnl"]),
                        "holding_sessions": index - pos["entry_index"],
                        "reason": pos["pending_reason"],
                    })
                    del positions[code]
                else:
                    pos["failed_exit_attempts"] += 1
                    skip(day, code, "exit_unfillable", direction="SELL",
                         exit_reason=pos["pending_reason"],
                         failed_attempts=pos["failed_exit_attempts"])
                    if pos["failed_exit_attempts"] > 5:
                        block(day, code, "exit_unfilled_after_5_retries")
                        break
            if halted:
                actual_end = day
                break

            source_index = index - 1 - entry_delay
            if source_index >= 0:
                candidates: list[tuple[float, str, int]] = []
                if (not bool(spec.get("weekly_only", False)) or week_last[source_index]) \
                        and bool(market_ok[source_index]):
                    for stock, code in enumerate(codes):
                        if entry[source_index, stock] and _finite(score[source_index, stock]):
                            candidates.append((float(score[source_index, stock]), code, stock))
                candidates.sort(key=lambda row: (-row[0], row[1]))
                signal_day = dates[source_index].strftime("%Y-%m-%d")
                for rank, (candidate_score, code, stock) in enumerate(candidates[:20], 1):
                    base = {"signal_date": signal_day, "score": candidate_score, "rank": rank}
                    if code in positions:
                        skip(day, code, "already_held", **base)
                        continue
                    if len(positions) >= 2:
                        skip(day, code, "position_limit", **base)
                        continue
                    if operations[week] + len(positions) + 2 > 4:
                        skip(day, code, "weekly_exit_capacity_reserved", **base)
                        continue
                    if arrays["factor_changed"][index, stock]:
                        skip(day, code, "factor_changed_on_entry_day", **base)
                        continue
                    blacked_out = any(day <= event["ex_date"]
                                      for event in visible_events(code, previous_market_day))
                    if blacked_out:
                        skip(day, code, "corporate_action_blackout", **base)
                        continue
                    bar = _opening_bar(code, day, index, stock, arrays)
                    if bar is None:
                        skip(day, code, "invalid_or_missing_open", **base)
                        continue
                    probe = {"strategy_id": _STRATEGY, "code": code, "direction": "BUY",
                             "qty": 100, "execution_mode": "at_open", "type": "MARKET"}
                    match, fill = engine.order_matcher.match_order(probe, bar)
                    if not match:
                        skip(day, code, "opening_unfillable", **base)
                        continue
                    budget = min(.45 * prior_nav, cash_at_open - .10 * prior_nav)
                    budget = min(budget, float(account.cash))
                    qty = _affordable_qty(account, float(fill), max(0., budget))
                    if qty < 100:
                        skip(day, code, "insufficient_fee_aware_lot_budget", **base)
                        continue
                    before = float(account.cash)
                    if not engine.execute_order(_STRATEGY, {**probe, "qty": qty}, bar,
                                                account=account):
                        skip(day, code, "execution_rejected", **base)
                        continue
                    tx = account.transactions[-1]
                    operations[week] += 1
                    cash_at_open -= float(tx["amount"] + tx["cost"])
                    positions[code] = {
                        "code": code, "entry_date": day.strftime("%Y-%m-%d"),
                        "entry_index": index, "entry_price": float(tx["price"]),
                        "entry_fee": float(tx["cost"]), "qty": int(tx["quantity"]),
                        "peak_close": None, "pending_reason": None,
                        "pending_index": None, "failed_exit_attempts": 0,
                    }
                    order_rows.append({
                        "date": day.strftime("%Y-%m-%d"), "code": code,
                        "direction": "BUY", "qty": int(tx["quantity"]),
                        "price": float(tx["price"]), "fees": float(tx["cost"]),
                        "cash_before": before, "cash_after": float(account.cash),
                        "reason": "entry_signal", "signal_date": signal_day,
                    })

            # Validate and consume completed closes only after all opening fills.
            for code in sorted(positions):
                stock = codes.index(code)
                close = arrays["close"][index, stock]
                volume = arrays["vol"][index, stock]
                # A suspension has no new executable/observable close.  Some
                # vendors repeat the prior close on zero-volume rows, so retain
                # the last genuine raw mark instead of treating it as new.
                if not _finite(volume) or float(volume) <= 0:
                    if positions[code]["entry_index"] == index:
                        block(day, code, "invalid_completed_bar")
                        break
                    continue
                if not arrays["day_valid"][index, stock]:
                    if _finite(close):
                        block(day, code, "invalid_completed_bar")
                        break
                    continue
                if not _finite(close) or float(close) <= 0:
                    continue
                close = float(close)
                last_marks[code] = close
                last_mark_dates[code] = day
                pos = positions[code]
                if pos.get("pending_reason") is None:
                    reason = None
                    if close < pos["entry_price"] * (1 - float(spec["stop"])):
                        reason = "initial_stop"
                    elif (spec.get("trail") is not None and pos["peak_close"] is not None
                          and close < pos["peak_close"] * (1 - float(spec["trail"]))):
                        reason = "trailing_stop"
                    elif (spec.get("take") is not None
                          and close > pos["entry_price"] * (1 + float(spec["take"]))):
                        reason = "take_profit"
                    elif exit_signal[index, stock]:
                        reason = "exit_signal"
                    if reason is not None:
                        pos["pending_reason"] = reason
                        pos["pending_index"] = index + 1
                        pos["failed_exit_attempts"] = 0
                pos["peak_close"] = close if pos["peak_close"] is None else max(
                    pos["peak_close"], close)
            if halted:
                actual_end = day
                break

            holdings = 0.
            stale = []
            for code, pos in positions.items():
                if code not in last_marks:
                    stale.append(code)
                    continue
                holdings += pos["qty"] * last_marks[code]
                if last_mark_dates.get(code) is None or last_mark_dates[code] < day:
                    stale.append(code)
            nav_value = float(account.cash + holdings)
            account.update_daily_nav(day, holdings)
            nav_rows.append({"date": day.strftime("%Y-%m-%d"), "nav": nav_value,
                             "cash": float(account.cash), "holdings": holdings,
                             "open_positions": len(positions),
                             "stale_marks": sorted(stale)})
            actual_end = day

    if actual_end is None:
        actual_end = dates[selected[0]]
    # A halt before close still gets an auditable mark using the last completed
    # raw close.  Unknown marks make the account incomplete and NAV undefined.
    if halted and (not nav_rows or nav_rows[-1]["date"] != actual_end.strftime("%Y-%m-%d")):
        holdings = 0.
        missing = []
        for code, pos in positions.items():
            if code not in last_marks:
                missing.append(code)
            else:
                holdings += pos["qty"] * last_marks[code]
        nav_rows.append({"date": actual_end.strftime("%Y-%m-%d"),
                         "nav": np.nan if missing else float(account.cash + holdings),
                         "cash": float(account.cash), "holdings": np.nan if missing else holdings,
                         "open_positions": len(positions), "stale_marks": sorted(missing)})

    nav = _frame(nav_rows, ["date", "nav", "cash", "holdings", "open_positions", "stale_marks"])
    orders = _frame(order_rows, ["date", "code", "direction", "qty", "price", "fees",
                                      "cash_before", "cash_after", "reason", "signal_date"])
    trades = _frame(trade_rows, ["code", "entry_date", "exit_date", "entry_price",
                                  "exit_price", "qty", "entry_fee", "exit_fee", "fees",
                                  "pnl", "holding_sessions", "reason"])
    skips = pd.DataFrame.from_records(skip_rows)
    end_nav = float(nav.iloc[-1]["nav"]) if len(nav) and _finite(nav.iloc[-1]["nav"]) else np.nan
    full_period = (not halted and actual_end == dates[selected[-1]]
                   and requested_end <= dates[-1] and _finite(end_nav))
    total_return = end_nav / float(capital) - 1 if _finite(end_nav) else np.nan
    elapsed = (actual_end - requested_start).days
    cagr = ((end_nav / float(capital)) ** (365.25 / elapsed) - 1
            if full_period and elapsed > 0 and end_nav > 0 else np.nan)
    if len(nav):
        nav_values = np.r_[float(capital), pd.to_numeric(nav["nav"], errors="coerce").to_numpy()]
        peaks = np.maximum.accumulate(np.where(np.isfinite(nav_values), nav_values, -np.inf))
        dd = np.where(np.isfinite(nav_values), nav_values / peaks - 1, np.nan)
        max_drawdown = float(np.nanmin(dd))
    else:
        max_drawdown = np.nan
    closed_pnl = float(trades["pnl"].sum()) if len(trades) else 0.
    open_pnl = 0.
    reconciliation_ok = True
    open_positions = []
    for code, pos in sorted(positions.items()):
        mark = last_marks.get(code)
        if mark is None:
            reconciliation_ok = False
            unrealized = np.nan
        else:
            unrealized = (mark - pos["entry_price"]) * pos["qty"] - pos["entry_fee"]
            open_pnl += unrealized
        open_positions.append({"code": code, "entry_date": pos["entry_date"],
                               "entry_price": pos["entry_price"], "qty": pos["qty"],
                               "mark": mark,
                               "stale": (code not in last_marks
                                         or last_mark_dates[code] < actual_end),
                               "unrealized_pnl": unrealized,
                               "pending_reason": pos.get("pending_reason")})
    reconciliation_error = (end_nav - float(capital) - closed_pnl - open_pnl
                            if _finite(end_nav) and reconciliation_ok else np.nan)
    if _finite(reconciliation_error) and abs(reconciliation_error) > 1e-6:
        blockers.append({"code": "ACCOUNT", "year": int(actual_end.year),
                         "reason": "incomplete_account", "date": actual_end.strftime("%Y-%m-%d")})
        full_period = False
        cagr = np.nan

    buy_count = int((orders["direction"] == "BUY").sum()) if len(orders) else 0
    sell_count = int((orders["direction"] == "SELL").sum()) if len(orders) else 0
    summary = {
        "valid": bool(full_period), "completed": bool(full_period),
        "requested_start": requested_start.strftime("%Y-%m-%d"),
        "requested_end": requested_end.strftime("%Y-%m-%d"),
        "actual_end": actual_end.strftime("%Y-%m-%d"),
        "initial_capital": float(capital), "cash": float(account.cash),
        "nav": end_nav, "end_nav": end_nav, "total_return": total_return,
        "cagr": cagr, "year_performance": total_return if full_period else np.nan,
        "max_drawdown": max_drawdown, "trades_closed": len(trades),
        "buy_count": buy_count, "sell_count": sell_count,
        "buy_fills": buy_count, "sell_fills": sell_count,
        "max_weekly_operations": max(operations.values(), default=0),
        "fees": float(account.total_commission + account.total_stamp_duty
                      + account.total_transfer_fee),
        "open_positions": open_positions,
        "net_pnl_reconciliation_error": reconciliation_error,
    }
    return {"summary": summary, "nav": nav, "orders": orders, "trades": trades,
            "skips": skips, "blockers": blockers}


__all__ = ["simulate_portfolio"]
