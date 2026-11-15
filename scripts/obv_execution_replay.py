"""Causal execution replay for close-observed OBV resonance signals.

Each call is an independent cash account.  A signal observed at a completed
close can buy only the next market-session open; it is never carried forward
to a later stock bar.  Prices and accounting remain raw throughout.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.execution_engine import ExecutionEngine
from src.services.account_ledger import AccountLedger
from src.services.order_matcher import OrderMatcher
from src.services.risk_audit_log import RiskAuditLog
from src.utils.runtime_params import parameter_overrides


COSTS = {
    "trading_cost.commission_rate": 0.00025,
    "trading_cost.min_commission": 5.0,
    "trading_cost.stamp_duty": 0.0005,
    "trading_cost.transfer_fee": 0.00001,
    "execution.slippage": 0.001,
}

_MAIN_BOARD = re.compile(r"(?:60[0135]\d{3}\.SH|00[0123]\d{3}\.SZ)\Z")
_STRATEGY = "obv_resonance_replay"
_FACTOR_TOLERANCE = 1e-12


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value) if math.isfinite(float(value)) else False
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "t"}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _bar(code: str, day: str, source: dict[str, Any]) -> dict[str, Any]:
    """Copy a caller bar and provide the names expected by project services."""
    result = dict(source)
    result["code"] = code
    result["dt"] = day
    if "vol" not in result and "volume" in result:
        result["vol"] = result["volume"]
    if "volume" not in result and "vol" in result:
        result["volume"] = result["vol"]
    return result


def _factor(bar: dict[str, Any] | None) -> float | None:
    if not bar or "adjustment_factor" not in bar:
        return None
    value = _number(bar.get("adjustment_factor"), default=float("nan"))
    return value if math.isfinite(value) and value > 0 else None


def _factor_changed(left: float, right: float) -> bool:
    return not math.isclose(left, right, rel_tol=_FACTOR_TOLERANCE,
                            abs_tol=_FACTOR_TOLERANCE)


def _base_result(code: str, signal_day: str, horizon: int) -> dict[str, Any]:
    return {
        "code": code,
        "signal_date": signal_day,
        "entry_date": None,
        "scheduled_exit_date": None,
        "exit_date": None,
        "status": "no_entry",
        "rejection_reason": None,
        "horizon": horizon,
        "entry_price": None,
        "exit_price": None,
        "qty": 0,
        "invested": 0.0,
        "realized_pnl": None,
        "net_return_pct": None,
        "fees": 0.0,
        "delayed_sell_days": 0,
        "holding_days": 0,
        "economic_price_return_pct": None,
        "raw_reference_anomaly": False,
        "raw_reference_anomaly_dates": [],
        "exit_reason": None,
    }


def _reject(result: dict[str, Any], reason: str) -> dict[str, Any]:
    result["rejection_reason"] = reason
    return result


def _entry_rejection(bar: dict[str, Any] | None) -> str | None:
    if bar is None:
        return "missing_next_market_bar"
    volume = _number(bar.get("vol", bar.get("volume", 0.0)))
    if volume <= 0 or _truthy(bar.get("is_suspended")):
        return "suspended_or_no_volume"
    # Entry happens at the open.  When supplied, this field is based only on
    # the prior-known raw reference and opening price; the completed day's
    # high/low must not be used to retroactively erase an opening fill.
    opening_reference_valid = bar.get(
        "open_price_reference_valid", bar.get("price_reference_valid"))
    if not _truthy(opening_reference_valid):
        return "unverified_price_reference"
    if int(_number(bar.get("history_sessions"), -1)) < 120:
        return "insufficient_history_sessions"
    opening = _number(bar.get("open"))
    upper = _number(bar.get("up_limit"))
    if opening <= 0:
        return "invalid_open"
    if upper <= 0:
        return "unverified_upper_limit"
    if opening >= upper - 1e-8 or _truthy(bar.get("is_limit_up_at_open")):
        return "open_at_limit_up"
    return None


def _affordable_qty(account: AccountLedger, fill_price: float, capital: float) -> int:
    """Largest fee-aware 100-share quantity that cash can actually settle."""
    high = max(0, int(capital / fill_price) // 100)
    low = 0
    while low < high:
        mid = (low + high + 1) // 2
        qty = mid * 100
        amount = fill_price * qty
        cost, _, _, _ = account.calculate_cost(amount, "BUY", fill_price, qty)
        if amount + cost <= capital + 1e-9:
            low = mid
        else:
            high = mid - 1
    return low * 100


def _sell_order(code: str, qty: int, mode: str) -> dict[str, Any]:
    return {
        "strategy_id": _STRATEGY,
        "code": code,
        "direction": "SELL",
        "qty": qty,
        "execution_mode": mode,
        "type": "MARKET",
    }


def replay_trade(
    code: str,
    signal_day: str,
    bars: dict[str, dict[str, Any]],
    calendar: list[str],
    horizon: int,
    capital: float = 100000.0,
    slippage: float = 0.001,
    *,
    exit_on_factor_loss: bool = False,
) -> dict[str, Any]:
    """Replay one OBV signal through the project's matcher, engine, and ledger.

    Fixed-horizon exits use the close ``horizon`` market sessions after entry.
    With ``exit_on_factor_loss``, the first completed close where the factor is
    false plans an exit for the next market open, while the close cap is 20
    sessions (or the requested horizon when shorter).
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be an integer >= 1")
    if not math.isfinite(float(capital)) or capital <= 0:
        raise ValueError("capital must be finite and positive")
    if not math.isfinite(float(slippage)) or not 0 <= slippage < 1:
        raise ValueError("slippage must be finite and in [0, 1)")
    if len(calendar) != len(set(calendar)) or calendar != sorted(calendar):
        raise ValueError("calendar must contain unique ascending ISO dates")

    result = _base_result(code, signal_day, horizon)
    if not _MAIN_BOARD.fullmatch(str(code)):
        return _reject(result, "out_of_scope_security")
    if signal_day not in calendar:
        return _reject(result, "signal_not_in_market_calendar")
    signal_index = calendar.index(signal_day)
    if signal_index + 1 >= len(calendar):
        return _reject(result, "next_market_day_out_of_range")

    signal_source = bars.get(signal_day)
    if signal_source is None:
        return _reject(result, "missing_signal_bar")
    signal_bar = _bar(code, signal_day, signal_source)
    if not _truthy(signal_bar.get("obv_resonance_factor")):
        return _reject(result, "signal_factor_not_active")
    signal_factor = _factor(signal_bar)
    if signal_factor is None:
        return _reject(result, "unverified_signal_adjustment_factor")

    entry_index = signal_index + 1
    entry_day = calendar[entry_index]
    entry_source = bars.get(entry_day)
    entry_bar = None if entry_source is None else _bar(code, entry_day, entry_source)
    if reason := _entry_rejection(entry_bar):
        return _reject(result, reason)
    assert entry_bar is not None
    entry_factor = _factor(entry_bar)
    if entry_factor is None:
        return _reject(result, "unverified_entry_adjustment_factor")
    if _factor_changed(signal_factor, entry_factor):
        return _reject(result, "corporate_action_on_entry")

    costs = dict(COSTS)
    costs["execution.slippage"] = float(slippage)
    account = AccountLedger(float(capital))
    matcher = OrderMatcher()
    engine = ExecutionEngine(account, matcher, RiskAuditLog())
    buy_probe = {
        "strategy_id": _STRATEGY,
        "code": code,
        "direction": "BUY",
        "qty": 100,
        "execution_mode": "at_open",
        "type": "MARKET",
    }

    with parameter_overrides(costs):
        match_ok, expected_fill = matcher.match_order(buy_probe, entry_bar)
        if not match_ok:
            return _reject(result, "entry_unfillable")
        qty = _affordable_qty(account, expected_fill, float(capital))
        if qty < 100:
            return _reject(result, "insufficient_cash_for_fee_aware_lot")
        buy_order = dict(buy_probe, qty=qty)
        if not engine.execute_order(_STRATEGY, buy_order, entry_bar, account=account):
            return _reject(result, "entry_execution_rejected")

        bought = account.transactions[-1]
        result.update(
            entry_date=entry_day,
            entry_price=float(bought["price"]),
            qty=int(bought["quantity"]),
            invested=float(bought["amount"] + bought["cost"]),
            fees=float(bought["cost"]),
            rejection_reason=None,
        )

        # The opening decision above remains valid.  A later raw-reference
        # anomaly on the same completed bar censors interpretation from this
        # point forward while retaining the entry and its cash cost.
        if not _truthy(entry_bar.get("price_reference_valid")):
            result.update(
                status="censored_unverified_price_reference",
                rejection_reason="raw_price_reference_anomaly",
                raw_reference_anomaly=True,
                raw_reference_anomaly_dates=[entry_day],
            )
            return result

        cap = min(horizon, 20) if exit_on_factor_loss else horizon
        planned_index = entry_index + cap
        if planned_index >= len(calendar):
            if not exit_on_factor_loss:
                result.update(status="censored_data_end", rejection_reason="fixed_exit_beyond_calendar")
                return result
            # An earlier factor-loss exit may still occur inside the observed
            # tail.  Keep scanning causally; without one, the result is censored.
            fixed_exit_day = None
        else:
            fixed_exit_day = calendar[planned_index]
            result["scheduled_exit_date"] = fixed_exit_day

        pending_index: int | None = None
        pending_reason: str | None = None
        attempts_after_plan = 0
        max_index = min(len(calendar) - 1, planned_index + 5)

        # The entry was decided before today's close.  If resonance is gone at
        # that close, T+1 permits an order only at the next market-session open.
        if exit_on_factor_loss and not _truthy(entry_bar.get("obv_resonance_factor")):
            if entry_index + 1 >= len(calendar):
                result.update(status="censored_data_end",
                              rejection_reason="factor_loss_exit_beyond_calendar",
                              exit_reason="factor_loss_next_open")
                return result
            pending_index = entry_index + 1
            pending_reason = "factor_loss_next_open"
            result["scheduled_exit_date"] = calendar[pending_index]

        for index in range(entry_index + 1, max_index + 1):
            day = calendar[index]
            source = bars.get(day)
            current = None if source is None else _bar(code, day, source)
            opening_exit_due = ((pending_index is not None and index >= pending_index)
                                or index > planned_index)

            if current is not None:
                current_factor = _factor(current)
                if current_factor is None:
                    result.update(status="censored_unverified_adjustment_factor",
                                  rejection_reason="missing_or_invalid_adjustment_factor",
                                  holding_days=index - entry_index)
                    return result
                reference_key = "open_price_reference_valid" if opening_exit_due and "open_price_reference_valid" in current else "price_reference_valid"
                if not _truthy(current.get(reference_key)):
                    result["raw_reference_anomaly"] = True
                    result["raw_reference_anomaly_dates"].append(day)
                if _factor_changed(entry_factor, current_factor):
                    result.update(status="censored_corporate_action",
                                  rejection_reason="adjustment_factor_changed_while_holding",
                                  holding_days=index - entry_index)
                    return result
                if result["raw_reference_anomaly"]:
                    result.update(status="censored_unverified_price_reference",
                                  rejection_reason="raw_price_reference_anomaly",
                                  holding_days=index - entry_index)
                    return result

            # Factor loss is observed only after this day's close, so its
            # earliest eligible order is the following market open.
            if (exit_on_factor_loss and pending_index is None and index < planned_index
                    and current is not None
                    and not _truthy(current.get("obv_resonance_factor"))):
                if index + 1 >= len(calendar):
                    result.update(status="censored_data_end",
                                  rejection_reason="factor_loss_exit_beyond_calendar",
                                  holding_days=index - entry_index,
                                  scheduled_exit_date=None,
                                  exit_reason="factor_loss_next_open")
                    return result
                pending_index = index + 1
                pending_reason = "factor_loss_next_open"
                result["scheduled_exit_date"] = calendar[pending_index]

            due_index = pending_index if pending_index is not None else planned_index
            if index < due_index:
                continue
            mode = "at_open" if pending_index is not None or index > planned_index else "at_close"
            reason = pending_reason or ("fixed_horizon_close" if index == planned_index
                                        else "delayed_fixed_horizon_open")
            if index > due_index:
                attempts_after_plan = index - due_index
                reason = "delayed_" + (pending_reason or "fixed_horizon_open")

            if current is not None:
                sell = _sell_order(code, qty, mode)
                if engine.execute_order(_STRATEGY, sell, current, account=account):
                    sold = account.transactions[-1]
                    pnl = float(account.cash - capital)
                    assert abs(pnl - float(sold["pnl"])) <= max(1e-7, abs(pnl) * 1e-10)
                    fees = float(account.total_commission + account.total_stamp_duty
                                 + account.total_transfer_fee)
                    exit_price = float(sold["price"])
                    entry_price = float(result["entry_price"])
                    result.update(
                        status="closed",
                        rejection_reason=None,
                        exit_date=day,
                        exit_price=exit_price,
                        realized_pnl=pnl,
                        net_return_pct=pnl / float(capital) * 100.0,
                        fees=fees,
                        delayed_sell_days=attempts_after_plan,
                        holding_days=index - entry_index,
                        economic_price_return_pct=(exit_price / entry_price - 1.0) * 100.0,
                        exit_reason=reason,
                    )
                    return result

            if index - due_index >= 5:
                result.update(status="censored_unresolved_exit",
                              rejection_reason="sell_unfilled_after_5_market_days",
                              delayed_sell_days=5,
                              holding_days=index - entry_index,
                              exit_reason=pending_reason or "fixed_horizon_exit")
                return result

        # The calendar ended before five retry sessions were observable.
        result.update(status="censored_data_end",
                      rejection_reason="sell_retry_window_truncated_by_calendar",
                      delayed_sell_days=max(0, min(5, len(calendar) - 1 - (pending_index or planned_index))),
                      holding_days=len(calendar) - 1 - entry_index,
                      exit_reason=pending_reason or "fixed_horizon_exit")
        return result


__all__ = ["COSTS", "replay_trade"]
