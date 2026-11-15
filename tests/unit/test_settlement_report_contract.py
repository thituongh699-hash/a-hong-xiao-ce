"""Settlement reporting keeps NAV, executions, and account reconciliation explicit."""

from datetime import datetime

import pandas as pd
import pytest

from src.services.performance_reporter import PerformanceReporter


class Ledger:
    def __init__(self, nav=(), transactions=()):
        self.nav = list(nav)
        self.transactions = list(transactions)

    def get_nav_history(self):
        return pd.DataFrame(self.nav)


class Audit:
    def get_rejection_count(self, _):
        return 0

    def get_violation_count(self, _):
        return 0

    def get_circuit_break_count(self, _):
        return 0


def summary(nav=(), transactions=()):
    return PerformanceReporter().generate_report(
        "s", Ledger(nav, transactions), Audit(), 100_000,
    )


def test_mixed_iso_nav_is_complete_and_invalid_date_fails_closed():
    nav = [
        {"dt": "2026-01-02T10:00:00", "nav": 100_000},
        {"dt": "2026-01-02T15:00:00", "nav": 90_000},
        {"dt": "2026-01-05", "nav": 99_000},
        {"dt": "2026-01-06T15:00:00", "nav": 89_100},
    ]
    result = summary(nav)
    assert result["final_capital"] == pytest.approx(89_100)
    assert result["roi"] == pytest.approx(-0.109)
    assert result["max_dd"] == pytest.approx(0.109)

    with pytest.raises(ValueError, match="invalid dt"):
        summary(nav + [{"dt": "not-an-iso-date", "nav": 123_456}])


def test_breakeven_is_not_a_loss_and_breaks_a_losing_streak():
    trades = [
        {"strategy_id": "s", "direction": "SELL", "dt": datetime(2026, 1, 2), "pnl": 100},
        {"strategy_id": "s", "direction": "SELL", "dt": datetime(2026, 1, 3), "pnl": -50},
        {"strategy_id": "s", "direction": "SELL", "dt": datetime(2026, 1, 4), "pnl": 0},
        {"strategy_id": "s", "direction": "SELL", "dt": datetime(2026, 1, 5), "pnl": -20},
    ]
    result = summary(transactions=trades)
    assert result["total_trades"] == 4
    assert result["trade_count"] == 4
    assert result["execution_count"] == 4
    assert result["buy_count"] == 0
    assert result["sell_count"] == 4
    assert result["breakeven_num"] == 1
    assert result["avg_loss"] == pytest.approx(35)
    assert result["max_loss_streak"] == 1

    formatted = PerformanceReporter().generate_backtest_report("s", trades, 100_000)
    assert formatted["trade_count"] == 4
    assert formatted["execution_count"] == 4
    assert formatted["loss_num"] == 2
    assert formatted["breakeven_num"] == 1
    assert formatted["avg_loss"] == pytest.approx(35)
    assert formatted["max_loss_streak"] == 1


def test_execution_counts_distinguish_buy_only_from_no_execution():
    buy = {
        "strategy_id": "s", "direction": "buy", "dt": datetime(2026, 1, 2),
        "price": 10, "quantity": 100, "amount": 1000, "cost": 5,
    }
    formatted = PerformanceReporter().generate_backtest_report("s", [buy], 100_000)
    assert formatted["trade_count"] == 0
    assert formatted["execution_count"] == 1
    assert formatted["buy_count"] == 1
    assert formatted["sell_count"] == 0
    assert formatted["breakeven_num"] == 0

    empty = PerformanceReporter().generate_backtest_report("s", [], 100_000)
    assert empty["execution_count"] == empty["buy_count"] == empty["sell_count"] == 0


def test_settlement_snapshot_is_validated_and_returned_without_recomputing_fees():
    settlement = {
        "as_of": "2026-01-31T15:00:00",
        "state": "open",
        "cash": 79_995.0,
        "holdings_value": 21_000.0,
        "open_position_count": 1,
        "open_quantity": 2_000,
        "realized_pnl": 100.0,
        "unrealized_pnl": 900.0,
        "fees_paid": 5.0,
        "reconciliation_delta": 0.0,
        "positions": [{"code": "000001.SZ", "quantity": 2_000}],
    }
    formatted = PerformanceReporter().generate_backtest_report(
        "s", [], 100_000, settlement=settlement,
    )
    assert formatted["settlement_schema_version"] == 1
    assert formatted["settlement"] == settlement
    assert formatted["settlement"]["fees_paid"] == 5.0
    assert formatted["realized_pnl"] == 0.0  # Trade-derived compatibility field remains independent.

    legacy = PerformanceReporter().generate_backtest_report("s", [], 100_000)
    assert legacy["settlement_schema_version"] == 1
    assert legacy["settlement"] is None

    bad = dict(settlement, reconciliation_delta=float("nan"))
    with pytest.raises(ValueError, match="reconciliation_delta must be finite"):
        PerformanceReporter().generate_backtest_report("s", [], 100_000, settlement=bad)


def test_force_close_reason_count_is_case_compatible():
    trade = {
        "strategy_id": "s", "direction": "SELL", "dt": datetime(2026, 1, 2),
        "pnl": 0, "reason": "force_close_end",
    }
    formatted = PerformanceReporter().generate_backtest_report("s", [trade], 100_000)
    assert formatted["force_close_count"] == 1

