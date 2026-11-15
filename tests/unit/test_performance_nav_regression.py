"""Portfolio performance uses session NAV while trade quality uses closed PnL."""

from datetime import datetime

import pandas as pd
import pytest

from src.services.performance_reporter import PerformanceReporter


class Ledger:
    def __init__(self, nav, transactions=()):
        self.nav = nav
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


def report(nav, transactions=(), start=None, end=None):
    return PerformanceReporter().generate_report(
        "s", Ledger(nav, transactions), Audit(), 100_000,
        start_date=start, end_date=end,
    )


def test_nav_recovery_has_half_drawdown_and_no_realized_pnl():
    nav = [
        {"dt": "2026-10-01 15:00", "nav": 100_000},
        {"dt": "2026-10-02 15:00", "nav": 50_000},
        {"dt": "2026-10-08 15:00", "nav": 100_000},
    ]
    result = report(nav)
    assert result["max_dd"] == pytest.approx(0.5)
    assert result["roi"] == pytest.approx(0)
    assert result["total_pnl"] == 0
    assert result["return_basis"] == "daily_nav"


def test_open_loss_and_buy_fee_affect_portfolio_return():
    nav = [
        {"dt": "2026-10-01 10:00", "nav": 100_000},
        {"dt": "2026-10-01 15:00", "nav": 99_995},
        {"dt": "2026-10-02 15:00", "nav": 90_000},
    ]
    result = report(nav, [{"strategy_id": "s", "direction": "BUY", "dt": datetime(2026, 10, 1), "cost": 5}])
    assert result["roi"] == pytest.approx(-0.1)
    assert result["max_dd"] == pytest.approx(0.1)
    assert result["total_pnl"] == 0
    assert result["final_capital"] == 90_000
    reporter = PerformanceReporter()
    daily = reporter._daily_nav(nav)
    assert daily.iloc[0] == 99_995
    assert reporter._session_returns(daily, 100_000).iloc[0] == pytest.approx(-0.00005)


def test_weekend_has_no_synthetic_session():
    daily = PerformanceReporter()._daily_nav([
        {"dt": "2026-10-02", "nav": 100_000},  # Friday
        {"dt": "2026-10-05", "nav": 101_000},  # Monday
    ])
    assert list(daily.index.strftime("%Y-%m-%d")) == ["2026-10-02", "2026-10-05"]
    assert len(PerformanceReporter()._session_returns(daily, 100_000)) == 2


def test_calendar_year_annualization_and_formatted_contract():
    nav = [
        {"dt": "2025-01-01", "nav": 100_000},
        {"dt": "2026-01-01", "nav": 110_000},
    ]
    summary = report(nav, start=datetime(2025, 1, 1), end=datetime(2026, 1, 1))
    assert summary["annualized_roi"] == pytest.approx(0.10, abs=0.001)
    formatted = PerformanceReporter().generate_backtest_report(
        "s", [], 100_000, datetime(2025, 1, 1), datetime(2026, 1, 1),
        summary_metrics=summary, nav_history=nav,
    )
    assert formatted["total_return"] == pytest.approx(summary["roi"])
    assert formatted["annual_return"] == pytest.approx(summary["annualized_roi"])
    assert formatted["end_capital"] == 110_000
    assert formatted["return_basis"] == "daily_nav"
    assert {"trade_details", "scorecard", "report_notes", "force_close_count"} <= formatted.keys()


def test_profit_factor_uses_total_wins_and_losses():
    trades = [
        {"strategy_id": "s", "direction": "SELL", "dt": datetime(2026, 10, 1), "pnl": 100},
        {"strategy_id": "s", "direction": "SELL", "dt": datetime(2026, 10, 2), "pnl": 100},
        {"strategy_id": "s", "direction": "SELL", "dt": datetime(2026, 10, 3), "pnl": -50},
    ]
    summary = report([], trades)
    assert summary["profit_factor"] == pytest.approx(4.0)
    formatted = PerformanceReporter().generate_backtest_report("s", trades, 100_000)
    assert formatted["profit_ratio"] == pytest.approx(4.0)
    assert formatted["return_basis"] == "closed_trade_pnl"


def test_no_losses_has_json_safe_undefined_profit_factor():
    trades = [{"strategy_id": "s", "direction": "SELL", "dt": datetime(2026, 10, 1), "pnl": 100}]
    summary = report([], trades)
    assert summary["profit_factor"] is None
    assert summary["scorecard"]["metrics"]["profit_loss_ratio"] == 0.0


def test_transaction_only_caller_without_nav_method_still_works():
    class LegacyLedger:
        transactions = [{"strategy_id": "s", "direction": "SELL", "dt": datetime(2026, 10, 1), "pnl": 100}]

    result = PerformanceReporter().generate_report("s", LegacyLedger(), Audit(), 100_000)
    assert result["roi"] == pytest.approx(0.001)
    assert result["return_basis"] == "closed_trade_pnl"
