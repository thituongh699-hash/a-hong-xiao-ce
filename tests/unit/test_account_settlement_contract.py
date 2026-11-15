from datetime import datetime

import pytest

from src.core.execution_engine import ExecutionEngine
from src.core.risk_manager import RiskManager
from src.services.account_ledger import AccountLedger
from src.services.order_matcher import OrderMatcher
from src.services.risk_audit_log import RiskAuditLog
from src.utils.runtime_params import parameter_overrides


CODE = "000001.SZ"
ZERO_COST = {
    "execution.slippage": 0,
    "trading_cost.min_commission": 0,
    "trading_cost.commission_rate": 0,
    "trading_cost.stamp_duty": 0,
    "trading_cost.transfer_fee": 0,
}
FIXED_FEE = {**ZERO_COST, "trading_cost.min_commission": 5}
RISK_LIMITS = {
    "risk_control.max_stop_loss_pct": 1,
    "risk_control.max_pos_per_stock": 0.10,
    "risk_control.max_total_pos": 0.50,
    "risk_control.max_daily_loss_pct": 0.02,
}


def bar(day, price=10, *, volume=1000):
    return {
        "code": CODE,
        "dt": datetime(2026, 6, day, 10),
        "open": price,
        "high": price + 0.2,
        "low": price - 0.2,
        "close": price,
        "volume": volume,
    }


def order(direction, qty=100, price=10, **extra):
    value = {"strategy_id": "s", "code": CODE, "direction": direction, "qty": qty, "price": price}
    value.update(extra)
    return value


def make_execution(initial=100_000):
    account = AccountLedger(initial)
    audit = RiskAuditLog()
    return account, ExecutionEngine(account, OrderMatcher(), audit), audit


def test_risk_and_execution_share_normalized_direction_contract():
    with parameter_overrides(RISK_LIMITS):
        audit = RiskAuditLog()
        risk = RiskManager(audit)
        oversized = order("  buy  ", qty=2_000, dt=datetime(2026, 6, 1, 10))
        approved, reason = risk.check_signal(oversized, 100_000, {}, 0)
        assert not approved
        assert "单票仓位" in reason

        invalid = order("hold", dt=datetime(2026, 6, 1, 10))
        approved, reason = risk.check_signal(invalid, 100_000, {}, 0)
        assert not approved
        assert "非法交易方向" in reason
        assert audit.rejections[-1]["rule_id"] == "R_DIRECTION_INVALID"

    with parameter_overrides(ZERO_COST):
        account, execution, _ = make_execution()
        assert execution.execute_order("s", order("  buy  "), bar(1), account=account)
        assert account.transactions[-1]["direction"] == "BUY"


def test_filled_add_on_updates_position_protection_and_preserves_fifo_t1():
    with parameter_overrides(ZERO_COST):
        account, execution, _ = make_execution()
        assert execution.execute_order(
            "s", order("BUY", stop_loss=9, take_profit=13), bar(1), account=account
        )

        failed = order("BUY", price=12, stop_loss=11.5, take_profit=14)
        assert not execution.execute_order("s", failed, bar(2, 12, volume=0), account=account)
        pos = execution.positions["s"][CODE]
        assert pos["stop_loss"] == 9
        assert pos["take_profit"] == 13

        assert execution.execute_order("s", order("BUY", price=12), bar(2, 12), account=account)
        pos = execution.positions["s"][CODE]
        assert pos["stop_loss"] == 9
        assert pos["take_profit"] == 13

        assert execution.execute_order(
            "s", order("BUY", price=11, stop_loss=10.5, take_profit=0), bar(3, 11), account=account
        )
        pos = execution.positions["s"][CODE]
        assert pos["stop_loss"] == 10.5
        assert pos["take_profit"] is None
        assert account.transactions[-1]["effective_stop_loss"] == 10.5
        assert account.transactions[-1]["effective_take_profit"] is None

        assert execution.execute_order("s", order("SELL", price=11), bar(3, 11), account=account)
        pos = execution.positions["s"][CODE]
        assert pos["qty"] == 200
        assert [lot["buy_day"] for lot in pos["lots"]] == ["2026-06-02", "2026-06-03"]
        settlement = execution.get_strategy_settlement(
            "s", account, {CODE: 11}, as_of=datetime(2026, 6, 3, 15)
        )
        assert settlement["positions"][0]["sellable_qty"] == 100
        stop_bar = bar(4, 11)
        stop_bar["low"] = 10
        protective_orders = execution.check_stops(stop_bar)
        assert len(protective_orders) == 1
        assert protective_orders[0]["reason"] == "STOP_LOSS"
        assert protective_orders[0]["trigger_price"] == 10.5


def test_transaction_metadata_and_open_settlement_reconcile_with_fees():
    with parameter_overrides(FIXED_FEE):
        account, execution, _ = make_execution()
        signal_dt = datetime(2026, 5, 31, 15)
        assert execution.execute_order(
            "s",
            order(
                "BUY",
                qty=200,
                stop_loss=9,
                take_profit=13,
                signal_dt=signal_dt,
                execution_mode="at_open",
                reason="ENTRY",
            ),
            bar(1),
            account=account,
        )
        assert execution.execute_order(
            "s",
            order(
                "SELL",
                qty=100,
                price=11,
                signal_dt=datetime(2026, 6, 1, 15),
                execution_mode="at_open",
                reason="FORCE_CLOSE_END",
            ),
            bar(2, 11),
            account=account,
        )

        transaction = account.transactions[-1]
        assert transaction["reason"] == "FORCE_CLOSE_END"
        assert transaction["signal_dt"] == datetime(2026, 6, 1, 15)
        assert transaction["execution_mode"] == "at_open"
        assert transaction["effective_stop_loss"] == 9
        assert transaction["effective_take_profit"] == 13

        settlement = execution.get_strategy_settlement(
            "s", account, {CODE: 12}, as_of=datetime(2026, 6, 2, 15)
        )
        assert settlement["state"] == "open"
        assert settlement["open_position_count"] == 1
        assert settlement["open_quantity"] == 100
        assert settlement["cash"] == pytest.approx(99_090)
        assert settlement["holdings_value"] == pytest.approx(1_200)
        assert settlement["realized_pnl"] == pytest.approx(92.5)
        assert settlement["unrealized_pnl"] == pytest.approx(197.5)
        assert settlement["fees_paid"] == pytest.approx(10)
        assert settlement["reconciliation_delta"] == pytest.approx(0)
        assert settlement["as_of"] == "2026-06-02T15:00:00"
        assert settlement["positions"] == [{
            "code": CODE,
            "qty": 100,
            "market_value": pytest.approx(1_200),
            "cost_basis": pytest.approx(1_002.5),
            "unrealized_pnl": pytest.approx(197.5),
            "sellable_qty": 100,
        }]

        assert execution.execute_order(
            "s", order("SELL", qty=100, price=12, reason="EXIT"), bar(3, 12), account=account
        )
        flat = execution.get_strategy_settlement("s", account, {CODE: 12})
        assert flat["state"] == "flat"
        assert flat["positions"] == []
        assert flat["holdings_value"] == 0
        assert flat["open_position_count"] == 0
        assert flat["open_quantity"] == 0
        assert flat["unrealized_pnl"] == 0
        assert flat["realized_pnl"] == pytest.approx(account.cash - account.initial_capital)
        assert flat["reconciliation_delta"] == pytest.approx(0)


def test_record_transaction_old_call_shape_and_daily_nav_compaction():
    legacy = AccountLedger(10_000)
    legacy.record_transaction("s", datetime(2026, 6, 1, 10), "BUY", 10, 100, 0)
    assert "reason" not in legacy.transactions[-1]

    bar_ledger = AccountLedger(10_000)
    bar_ledger.update_daily_nav(datetime(2026, 6, 1, 10), 100)
    bar_ledger.update_daily_nav(datetime(2026, 6, 1, 15), 200)
    assert len(bar_ledger.daily_nav) == 2

    daily = AccountLedger(10_000, nav_frequency="daily")
    daily.update_daily_nav(datetime(2026, 6, 1, 10), 100)
    daily.update_daily_nav(datetime(2026, 6, 1, 15), 250)
    daily.update_daily_nav(datetime(2026, 6, 1, 15), 300)
    assert len(daily.daily_nav) == 1
    assert daily.daily_nav[-1] == {
        "dt": datetime(2026, 6, 1, 15),
        "nav": 10_300,
        "cash": 10_000,
        "holdings": 300,
    }
    daily.update_daily_nav(datetime(2026, 6, 2, 15), 400)
    assert len(daily.daily_nav) == 2
    with pytest.raises(ValueError, match="monotonic"):
        daily.update_daily_nav(datetime(2026, 6, 2, 14), 350)
    with pytest.raises(ValueError, match="nav_frequency"):
        AccountLedger(10_000, nav_frequency="weekly")
