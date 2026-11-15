from datetime import datetime

import pytest

from src.core.execution_engine import ExecutionEngine
from src.services.account_ledger import AccountLedger
from src.services.order_matcher import OrderMatcher
from src.services.risk_audit_log import RiskAuditLog
from src.utils.runtime_params import parameter_overrides


CODE = "000001.SZ"
COSTS = {
    "execution.slippage": 0,
    "trading_cost.min_commission": 5,
    "trading_cost.commission_rate": 0,
    "trading_cost.stamp_duty": 0,
    "trading_cost.transfer_fee": 0,
}


def bar(day, open_price, high, low, close):
    return dict(code=CODE, dt=datetime(2026, 6, day, 10), open=open_price,
                high=high, low=low, close=close, volume=1000)


def setup_position(qty=100, stop_loss=9, take_profit=None):
    ledger = AccountLedger(100000)
    execution = ExecutionEngine(ledger, OrderMatcher(), RiskAuditLog())
    buy = dict(code=CODE, direction="BUY", qty=qty, price=10,
               stop_loss=stop_loss, take_profit=take_profit)
    assert execution.execute_order("s", buy, bar(1, 10, 10.5, 9.5, 10), account=ledger)
    return ledger, execution


@pytest.mark.parametrize(
    "sell_bar,expected_fill",
    [
        (bar(2, 10, 12, 8, 10), 9),
        (bar(2, 8, 10, 7, 8), 8),
    ],
)
def test_stop_uses_trigger_or_worse_gap_open(sell_bar, expected_fill):
    with parameter_overrides(COSTS):
        ledger, execution = setup_position(take_profit=11)
        orders = execution.check_stops(sell_bar)
        assert len(orders) == 1
        assert orders[0]["execution_mode"] == "sell_stop"
        assert orders[0]["trigger_price"] == 9
        assert execution.execute_order("s", orders[0], sell_bar, account=ledger)
        assert ledger.transactions[-1]["price"] == pytest.approx(expected_fill)


def test_full_round_trip_pnl_reconciles_to_cash_including_buy_fee():
    with parameter_overrides(COSTS):
        ledger, execution = setup_position(stop_loss=None)
        sell = dict(code=CODE, direction="SELL", qty=100, price=11)
        assert execution.execute_order("s", sell, bar(2, 11, 11.5, 10.5, 11), account=ledger)
        assert ledger.transactions[-1]["pnl"] == pytest.approx(ledger.cash - 100000)
        assert ledger.transactions[-1]["pnl"] == pytest.approx(90)


def test_partial_sales_allocate_buy_fee_to_each_closed_lot():
    with parameter_overrides(COSTS):
        ledger, execution = setup_position(qty=200, stop_loss=None)
        sell = dict(code=CODE, direction="SELL", qty=100, price=11)
        assert execution.execute_order("s", sell, bar(2, 11, 11.5, 10.5, 11), account=ledger)
        assert ledger.transactions[-1]["pnl"] == pytest.approx(92.5)
        assert execution.positions["s"][CODE]["qty"] == 100
        sell["price"] = 9
        assert execution.execute_order("s", sell, bar(3, 9, 9.5, 8.5, 9), account=ledger)
        realized = sum(t["pnl"] for t in ledger.transactions if t["direction"] == "SELL")
        assert ledger.transactions[-1]["pnl"] == pytest.approx(-107.5)
        assert realized == pytest.approx(ledger.cash - 100000)


def test_buy_lot_rounding_and_t1_sell_remain_enforced():
    with parameter_overrides(COSTS):
        ledger, execution = setup_position(qty=150, stop_loss=None)
        assert execution.positions["s"][CODE]["qty"] == 100
        sell = dict(code=CODE, direction="SELL", qty=100, price=10)
        assert not execution.execute_order(
            "s", sell, bar(1, 10, 10.5, 9.5, 10), account=ledger
        )
        assert execution.positions["s"][CODE]["qty"] == 100
        assert execution.execute_order(
            "s", sell, bar(2, 10, 10.5, 9.5, 10), account=ledger
        )
