import pandas as pd
import pytest

from src.services.order_matcher import OrderMatcher
from src.services.account_ledger import AccountLedger
from src.services.risk_audit_log import RiskAuditLog
from src.core.execution_engine import ExecutionEngine
from src.utils.runtime_params import parameter_overrides


def daily(**changes):
    row = dict(code='000001.SZ', dt=pd.Timestamp('2026-08-03'), open=10.1,
               high=11., low=10., close=11., vol=100000., up_limit=11., down_limit=9.)
    row.update(changes)
    return row


def order(mode, direction='BUY', **changes):
    row = dict(code='000001.SZ', strategy_id='test', direction=direction, qty=1000,
               price=10.3, execution_mode=mode, trigger_price=10.31)
    row.update(changes)
    return row


def test_close_limit_up_is_not_a_buy():
    assert not OrderMatcher().match_order(order('at_close'), daily())[0]


def test_intraday_threshold_can_fill_before_close_limit_up():
    with parameter_overrides({'execution.slippage': 0}):
        ok, price = OrderMatcher().match_order(order('buy_stop'), daily(is_limit_up=True))
        assert ok and price == pytest.approx(10.31)


def test_open_limit_up_rejected_even_if_it_later_breaks():
    assert not OrderMatcher().match_order(order('buy_stop'), daily(open=11., low=10.1, close=10.5))[0]


def test_one_word_limit_up_no_buy_in_either_mode():
    for mode in ['buy_stop', 'at_close', 'at_open']:
        assert not OrderMatcher().match_order(order(mode), daily(open=11., low=11.))[0]


def test_gap_above_three_percent_pays_actual_open():
    with parameter_overrides({'execution.slippage': 0}):
        ok, price = OrderMatcher().match_order(order('buy_stop'), daily(open=10.6, close=10.5))
        assert ok and price == pytest.approx(10.6)


def test_high_below_trigger_cannot_fill():
    assert not OrderMatcher().match_order(order('buy_stop'), daily(high=10.30, close=10.2))[0]


def test_close_limit_down_does_not_sell_but_earlier_stop_can():
    row = daily(open=9.5, high=10., low=9., close=9.)
    with parameter_overrides({'execution.slippage': 0}):
        matcher = OrderMatcher()
        assert not matcher.match_order(order('at_close', 'SELL'), row)[0]
        ok, price = matcher.match_order(order('sell_stop', 'SELL', trigger_price=9.4), row)
        assert ok and price == pytest.approx(9.4)


def test_open_limit_down_blocks_sale():
    assert not OrderMatcher().match_order(order('sell_stop', 'SELL', trigger_price=9.4), daily(open=9., low=9., close=9.2))[0]


def test_flat_nonlimit_day_not_misclassified_with_boundaries():
    with parameter_overrides({'execution.slippage': 0}):
        assert OrderMatcher().match_order(order('at_close'), daily(open=10., high=10., low=10., close=10.))[0]


def test_unknown_timing_rejected():
    assert not OrderMatcher().match_order(order('unknown'), daily(close=10.5))[0]


def test_legacy_behavior_preserved_and_missing_metadata_gap_reproduced():
    row = daily()
    row.pop('up_limit'); row.pop('down_limit')
    legacy = order(''); legacy.pop('execution_mode')
    assert OrderMatcher().match_order(legacy, row)[0]


def test_project_execution_enforces_t_plus_one_and_costs():
    account = AccountLedger(100000.)
    log = RiskAuditLog()
    engine = ExecutionEngine(account, OrderMatcher(), log)
    with parameter_overrides({'execution.slippage': 0, 'trading_cost.commission_rate': .00025,
                              'trading_cost.min_commission': 5., 'trading_cost.stamp_duty': .001,
                              'trading_cost.transfer_fee': 0}):
        row = daily(close=10.5)
        assert engine.execute_order('test', order('buy_stop'), row)
        assert not engine.execute_order('test', order('at_close', 'SELL'), row)
        assert log.rejections[-1]['rule_id'] == 'EXEC_T1_BLOCK'
        assert engine.execute_order('test', order('at_close', 'SELL'), daily(dt=row['dt']+pd.Timedelta(days=1), close=10.5))
        assert len(account.transactions) == 2
        assert account.cash == pytest.approx(100000 + (10.5 - 10.31)*1000 - 5 - 5 - 10.5)
