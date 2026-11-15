import asyncio
from datetime import datetime

import pandas as pd
import pytest

import src.core.backtest_engine as module
from src.core.backtest_engine import BacktestEngine
from src.core.execution_engine import ExecutionEngine
from src.core.risk_manager import RiskManager
from src.core.strategy_engine import StrategyEngine
from src.services.account_ledger import AccountLedger
from src.services.order_matcher import OrderMatcher
from src.services.risk_audit_log import RiskAuditLog
from src.strategies.implemented_strategies import BaseImplementedStrategy
from src.utils.runtime_params import parameter_overrides


class OneSignal(BaseImplementedStrategy):
    def __init__(self, signal_bar=1):
        super().__init__('causal', 'causal regression', 'D')
        self.count = 0
        self.signal_bar = signal_bar

    def on_bar(self, bar):
        self.count += 1
        if self.count == self.signal_bar:
            return dict(strategy_id=self.id, code=bar['code'], dt=bar['dt'],
                        direction='BUY', price=bar['close'], qty=100)


class InMemoryProvider:
    price_mode = 'none'

    def check_connectivity(self, code):
        return True, 'in memory'

    def fetch_kline_data(self, *args):
        return pd.DataFrame([
            dict(code='000001.SZ', dt=pd.Timestamp('2026-06-01'), open=10, high=11.5, low=9.8, close=11, vol=10000, amount=100000),
            dict(code='000001.SZ', dt=pd.Timestamp('2026-06-02'), open=12, high=12.5, low=11.8, close=12.2, vol=10000, amount=120000),
            dict(code='000001.SZ', dt=pd.Timestamp('2026-06-03'), open=12.3, high=12.5, low=12, close=12.1, vol=10000, amount=120000),
        ])


def run_engine(monkeypatch, signal_bar, trading_scope=None):
    strategy = OneSignal(signal_bar)
    monkeypatch.setattr(module, 'create_strategies', lambda **kwargs: [strategy])
    monkeypatch.setattr(module.ConfigLoader, 'reload', lambda: {
        'risk_control.max_drawdown_pct': 0,
        'execution.force_close_on_backtest_end': False,
        'system.backtest_flow_emit_stride': 20,
    })
    events = []

    async def emit(kind, value):
        events.append((kind, value))

    engine = BacktestEngine('000001.SZ', strategy_ids=['causal'], initial_capital=100000,
                            event_callback=emit, provider_override=InMemoryProvider(),
                            provider_source_override='tdx', adjustment_mode='none',
                            trading_scope=trading_scope,
                            security_records=[{'code': '000001.SZ', 'name': '普通主板公司'}])
    async def no_cache(*args):
        pass
    engine._persist_backtest_cache_to_db = no_cache
    with parameter_overrides({'execution.slippage': 0, 'risk_control.max_pos_per_stock': 1,
                              'risk_control.max_total_pos': 1}):
        asyncio.run(engine.run(datetime(2026, 6, 1), datetime(2026, 6, 3)))
    assert not [e for e in events if e[0] == 'backtest_failed']
    return engine, events


def test_close_signal_fills_only_at_following_open(monkeypatch):
    engine, events = run_engine(monkeypatch, 1)
    trades = engine.strategy_accounts['causal'].transactions
    assert len(trades) == 1
    assert trades[0]['dt'] == pd.Timestamp('2026-06-02')
    assert trades[0]['price'] == 12
    event = next(data for kind, data in events if kind == 'backtest_trade')
    assert pd.Timestamp(event['signal_dt']) == pd.Timestamp('2026-06-01')
    assert event['price'] == 12
    nav = engine.strategy_accounts['causal'].get_nav_history()
    assert nav.dt.nunique() == 3


def test_final_bar_signal_is_not_filled_without_future_bar(monkeypatch):
    engine, _ = run_engine(monkeypatch, 3)
    assert engine.strategy_accounts['causal'].transactions == []
    assert not engine.pending_signals


def test_drawdown_liquidation_retries_after_t1_and_blocks_new_entries():
    engine = BacktestEngine.__new__(BacktestEngine)
    account = AccountLedger(100000)
    account.cash = 90000
    audit = RiskAuditLog()
    engine.initial_capital = 100000
    engine.strategy_accounts = {'s': account}
    engine.execution_engine = ExecutionEngine(account, OrderMatcher(), audit)
    engine.risk_manager = RiskManager(audit)
    engine.strategy_engine = StrategyEngine([])
    engine.execution_engine.positions = {'s': {'000001.SZ': {
        'qty': 1000, 'avg_price': 10, 'market_value': 9000, 'direction': 'BUY',
        'lots': [{'qty': 1000, 'buy_day': '2026-06-01', 'unit_cost': 10}],
    }}}
    engine.config = {'risk_control.max_drawdown_pct': .005}
    engine.aggregate_nav = []
    engine.drawdown_above_limit = False
    async def silent(*args, **kwargs):
        pass
    engine._emit = silent
    def bar(day):
        return dict(code='000001.SZ', dt=datetime(2026, 6, day), open=9, high=9.2,
                    low=8.8, close=9, volume=10000)
    with parameter_overrides({'execution.slippage': 0}):
        assert asyncio.run(engine._enforce_drawdown_limit(bar(1))) is True
        assert account.transactions == []
        same_day_next_bar = bar(1)
        same_day_next_bar['dt'] = datetime(2026, 6, 1, 10)
        asyncio.run(engine._execute_drawdown_orders(same_day_next_bar))
        assert engine.execution_engine.positions['s']['000001.SZ']['qty'] == 1000
        next_day = bar(2)
        next_day['open'] = 9.1
        asyncio.run(engine._execute_drawdown_orders(next_day))
    assert not engine.execution_engine.positions['s']
    assert account.transactions[-1]['price'] == pytest.approx(9.1)


def test_pending_buy_risk_uses_open_without_future_close():
    engine = BacktestEngine.__new__(BacktestEngine)
    account = AccountLedger(100000)
    audit = RiskAuditLog()
    engine.strategy_accounts = {'s': account}
    engine.execution_engine = ExecutionEngine(account, OrderMatcher(), audit)
    engine.risk_manager = RiskManager(audit)
    engine.strategy_engine = StrategyEngine([])
    engine.initial_capital = 100000
    engine.aggregate_nav = []
    engine.config = {'risk_control.max_drawdown_pct': 0}
    engine._session_start_nav = {'s': 100000}
    engine.pending_signals = [dict(strategy_id='s', code='000001.SZ', dt=datetime(2026, 6, 1), direction='BUY', price=10, qty=100)]
    async def silent(*args, **kwargs):
        pass
    engine._emit = silent
    observed = []
    def approve(signal, value, positions, daily_pnl, **kwargs):
        observed.append((signal['price'], value, daily_pnl))
        return True, 'ok'
    engine.risk_manager.check_signal = approve
    with parameter_overrides({'execution.slippage': 0}):
        asyncio.run(engine._execute_pending_signals(dict(code='000001.SZ', dt=datetime(2026, 6, 2), open=11, high=30, low=10, close=29, volume=10000)))
    assert observed == [(11, 100000, 0)]
    assert account.transactions[0]['price'] == 11


def test_scope_accepted_engine_run_retains_scope_in_actual_result(monkeypatch):
    scope = {"boards": ["main_board"], "stock_types": ["normal"]}
    engine, events = run_engine(monkeypatch, 1, trading_scope=scope)
    assert len(engine.strategy_accounts['causal'].transactions) == 1
    result = next(value for kind, value in events if kind == 'backtest_result')
    assert result['baseline_meta']['trading_scope'] == scope
    assert result['baseline_meta']['trading_scope_metadata']['historical_st_status_verified'] is False
