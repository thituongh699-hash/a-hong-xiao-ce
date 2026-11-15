import asyncio
from datetime import datetime

import pandas as pd
import pytest

import src.core.backtest_engine as engine_module
from src.core.backtest_engine import BacktestEngine
from src.strategies.implemented_strategies import BaseImplementedStrategy, Strategy00
from src.core.strategy_engine import StrategyEngine
from src.utils.runtime_params import parameter_overrides


def raw_prices():
    dates = pd.date_range('2026-06-01', periods=13)
    closes = [100.] * 10 + [51., 53., 54.]
    return pd.DataFrame([dict(code='000001.SZ', dt=dt, open=p, high=p+0.2, low=p-0.2,
                              close=p, vol=100000, amount=100000*p) for dt, p in zip(dates, closes)])


def factors():
    frame = pd.DataFrame({'dt': pd.to_datetime(['2026-06-01', '2026-06-11']), 'factor': [1., 2.]})
    frame.attrs.update(source='test', verified=True, coverage_start='1990-01-01', coverage_end='2026-06-30')
    return frame


class Provider:
    price_mode = 'none'

    def check_connectivity(self, code):
        return True, 'ok'

    def fetch_kline_data(self, code, start, end, interval):
        return raw_prices()


class TenDayReturn(BaseImplementedStrategy):
    supports_dynamic_qfq = True
    def __init__(self):
        super().__init__('test', 'ten day return', 'D')

    def on_bar(self, bar):
        code = bar['code']
        self.history[code] = pd.concat([self.history.get(code, pd.DataFrame()), pd.DataFrame([bar])])
        frame = self.history[code]
        if len(frame) < 11 or self.positions.get(code, 0):
            return None
        if frame['close'].iloc[-1] / frame['close'].iloc[-11] - 1 < .05:
            return None
        # Sizing deliberately uses the real cash/price context.
        qty = int((self.current_cash * .1 // self.last_price) // 100) * 100
        return dict(strategy_id=self.id, code=code, dt=bar['dt'], direction='BUY',
                    price=bar['close'], qty=qty, stop_loss=bar['close']*.9)


@pytest.mark.parametrize('mode, expected_trades', [('none', 0), ('qfq', 1), ('hfq', 1)])
def test_return_signal_adjusted_but_execution_and_sizing_raw(monkeypatch, mode, expected_trades):
    strategy = TenDayReturn()
    monkeypatch.setattr(engine_module, 'create_strategies', lambda **kwargs: [strategy])
    events = []

    async def emit(kind, data, **kwargs):
        events.append((kind, data))

    engine = BacktestEngine('000001.SZ', strategy_ids=['test'], initial_capital=100000,
                              provider_override=Provider(), provider_source_override='tdx',
                              adjustment_mode=mode, adjustment_factors=factors(), event_callback=emit)
    with parameter_overrides({'execution.slippage': 0, 'risk_control.max_stop_loss_pct': 1}):
        asyncio.run(engine.run(datetime(2026, 6, 1), datetime(2026, 6, 13)))
    assert not [e for e in events if e[0] == 'backtest_failed']
    trades = [data for kind, data in events if kind == 'backtest_trade']
    assert len(trades) == expected_trades
    if expected_trades:
        assert trades[0]['price'] == pytest.approx(54.)
        assert pd.Timestamp(trades[0]['dt']) == pd.Timestamp('2026-06-13')
        assert trades[0]['qty'] == 100
        assert engine.execution_engine.positions['test']['000001.SZ']['stop_loss'] == pytest.approx(47.7)
    result = [data for kind, data in events if kind == 'backtest_result'][0]
    assert result['baseline_meta']['price_policy']['execution_price_mode'] == 'none'


def test_qfq_history_rebase_leaves_cash_shares_and_volume_unchanged():
    strategy = TenDayReturn()
    strategy.history['000001.SZ'] = raw_prices().iloc[:2].copy()
    strategy.entry_price_local = {'000001.SZ': 100.}
    strategy.positions['000001.SZ'] = 100
    strategy.current_cash = 1000
    strategy.rebase_signal_prices('000001.SZ', .5)
    assert strategy.history['000001.SZ']['close'].tolist() == [50., 50.]
    assert strategy.history['000001.SZ']['vol'].tolist() == [100000, 100000]
    assert strategy.entry_price_local['000001.SZ'] == 50.
    assert strategy.positions['000001.SZ'] == 100
    assert strategy.current_cash == 1000


@pytest.mark.parametrize('tf', ['W', '2D', 'M'])
def test_multi_day_aggregates_are_rejected_before_incorrect_adjustment(tf):
    engine = BacktestEngine.__new__(BacktestEngine)
    with pytest.raises(engine_module.PriceAdjustmentError, match='unsupported strategy timeframe'):
        engine._normalize_trigger_tf(tf)


def test_task_cost_overrides_are_isolated_across_concurrent_runs(monkeypatch):
    from src.utils import runtime_params
    monkeypatch.setattr(runtime_params.ConfigLoader, 'reload', lambda: {'execution.slippage': .003})
    async def task(value):
        with runtime_params.parameter_overrides({'execution.slippage': value}):
            await asyncio.sleep(0)
            assert runtime_params.get_value('execution.slippage') == value
            assert await asyncio.to_thread(runtime_params.get_value, 'execution.slippage') == value
        assert runtime_params.get_value('execution.slippage') == .003
    async def both():
        await asyncio.gather(task(.001), task(.002))
    asyncio.run(both())
    assert runtime_params.get_value('execution.slippage') == .003


def test_task_cost_override_resets_after_failure(monkeypatch):
    from src.utils import runtime_params
    monkeypatch.setattr(runtime_params.ConfigLoader, 'reload', lambda: {'execution.slippage': .003})
    with pytest.raises(ValueError):
        with runtime_params.parameter_overrides({'execution.slippage': .001}):
            raise ValueError('run failed')
    assert runtime_params.get_value('execution.slippage') == .003


def test_buy_and_hold_sizes_with_raw_price_in_hfq():
    strategy = Strategy00()
    strategy.set_backtest_context(current_cash=10000., last_price=10.)
    signal = StrategyEngine([strategy]).generate_signals(
        dict(code='000001.SZ', dt=datetime(2026, 6, 12), close=10.),
        strategy_context={'__kline_by_strategy__': {'00': dict(code='000001.SZ', dt=datetime(2026, 6, 12), close=20., raw_close=10., signal_multiplier=2.)}},
    )[0]
    assert signal['qty'] == 1000
    assert signal['price'] == 10.


def test_held_position_crossing_exdate_fails_instead_of_false_return(monkeypatch):
    strategy = Strategy00()
    monkeypatch.setattr(engine_module, 'create_strategies', lambda **kwargs: [strategy])
    events = []

    async def emit(kind, data, **kwargs):
        events.append((kind, data))

    engine = BacktestEngine('000001.SZ', strategy_ids=['00'], initial_capital=1000000,
                              provider_override=Provider(), provider_source_override='tdx',
                              adjustment_mode='hfq', adjustment_factors=factors(), event_callback=emit)
    # Leave enough room for commission so the initial buy is actually filled.
    original = strategy.on_bar
    def with_reserve(bar):
        signal = original(bar)
        if signal and signal['direction'] == 'BUY':
            signal['qty'] = 100
        return signal
    strategy.on_bar = with_reserve
    asyncio.run(engine.run(datetime(2026, 6, 1), datetime(2026, 6, 12)))
    failures = [data for kind, data in events if kind == 'backtest_failed']
    assert failures and '除权' in failures[-1]['msg']
    assert not [data for kind, data in events if kind == 'backtest_result']
