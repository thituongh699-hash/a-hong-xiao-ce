from types import SimpleNamespace

import pandas as pd
import pytest

import src.core.live_trading_engine as live_module
from src.core.live_trading_engine import LiveTradingEngine
from src.core.strategy_engine import StrategyEngine
from src.strategies.implemented_strategies import BaseImplementedStrategy
from src.utils.price_adjustment import PriceAdjustmentError


CODE = "000001.SZ"


def _raw_history():
    return pd.DataFrame([
        dict(code=CODE, dt=pd.Timestamp("2026-06-10 15:00"), open=100., high=101., low=99., close=100., vol=1000, amount=100000),
        dict(code=CODE, dt=pd.Timestamp("2026-06-11 15:00"), open=51., high=52., low=50., close=51., vol=2000, amount=102000),
    ])


def _factors():
    frame = pd.DataFrame({"dt": pd.to_datetime(["1990-01-01", "2026-06-11"]), "factor": [1., 2.]})
    frame.attrs.update(source="test", verified=True, coverage_start="1990-01-01", coverage_end="2026-06-11")
    return frame


class CashSizedStrategy(BaseImplementedStrategy):
    supports_dynamic_qfq = True

    def __init__(self):
        super().__init__("test", "cash sized", "D")
        self.seen_close = None

    def _cfg(self, key, default):
        return {"order_qty_mode": "cash_pct", "order_cash_pct": .8}.get(key, default)

    def on_bar(self, bar):
        self.seen_close = bar["close"]
        return dict(strategy_id=self.id, code=bar["code"], dt=bar["dt"],
                    direction="BUY", price=bar["close"], qty=self._qty(),
                    stop_loss=bar["close"] * .9)


def _engine(mode, strategy=None):
    engine = LiveTradingEngine.__new__(LiveTradingEngine)
    engine.stock_code = CODE
    engine.config = {}
    engine.adjustment_mode = mode
    engine._raw_strategy_histories = {"D": _raw_history()}
    engine._price_plan = None
    engine._price_factor_day = None
    engine._last_signal_factor = None
    engine.execution_engine = SimpleNamespace(positions={})
    engine.strategies = [strategy] if strategy else []
    engine.active_strategy_ids = [strategy.id] if strategy else []
    engine.strategy_trigger_tf = {strategy.id: "D"} if strategy else {}
    engine.account_ledger = SimpleNamespace(cash=20000.)
    return engine


def test_live_warmup_history_uses_verified_factor_but_keeps_raw_history(monkeypatch):
    monkeypatch.setattr(live_module, "load_adjustment_factors", lambda *args: _factors())
    engine = _engine("qfq")
    engine._refresh_price_plan(pd.Timestamp("2026-06-11 15:00"))

    signal = engine._signal_history(engine._raw_strategy_histories["D"], pd.Timestamp("2026-06-11"))
    assert signal["close"].tolist() == pytest.approx([50., 51.])
    assert signal["raw_close"].tolist() == pytest.approx([100., 51.])
    assert signal["vol"].tolist() == [1000, 2000]
    assert engine._raw_strategy_histories["D"]["close"].tolist() == [100., 51.]


def test_live_qfq_rebases_existing_indicator_state_without_changing_cash_or_shares(monkeypatch):
    monkeypatch.setattr(live_module, "load_adjustment_factors", lambda *args: _factors())
    strategy = CashSizedStrategy()
    engine = _engine("qfq", strategy)
    engine._refresh_price_plan(pd.Timestamp("2026-06-10"))
    strategy.history[CODE] = _raw_history().iloc[:1].copy()
    strategy.entry_price[CODE] = 100.
    strategy.positions[CODE] = 100

    engine._refresh_price_plan(pd.Timestamp("2026-06-11"))

    assert strategy.history[CODE]["close"].tolist() == [50.]
    assert strategy.history[CODE]["vol"].tolist() == [1000]
    assert strategy.entry_price[CODE] == 50.
    assert strategy.positions[CODE] == 100
    assert engine.account_ledger.cash == 20000.


def test_live_restart_blocks_held_position_across_exdate(monkeypatch):
    monkeypatch.setattr(live_module, "load_adjustment_factors", lambda *args: _factors())
    engine = _engine("hfq")
    engine.execution_engine.positions = {"test": {CODE: {
        "qty": 100, "lots": [{"qty": 100, "buy_day": "2026-06-10"}]
    }}}

    with pytest.raises(PriceAdjustmentError, match="corporate action with open holdings"):
        engine._refresh_price_plan(pd.Timestamp("2026-06-11"))
    assert engine._price_plan is None


def test_live_restart_allows_holding_bought_after_exdate(monkeypatch):
    monkeypatch.setattr(live_module, "load_adjustment_factors", lambda *args: _factors())
    engine = _engine("hfq")
    engine.execution_engine.positions = {"test": {CODE: {
        "qty": 100, "lots": [{"qty": 100, "buy_day": "2026-06-11"}]
    }}}

    engine._refresh_price_plan(pd.Timestamp("2026-06-11"))
    assert engine._last_signal_factor == 2.


def test_live_restart_blocks_holding_without_buy_date(monkeypatch):
    monkeypatch.setattr(live_module, "load_adjustment_factors", lambda *args: _factors())
    engine = _engine("hfq")
    engine.execution_engine.positions = {"test": {CODE: {"qty": 100}}}

    with pytest.raises(PriceAdjustmentError, match="no buy date"):
        engine._refresh_price_plan(pd.Timestamp("2026-06-11"))


def test_other_stock_holding_does_not_use_current_stocks_factor(monkeypatch):
    monkeypatch.setattr(live_module, "load_adjustment_factors", lambda *args: _factors())
    engine = _engine("hfq")
    engine.execution_engine.positions = {"test": {"600000.SH": {"qty": 100}}}
    engine._refresh_price_plan(pd.Timestamp("2026-06-10"))
    engine._refresh_price_plan(pd.Timestamp("2026-06-11"))
    assert engine._last_signal_factor == 2.


def test_live_hfq_strategy_sees_adjusted_prices_but_sizing_and_order_are_raw(monkeypatch):
    monkeypatch.setattr(live_module, "load_adjustment_factors", lambda *args: _factors())
    strategy = CashSizedStrategy()
    engine = _engine("hfq", strategy)
    engine._refresh_price_plan(pd.Timestamp("2026-06-11"))
    raw_bar = _raw_history().iloc[-1].to_dict()
    engine._fetch_latest_bar_for_timeframe = lambda tf, dt: raw_bar

    context = engine._build_strategy_context(raw_bar["dt"], raw_bar, [strategy.id])
    signals = StrategyEngine([strategy]).generate_signals(
        raw_bar, runnable_strategy_ids=[strategy.id], strategy_context=context
    )

    assert strategy.seen_close == pytest.approx(102.)
    assert strategy.last_price == pytest.approx(51.)
    assert len(signals) == 1
    assert signals[0]["qty"] == 300
    assert signals[0]["price"] == pytest.approx(51.)
    assert signals[0]["stop_loss"] == pytest.approx(45.9)


@pytest.mark.parametrize("timeframe", ["W", "2D", "90min", ""])
def test_unsupported_live_timeframe_does_not_silently_become_one_minute(timeframe):
    engine = _engine("none")
    with pytest.raises(PriceAdjustmentError, match="unsupported live strategy timeframe"):
        engine._normalize_trigger_tf(timeframe)
