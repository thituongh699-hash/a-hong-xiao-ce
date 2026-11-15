import pandas as pd

from src.core.live_trading_engine import LiveTradingEngine
from src.strategies.base_strategy import BaseStrategy


def test_prime_strategy_history_converts_list_container_to_dict():
    strategy = BaseStrategy("X1")
    strategy.history = []
    df = pd.DataFrame([{"code": "002432.SZ", "close": 10.2}])

    LiveTradingEngine._prime_strategy_history(strategy, "002432.SZ", df)

    assert isinstance(strategy.history, dict)
    assert "002432.SZ" in strategy.history
    assert strategy.history["002432.SZ"].equals(df)
    assert strategy.history["002432.SZ"] is not df


def test_prime_strategy_history_keeps_existing_dict_entries():
    strategy = BaseStrategy("X2")
    strategy.history = {"000001.SZ": pd.DataFrame([{"close": 8.8}])}
    df = pd.DataFrame([{"code": "002432.SZ", "close": 10.5}])

    LiveTradingEngine._prime_strategy_history(strategy, "002432.SZ", df)

    assert "000001.SZ" in strategy.history
    assert "002432.SZ" in strategy.history
    assert strategy.history["002432.SZ"].equals(df)


def test_is_pure_daily_mode_with_active_daily_only():
    engine = LiveTradingEngine.__new__(LiveTradingEngine)
    engine.active_strategy_ids = ["04"]

    class _S:
        def __init__(self, sid, tf):
            self.id = sid
            self.trigger_timeframe = tf

    engine.strategies = [_S("04", "D"), _S("06", "1min")]
    assert engine._is_pure_daily_mode() is True


def test_daily_tick_uses_1459_when_minute_close_confirm_enabled():
    engine = LiveTradingEngine.__new__(LiveTradingEngine)
    engine._minute_close_confirm_enabled = True
    dt_ok = pd.Timestamp("2026-04-15 14:59:00")
    dt_bad = pd.Timestamp("2026-04-15 15:00:00")
    assert engine._is_timeframe_tick(dt_ok, "D") is True
    assert engine._is_timeframe_tick(dt_bad, "D") is False
