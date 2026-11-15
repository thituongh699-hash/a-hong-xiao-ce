from src.core.backtest_engine import BacktestEngine


def _cab_with_cfg(cfg):
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.combination_config = engine._normalize_combination_config(cfg)
    return engine


def test_and_mode_requires_all_runnable_and_same_direction():
    engine = _cab_with_cfg({"enabled": True, "mode": "and"})
    signals = [
        {"strategy_id": "S1", "direction": "BUY", "qty": 100},
        {"strategy_id": "S2", "direction": "BUY", "qty": 100},
    ]
    kept, meta = engine._apply_signal_combination(signals, runnable_strategy_ids=["S1", "S2"])
    assert len(kept) == 2
    assert meta.get("mode") == "and"
    blocked, blocked_meta = engine._apply_signal_combination(signals[:1], runnable_strategy_ids=["S1", "S2"])
    assert blocked == []
    assert blocked_meta.get("reason") == "not_all_signaled"


def test_vote_mode_keeps_winner_direction():
    engine = _cab_with_cfg(
        {"enabled": True, "mode": "vote", "weights": {"S1": 2.0, "S2": 1.0, "S3": 1.0}}
    )
    signals = [
        {"strategy_id": "S1", "direction": "BUY", "qty": 100},
        {"strategy_id": "S2", "direction": "BUY", "qty": 100},
        {"strategy_id": "S3", "direction": "SELL", "qty": 100},
    ]
    kept, meta = engine._apply_signal_combination(signals, runnable_strategy_ids=["S1", "S2", "S3"])
    assert len(kept) == 2
    assert all(x["direction"] == "BUY" for x in kept)
    assert meta.get("winner") == "BUY"
