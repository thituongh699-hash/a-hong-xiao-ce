import numpy as np
import pandas as pd
import pytest

from scripts.lowfreq_portfolio import simulate_portfolio


SPEC = dict(id="x", family="test", lookback=1, stop=.10, trail=.10,
            take=None, max_hold=2, exit_ma=0, weekly_only=False)


def dataset(dates, codes=("000001.SZ",), price=10.):
    t, s = len(dates), len(codes)
    shape = (t, s)
    return dict(
        dates=list(dates), codes=list(codes),
        open=np.full(shape, price, float), close=np.full(shape, price, float),
        vol=np.full(shape, 10000., float), up_limit=np.full(shape, price * 1.1),
        down_limit=np.full(shape, price * .9), factor_changed=np.zeros(shape, bool),
        open_valid=np.ones(shape, bool), day_valid=np.ones(shape, bool),
        entry={"x": np.zeros(shape, bool)}, score={"x": np.zeros(shape, float)},
        exit={"x": np.zeros(shape, bool)}, market_ok=np.ones(t, bool),
        week_last=np.zeros(t, bool),
    )


def test_signal_is_t_plus_one_and_open_fill_does_not_read_later_close():
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    data = dataset(dates)
    data["entry"]["x"][0, 0] = True
    data["score"]["x"][0, 0] = 1
    data["close"][1, 0] = 1000  # known only after the opening fill
    data["up_limit"][1, 0] = 11
    result = simulate_portfolio(data, SPEC, dates[0], dates[-1], slippage=0)
    buy = result["orders"].query("direction == 'BUY'").iloc[0]
    assert buy.date == dates[1]
    assert buy.price == pytest.approx(10)
    assert result["summary"]["buy_count"] == 1


def test_fee_aware_lots_and_whole_account_nav_reconcile():
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    data = dataset(dates)
    data["entry"]["x"][0, 0] = True
    data["score"]["x"][0, 0] = 1
    data["close"][1:, 0] = [10., 12.]
    result = simulate_portfolio(data, SPEC, dates[0], dates[-1], capital=50000, slippage=0)
    buy = result["orders"].iloc[0]
    assert buy.qty % 100 == 0 and buy.qty == 2200
    expected_fee = max(5, buy.qty * buy.price * .00025) + buy.qty * buy.price * .00001
    assert buy.fees == pytest.approx(expected_fee)
    assert result["summary"]["nav"] == pytest.approx(
        result["summary"]["cash"] + buy.qty * 12)
    assert result["summary"]["net_pnl_reconciliation_error"] == pytest.approx(0)


def test_same_day_sale_proceeds_are_not_reused():
    dates = [f"2026-01-{day:02d}" for day in range(5, 9)]
    codes = ("000001.SZ", "000002.SZ")
    data = dataset(dates, codes)
    data["entry"]["x"][0, 0] = True
    data["score"]["x"][0, 0] = 2
    data["exit"]["x"][1, 0] = True
    data["entry"]["x"][1, 1] = True
    data["score"]["x"][1, 1] = 1
    data["open"][2, 1] = data["close"][2, 1] = 20
    data["up_limit"][2, 1], data["down_limit"][2, 1] = 22, 18
    result = simulate_portfolio(data, SPEC, dates[0], dates[-1], capital=3000, slippage=0)
    assert list(result["orders"].direction) == ["BUY", "SELL"]
    assert "insufficient_fee_aware_lot_budget" in set(result["skips"].reason)


def test_two_buys_two_sells_fit_weekly_limit_and_reserve_blocks_more():
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    codes = ("000001.SZ", "000002.SZ", "000003.SZ")
    data = dataset(dates, codes)
    data["entry"]["x"][0, :3] = True
    data["score"]["x"][0, :] = [3, 2, 1]
    data["exit"]["x"][1, :2] = True
    result = simulate_portfolio(data, SPEC, dates[0], dates[-1], capital=50000, slippage=0)
    assert list(result["orders"].direction).count("BUY") == 2
    assert list(result["orders"].direction).count("SELL") == 2
    assert result["summary"]["max_weekly_operations"] == 4


def test_announced_action_is_visible_only_after_announcement_and_unknown_halts():
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    data = dataset(dates)
    data["entry"]["x"][0, 0] = True
    data["score"]["x"][0, 0] = 1
    data["factor_changed"][3, 0] = True
    event = {"000001.SZ": [dict(announcement_date=dates[1], ex_date=dates[3],
                                    record_date=dates[2], source_cache="test")]}
    avoided = simulate_portfolio(data, SPEC, dates[0], dates[-1], events=event, slippage=0)
    assert avoided["summary"]["valid"]
    assert avoided["orders"].iloc[-1].date == dates[2]
    assert avoided["orders"].iloc[-1].reason == "announced_corporate_action"
    unknown = simulate_portfolio(data, SPEC, dates[0], dates[-1], slippage=0)
    assert not unknown["summary"]["valid"]
    assert unknown["blockers"][0]["reason"] == "corporate_action"


def test_max_hold_counts_market_sessions_and_sell_retries_five_days():
    dates = [f"2026-02-{day:02d}" for day in range(2, 11)]
    data = dataset(dates)
    data["entry"]["x"][0, 0] = True
    data["score"]["x"][0, 0] = 1
    # Entry is index 1 and max_hold=2, so first sell is index 3. Block it and
    # five retry opens; the sixth failure halts the incomplete account.
    data["open"][3:, 0] = 9
    data["down_limit"][3:, 0] = 9
    result = simulate_portfolio(data, SPEC, dates[0], dates[-1], slippage=0)
    assert result["orders"].iloc[0].date == dates[1]
    assert result["blockers"][-1]["reason"] == "exit_unfilled_after_5_retries"
    attempts = result["skips"].query("reason == 'exit_unfillable'")
    assert len(attempts) == 6


def test_invalid_completed_entry_day_preserves_buy_then_halts():
    dates = ["2026-03-02", "2026-03-03", "2026-03-04"]
    data = dataset(dates)
    data["entry"]["x"][0, 0] = True
    data["score"]["x"][0, 0] = 1
    data["day_valid"][1, 0] = False
    result = simulate_portfolio(data, SPEC, dates[0], dates[-1], slippage=0)
    assert len(result["orders"]) == 1
    assert result["orders"].iloc[0].direction == "BUY"
    assert result["blockers"][0]["reason"] == "invalid_completed_bar"
    assert np.isnan(result["summary"]["cagr"])


def test_suspension_keeps_last_raw_mark_and_reports_it_stale():
    dates = ["2026-04-06", "2026-04-07", "2026-04-08"]
    data = dataset(dates)
    data["entry"]["x"][0, 0] = True
    data["score"]["x"][0, 0] = 1
    data["vol"][2, 0] = 0
    data["close"][2, 0] = 99  # repeated/vendor value must not become the mark
    result = simulate_portfolio(data, SPEC, dates[0], dates[-1], slippage=0)
    holding = result["summary"]["open_positions"][0]
    assert holding["mark"] == 10
    assert holding["stale"] is True
    assert result["nav"].iloc[-1].stale_marks == ["000001.SZ"]


def test_announced_action_unfilled_exit_retries_are_not_reset_each_day():
    dates = [f"2026-05-{day:02d}" for day in range(4, 13)]
    data = dataset(dates)
    data["entry"]["x"][0, 0] = True
    data["score"]["x"][0, 0] = 1
    # The announcement becomes visible at index 2.  Keep every attempted open
    # locked at the lower limit through the initial attempt plus five retries.
    data["open"][2:, 0] = 9
    data["down_limit"][2:, 0] = 9
    event = {"000001.SZ": [dict(announcement_date=dates[1],
                                    ex_date="2026-05-29", record_date="2026-05-28",
                                    source_cache="test")]}
    long_hold = {**SPEC, "max_hold": 20}

    result = simulate_portfolio(data, long_hold, dates[0], dates[-1],
                                events=event, slippage=0)

    attempts = result["skips"].query("reason == 'exit_unfillable'")
    assert list(attempts.failed_attempts) == [1, 2, 3, 4, 5, 6]
    assert result["blockers"][-1]["reason"] == "exit_unfilled_after_5_retries"
    assert result["summary"]["valid"] is False


def test_opening_buy_does_not_use_execution_day_completed_volume():
    dates = ["2026-06-01", "2026-06-02", "2026-06-03"]
    positive = dataset(dates)
    positive["entry"]["x"][0, 0] = True
    positive["score"]["x"][0, 0] = 1
    zero = {key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in positive.items()}
    zero["entry"] = {"x": positive["entry"]["x"].copy()}
    zero["score"] = {"x": positive["score"]["x"].copy()}
    zero["exit"] = {"x": positive["exit"]["x"].copy()}
    zero["vol"][1, 0] = 0

    normal = simulate_portfolio(positive, SPEC, dates[0], dates[-1], slippage=0)
    suspended = simulate_portfolio(zero, SPEC, dates[0], dates[-1], slippage=0)

    normal_buy = normal["orders"].query("direction == 'BUY'").iloc[0]
    suspended_buy = suspended["orders"].query("direction == 'BUY'").iloc[0]
    assert suspended_buy[["date", "code", "qty", "price"]].to_dict() == \
        normal_buy[["date", "code", "qty", "price"]].to_dict()
    assert suspended["blockers"][0]["reason"] == "invalid_completed_bar"
    assert len(suspended["orders"]) == 1  # the opening buy remains auditable
