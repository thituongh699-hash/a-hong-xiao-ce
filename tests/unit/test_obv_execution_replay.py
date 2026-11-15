import pytest

from scripts.obv_execution_replay import COSTS, replay_trade


def bar(price=10.0, *, factor=True, adjustment=1.0, volume=10000.0,
        history=130, reference=True, up_limit=None, down_limit=None):
    upper = price * 1.1 if up_limit is None else up_limit
    lower = price * 0.9 if down_limit is None else down_limit
    return {
        "open": price,
        "high": price * 1.02,
        "low": price * 0.98,
        "close": price,
        "vol": volume,
        "amount": price * volume,
        "price_reference_valid": reference,
        "up_limit": upper,
        "down_limit": lower,
        "history_sessions": history,
        "adjustment_factor": adjustment,
        "obv_resonance_factor": factor,
    }


def test_signal_close_buys_only_next_market_open_and_horizon_is_t_plus_one():
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07"]
    bars = {
        calendar[0]: bar(8.0),
        calendar[1]: bar(10.0),
        calendar[2]: bar(11.0),
    }
    result = replay_trade("600000.SH", calendar[0], bars, calendar, horizon=1)

    assert result["status"] == "closed"
    assert result["entry_date"] == calendar[1]
    assert result["scheduled_exit_date"] == calendar[2]
    assert result["exit_date"] == calendar[2]
    assert result["entry_price"] == pytest.approx(10.0 * 1.001)
    assert result["holding_days"] == 1
    with pytest.raises(ValueError, match="horizon"):
        replay_trade("600000.SH", calendar[0], bars, calendar, horizon=0)


def test_fee_aware_affordability_rejects_nominal_100_share_cash():
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07"]
    bars = {day: bar(10.0) for day in calendar}
    result = replay_trade("000001.SZ", calendar[0], bars, calendar,
                          horizon=1, capital=1005.0)
    assert result["status"] == "no_entry"
    assert result["rejection_reason"] == "insufficient_cash_for_fee_aware_lot"
    assert result["qty"] == 0


def test_corporate_action_is_causal_and_censors_only_after_entry():
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    future_split = {
        calendar[0]: bar(10.0, adjustment=1.0),
        calendar[1]: bar(10.0, adjustment=1.0),
        calendar[2]: bar(10.5, adjustment=1.0),
        calendar[3]: bar(5.3, adjustment=2.0, up_limit=5.8, down_limit=4.8),
    }
    closed_before_split = replay_trade("600000.SH", calendar[0], future_split,
                                       calendar, horizon=1)
    assert closed_before_split["status"] == "closed"

    held_through_split = replay_trade("600000.SH", calendar[0], future_split,
                                      calendar, horizon=2)
    assert held_through_split["status"] == "censored_corporate_action"
    assert held_through_split["entry_date"] == calendar[1]
    assert held_through_split["holding_days"] == 2


def test_adjustment_change_on_entry_and_suspended_next_open_are_no_entry():
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07"]
    action = {calendar[0]: bar(adjustment=1), calendar[1]: bar(adjustment=2),
              calendar[2]: bar(adjustment=2)}
    assert replay_trade("600000.SH", calendar[0], action, calendar, 1)[
        "rejection_reason"] == "corporate_action_on_entry"

    suspended = {calendar[0]: bar(), calendar[1]: bar(volume=0), calendar[2]: bar()}
    result = replay_trade("600000.SH", calendar[0], suspended, calendar, 1)
    assert result["status"] == "no_entry"
    assert result["rejection_reason"] == "suspended_or_no_volume"


def test_open_reference_allows_entry_then_later_same_day_anomaly_censors():
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07"]
    bars = {day: bar(10.0) for day in calendar}
    bars[calendar[1]]["open_price_reference_valid"] = True
    bars[calendar[1]]["price_reference_valid"] = False

    result = replay_trade("600000.SH", calendar[0], bars, calendar, horizon=1)

    assert result["status"] == "censored_unverified_price_reference"
    assert result["entry_date"] == calendar[1]
    assert result["qty"] >= 100
    assert result["entry_price"] == pytest.approx(10.0 * 1.001)
    assert result["invested"] > 0
    assert result["raw_reference_anomaly_dates"] == [calendar[1]]


def test_unfillable_scheduled_sell_retries_subsequent_opens():
    calendar = [f"2026-01-{day:02d}" for day in range(5, 11)]
    bars = {day: bar(10.0) for day in calendar}
    # Scheduled close is locked down.  The first retry open is also locked;
    # the second retry open is tradable.
    bars[calendar[2]] = bar(9.0, down_limit=9.0)
    bars[calendar[2]].update(open=9.0, high=9.0, low=9.0, close=9.0)
    bars[calendar[3]] = bar(8.1, down_limit=8.1)
    bars[calendar[3]].update(open=8.1, high=8.2, low=8.1, close=8.15)
    bars[calendar[4]] = bar(8.3, down_limit=7.4, up_limit=9.0)

    result = replay_trade("000001.SZ", calendar[0], bars, calendar, horizon=1)
    assert result["status"] == "closed"
    assert result["scheduled_exit_date"] == calendar[2]
    assert result["exit_date"] == calendar[4]
    assert result["delayed_sell_days"] == 2
    assert result["exit_reason"] == "delayed_fixed_horizon_open"


def test_cash_pnl_fees_and_economic_price_return_reconcile():
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07"]
    bars = {calendar[0]: bar(10), calendar[1]: bar(10), calendar[2]: bar(12)}
    capital = 100000.0
    result = replay_trade("600000.SH", calendar[0], bars, calendar, 1, capital)

    buy_amount = result["entry_price"] * result["qty"]
    buy_fee = max(5.0, buy_amount * COSTS["trading_cost.commission_rate"])
    buy_fee += buy_amount * COSTS["trading_cost.transfer_fee"]
    sell_amount = result["exit_price"] * result["qty"]
    sell_fee = max(5.0, sell_amount * COSTS["trading_cost.commission_rate"])
    sell_fee += sell_amount * (COSTS["trading_cost.stamp_duty"]
                               + COSTS["trading_cost.transfer_fee"])
    expected_pnl = sell_amount - sell_fee - buy_amount - buy_fee

    assert result["realized_pnl"] == pytest.approx(expected_pnl)
    assert result["fees"] == pytest.approx(buy_fee + sell_fee)
    assert result["net_return_pct"] == pytest.approx(expected_pnl / capital * 100)
    assert result["economic_price_return_pct"] == pytest.approx(
        (result["exit_price"] / result["entry_price"] - 1) * 100)


def test_factor_loss_observed_at_close_exits_next_market_open():
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
                "2026-01-09"]
    bars = {day: bar(10 + i, factor=True) for i, day in enumerate(calendar)}
    bars[calendar[2]]["obv_resonance_factor"] = False
    result = replay_trade("600000.SH", calendar[0], bars, calendar, horizon=20,
                          exit_on_factor_loss=True)

    assert result["status"] == "closed"
    assert result["scheduled_exit_date"] == calendar[3]
    assert result["exit_date"] == calendar[3]
    assert result["exit_reason"] == "factor_loss_next_open"
    assert result["holding_days"] == 2


def test_entry_day_factor_loss_exits_only_at_following_market_open():
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    bars = {
        calendar[0]: bar(8.0, factor=True),
        calendar[1]: bar(10.0, factor=False),
        calendar[2]: bar(12.0, factor=True),
        calendar[3]: bar(13.0, factor=True),
    }

    result = replay_trade("600000.SH", calendar[0], bars, calendar, horizon=20,
                          exit_on_factor_loss=True)

    assert result["status"] == "closed"
    assert result["entry_date"] == calendar[1]
    assert result["entry_price"] == pytest.approx(10.0 * 1.001)
    assert result["scheduled_exit_date"] == calendar[2]
    assert result["exit_date"] == calendar[2]
    assert result["exit_price"] == pytest.approx(12.0 * (1 - 0.001))
    assert result["holding_days"] == 1


def test_raw_reference_anomaly_is_reported_separately():
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07"]
    bars = {calendar[0]: bar(), calendar[1]: bar(),
            calendar[2]: bar(reference=False)}
    result = replay_trade("600000.SH", calendar[0], bars, calendar, 1)
    assert result["status"] == "censored_unverified_price_reference"
    assert result["raw_reference_anomaly"] is True
    assert result["raw_reference_anomaly_dates"] == [calendar[2]]


def test_open_exit_cannot_be_erased_by_later_same_day_range_anomaly():
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    bars = {calendar[0]: bar(10), calendar[1]: bar(10, factor=False),
            calendar[2]: bar(11, reference=False), calendar[3]: bar(11)}
    bars[calendar[2]].update(open_price_reference_valid=True, high=100.)
    result = replay_trade("600000.SH", calendar[0], bars, calendar, 20,
                          exit_on_factor_loss=True)
    assert result["status"] == "closed"
    assert result["exit_date"] == calendar[2]
    assert result["exit_price"] == pytest.approx(11 * .999)
