import numpy as np
import pandas as pd
import pytest

from src.strategies.obv_resonance_factor import features, signal_mask
from src.utils.price_adjustment import PriceAdjustmentError, _factor_frame


def bars(count=115):
    days = pd.bdate_range("2025-01-06", periods=count)
    close = 10 + np.arange(count) * .1
    frame = pd.DataFrame(dict(code="000001.SZ", dt=days, open=close,
                              high=close + .1, low=close - .1, close=close,
                              vol=np.arange(count) * 100 + 10000.))
    frame.attrs["raw_price_verified_external"] = True
    return frame


def factors(raw, events=()):
    return _factor_frame([dict(dt=d, factor=f) for d, f in events], raw.dt.max())


def reference(raw, ma_period):
    """Slow chart calculation, recomputed from only each day's input prefix."""
    result = []
    for end in range(1, len(raw) + 1):
        subset = raw.iloc[:end]
        daily = (np.sign(subset.close.diff()).fillna(0) * subset.vol).cumsum() / 10000
        weekly = subset.groupby(subset.dt.dt.to_period("W-FRI")).agg(
            close=("close", "last"), vol=("vol", "sum"))
        obv = (np.sign(weekly.close.diff()).fillna(0) * weekly.vol).cumsum() / 10000
        d_ma, w_ma = daily.rolling(ma_period).mean().iloc[-1], obv.rolling(ma_period).mean().iloc[-1]
        rising = len(obv) >= 3 and obv.iloc[-1] > obv.iloc[-2] > obv.iloc[-3]
        result.append(dict(daily_obv=daily.iloc[-1], daily_maobv=d_ma,
                           weekly_obv=obv.iloc[-1], weekly_maobv=w_ma,
                           weekly_volume=weekly.vol.iloc[-1],
                           obv_resonance_factor=int(rising and obv.iloc[-1] > w_ma
                                                    and daily.iloc[-1] > d_ma)))
    return pd.DataFrame(result)


def set_close(raw, index, value):
    raw.loc[index, ["open", "high", "low", "close"]] = [value, value + .1, value - .1, value]


def test_default_18_warmup_and_persistent_state():
    raw = bars()
    work = features(raw, factors(raw))
    assert not work.history_ready.iloc[:85].any()
    assert work.history_ready.iloc[85:].all()  # 17 completed weeks + current week
    assert work.obv_resonance_factor.iloc[85:].eq(1).all()
    assert signal_mask(work, first_only=True).sum() == 1
    assert signal_mask(work, first_only=True).iloc[85]
    pd.testing.assert_series_equal(signal_mask(work), work.obv_resonance_factor.eq(1))


def test_matches_independent_per_day_chart_recalculation():
    raw = bars(44)
    for i in range(len(raw)):
        set_close(raw, i, 10 + np.sin(i * .8) + i * .015)
    # Holiday-shortened week, empty week, and a tied daily close.
    raw = raw.drop(index=[3, 4, 15, 16, 17, 18, 19]).reset_index(drop=True)
    set_close(raw, 8, raw.close.iloc[7])
    expected = reference(raw, 4)
    actual = features(raw, factors(raw), ma_period=4)
    for column in expected:
        np.testing.assert_allclose(actual[column], expected[column], equal_nan=True,
                                   err_msg=column)


@pytest.mark.parametrize("end", [1, 5, 85, 86, 87, 89, 90, 91, 93, 110])
def test_history_is_prefix_invariant_even_with_future_extremes_and_split(end):
    raw = bars()
    raw.loc[95:, ["open", "high", "low", "close"]] /= 2
    raw.loc[99, "vol"] = 1e12
    set_close(raw, 109, 500)
    adjustment = factors(raw, [(str(raw.dt.iloc[95].date()), 2.)])
    full = features(raw, adjustment)
    partial = features(raw.iloc[:end], adjustment)
    pd.testing.assert_frame_equal(full.iloc[:end].reset_index(drop=True), partial)


def test_partial_week_can_turn_off_then_back_on_without_rewriting_past():
    raw = bars(95)
    set_close(raw, 86, raw.close.iloc[84] - 1)  # Tuesday below last Friday
    set_close(raw, 87, raw.close.iloc[84] + 1)  # Wednesday above last Friday
    work = features(raw, factors(raw))
    assert work.obv_resonance_factor.iloc[85:88].tolist() == [1, 0, 1]
    assert work.resonance_trigger.iloc[85:88].tolist() == [True, False, True]
    assert work.weekly_obv.iloc[86] < work.weekly_prev_obv.iloc[86]
    assert work.weekly_obv.iloc[87] > work.weekly_prev_obv.iloc[87]
    assert work.weekly_volume.iloc[87] == raw.vol.iloc[85:88].sum()


def test_two_weekly_increases_need_three_obv_points():
    raw = bars(10)
    work = features(raw, factors(raw), ma_period=2)
    assert work.weekly_above_maobv.iloc[-1]
    assert not work.weekly_rising_two.any()
    assert not signal_mask(work).any()


def test_equal_weekly_close_and_obv_do_not_count_as_rising():
    raw = bars(95)
    set_close(raw, 85, raw.close.iloc[84])
    work = features(raw, factors(raw))
    assert work.weekly_obv.iloc[85] == work.weekly_prev_obv.iloc[85]
    assert not work.weekly_rising_two.iloc[85]
    assert not signal_mask(work).iloc[85]


def test_daily_obv_above_ma_is_required_even_with_two_rising_weeks():
    raw = bars(115)
    # One very large sell-volume day leaves daily OBV below its MA, but its
    # close is still above last Friday: the weekly condition remains true.
    set_close(raw, 87, raw.close.iloc[86] - .05)
    raw.loc[87, "vol"] = 1e8
    work = features(raw, factors(raw))
    assert work.weekly_rising_two.iloc[87] and work.weekly_above_maobv.iloc[87]
    assert not work.daily_above_maobv.iloc[87]
    assert not signal_mask(work).iloc[87]


def test_weekly_above_ma_is_required_even_after_two_rising_weeks():
    raw = bars(20)
    for i in range(20):
        set_close(raw, i, [20., 10., 11., 12.][i // 5] + (i % 5) * .01)
    raw["vol"] = 1000.
    raw.loc[5:9, "vol"] = 1e7  # Large earlier weekly OBV drawdown
    work = features(raw, factors(raw), ma_period=4)
    assert work.weekly_rising_two.iloc[-1] and work.daily_above_maobv.iloc[-1]
    assert not work.weekly_above_maobv.iloc[-1]
    assert not signal_mask(work).iloc[-1]


def test_weekly_obv_is_recomputed_from_week_volume_not_sampled_daily_obv():
    raw = bars(95)
    work = features(raw, factors(raw))
    assert work.weekly_obv.iloc[89] - work.weekly_prev_obv.iloc[89] == pytest.approx(raw.vol.iloc[85:90].sum() / 10000)
    assert work.weekly_obv.iloc[89] != work.daily_obv.iloc[89]


def test_adjust_daily_prices_before_week_aggregation_and_keep_volume_raw():
    raw = bars(95)
    split_index = 87  # Wednesday, inside the partial week
    split = raw.copy(deep=True)
    split.loc[split_index:, ["open", "high", "low", "close"]] /= 2
    original = split.copy(deep=True)
    split_factors = factors(split, [(str(split.dt.iloc[split_index].date()), 2.)])
    plain = features(raw, factors(raw))
    adjusted = features(split, split_factors)
    cols = ["daily_obv", "daily_maobv", "weekly_obv", "weekly_maobv", "weekly_volume", "obv_resonance_factor"]
    pd.testing.assert_frame_equal(plain[cols], adjusted[cols])
    pd.testing.assert_series_equal(adjusted.vol, raw.vol)
    assert adjusted.close.iloc[87] == raw.close.iloc[87] / 2
    assert adjusted.signal_close.iloc[87] == raw.close.iloc[87]
    pd.testing.assert_frame_equal(split, original)


def test_flat_and_zero_volume_do_not_trigger():
    raw = bars(100)
    raw[["open", "high", "low", "close"]] = [10., 10.1, 9.9, 10.]
    flat = features(raw, factors(raw))
    assert flat.daily_obv.eq(0).all() and flat.weekly_obv.eq(0).all()
    assert not signal_mask(flat).any()
    raw = bars(100)
    raw["vol"] = 0.
    assert not signal_mask(features(raw, factors(raw))).any()


@pytest.mark.parametrize("period", [True, 1, 0, -1, 18.0, "18"])
def test_invalid_ma_period_rejected(period):
    raw = bars()
    with pytest.raises(ValueError, match="integer"):
        features(raw, factors(raw), ma_period=period)


@pytest.mark.parametrize("kind", ["empty", "duplicate", "multi_code", "negative_vol", "nan_vol", "bad_ohlc", "bad_date", "weekend", "missing_vol"])
def test_bad_input_fails_closed(kind):
    raw = bars()
    if kind == "empty":
        raw = raw.iloc[:0]
    elif kind == "duplicate":
        raw.loc[1, "dt"] = raw.dt.iloc[0] + pd.Timedelta(hours=15)
    elif kind == "multi_code":
        raw.loc[1, "code"] = "600000.SH"
    elif kind == "negative_vol":
        raw.loc[1, "vol"] = -1
    elif kind == "nan_vol":
        raw.loc[1, "vol"] = np.nan
    elif kind == "bad_ohlc":
        raw.loc[1, "high"] = 0
    elif kind == "bad_date":
        raw["dt"] = raw.dt.astype(str)
        raw.loc[1, "dt"] = "bad"
    elif kind == "weekend":
        raw.loc[1, "dt"] = pd.Timestamp("2025-01-11")
    elif kind == "missing_vol":
        raw = raw.drop(columns="vol")
    with pytest.raises(ValueError):
        features(raw, factors(bars()))


def test_raw_provenance_and_verified_factor_coverage_required():
    raw = bars()
    unverified = raw.copy()
    unverified.attrs.clear()
    with pytest.raises(PriceAdjustmentError, match="raw price mode is unverified"):
        features(unverified, factors(raw))
    raw.attrs["price_mode"] = "qfq"
    with pytest.raises(PriceAdjustmentError, match="adjusted prices"):
        features(raw, factors(raw))
    raw = bars()
    unverified_factors = factors(raw)
    unverified_factors.attrs.clear()
    with pytest.raises(PriceAdjustmentError, match="not verified"):
        features(raw, unverified_factors)
    with pytest.raises(PriceAdjustmentError, match="coverage"):
        features(raw, factors(raw.iloc[:-1]))


def test_sorted_input_timezone_and_original_data_preserved():
    raw = bars()
    raw["dt"] = raw.dt.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    original = raw.copy(deep=True)
    actual = features(raw.iloc[::-1], factors(bars()))
    expected = features(bars(), factors(bars()))
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(raw, original)
