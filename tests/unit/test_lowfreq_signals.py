import numpy as np
import pandas as pd
import pytest

from scripts.lowfreq_signals import SPECS, candidate_signals, prepare_frame


def bars(count=150):
    close = 10 + np.arange(count) * .02
    return pd.DataFrame({
        "dt": pd.bdate_range("2024-01-02", periods=count),
        "open": close - .02,
        "high": close + .05,
        "low": close - .05,
        "close": close,
        "vol": np.full(count, 1_000_000.0),
        "amount": np.full(count, 80_000_000.0),
        "adjustment_factor": np.ones(count),
    })


def spec(candidate_id):
    return next(item for item in SPECS if item["id"] == candidate_id)


def test_exact_ten_fixed_interpretable_specs():
    assert [item["id"] for item in SPECS] == [
        "breakout20", "breakout60", "pullback3", "pullback5", "reversal3",
        "reversal5", "compression20", "compression40", "rotation20", "rotation60",
    ]
    assert sum(item["weekly_only"] for item in SPECS) == 2
    modified = dict(spec("breakout20"), lookback=21)
    with pytest.raises(ValueError, match="predefined"):
        candidate_signals(prepare_frame(bars()), modified)


def test_prepare_keeps_raw_execution_prices_and_is_split_scale_invariant():
    plain = bars()
    split = plain.copy(deep=True)
    split.loc[75:, ["open", "high", "low", "close"]] /= 2
    split.loc[75:, "adjustment_factor"] = 2
    before = split.copy(deep=True)
    expected = prepare_frame(plain)
    actual = prepare_frame(split)
    pd.testing.assert_frame_equal(split, before)
    pd.testing.assert_frame_equal(actual[["s_open", "s_high", "s_low", "s_close"]],
                                  expected[["s_open", "s_high", "s_low", "s_close"]])
    pd.testing.assert_series_equal(actual["close"], split["close"])
    assert actual.loc[75, "close"] != actual.loc[75, "s_close"]


@pytest.mark.parametrize("candidate_id", [item["id"] for item in SPECS])
@pytest.mark.parametrize("end", [1, 20, 120, 121, 139])
def test_features_and_signals_are_prefix_causal(candidate_id, end):
    raw = bars()
    raw.loc[140:, ["open", "high", "low", "close"]] *= 100
    raw.loc[145:, "vol"] = 1e12
    full_frame = prepare_frame(raw)
    prefix_frame = prepare_frame(raw.iloc[:end])
    pd.testing.assert_frame_equal(full_frame.iloc[:end], prefix_frame)
    full = candidate_signals(full_frame, spec(candidate_id)).iloc[:end]
    prefix = candidate_signals(prefix_frame, spec(candidate_id))
    pd.testing.assert_frame_equal(full, prefix)


def test_breakout_uses_strict_prior_high_and_excludes_current_candle():
    raw = bars(125)
    raw.loc[124, "close"] = raw.loc[103:123, "high"].max() + .01
    raw.loc[124, "open"] = raw.loc[124, "close"] - .01
    raw.loc[124, "high"] = raw.loc[124, "close"] + 100  # must not enter prior window
    raw.loc[124, "low"] = raw.loc[124, "open"] - .01
    raw.loc[124, "vol"] = 1_300_000
    prepared = prepare_frame(raw)
    signals = candidate_signals(prepared, spec("breakout20"))
    assert prepared.loc[124, "prior_high20"] == pytest.approx(raw.loc[104:123, "high"].max())
    assert signals.loc[124, "entry"]
    raw.loc[124, "close"] = prepared.loc[124, "prior_high20"]
    raw.loc[124, "open"] = raw.loc[124, "close"]
    raw.loc[124, "low"] = raw.loc[124, "close"] - .01
    assert not candidate_signals(prepare_frame(raw), spec("breakout20")).loc[124, "entry"]


@pytest.mark.parametrize("candidate_id", [item["id"] for item in SPECS])
def test_no_entry_before_120_completed_bars(candidate_id):
    signals = candidate_signals(prepare_frame(bars()), spec(candidate_id))
    assert not signals["entry"].iloc[:120].any()
    assert signals["entry"].dtype == bool
    assert signals["score"].dtype == float
    assert signals["exit"].dtype == bool


def test_return_and_score_units_are_decimal_not_percent():
    raw = bars(125)
    raw.loc[124, ["open", "high", "low", "close"]] = [12.9, 13.1, 12.8, 13.0]
    prepared = prepare_frame(raw)
    expected = prepared.loc[124, "s_close"] / prepared.loc[104, "s_close"] - 1
    assert prepared.loc[124, "momentum20"] == pytest.approx(expected)
    rotation = candidate_signals(prepared, spec("rotation20"))
    assert rotation.loc[124, "score"] == pytest.approx(expected)
    assert 0 < rotation.loc[124, "score"] < 1


@pytest.mark.parametrize("missing", [
    "dt", "open", "high", "low", "close", "vol", "amount", "adjustment_factor",
])
def test_missing_raw_input_fails_closed(missing):
    with pytest.raises(ValueError, match="lack columns"):
        prepare_frame(bars().drop(columns=missing))


@pytest.mark.parametrize("kind", ["nan", "zero_factor", "negative_vol", "unsorted", "bad_ohlc"])
def test_invalid_raw_input_fails_closed(kind):
    raw = bars()
    if kind == "nan":
        raw.loc[3, "amount"] = np.nan
    elif kind == "zero_factor":
        raw.loc[3, "adjustment_factor"] = 0
    elif kind == "negative_vol":
        raw.loc[3, "vol"] = -1
    elif kind == "unsorted":
        raw.loc[3, "dt"], raw.loc[4, "dt"] = raw.loc[4, "dt"], raw.loc[3, "dt"]
    elif kind == "bad_ohlc":
        raw.loc[3, "high"] = raw.loc[3, "close"] - 1
    with pytest.raises(ValueError):
        prepare_frame(raw)


def test_common_liquidity_uses_prior_amount_and_positive_current_volume():
    raw = bars(125)
    raw.loc[124, ["open", "high", "low", "close"]] = [13.0, 13.2, 12.9, 13.1]
    prepared = prepare_frame(raw)
    liquid = candidate_signals(prepared, spec("rotation20"))
    assert liquid.loc[124, "entry"]
    raw.loc[104:123, "amount"] = 49_999_999
    assert not candidate_signals(prepare_frame(raw), spec("rotation20")).loc[124, "entry"]
    raw = bars(125)
    raw.loc[124, ["open", "high", "low", "close"]] = [13.0, 13.2, 12.9, 13.1]
    raw.loc[124, "vol"] = 0
    assert not candidate_signals(prepare_frame(raw), spec("rotation20")).loc[124, "entry"]


def test_candidate_output_preserves_custom_index_and_dt():
    raw = bars()
    raw.index = np.arange(1000, 1000 + len(raw))
    prepared = prepare_frame(raw)
    result = candidate_signals(prepared, spec("rotation20"))
    pd.testing.assert_index_equal(result.index, raw.index)
    pd.testing.assert_series_equal(result["dt"], prepared["dt"])
    assert list(result.columns) == ["dt", "entry", "score", "exit"]


def test_missing_prepared_input_fails_closed():
    with pytest.raises(ValueError, match="lacks columns"):
        candidate_signals(prepare_frame(bars()).drop(columns="ma60"), spec("breakout20"))
