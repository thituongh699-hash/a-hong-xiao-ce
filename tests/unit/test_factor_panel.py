import numpy as np
import pandas as pd
import pytest

from src.factor.panel import (
    build_panel,
    compute_factor,
    forward_returns,
    price_limit_pct,
    rebalance_dates,
    tradability,
    attach_labels,
)
from src.factor.spec import FactorSpec


def _bars(closes, code="600000", start="2024-01-02"):
    close = np.asarray(closes, dtype=float)
    dates = pd.bdate_range(start, periods=len(close))
    return {
        code: pd.DataFrame({
            "dt": dates,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "vol": np.full(len(close), 1_000_000.0),
            "amount": close * 1_000_000.0,
        })
    }


def test_forward_return_starts_after_entry_lag_not_on_signal_day():
    bars = _bars([10, 11, 12, 13, 14, 15])
    panel = build_panel(bars).panel
    result = forward_returns(panel, horizon=2, entry_lag=1)
    # T=0 信号，T+1 收盘 11 建仓，T+3 收盘 13 离场
    assert result.iloc[0] == pytest.approx(13 / 11 - 1)


def test_forward_return_with_zero_entry_lag_uses_signal_close():
    panel = build_panel(_bars([10, 11, 12, 13])).panel
    result = forward_returns(panel, horizon=2, entry_lag=0)
    assert result.iloc[0] == pytest.approx(12 / 10 - 1)


def test_forward_return_ignores_the_untradable_signal_day_move():
    base = build_panel(_bars([10, 11, 12, 13, 14])).panel
    shifted = build_panel(_bars([99, 11, 12, 13, 14])).panel
    assert forward_returns(base, 2, 1).iloc[0] == pytest.approx(
        forward_returns(shifted, 2, 1).iloc[0]
    )


def test_limit_up_entry_day_is_marked_untradable_on_main_board():
    panel = build_panel(_bars([10, 11, 12, 12])).panel
    flags = tradability(panel, horizon=1, entry_lag=1)
    assert bool(flags["entry_ok"].iloc[0]) is False
    # 次日为 10% 涨停，无法以收盘价买入


def test_same_move_is_tradable_on_20pct_board():
    panel = build_panel(_bars([10, 11, 12, 12], code="300001")).panel
    flags = tradability(panel, horizon=1, entry_lag=1)
    assert bool(flags["entry_ok"].iloc[0]) is True


def test_untradable_entry_masks_forward_return_instead_of_reporting_it():
    panel = build_panel(_bars([10, 11, 12, 12])).panel
    labelled = attach_labels(panel, horizon=1, entry_lag=1)
    assert pd.isna(labelled["fwd_ret"].iloc[0])
    assert labelled["fwd_ret"].iloc[1] == pytest.approx(0.0)


def test_price_limit_matches_board_prefix():
    assert price_limit_pct("600519") == pytest.approx(0.10)
    assert price_limit_pct("300750") == pytest.approx(0.20)
    assert price_limit_pct("688981") == pytest.approx(0.20)
    assert price_limit_pct("830799") == pytest.approx(0.30)
    assert price_limit_pct("600519.SH") == pytest.approx(0.10)


def test_duplicate_dates_reject_the_whole_symbol():
    bars = _bars([10, 11, 12])
    bars.update(_bars([20, 21, 22], code="600001"))
    bars["600000"] = pd.concat([bars["600000"], bars["600000"].iloc[[1]]], ignore_index=True)
    build = build_panel(bars)
    assert build.quality["codes_used"] == 1
    assert "存在重复日期" in build.quality["codes_rejected"]["600000"]


def test_broken_high_low_envelope_is_rejected():
    bars = _bars([10, 11, 12])
    bars.update(_bars([20, 21, 22], code="600001"))
    bars["600000"].loc[1, "high"] = 5.0
    build = build_panel(bars)
    assert "OHLC" in build.quality["codes_rejected"]["600000"]
    assert build.quality["codes_used"] == 1


def test_panel_raises_when_every_symbol_is_rejected():
    with pytest.raises(ValueError):
        build_panel({"600000": pd.DataFrame()})


def test_rebalance_dates_takes_non_overlapping_steps():
    panel = build_panel(_bars(list(np.linspace(10, 20, 10)))).panel
    dates = rebalance_dates(panel, step=3)
    assert len(dates) == 4
    assert dates[1] == panel["dt"].iloc[3]


def test_compute_factor_is_invariant_to_future_rows():
    spec = FactorSpec(
        factor_id="rev_test", name="测试反转", category="量价", direction=-1,
        hypothesis="仅用于测试因果性", lookback=3,
        compute=lambda df, p: df["close"] / df["close"].shift(2) - 1,
    )
    full = build_panel(_bars([10, 11, 12, 13, 14, 15, 16, 17])).panel
    truncated = full.loc[full["dt"] <= full["dt"].iloc[4]].copy()
    values_full = compute_factor(full, spec)
    values_truncated = compute_factor(truncated, spec)
    assert np.array_equal(
        values_full.iloc[:5].to_numpy(dtype=float),
        values_truncated.to_numpy(dtype=float),
        equal_nan=True,
    )


def test_compute_factor_rejects_output_with_wrong_length():
    spec = FactorSpec(
        factor_id="bad", name="坏因子", category="量价", direction=1,
        hypothesis="输出长度不对", lookback=1,
        compute=lambda df, p: pd.Series([1.0, 2.0]),
    )
    panel = build_panel(_bars([10, 11, 12, 13])).panel
    with pytest.raises(ValueError):
        compute_factor(panel, spec)
