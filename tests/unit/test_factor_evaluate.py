import numpy as np
import pandas as pd
import pytest

from src.factor.evaluate import (
    group_turnover,
    ic_series,
    ic_summary,
    layered_stats,
    quantile_returns,
)


def _cross_section_frame(n_names=20, n_periods=3, slope=0.001, direction=1):
    rows = []
    for period in range(n_periods):
        dt = pd.Timestamp("2024-01-02") + pd.Timedelta(days=10 * period)
        for i in range(n_names):
            score = float(i + 1)
            rows.append({
                "dt": dt,
                "code": f"{600000 + i}",
                "score": score,
                "fwd_ret": direction * slope * score,
            })
    return pd.DataFrame(rows)


def test_rank_ic_is_one_when_factor_order_matches_return_order():
    frame = _cross_section_frame()
    ic = ic_series(frame, min_names=5)
    assert len(ic) == 3
    assert ic.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_rank_ic_skips_cross_sections_below_minimum_names():
    frame = _cross_section_frame(n_names=4)
    assert ic_series(frame, min_names=5).empty


def test_rank_ic_is_undefined_for_constant_factor():
    frame = _cross_section_frame()
    frame["score"] = 1.0
    assert ic_series(frame, min_names=5).empty


def test_ic_summary_reports_nan_ratio_when_dispersion_is_zero():
    summary = ic_summary(pd.Series([0.1, 0.1, 0.1]), direction=1)
    assert summary["mean"] == pytest.approx(0.1)
    assert summary["n"] == 3
    assert summary["hit_rate"] == pytest.approx(1.0)
    assert np.isnan(summary["icir"])
    assert np.isnan(summary["t"])


def test_ic_summary_direction_adjustment_flips_sign_and_hit_rate():
    ic = pd.Series([0.1, 0.1, -0.1])
    adjusted = ic_summary(ic, direction=-1)
    assert adjusted["mean"] == pytest.approx(0.1 / 3)
    assert adjusted["mean_adj"] == pytest.approx(-0.1 / 3)
    # 方向为负时，只有 IC<0 的那一期算「方向正确」
    assert adjusted["hit_rate_adj"] == pytest.approx(1 / 3)


def test_quantile_groups_are_equal_sized_and_ordered():
    groups = quantile_returns(_cross_section_frame(), quantiles=5, min_names=20, min_names_per_group=4)
    assert list(groups.columns) == [1, 2, 3, 4, 5]
    means = groups.mean()
    assert means[1] == pytest.approx(0.0025)
    assert means[5] == pytest.approx(0.0185)


def test_layered_stats_reports_perfect_monotonicity_and_spread():
    groups = quantile_returns(_cross_section_frame(), quantiles=5, min_names=20, min_names_per_group=4)
    stats = layered_stats(groups, direction=1)
    assert stats["monotone"] == pytest.approx(1.0)
    assert stats["monotone_adj"] == pytest.approx(1.0)
    assert stats["reversals"] == 0
    assert stats["spread_mean"] == pytest.approx(0.016)
    assert stats["best_group"] == "Q5"
    assert stats["long_excess"] == pytest.approx(0.008)


def test_layered_stats_reverses_group_order_for_inverse_factor():
    frame = _cross_section_frame(direction=-1)
    groups = quantile_returns(frame, quantiles=5, min_names=20, min_names_per_group=4)
    stats = layered_stats(groups, direction=-1)
    assert stats["monotone"] == pytest.approx(-1.0)
    assert stats["monotone_adj"] == pytest.approx(1.0)
    assert stats["best_group"] == "Q1"
    assert stats["spread_mean"] == pytest.approx(0.016)


def test_layered_stats_counts_adjacent_reversals():
    frame = _cross_section_frame()
    # 打乱中间组，使组均收益不再单调
    frame.loc[frame["score"].between(5, 8), "fwd_ret"] = 0.05
    groups = quantile_returns(frame, quantiles=5, min_names=20, min_names_per_group=4)
    stats = layered_stats(groups, direction=1)
    assert stats["reversals"] >= 1
    assert stats["monotone_adj"] < 1.0


def test_turnover_is_zero_when_group_membership_never_changes():
    frame = _cross_section_frame(n_periods=4)
    frame["score"] = 1.0
    assert group_turnover(frame, quantiles=5, min_names=20) == pytest.approx(0.0)


def test_turnover_is_one_when_membership_fully_rotates():
    rows = []
    for period in range(2):
        dt = pd.Timestamp("2024-01-02") + pd.Timedelta(days=10 * period)
        for i in range(20):
            base = i + 1 if period == 0 else 20 - i
            rows.append({"dt": dt, "code": f"{600000 + i}", "score": float(base), "fwd_ret": 0.0})
    assert group_turnover(pd.DataFrame(rows), quantiles=5, min_names=20) == pytest.approx(1.0)
