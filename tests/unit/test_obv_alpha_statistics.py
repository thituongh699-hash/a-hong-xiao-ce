import numpy as np
import pandas as pd
import pytest

from scripts.obv_alpha_statistics import daily_effects, mean_inference


def test_hac_zero_lag_equals_sample_variance_over_n():
    x = np.array([1.0, 3.0, 2.0, 6.0])
    got = mean_inference(x, maxlags=0, draws=20, seed=4)
    assert got["hac_se_pct"] == pytest.approx(x.std(ddof=1) / np.sqrt(len(x)))


def test_hac_lag_one_uses_calendar_lag_covariance_and_bartlett_weight():
    x = np.array([1.0, 2.0, 4.0, 3.0])
    n = len(x)
    r = x - x.mean()
    expected_var = (r @ r + (1 - 1 / 2) * 2 * (r[1:] @ r[:-1])) / n**2 * n / (n - 1)
    got = mean_inference(x, maxlags=1, draws=20)
    assert got["hac_se_pct"] ** 2 == pytest.approx(expected_var)


def test_hac_respects_missing_calendar_slots_instead_of_compressing():
    spaced = np.array([1.0, np.nan, 1.0, np.nan, 1.0, np.nan, 1.0])
    compact = np.array([1.0, 1.0, 1.0, 1.0])
    spaced_result = mean_inference(spaced, maxlags=1, draws=20)
    compact_result = mean_inference(compact, maxlags=1, draws=20)
    assert spaced_result["hac_se_pct"] == 0
    assert compact_result["hac_se_pct"] == 0

    # Alternating observations are five calendar slots apart. Their lag-one
    # HAC covariance must not be treated as adjacent as it is after compression.
    spaced = np.array([1.0, np.nan, np.nan, np.nan, np.nan, 2.0,
                       3.0, np.nan, np.nan, np.nan, np.nan, 4.0])
    full = mean_inference(spaced, maxlags=1, draws=20)
    compressed = mean_inference(spaced[np.isfinite(spaced)], maxlags=1, draws=20)
    assert full["hac_se_pct"] != pytest.approx(compressed["hac_se_pct"])


def test_hac_se_increases_for_positive_calendar_autocorrelation():
    independent = np.array([-1., 1.] * 50)
    persistent = np.repeat(np.array([-1., 1.] * 10), 5)
    se_i = mean_inference(independent, maxlags=5, draws=20)["hac_se_pct"]
    se_p = mean_inference(persistent, maxlags=5, draws=20)["hac_se_pct"]
    assert se_p > se_i


def test_block_bootstrap_is_seed_deterministic_and_handles_empty_values():
    x = np.array([1., 2., np.nan, 8., -1., 4.])
    first = mean_inference(x, maxlags=2, block_length=3, draws=100, seed=91)
    second = mean_inference(x, maxlags=2, block_length=3, draws=100, seed=91)
    assert first == second
    empty = mean_inference([np.nan, np.inf], maxlags=2, draws=10)
    assert empty["n"] == 0 and np.isnan(empty["mean_pct"])


def test_vectorized_bootstrap_matches_independent_scalar_indices():
    x = np.array([1., 2., np.nan, 8., -1., 4., 2.])
    rng = np.random.default_rng(91)
    values = []
    for _ in range(600):
        starts = rng.integers(0, len(x), size=3)
        indices = np.concatenate([(start + np.arange(3)) % len(x) for start in starts])[:len(x)]
        values.append(np.nanmean(x[indices]))
    got = mean_inference(x, maxlags=2, block_length=3, draws=600, seed=91)
    low, high = np.quantile(values, [.025, .975])
    assert got["bootstrap_ci_low_pct"] == pytest.approx(low)
    assert got["bootstrap_ci_high_pct"] == pytest.approx(high)


def _panel(rows):
    frame = pd.DataFrame(rows)
    frame["dt"] = pd.Timestamp("2025-01-02")
    frame["eligible"] = True
    frame["trigger"] = False
    frame["daily_condition"] = True
    frame["weekly_condition"] = True
    for h in (1, 5, 10, 20):
        frame[f"fwd_close_{h}"] = frame["y"]
        frame[f"label_end_{h}"] = frame.dt + pd.to_timedelta(h, unit="D")
    return frame


def test_matched_and_conditional_effect_remove_momentum_confounding():
    rng = np.random.default_rng(123)
    n = 3000
    mom = rng.normal(size=n)
    vol = rng.normal(size=n)
    probability = 1 / (1 + np.exp(-0.9 * mom))
    factor = rng.binomial(1, probability)
    y = 2.0 * mom + 0.3 * vol + rng.normal(0, 0.5, n)
    frame = _panel([{
        "code": f"{i:06d}", "factor": factor[i], "mom10": mom[i] + rng.normal(0, .01),
        "mom20": mom[i], "mom60": mom[i] + rng.normal(0, .02), "vol20": vol[i], "y": y[i],
    } for i in range(n)])
    got = daily_effects(frame, horizons=(10,), min_selected=10, min_controls=10,
                        min_match_per_side=3, min_ols=100).iloc[0]
    assert got.spread_pct > 0.1
    assert abs(got.matched_spread_pct) < 0.22
    assert abs(got.conditional_beta_pct) < 0.15
    assert got.matched_selected_coverage > 0.5


def test_conditional_ols_recovers_binary_effect_and_returns_nan_if_unidentified():
    rng = np.random.default_rng(8)
    n = 1200
    factor = np.tile([0, 1], n // 2)
    mom = rng.normal(size=n)
    vol = rng.normal(size=n)
    y = 0.8 * factor + 1.2 * mom - 0.4 * vol + rng.normal(0, .1, n)
    rows = [{"code": str(i), "factor": factor[i], "mom10": mom[i] + rng.normal(0, .2), "mom20": mom[i],
             "mom60": rng.normal(), "vol20": vol[i], "log_amount20": rng.normal(), "y": y[i]}
            for i in range(n)]
    got = daily_effects(_panel(rows), horizons=(10,), min_selected=10, min_controls=10, min_ols=100).iloc[0]
    assert got.conditional_beta_pct == pytest.approx(0.8, abs=0.04)

    # Factor is an exact linear combination of a supplied covariate, so its
    # incremental coefficient is not identified.
    for row in rows:
        row["mom10"] = float(row["factor"])
    unidentified = daily_effects(_panel(rows), horizons=(10,), min_ols=100).iloc[0]
    assert np.isnan(unidentified.conditional_beta_pct)


def test_ineligible_rows_are_excluded_from_all_daily_estimates():
    rows = []
    for i in range(40):
        rows.append({"code": str(i), "factor": int(i < 20), "mom10": i, "mom20": i,
                     "mom60": i, "vol20": i % 5, "y": 1.0 if i < 20 else 0.0})
    frame = _panel(rows)
    frame.loc[0, ["eligible", "factor", "y"]] = [False, 0, -999.]
    got = daily_effects(frame, horizons=(10,), min_selected=10, min_controls=10,
                        min_match_per_side=1, min_ols=100).iloc[0]
    assert got.all_selected == 19
    assert got.all_control == 20
    assert got.selected_mean == pytest.approx(1.0)
    assert got.control_mean == pytest.approx(0.0)

