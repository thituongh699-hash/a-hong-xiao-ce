import numpy as np
import pandas as pd
import pytest

from src.factor.preprocess import (
    PreprocessConfig,
    neutralize,
    preprocess_cross_section,
    standardize,
    winsorize_mad,
)


def test_mad_winsorize_clips_outlier_at_scaled_median_absolute_deviation():
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=float)
    result = winsorize_mad(values, n=3.0)
    # 中位数 5.5，绝对偏差中位数 2.5，边界 5.5 ± 3 × 1.4826 × 2.5
    assert result.max() == pytest.approx(5.5 + 3 * 1.4826 * 2.5)
    assert result.min() == pytest.approx(1.0)
    assert result.iloc[:9].tolist() == values.iloc[:9].tolist()


def test_mad_winsorize_returns_input_when_mad_is_zero():
    values = pd.Series([2.0, 2.0, 2.0, 9.0])
    assert winsorize_mad(values).tolist() == values.tolist()


def test_zscore_standardization_has_zero_mean_and_unit_population_std():
    result = standardize(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert float(result.mean()) == pytest.approx(0.0)
    assert float(result.std(ddof=0)) == pytest.approx(1.0)


def test_zero_variance_cross_section_standardizes_to_zero_not_nan():
    result = standardize(pd.Series([5.0, 5.0, 5.0]))
    assert result.tolist() == [0.0, 0.0, 0.0]


def test_rank_standardization_maps_to_symmetric_unit_interval():
    result = standardize(pd.Series([30.0, 10.0, 20.0]), method="rank")
    # 秩为 3/1/2，减中点 2 后除以 (n-1)/2 = 1
    assert result.tolist() == pytest.approx([1.0, -1.0, 0.0])
    assert float(result.min()) == pytest.approx(-1.0)
    assert float(result.max()) == pytest.approx(1.0)


def test_neutralize_removes_exact_industry_and_size_effects():
    index = pd.RangeIndex(4)
    industry = pd.Series(["A", "B", "A", "B"], index=index)
    size = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    values = pd.Series(
        [3.0 * (1 if ind == "A" else 0) + 2.0 * np.log(sz) for ind, sz in zip(industry, size)],
        index=index,
    )
    residual = neutralize(values, industry=industry, size=size)
    assert float(residual.abs().max()) < 1e-9


def test_neutralize_falls_back_to_demeaning_when_sample_too_small():
    values = pd.Series([1.0, 3.0])
    residual = neutralize(values, size=pd.Series([1.0, 2.0]))
    assert residual.tolist() == pytest.approx([-1.0, 1.0])


def test_preprocess_chain_matches_documented_order():
    index = pd.RangeIndex(12)
    generator = np.random.default_rng(7)
    values = pd.Series(generator.normal(0, 1, 12), index=index)
    values.iloc[3] = 40.0
    size = pd.Series(generator.uniform(1e8, 5e9, 12), index=index)
    industry = pd.Series(["A", "B", "C"] * 4, index=index)
    config = PreprocessConfig(winsor_n=3.0, method="zscore", neutralize=True, min_valid=5)

    expected = standardize(
        neutralize(
            standardize(winsorize_mad(values, 3.0), "zscore"),
            industry=industry,
            size=size,
        ),
        "zscore",
    )
    assert preprocess_cross_section(values, industry, size, config).tolist() == pytest.approx(expected.tolist())


def test_preprocess_returns_all_nan_below_minimum_valid_samples():
    values = pd.Series([1.0, 2.0, 3.0, np.nan, np.nan])
    result = preprocess_cross_section(values, config=PreprocessConfig(min_valid=5))
    assert result.isna().all()


def test_preprocess_keeps_missing_values_missing():
    values = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    result = preprocess_cross_section(values, config=PreprocessConfig(min_valid=5))
    assert pd.isna(result.iloc[1])
    assert result.notna().sum() == 5
