import numpy as np
import pandas as pd
import pytest

from src.factor.combine import (
    build_combination,
    combine_scores,
    combine_weights,
    correlation_matrix,
    select_decorrelated,
)


def _scores():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=3), [f"{600000 + i}" for i in range(12)]],
        names=["dt", "code"],
    )
    generator = np.random.default_rng(0)
    base = generator.normal(size=len(index))
    return pd.DataFrame(
        {"a": base, "b": base * 2.0 + 1.0, "c": generator.normal(size=len(index))},
        index=index,
    )


def test_correlation_matrix_detects_duplicated_signal():
    matrix = correlation_matrix(_scores(), min_names=5)
    assert matrix.loc["a", "b"] == pytest.approx(1.0)
    assert abs(matrix.loc["a", "c"]) < 0.8
    assert matrix.loc["a", "a"] == pytest.approx(1.0)


def test_select_decorrelated_drops_the_redundant_factor_first():
    kept, dropped = select_decorrelated(
        correlation_matrix(_scores(), min_names=5),
        max_abs_corr=0.7,
        priority=["a", "b", "c"],
    )
    assert "a" in kept
    assert "b" not in kept
    assert dropped[0]["factor_id"] == "b"
    assert dropped[0]["against"] == "a"


def test_icir_weighting_shares_sum_to_one_and_respect_strength():
    weights, warnings = combine_weights(
        {"a": {"ic": {"icir_adj": 1.0}}, "b": {"ic": {"icir_adj": 3.0}}}, method="icir"
    )
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] == pytest.approx(0.25)
    assert weights["b"] == pytest.approx(0.75)
    assert warnings == []


def test_icir_weighting_caps_a_single_dominant_factor():
    weights, _ = combine_weights(
        {"a": {"ic": {"icir_adj": 100.0}}, "b": {"ic": {"icir_adj": 1.0}}}, method="icir", cap=3.0
    )
    assert weights["a"] == pytest.approx(0.75)
    assert weights["b"] == pytest.approx(0.25)


def test_zero_strength_falls_back_to_equal_weight_with_a_warning():
    weights, warnings = combine_weights(
        {"a": {"ic": {"icir_adj": 0.0}}, "b": {"ic": {"icir_adj": 0.0}}}, method="icir"
    )
    assert weights == pytest.approx({"a": 0.5, "b": 0.5})
    assert warnings and "等权" in warnings[0]


def test_unknown_weighting_method_raises():
    with pytest.raises(ValueError):
        combine_weights({"a": {"ic": {"icir_adj": 1.0}}}, method="magic")


def test_combined_scores_are_standardized_per_cross_section():
    scores = _scores()
    combined = combine_scores(scores[["a", "c"]], {"a": 0.5, "c": 0.5})
    per_date = combined.groupby(level=0)
    assert per_date.mean().abs().max() < 1e-9
    assert (per_date.std(ddof=0) - 1.0).abs().max() < 1e-9


def test_build_combination_reports_weights_and_drops():
    scores = _scores()
    metrics = {
        "a": {"ic": {"icir_adj": 1.0}},
        "b": {"ic": {"icir_adj": 2.0}},
        "c": {"ic": {"icir_adj": 0.5}},
    }
    result = build_combination(scores, metrics, method="icir", max_abs_corr=0.7)
    assert "b" in result["dropped"] or "a" in result["kept"]
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    assert not result["scores"].empty
