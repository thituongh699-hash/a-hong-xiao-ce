import json

import pytest

from src.factor.dataset import synthetic_bars
from src.factor.panel import build_panel
from src.factor.pipeline import SCHEMA, StudyConfig, run_study
from src.factor.spec import FactorRegistryError, FactorSpec, get_factor


@pytest.fixture(scope="module")
def report():
    bars = synthetic_bars(n_codes=40, n_days=220, seed=11)
    return run_study(
        build_panel(bars),
        StudyConfig(
            factor_ids=("vol_20", "rev_5"),
            horizon=10,
            min_names=20,
            universe="单元测试面板",
            survivorship_complete=True,
        ),
    )


def test_report_is_strictly_json_serializable(report):
    json.dumps(report, allow_nan=False, ensure_ascii=False)


def test_report_declares_schema_and_sample_window(report):
    assert report["schema"] == SCHEMA
    assert report["sample"]["rebalance_dates"] >= 6
    assert report["sample"]["date_start"] <= report["sample"]["date_end"]
    assert report["fingerprint"]


def test_each_factor_carries_hypothesis_evidence_and_four_gates(report):
    assert report["factors"]
    for item in report["factors"]:
        assert item["hypothesis"]
        assert item["gates"] is not None
        assert len(item["gates"]["gates"]) == 4
        assert item["gates"]["verdict"] in {"通过", "待补证", "淘汰"}


def test_counts_cover_every_requested_factor(report):
    counts = report["summary"]["counts"]
    assert sum(counts.values()) == len(report["factors"])


def test_synthetic_warning_is_always_present_in_demo_mode():
    bars = synthetic_bars(n_codes=30, n_days=200, seed=4)
    result = run_study(
        build_panel(bars),
        StudyConfig(factor_ids=("vol_20",), horizon=10, min_names=20, demo=True),
    )
    assert any("合成数据" in warning for warning in result["warnings"])


def test_incomplete_universe_defers_approval():
    bars = synthetic_bars(n_codes=40, n_days=200, seed=9)
    result = run_study(
        build_panel(bars),
        StudyConfig(factor_ids=("vol_20",), horizon=10, min_names=20, survivorship_complete=False),
    )
    assert any("幸存者偏差" in warning for warning in result["warnings"])
    gates = result["factors"][0]["gates"]
    coverage = [
        check for gate in gates["gates"] for check in gate["checks"]
        if check["name"].startswith("样本覆盖")
    ]
    assert coverage and coverage[0]["passed"] is False
    # 覆盖度属于「证据不足」而不是「因子无效」，因此不允许判定为通过
    assert coverage[0]["blocking"] is False
    assert gates["verdict"] != "通过"


def test_same_seed_produces_identical_statistics():
    config = StudyConfig(factor_ids=("vol_20",), horizon=10, min_names=20, survivorship_complete=True)
    first = run_study(build_panel(synthetic_bars(n_codes=36, n_days=180, seed=5)), config)
    second = run_study(build_panel(synthetic_bars(n_codes=36, n_days=180, seed=5)), config)
    assert first["factors"][0]["ic"]["mean"] == second["factors"][0]["ic"]["mean"]
    assert first["factors"][0]["layered"]["spread_mean"] == second["factors"][0]["layered"]["spread_mean"]


def test_unknown_factor_is_recorded_instead_of_aborting_the_study():
    result = run_study(
        build_panel(synthetic_bars(n_codes=36, n_days=180, seed=3)),
        StudyConfig(factor_ids=("not_a_factor",), horizon=10, min_names=20),
    )
    assert result["factors"][0]["gates"] is None
    assert result["summary"]["counts"]["计算失败"] == 1


def test_too_few_cross_sections_raise_actionable_error():
    with pytest.raises(ValueError, match="截面"):
        run_study(
            build_panel(synthetic_bars(n_codes=30, n_days=50)),
            StudyConfig(factor_ids=("rev_5",), horizon=20, min_names=10),
        )


def test_factor_without_economic_hypothesis_cannot_be_registered():
    with pytest.raises(FactorRegistryError, match="经济假设"):
        FactorSpec(
            factor_id="nohypo", name="无假设", category="量价", direction=1,
            hypothesis="   ", lookback=2, compute=lambda frame, params: frame["close"],
        )


def test_invalid_direction_is_rejected():
    with pytest.raises(FactorRegistryError):
        FactorSpec(
            factor_id="way", name="错方向", category="量价", direction=0,
            hypothesis="方向必须为 ±1", lookback=2, compute=lambda frame, params: frame["close"],
        )


def test_builtin_factor_catalog_contains_all_registered_ids():
    from src.factor.spec import BUILTIN_FACTORS, catalog

    ids = {item["factor_id"] for item in catalog()}
    assert {spec.factor_id for spec in BUILTIN_FACTORS} <= ids
    assert get_factor("vol_20").direction == -1
