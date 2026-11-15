import pytest

from src.factor.gates import GateConfig, judge


def _metrics(
    ic=0.06, icir=0.8, hit=0.6, t=3.2,
    mono=0.9, reversals=0, spread_t=2.4, spread_win=0.6,
    net=0.006, turnover=0.5,
):
    return {
        "ic": {"mean_adj": ic, "icir_adj": icir, "hit_rate_adj": hit, "t_adj": t},
        "layered": {
            "monotone_adj": mono, "reversals": reversals,
            "spread_t": spread_t, "spread_win": spread_win,
        },
        "cost": {"net_long_excess": net, "turnover": turnover},
    }


def _judge(metrics, **kwargs):
    params = {"oos": {"ic": {"mean_adj": 0.05}, "base_mean_adj": 0.06},
              "param_stability": 1.0, "universe_stability": 1.0,
              "survivorship_complete": True}
    params.update(kwargs)
    return judge(metrics, GateConfig(), **params)


def test_all_gates_pass_yields_approval():
    result = _judge(_metrics())
    assert result["verdict"] == "通过"
    assert [g["status"] for g in result["gates"]] == ["passed"] * 4
    assert result["reasons"] == []


def test_weak_predictive_power_is_rejected_outright():
    result = _judge(_metrics(ic=0.01, icir=0.1, hit=0.5, t=0.4))
    assert result["verdict"] == "淘汰"
    assert result["gates"][0]["status"] == "failed"
    assert "预测力" in result["reasons"][0]


def test_missing_out_of_sample_evidence_blocks_approval():
    result = _judge(_metrics(), oos=None)
    assert result["gates"][2]["status"] == "incomplete"
    assert result["verdict"] == "待补证"


def test_incomplete_survivorship_is_reported_as_weak_not_as_factor_failure():
    result = _judge(_metrics(), survivorship_complete=False)
    assert result["gates"][2]["status"] == "weak"
    assert result["verdict"] == "待补证"
    assert result["gates"][0]["status"] == "passed"


def test_unstable_parameters_fail_the_robustness_gate():
    result = _judge(_metrics(), param_stability=0.2)
    assert result["gates"][2]["status"] == "failed"
    assert result["verdict"] == "淘汰"


def test_out_of_sample_decay_fails_the_robustness_gate():
    result = _judge(_metrics(), oos={"ic": {"mean_adj": 0.01}, "base_mean_adj": 0.06})
    assert result["gates"][2]["status"] == "failed"


def test_out_of_sample_sign_flip_fails_even_when_magnitude_is_large():
    result = _judge(_metrics(), oos={"ic": {"mean_adj": -0.05}, "base_mean_adj": 0.06})
    assert result["gates"][2]["status"] == "failed"


def test_turnover_above_cap_fails_implementability():
    result = _judge(_metrics(turnover=0.95))
    assert result["gates"][3]["status"] == "failed"
    assert result["verdict"] == "淘汰"


def test_negative_net_excess_fails_implementability():
    result = _judge(_metrics(net=-0.001))
    assert result["gates"][3]["status"] == "failed"


def test_non_monotone_layers_fail_the_second_gate():
    result = _judge(_metrics(mono=0.2, reversals=3, spread_t=0.3, spread_win=0.4))
    assert result["gates"][1]["status"] == "failed"


def test_report_payload_shape_is_stable():
    result = _judge(_metrics())
    assert set(result) == {"gates", "status", "verdict", "reasons", "config"}
    gate = result["gates"][0]
    assert set(gate) == {"gate", "name", "status", "status_label", "reason", "checks"}
    assert set(gate["checks"][0]) == {"name", "value", "threshold", "passed", "blocking", "note"}
    assert result["config"]["IC 绝对值下限"] == pytest.approx(0.03)


def test_thresholds_are_configurable():
    strict = GateConfig(min_abs_ic=0.10)
    result = judge(
        _metrics(ic=0.06), strict,
        oos={"ic": {"mean_adj": 0.05}, "base_mean_adj": 0.06},
        param_stability=1.0, universe_stability=1.0, survivorship_complete=True,
    )
    assert result["gates"][0]["status"] == "failed"
