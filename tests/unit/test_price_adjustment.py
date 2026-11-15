from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from src.utils import price_adjustment as adjustment
from src.utils import price_factor_worker


def _bars() -> pd.DataFrame:
    return pd.DataFrame([
        {"code": "600000.SH", "dt": "2024-06-03 15:00", "open": 100.0,
         "high": 105.0, "low": 95.0, "close": 100.0, "vol": 2000, "amount": 200000},
        {"code": "600000.SH", "dt": "2024-06-04 15:00", "open": 50.0,
         "high": 52.5, "low": 47.5, "close": 50.0, "vol": 4000, "amount": 200000},
    ])


def _factors(end="2024-06-04") -> pd.DataFrame:
    frame = pd.DataFrame({"dt": pd.to_datetime(["1990-01-01", "2024-06-04"]),
                          "factor": [1.0, 2.0]})
    frame.attrs.update(source="baostock.backAdjustFactor", verified=True,
                       coverage_start="1990-01-01", coverage_end=end)
    return frame


def test_hfq_signal_continuity_and_raw_execution_prices() -> None:
    original = _bars()
    plan = adjustment.AdjustmentPlan(original, _factors(), "hfq")
    before = plan.signal_bar(plan.raw_df.iloc[0].to_dict())
    after = plan.signal_bar(plan.raw_df.iloc[1].to_dict())

    assert before["close"] == after["close"] == 100.0
    assert after["high"] == 105.0
    assert after["vol"] == 4000
    assert after["amount"] == 200000
    assert after["execution_price_mode"] == "none"
    assert after["raw_close"] == 50.0
    assert plan.raw_df["close"].tolist() == original["close"].tolist()
    assert original.columns.tolist() == ["code", "dt", "open", "high", "low", "close", "vol", "amount"]
    assert plan.raw_df["__factor"].tolist() == [1.0, 2.0]
    assert plan.raw_df["__signal_multiplier"].tolist() == [1.0, 2.0]

    signal = {"price": 100.0, "stop_loss": 90.0, "take_profit": 120.0,
              "qty": 300, "reason": "synthetic split"}
    executed = adjustment.signal_to_execution(signal, after, plan.raw_df.iloc[1].to_dict())
    assert (executed["price"], executed["stop_loss"], executed["take_profit"]) == (50, 45, 60)
    assert (executed["qty"], executed["reason"]) == (300, "synthetic split")
    assert signal["price"] == 100.0
    sentinel = adjustment.signal_to_execution(
        {"price": 100, "stop_loss": 0, "take_profit": 0, "qty": 100},
        after, plan.raw_df.iloc[1].to_dict())
    assert sentinel["stop_loss"] == sentinel["take_profit"] == 0


def test_qfq_uses_raw_current_bar_and_rebase_ratio() -> None:
    plan = adjustment.AdjustmentPlan(_bars(), _factors(), "qfq")
    current = plan.signal_bar(_bars().iloc[1].to_dict())
    assert current["close"] == 50.0
    assert current["signal_multiplier"] == 1.0
    assert plan.rebase_ratio(plan.factor_at("2024-06-03"), plan.factor_at("2024-06-04")) == .5
    assert 100 * plan.rebase_ratio(1, 2) == current["close"]
    assert plan.metadata()["corporate_action_accounting"] == "unsupported"


def test_future_factor_does_not_change_historical_signals() -> None:
    original = _factors('2024-06-07')
    extended = pd.concat([original, pd.DataFrame({'dt': [pd.Timestamp('2024-06-06')], 'factor': [3.0]})], ignore_index=True)
    extended.attrs.update(original.attrs)
    for mode in ('hfq', 'qfq'):
        before = adjustment.AdjustmentPlan(_bars(), original, mode)
        after = adjustment.AdjustmentPlan(_bars(), extended, mode)
        assert [before.signal_bar(row)['close'] for row in _bars().to_dict('records')] == [after.signal_bar(row)['close'] for row in _bars().to_dict('records')]


@pytest.mark.parametrize('code', ['000001.SH', '510300.SH', '399001.SZ', '200001.SZ'])
def test_non_stock_instruments_do_not_get_identity_factors(code, monkeypatch) -> None:
    monkeypatch.setattr(adjustment, '_worker_payload', lambda *args: pytest.fail('unsupported instrument must not query'))
    with pytest.raises(adjustment.PriceAdjustmentError, match='A-share'):
        adjustment.load_adjustment_factors(code, '2024-06-03', '2024-06-04')


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_factors_fail_closed(bad: float) -> None:
    factors = _factors()
    factors.loc[1, "factor"] = bad
    with pytest.raises(adjustment.PriceAdjustmentError):
        adjustment.AdjustmentPlan(_bars(), factors, "hfq")


def test_missing_coverage_and_unverified_factors_fail_closed() -> None:
    with pytest.raises(adjustment.PriceAdjustmentError, match="coverage"):
        adjustment.AdjustmentPlan(_bars(), _factors("2024-06-03"), "hfq")
    factors = _factors()
    factors.attrs["verified"] = False
    with pytest.raises(adjustment.PriceAdjustmentError, match="verified"):
        adjustment.AdjustmentPlan(_bars(), factors, "qfq")
    with pytest.raises(adjustment.PriceAdjustmentError, match="coverage"):
        adjustment.AdjustmentPlan(_bars(), _factors(), "hfq").factor_at("2024-06-05")


def test_conflicting_factor_dates_fail_closed() -> None:
    factors = _factors()
    extra = pd.DataFrame({"dt": [pd.Timestamp("2024-06-04")], "factor": [3.0]})
    conflicting = pd.concat([factors, extra], ignore_index=True)
    conflicting.attrs.update(factors.attrs)
    with pytest.raises(adjustment.PriceAdjustmentError, match="conflicting"):
        adjustment.AdjustmentPlan(_bars(), conflicting, "hfq")


def test_raw_price_provenance_requires_evidence() -> None:
    data = _bars()
    assert adjustment.assert_raw_prices(SimpleNamespace(price_mode="none"), "tdx", data)
    with pytest.raises(adjustment.PriceAdjustmentError):
        adjustment.assert_raw_prices(SimpleNamespace(price_mode="none"), "default", data)
    assert adjustment.assert_raw_prices(None, "default", data,
                                        {"data_provider": {"stored_price_mode": "none"}})
    with pytest.raises(adjustment.PriceAdjustmentError, match="adjusted prices"):
        adjustment.assert_raw_prices(SimpleNamespace(price_mode="qfq"), "tdx", data,
                                     {"data_provider.stored_price_mode": "none"})
    adjusted = data.copy()
    adjusted.attrs["price_mode"] = "hfq"
    with pytest.raises(adjustment.PriceAdjustmentError, match="adjusted prices"):
        adjustment.assert_raw_prices(None, "default", adjusted,
                                     {"data_provider.stored_price_mode": "none"})


def test_factor_loader_uses_only_back_factor_and_cached_prior_anchor(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_worker(code, end, timeout):
        calls.append((code, end.date().isoformat(), timeout))
        return {"ok": True,
                "fields": ["code", "dividOperateDate", "foreAdjustFactor", "backAdjustFactor"],
                "rows": [["sh.600000", "2024-06-04", "0.5", "2.0"]]}

    monkeypatch.setattr(adjustment, "_worker_payload", fake_worker)
    config = {"data_provider.factor_cache_dir": str(tmp_path)}
    factors = adjustment.load_adjustment_factors("600000.SH", "2024-06-03", "2024-06-05", config)
    assert calls == [("sh.600000", "2024-06-05", 20)]
    assert factors["factor"].tolist() == [1.0, 2.0]
    assert factors.attrs["source"] == "baostock.backAdjustFactor"
    assert factors.attrs["coverage_end"] == "2024-06-05"
    assert factors.attrs["verified"] is True
    cached = adjustment.load_adjustment_factors("sh.600000", "2024-06-04", "2024-06-05", config)
    assert cached["factor"].tolist() == [1.0, 2.0]
    assert len(calls) == 1
    files = list(tmp_path.glob("adjustment_factors_v1_*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["coverage_end"] == "2024-06-05"
    shorter = adjustment.load_adjustment_factors("600000.SH", "2024-06-03", "2024-06-04", config)
    assert shorter.attrs["coverage_end"] == "2024-06-04"
    with pytest.raises(adjustment.PriceAdjustmentError, match="coverage"):
        adjustment.AdjustmentPlan(_bars(), shorter, "hfq").factor_at("2024-06-05")


def test_factor_loader_rejects_schema_error_without_identity_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(adjustment, "_worker_payload", lambda *_: {
        "ok": True, "fields": ["dividOperateDate", "foreAdjustFactor"],
        "rows": [["2024-06-04", "0.5"]]})
    with pytest.raises(adjustment.PriceAdjustmentError, match="schema"):
        adjustment.load_adjustment_factors("600000.SH", "2024-06-03", "2024-06-04",
                                           {"data_provider.factor_cache_dir": str(tmp_path)})
    assert not list(tmp_path.iterdir())


def test_worker_timeout_is_fatal_and_uses_absolute_runtime(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        assert args[0][0] == sys.executable
        assert kwargs["timeout"] == 0.01
        raise subprocess.TimeoutExpired(args[0], 0.01)

    monkeypatch.setattr(adjustment.subprocess, "run", timeout)
    with pytest.raises(adjustment.PriceAdjustmentError, match="timed out"):
        adjustment._worker_payload("sh.600000", pd.Timestamp("2024-06-04"), .01)


def test_worker_drains_cursor_before_logout(monkeypatch) -> None:
    state = {"logged_in": False, "rows": 0}

    class Cursor:
        error_code = "0"
        error_msg = ""
        fields = ["dividOperateDate", "backAdjustFactor"]

        def next(self):
            assert state["logged_in"]
            state["rows"] += 1
            return state["rows"] == 1

        def get_row_data(self):
            return ["2024-06-04", "2.0"]

    fake = SimpleNamespace(
        login=lambda: (state.__setitem__("logged_in", True) or SimpleNamespace(error_code="0")),
        query_adjust_factor=lambda **kwargs: Cursor(),
        logout=lambda: state.__setitem__("logged_in", False),
    )
    monkeypatch.setitem(sys.modules, "baostock", fake)
    result = price_factor_worker.query("sh.600000", "2024-06-04")
    assert result["rows"] == [["2024-06-04", "2.0"]]
    assert state == {"logged_in": False, "rows": 2}


def test_qfq_strategy_rebase_requires_custom_strategy_hook_or_opt_in() -> None:
    def default_hook(self, code, ratio):
        self.last_call = (code, ratio)

    default_hook.__qualname__ = "BaseImplementedStrategy.rebase_signal_prices"

    class CustomStrategy:
        rebase_signal_prices = default_hook

    custom = CustomStrategy()
    with pytest.raises(adjustment.PriceAdjustmentError, match="custom strategy"):
        adjustment.rebase_strategy_prices(custom, "600000.SH", 0.5)
    assert not hasattr(custom, "last_call")
    custom.supports_dynamic_qfq = True
    adjustment.rebase_strategy_prices(custom, "600000.SH", 0.5)
    assert custom.last_call == ("600000.SH", 0.5)

    class OverriddenStrategy:
        def rebase_signal_prices(self, code, ratio):
            self.last_call = (code, ratio)

    overridden = OverriddenStrategy()
    adjustment.rebase_strategy_prices(overridden, "600000.SH", 0.25)
    assert overridden.last_call == ("600000.SH", 0.25)


def test_factor_worker_exits_on_its_own_deadline_without_parent_cleanup() -> None:
    program = ('import sys,time; from src.utils import price_factor_worker as w; '
               'w.query=lambda *args:time.sleep(60); '
               'sys.argv=["worker", "sz.000001", "2026-09-30", "0.2"]; w.main()')
    child = subprocess.run([sys.executable, '-c', program], timeout=5, capture_output=True)
    assert child.returncode == 124
