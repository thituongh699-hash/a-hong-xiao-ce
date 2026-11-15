import json
import pickle

import numpy as np
import pandas as pd
import pytest

from scripts.research_obv_resonance import labeled_panel, build_bars, frozen_design, generator_hashes, reuse_dataset, sha
from src.strategies.obv_resonance_factor import features
from src.utils.price_adjustment import _factor_frame


def work_frame():
    days = pd.bdate_range("2023-01-02", periods=230)
    close = 10 + np.arange(230) * .01
    raw = pd.DataFrame(dict(code="000001.SZ", dt=days, open=close, close=close,
                             high=close+.01, low=close-.01, vol=10000., amount=100000.))
    raw.attrs["raw_price_verified_external"] = True
    return features(raw, _factor_frame([], raw.dt.max()))


def test_labels_follow_market_calendar_not_suspended_stock_bar_count():
    work = work_frame()
    calendar = pd.DatetimeIndex(work.dt)
    signal_index = 200
    signal_day = calendar[signal_index]
    work = work.drop(index=[signal_index+1, signal_index+5]).reset_index(drop=True)
    panel = labeled_panel(work, calendar).set_index("dt")
    assert np.isnan(panel.loc[signal_day, "fwd_close_5"])
    assert np.isnan(panel.loc[signal_day, "fwd_open_5"])
    assert panel.loc[signal_day, "label_end_5"] == calendar[signal_index+5]
    expected = (work.set_index("dt").signal_close.loc[calendar[signal_index+10]]
                / work.set_index("dt").signal_close.loc[signal_day]-1)*100
    assert np.isclose(panel.loc[signal_day, "fwd_close_10"], expected)


def test_predeclared_primary_fixed_factor_and_cost_timing():
    design = frozen_design()
    assert design["ma_period"] == 18 and not design["tuning"]
    assert design["primary_horizon"] == 10 and design["split_label_purge"]
    assert design["transfer_per_side"] == .00001
    assert "next market-day open" in design["entry"]


def test_open_reference_does_not_use_later_high_or_low():
    work = work_frame()
    work.loc[201, "high"] = work.loc[201, "close"] * 2
    bars = build_bars(work)
    row = bars[str(work.dt.iloc[201].date())]
    assert row["open_price_reference_valid"]
    assert not row["price_reference_valid"]


def test_reuse_binds_generator_and_dataset_bytes_before_deserialization(tmp_path):
    artifact = tmp_path / "dataset.pkl"
    artifact.write_bytes(pickle.dumps({"fingerprint": "test"}))
    manifest = dict(fingerprint="test", files={}, generator_sha256=generator_hashes(),
                    dataset_sha256=sha(artifact))
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert reuse_dataset(tmp_path)["fingerprint"] == "test"
    artifact.write_bytes(artifact.read_bytes() + b"unexpected change")
    with pytest.raises(ValueError, match="checksum"):
        reuse_dataset(tmp_path)
    manifest["generator_sha256"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="generating code"):
        reuse_dataset(tmp_path)


def test_reuse_rejects_changed_stock_scope_dependency(tmp_path, monkeypatch):
    from scripts import research_obv_resonance as study

    artifact = tmp_path / "dataset.pkl"
    artifact.write_bytes(pickle.dumps({"fingerprint": "test"}))
    manifest = dict(fingerprint="test", files={}, generator_sha256=generator_hashes(),
                    dataset_sha256=sha(artifact))
    (tmp_path / "input-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    scope_path = study.ROOT / "scripts/backtest_turning_points.py"
    original_digest = study.sha

    def changed_scope_digest(path):
        return "simulated changed stock scope implementation" if path == scope_path else original_digest(path)

    monkeypatch.setattr(study, "sha", changed_scope_digest)
    with pytest.raises(ValueError, match="generating code"):
        reuse_dataset(tmp_path)
