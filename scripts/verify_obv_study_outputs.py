"""Audit the completed study's exported cash records and paired cost stress."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((out / "input-manifest.json").read_text(encoding="utf-8"))
    snapshot_root = out / "executed-source"
    snapshot_path = snapshot_root / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else None
    source_differences = []
    if snapshot:
        for relative, expected in snapshot["files"].items():
            assert digest(snapshot_root / relative) == expected, f"changed source snapshot: {relative}"
            if digest(root / relative) != expected:
                source_differences.append(relative)
    for path, expected in manifest["files"].items():
        assert digest(path) == expected, f"changed input: {Path(path).name}"
    assert digest(out / "dataset.pkl") == manifest["dataset_sha256"]
    for path, expected in manifest["generator_sha256"].items():
        relative = Path(path).relative_to(root)
        archived = snapshot_root / relative
        checked = archived if snapshot and archived.exists() else Path(path)
        assert digest(checked) == expected, f"changed generator: {Path(path).name}"
    trials = pd.read_csv(out / "trade-trials.csv.gz")
    assert not trials.duplicated(["code", "signal_date", "rule"]).any()
    index_paths = [Path(p) for p in manifest["files"] if Path(p).name.lower() == "sz399300.day"]
    assert len(index_paths) == 1
    index_bytes = index_paths[0].read_bytes()
    assert len(index_bytes) % 32 == 0
    date_codes = [int.from_bytes(index_bytes[i:i+4], "little") for i in range(0, len(index_bytes), 32)]
    calendar = [f"{d//10000:04d}-{d//100%100:02d}-{d%100:02d}" for d in date_codes if 20221001 <= d <= 20260930]
    assert calendar == sorted(set(calendar)) and calendar[-1] == "2026-09-30"
    session_index = {day: i for i, day in enumerate(calendar)}
    entered = trials[trials.entry_date.notna()]
    assert entered.qty.ge(100).all() and entered.qty.mod(100).eq(0).all()
    assert entered.invested.le(100000 + 1e-7).all()
    assert (entered.entry_date.map(session_index) - entered.signal_date.map(session_index)).eq(1).all()
    closed = trials[trials.status.eq("closed")].copy()
    assert (closed.exit_date > closed.entry_date).all(), "T+1 failed"
    assert closed.delayed_sell_days.between(0, 5).all()
    reconstruction = (closed.exit_price - closed.entry_price) * closed.qty - closed.fees
    pnl_error = float((reconstruction - closed.realized_pnl).abs().max())
    roi_error = float((closed.realized_pnl / 100000 * 100 - closed.net_return_pct).abs().max())
    buy_amount = closed.entry_price * closed.qty
    sell_amount = closed.exit_price * closed.qty
    expected_fees = (np.maximum(5, buy_amount * .00025) + np.maximum(5, sell_amount * .00025)
                     + sell_amount * .0005 + (buy_amount + sell_amount) * .00001)
    fee_error = float((closed.fees - expected_fees).abs().max())
    assert pnl_error < 1e-6 and roi_error < 1e-9 and fee_error < 1e-6
    fixed = closed[closed.rule.ne("factor_loss")]
    assert (fixed.scheduled_exit_date.map(session_index) - fixed.entry_date.map(session_index)).eq(fixed.horizon).all()
    assert (closed.exit_date.map(session_index) - closed.scheduled_exit_date.map(session_index)).eq(closed.delayed_sell_days).all()
    index_dates = sorted(pd.read_csv(out / "data-coverage.csv").last_date.dropna().unique())
    assert index_dates[-1] == "2026-09-30"
    execution = pd.read_csv(out / "execution-summary.csv")
    for r in execution.itertuples():
        statuses = json.loads(r.status_counts)
        assert sum(statuses.values()) == r.trials
        assert statuses.get("closed", 0) == r.closed
        assert r.closed + r.censored + r.no_entry == r.trials
    base = closed[closed.rule.eq("hold10")]
    stress = closed[closed.rule.eq("hold10_stress")]
    paired = base.merge(stress, on=["code", "signal_date"], suffixes=("_base", "_stress"), validate="one_to_one")
    paired["net_difference_pct"] = paired.net_return_pct_stress - paired.net_return_pct_base
    paired.to_csv(out / "stress-paired.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    recent = paired[paired.signal_date.ge("2025-10-01") & paired.exit_date_base.le("2026-09-30") & paired.exit_date_stress.le("2026-09-30")]
    hashes = ({str(root / f): expected for f, expected in snapshot["files"].items()} if snapshot else
              {str(root / f): digest(root / f) for f in (
                  "scripts/research_obv_resonance.py", "scripts/obv_alpha_statistics.py",
                  "scripts/obv_execution_replay.py", "scripts/backtest_turning_points.py",
                  "src/strategies/obv_resonance_factor.py")})
    report = dict(status="passed", exported_trials=len(trials), closed=len(closed),
                  max_cash_pnl_error=pnl_error, max_return_pct_error=roi_error, max_fee_error=fee_error,
                  next_market_open_entries_checked=len(entered), fixed_horizon_exits_checked=len(fixed),
                  paired_stress_count=len(paired), recent_paired_stress_count=len(recent),
                  recent_paired_base_mean_pct=float(recent.net_return_pct_base.mean()),
                  recent_paired_stress_mean_pct=float(recent.net_return_pct_stress.mean()),
                  recent_paired_cost_difference_pct=float(recent.net_difference_pct.mean()),
                  source_sha256=hashes, input_files_checked=len(manifest["files"]),
                  source_snapshot=str(snapshot_root) if snapshot else None,
                  current_source_differences=source_differences,
                  ths_client_backtest=False, shared_account_portfolio=False)
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
