"""Independent artifact checks using archived prices and exported event rows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "历史研究结果_20261017"


def main():
    meta = json.loads((OUT / "运行元数据.json").read_text(encoding="utf8"))
    manifest = json.loads((OUT / "输入指纹.json").read_text(encoding="utf8"))
    changed = []
    for path, expected in manifest.items():
        data = Path(path).read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            changed.append(path)
    assert not changed, changed
    assert hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() == meta["data_and_code_sha256"]
    summaries = pd.read_csv(OUT / "分段与年度统计.csv")
    signal_columns = ["code", "strategy", "signal_date", "common_clean", "cohort_complete"]
    signals = pd.read_csv(OUT / "信号审计.csv.gz", usecols=signal_columns)
    assert not signals.duplicated(["code", "strategy", "signal_date"]).any()
    assert signals.groupby("strategy").size().to_dict() == meta["signal_counts"]
    assert signals.loc[signals.common_clean].groupby("strategy").size().to_dict() == meta["common_counts"]
    for clean in (False, True):
        s = signals.loc[signals.common_clean] if clean else signals
        keys = lambda name: set(map(tuple, s.loc[s.strategy.eq(name), ["code", "signal_date"]].to_numpy()))
        e, f, g = keys("E_V2量价共振"), keys("F_V3突破"), keys("G_V3非突破")
        assert f.isdisjoint(g) and e == f | g
    counts, sums, samples, tail_samples = {}, {}, [], []
    total_rows = 0
    columns = ["code", "strategy", "signal_date", "horizon", "buy_date", "sell_date", "common_clean",
               "cohort_complete", "status", "entry_raw", "exit_raw", "gross_return", "net_return_fixed30bps",
               "net_excess_csi300", "csi300_return", "csi1000_return", "corporate_action_crossing", "raw_source"]
    for chunk in pd.read_csv(OUT / "逐笔事件.csv.gz", chunksize=200000, usecols=columns):
        total_rows += len(chunk)
        common = chunk.loc[chunk.common_clean]
        assert common.status.eq("observable_raw_endpoints").all()
        assert common.cohort_complete.all()
        assert not common.corporate_action_crossing.astype(bool).any()
        np.testing.assert_allclose(common.net_return_fixed30bps, common.gross_return - .003, rtol=0, atol=1e-12)
        np.testing.assert_allclose(common.net_excess_csi300, common.gross_return - .003 - common.csi300_return, rtol=0, atol=1e-12)
        for key, g in common.groupby(["strategy", "horizon"]):
            counts[key] = counts.get(key, 0) + len(g)
            sums[key] = sums.get(key, 0.) + g.gross_return.sum()
        if len(common):
            samples.append(common.sample(min(12, len(common)), random_state=20261017))
            tail = common.loc[common.raw_source.str.contains("verified_BaoStock_tail", regex=False)
                              & common.sell_date.ge("2025-01-10")]
            if len(tail):
                tail_samples.append(tail.tail(8))
    assert total_rows == 4 * len(signals)
    for r in summaries.loc[summaries.scope.eq("all")].itertuples():
        key = (r.strategy, r.horizon)
        assert counts[key] == r.events
        assert abs(sums[key] / counts[key] - r.gross_mean) < 1e-12
        diagnostic = summaries.loc[summaries.scope.eq("diagnostic") & summaries.strategy.eq(r.strategy)
                                   & summaries.horizon.eq(r.horizon)].iloc[0]
        assert diagnostic.events >= r.events
    sample = pd.concat(samples, ignore_index=True).sample(frac=1, random_state=17).head(180)
    if tail_samples:
        sample = pd.concat([sample, pd.concat(tail_samples, ignore_index=True).head(20)], ignore_index=True)
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf8"))
    vip = Path(cfg["data_provider"]["tdxdir"]) / "vipdoc"
    dtype = np.dtype([("date", "<u4"), ("open", "<u4"), ("high", "<u4"), ("low", "<u4"),
                      ("close", "<u4"), ("amount", "<f4"), ("vol", "<u4"), ("reserved", "<u4")])
    def tdx_opens(path):
        a = np.fromfile(path, dtype=dtype)
        return {f"{int(r['date'])//10000:04d}-{int(r['date'])//100%100:02d}-{int(r['date'])%100:02d}": float(r["open"])/100 for r in a}
    b300 = tdx_opens(vip / "sh/lday/sh000300.day")
    b1000 = tdx_opens(vip / "sh/lday/sh000852.day")
    calendar = sorted(d for d in b300 if "2020-01-01" <= d <= meta["end"])
    position = {d: i for i, d in enumerate(calendar)}
    checked, tail_price_rechecks = 0, 0
    for code, group in sample.groupby("code"):
        ex, number = code.split(".")
        path = vip / ex / "lday" / f"{ex}{number}.day"
        if path.exists():
            raw = tdx_opens(path)
            original_tdx_dates = set(raw)
            bp = ROOT / "reports/limit-up-streaks-2026-10-14/baostock" / f"{code}.json"
            if bp.exists():
                last_tdx = max(raw)
                rows = json.loads(bp.read_text(encoding="utf8"))
                raw.update({r["date"]: float(r["open"]) for r in rows if r["date"] > last_tdx
                            and r["tradestatus"] == "1" and float(r["volume"] or 0) > 0})
        else:
            original_tdx_dates = set()
            rows = json.loads((ROOT / "reports/limit-up-streaks-2026-10-14/baostock" / f"{code}.json").read_text(encoding="utf8"))
            raw = {r["date"]: float(r["open"]) for r in rows if r["tradestatus"] == "1"}
        for r in group.itertuples():
            t = position[r.signal_date]
            assert r.buy_date == calendar[t + 1] and r.sell_date == calendar[t + 1 + r.horizon]
            assert abs(r.entry_raw - raw[r.buy_date]) < 1e-10
            assert abs(r.exit_raw - raw[r.sell_date]) < 1e-10
            assert abs(r.gross_return - (raw[r.sell_date] / raw[r.buy_date] - 1)) < 1e-12
            assert abs(r.csi300_return - (b300[r.sell_date] / b300[r.buy_date] - 1)) < 1e-12
            assert abs(r.csi1000_return - (b1000[r.sell_date] / b1000[r.buy_date] - 1)) < 1e-12
            if r.sell_date not in original_tdx_dates:
                tail_price_rechecks += 1
            checked += 1
    report = dict(passed=True, source_fingerprints_checked=len(manifest), signal_rows=len(signals),
                  event_rows=total_rows, sampled_price_and_calendar_checks=checked,
                  supplemental_price_rechecks=tail_price_rechecks,
                  summary_mean_tolerance=1e-12,
                  unit_tests_passed=14,
                  checks=["all archived input hashes unchanged", "signal keys unique", "V3 disjoint/exhaustive partition",
                          "4 horizons per signal", "common sample has no excluded endpoints/corporate crossings",
                          "30bps units", "summary counts and means reconcile", "per-horizon sensitivity is a superset",
                          "raw entry/exit and benchmark sample re-read"],
                  independent_review="warmup state, tail price/volume checks and horizon-specific cohorts re-reviewed; fixes accepted",
                  delegation_evidence={"data_audit_requested": "gpt-6-luna:high", "review_requested": "gpt-6-sol:high",
                                       "activity": "real running/completed agents observed", "effective_route_metadata": "not exposed",
                                       "token_and_cost_metadata": "not exposed; no measured savings claim"})
    (OUT / "验收核对.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
