"""Frozen local-data event study; no network, broker, account or config mutation.

Uses listing/delisting metadata instead of today's stock list. Every unavailable
stock and event remains in audit output. Returns are RAW open/open returns, only
for windows without factor changes. Corporate-action factor returns are never
treated as realizable cash returns. See 研究结果.md for interpretation boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = sys.executable
SPEC = importlib.util.spec_from_file_location("pioneer_signal", HERE / "先锋上车事件研究.py")
SIGNAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIGNAL)
STRATEGIES = ["A_原版", "B_原版加均量", "C_再加收盘质量", "D_再加趋势斜率",
              "E_V2量价共振", "F_V3突破", "G_V3非突破"]
CORE = [STRATEGIES[i] for i in (0, 4, 5, 6)]
DTYPE = np.dtype([("date", "<u4"), ("open", "<u4"), ("high", "<u4"),
                  ("low", "<u4"), ("close", "<u4"), ("amount", "<f4"),
                  ("volume", "<u4"), ("reserved", "<u4")])
INPUTS = ROOT / "outputs/condition1-nextday-2026-10-15/inputs"
OLD = ROOT / "reports/limit-up-streaks-2026-10-14"


def allowed(code):
    return (code.startswith(("sh.600", "sh.601", "sh.603", "sh.605", "sh.688"))
            or code.startswith(("sz.000", "sz.001", "sz.002", "sz.003", "sz.300", "sz.301")))


def segment(date):
    year = int(str(date)[:4])
    return "2020-2021" if year <= 2021 else "2022-2024" if year <= 2024 else "2025-2026"


class Archive:
    def __init__(self):
        self.manifest = {}

    def read(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        self.manifest[str(path)] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    def json(self, path):
        return json.loads(self.read(path).decode("utf-8-sig"))

    def tdx(self, path):
        data = self.read(path)
        if len(data) % 32:
            raise ValueError("invalid TDX record size")
        a = np.frombuffer(data, dtype=DTYPE)
        frame = pd.DataFrame({k: a[k] for k in ("date", "open", "high", "low", "close", "volume")})
        frame["date"] = pd.to_datetime(frame.date.astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
        frame[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]] / 100.
        return frame


def validate_bars(df):
    if df.empty:
        return df
    if df.date.duplicated().any() or not df.date.is_monotonic_increasing:
        raise ValueError("duplicate/unsorted bars")
    values = df[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("non-finite/negative OHLCV")
    active = df.volume.gt(0) & df[["open", "high", "low", "close"]].gt(0).all(axis=1)
    if ((df.loc[active, "low"] > df.loc[active, ["open", "close"]].min(axis=1) + 1e-8).any()
            or (df.loc[active, "high"] + 1e-8 < df.loc[active, ["open", "close"]].max(axis=1)).any()):
        raise ValueError("invalid OHLC envelope")
    return df.loc[active].reset_index(drop=True)


def append_verified_tail(primary, supplement):
    """Keep TDX history, append saved raw terminal bars only after overlap QA."""
    tail = supplement.loc[supplement.date.gt(primary.date.iloc[-1])]
    if tail.empty:
        return primary
    overlap = primary.set_index("date").join(supplement.set_index("date"), how="inner", rsuffix="_bao")
    if overlap.empty:
        raise ValueError("tail source has no overlapping raw-price dates")
    error = max((overlap[k] - overlap[k + "_bao"]).abs().max() for k in ("open", "high", "low", "close"))
    if error > .011:
        raise ValueError("tail source overlapping raw OHLC mismatch")
    volume_error = (overlap.volume.astype(float) - overlap.volume_bao).abs()
    volume_relative_error = volume_error / overlap.volume.astype(float).clip(lower=1.)
    if ((volume_error > 100.) & (volume_relative_error > .001)).any():
        raise ValueError("tail source overlapping volume units/values mismatch")
    result = pd.concat([primary, tail], ignore_index=True)
    result.attrs.update(tail_overlap_bars=len(overlap), tail_max_ohlc_error=float(error), tail_appended_bars=len(tail),
                        tail_max_volume_error_shares=float(volume_error.max()))
    return result


def load_raw(archive, vip, code, end):
    ex, num = code.split(".")
    path = vip / ex / "lday" / f"{ex}{num}.day"
    primary = pd.DataFrame()
    if path.exists():
        primary = validate_bars(archive.tdx(path).loc[lambda x: x.date.le(end)].copy())
    path = OLD / "baostock" / f"{code}.json"
    if path.exists():
        rows = archive.json(path)
        if any(r["code"] != code for r in rows):
            raise ValueError("wrong stock raw file")
        df = pd.DataFrame(rows)
        if len(df):
            for k in ("open", "high", "low", "close", "volume"):
                df[k] = pd.to_numeric(df[k].replace("", "0"), errors="raise")
            # Empirical overlap checks establish BOTH binary TDX and BaoStock
            # volumes in shares. Reader-library display units are irrelevant.
            df = df.loc[df.date.le(end) & df.tradestatus.eq("1")].copy()
            df = validate_bars(df)
            if len(primary):
                combined = append_verified_tail(primary, df)
                source = "TDX_raw+verified_BaoStock_tail" if len(combined) > len(primary) else "TDX_raw"
                return combined, source
            return df, "BaoStock_raw_adjustflag3"
    if len(primary):
        return primary, "TDX_raw"
    return pd.DataFrame(), "missing"


def load_factors(archive, code, end):
    path = ROOT / "data/history/cache" / f"adjustment_factors_v1_{code.replace('.', '_')}.json"
    if path.exists():
        obj = archive.json(path)
        if (obj.get("version") != 1 or obj.get("code") != code
                or obj.get("source") != "baostock.backAdjustFactor" or obj["coverage_end"] < end):
            raise ValueError("factor identity/source/coverage mismatch")
        rows = {r["dt"]: float(r["factor"]) for r in obj["rows"] if r["dt"] <= end}
        source = "verified_cache_to_" + obj["coverage_end"]
    else:
        path = INPUTS / "factors" / f"{code}.json"
        if not path.exists():
            raise ValueError("missing official factor archive")
        obj = archive.json(path)
        if any(r["code"] != code for r in obj):
            raise ValueError("wrong stock factor file")
        # fetch_inputs.py records a successful full-history query to 2026-10-14.
        if end > "2026-10-14":
            raise ValueError("supplemental factor coverage exceeded")
        rows = {r["dividOperateDate"]: float(r["backAdjustFactor"])
                for r in obj if r["dividOperateDate"] <= end}
        source = "supplemental_BaoStock_full_query_to_2026-10-14"
    rows.setdefault("1990-01-01", 1.)
    dates = np.array(sorted(rows))
    factors = np.array([rows[d] for d in dates], float)
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("invalid factor")
    return dates, factors, source


def scheduled_outcome(raw, calendar, position, signal_day, horizon, change_dates, benchmark):
    t = position[signal_day]
    result = {"horizon": horizon, "status": "right_censored"}
    if t + 1 + horizon >= len(calendar):
        return result
    buy, sell = calendar[t + 1], calendar[t + 1 + horizon]
    result.update(buy_date=buy, sell_date=sell)
    raw_dates = raw.index if isinstance(raw, pd.DataFrame) else raw
    if buy not in raw_dates:
        result["status"] = "entry_missing_or_suspended"
        return result
    if sell not in raw_dates:
        result["status"] = "exit_missing_or_suspended_or_delisted"
        return result
    b, s = (raw.loc[buy], raw.loc[sell]) if isinstance(raw, pd.DataFrame) else (raw[buy], raw[sell])
    if isinstance(benchmark, pd.DataFrame):
        b300, s300 = benchmark.loc[buy, "csi300"], benchmark.loc[sell, "csi300"]
        b1000, s1000 = benchmark.loc[buy, "csi1000"], benchmark.loc[sell, "csi1000"]
    else:
        b300, s300 = benchmark[buy].csi300, benchmark[sell].csi300
        b1000, s1000 = benchmark[buy].csi1000, benchmark[sell].csi1000
    result.update(entry_raw=float(b.open), exit_raw=float(s.open),
                  entry_one_price=bool(b.high == b.low), exit_one_price=bool(s.high == s.low),
                  gross_return=float(s.open / b.open - 1),
                  csi300_return=float(s300 / b300 - 1),
                  csi1000_return=float(s1000 / b1000 - 1))
    crossing = np.searchsorted(change_dates, sell, side="right") - np.searchsorted(change_dates, buy, side="right")
    result["corporate_action_crossing"] = bool(crossing)
    result["status"] = "corporate_action_excluded" if crossing else "observable_raw_endpoints"
    return result


def block_interval(frame, value, daily_equal=False):
    if daily_equal:
        work = frame.groupby("signal_date", as_index=False)[value].mean()
    else:
        work = frame[["signal_date", value]].dropna().copy()
    work["block"] = pd.to_datetime(work.signal_date).dt.to_period("Q").astype(str)
    blocks = work.groupby("block")[value].agg(["sum", "count"])
    if len(blocks) < 8:
        return np.nan, np.nan, len(blocks)
    rng = np.random.default_rng(20261017)
    choices = rng.integers(0, len(blocks), size=(2000, len(blocks)))
    means = blocks["sum"].to_numpy()[choices].sum(axis=1) / blocks["count"].to_numpy()[choices].sum(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return float(lo), float(hi), len(blocks)


def summarize(events, signals):
    rows = []
    usable = events.loc[events.common_clean].copy()
    usable["net30"] = usable.gross_return - .003
    usable["excess300"] = usable.net30 - usable.csi300_return
    usable["excess1000"] = usable.net30 - usable.csi1000_return
    scopes = [("all", "all", usable)]
    scopes += [("segment", k, g) for k, g in usable.groupby("segment")]
    scopes += [("year", str(k), g) for k, g in usable.groupby("signal_year")]
    scopes += [("universe", "in_2026_08_snapshot", usable.loc[usable.in_current_snapshot])]
    scopes += [("universe", "outside_2026_08_snapshot", usable.loc[~usable.in_current_snapshot])]
    scopes += [("liquidity_flag", "no_one_price_endpoints",
                usable.loc[~usable.entry_one_price.fillna(False) & ~usable.exit_one_price.fillna(False)])]
    horizon_specific = events.loc[events.horizon_cohort_complete & events.status.eq("observable_raw_endpoints")].copy()
    horizon_specific["net30"] = horizon_specific.gross_return - .003
    horizon_specific["excess300"] = horizon_specific.net30 - horizon_specific.csi300_return
    horizon_specific["excess1000"] = horizon_specific.net30 - horizon_specific.csi1000_return
    scopes += [("diagnostic", "horizon_specific_clean", horizon_specific)]
    for scope, label, frame in scopes:
        for (strategy, horizon), g in frame.groupby(["strategy", "horizon"]):
            if not len(g):
                continue
            lo, hi, blocks = block_interval(g, "excess300")
            dlo, dhi, _ = block_interval(g, "excess300", True)
            rows.append(dict(scope=scope, label=label, strategy=strategy, horizon=horizon,
                             events=len(g), stocks=g.code.nunique(), signal_dates=g.signal_date.nunique(),
                             gross_mean=g.gross_return.mean(), net30_mean=g.net30.mean(),
                             net60_mean=g.gross_return.mean() - .006,
                             net30_median=g.net30.median(), net30_win_rate=g.net30.gt(0).mean(),
                             excess300_mean=g.excess300.mean(), excess1000_mean=g.excess1000.mean(),
                             excess300_ci_low=lo, excess300_ci_high=hi, quarter_blocks=blocks,
                             date_equal_excess300=g.groupby("signal_date").excess300.mean().mean(),
                             date_equal_ci_low=dlo, date_equal_ci_high=dhi,
                             delisted_by_cutoff_events=int(g.delisted_by_cutoff.sum())))
    return pd.DataFrame(rows), usable


def paired_comparison(usable):
    rows = []
    for horizon, h in usable.groupby("horizon"):
        for label, frame in [("all", h), *list(h.groupby("segment"))]:
            pivot = frame.groupby(["signal_date", "strategy"]).excess300.mean().unstack()
            for a, b in [(CORE[1], CORE[0]), (CORE[2], CORE[3])]:
                if a not in pivot or b not in pivot:
                    continue
                p = pivot[[a, b]].dropna()
                diff = pd.DataFrame({"signal_date": p.index, "difference": (p[a] - p[b]).to_numpy()})
                lo, hi, blocks = block_interval(diff, "difference")
                rows.append(dict(segment=label, horizon=horizon, comparison=f"{a} - {b}",
                                 paired_dates=len(p), difference=diff.difference.mean(),
                                 ci_low=lo, ci_high=hi, quarter_blocks=blocks))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=HERE / "历史研究结果_20261017")
    parser.add_argument("--end", default="2026-10-09")
    parser.add_argument("--warmup", type=int, default=120, help="统一丢弃最初N根有效日线")
    parser.add_argument("--max-stocks", type=int)
    args = parser.parse_args()
    if args.warmup < 60 or args.end > "2026-10-09":
        raise ValueError("archival study requires warmup>=60, end<=2026-10-09")
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = Archive()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    vip = Path(cfg["data_provider"]["tdxdir"]) / "vipdoc"
    for file in [Path(__file__), HERE / "先锋上车事件研究.py", HERE / "先锋上车V2_条件选股.txt",
                 HERE / "先锋上车V3_双信号副图.txt", INPUTS.parent / "fetch_inputs.py", OLD / "fetch_baostock.py"]:
        archive.read(file)
    b300 = archive.tdx(vip / "sh/lday/sh000300.day").set_index("date")
    b1000 = archive.tdx(vip / "sh/lday/sh000852.day").set_index("date")
    b300 = b300.loc[(b300.index >= "2020-01-01") & (b300.index <= args.end)]
    calendar = b300.index.to_numpy()
    if len(calendar) < 250 or not b300.index.equals(b1000.loc[b300.index].index):
        raise ValueError("invalid benchmark calendar")
    benchmark = pd.DataFrame({"csi300": b300.open, "csi1000": b1000.loc[b300.index].open})
    benchmark_lookup = {r.Index: r for r in benchmark.itertuples()}
    position = {d: i for i, d in enumerate(calendar)}
    basic = archive.json(INPUTS / "stock-basic.json")
    master = {r["code"]: r for r in basic if r["type"] == "1" and allowed(r["code"])
              and (not r["ipoDate"] or r["ipoDate"] <= args.end)
              and (not r["outDate"] or r["outDate"] >= calendar[0])}
    current = pd.read_csv(ROOT / "data/stock_list.csv", dtype=str)
    archive.read(ROOT / "data/stock_list.csv")
    current_codes = {c[-2:].lower() + "." + c[:6] for c in current.code}
    known_master = set(master)
    local_unknown = []
    for ex in ("sh", "sz"):
        for p in sorted((vip / ex / "lday").glob("*.day")):
            code = ex + "." + p.stem[2:]
            if allowed(code) and code not in known_master:
                local_unknown.append(code)
    selected = sorted(master.items())[:args.max_stocks] if args.max_stocks else sorted(master.items())
    coverage, annual, signal_chunks, event_chunks, source_checks = [], [], [], [], []
    total_signals = 0
    started = time.time()
    for j, (code, meta) in enumerate(selected, 1):
        cov = dict(code=code, name=meta["code_name"], ipo_date=meta["ipoDate"],
                   delist_date=meta["outDate"], in_current_snapshot=code in current_codes,
                   delisted_by_cutoff=bool(meta["outDate"] and meta["outDate"] <= args.end),
                   status="missing_raw", raw_bars=0, signals=0)
        df = pd.DataFrame()
        factored = False
        signal_rows, event_rows = [], []
        try:
            df, source = load_raw(archive, vip, code, args.end)
            cov["raw_source"] = source
            cov.update(df.attrs)
            if not len(df):
                raise ValueError("missing raw daily history")
            # Historical listing date is an eligibility rule, never today's name/ST.
            if meta["ipoDate"]:
                df = df.loc[df.date.ge(meta["ipoDate"])].reset_index(drop=True)
            if meta["outDate"]:
                df = df.loc[df.date.le(meta["outDate"])].reset_index(drop=True)
            df = df.loc[df.date.isin(calendar)].reset_index(drop=True)
            if df.empty:
                raise ValueError("no listed-history bars within benchmark calendar")
            cov.update(raw_bars=len(df), first_date=df.date.iloc[0], last_date=df.date.iloc[-1], status="missing_factor")
            dates, factors, factor_source = load_factors(archive, code, args.end)
            factor = factors[np.searchsorted(dates, df.date.to_numpy(), side="right") - 1]
            changed = np.r_[False, ~np.isclose(factors[1:], factors[:-1], rtol=1e-10, atol=1e-12)]
            change_dates = dates[changed]
            factored = True
            cov.update(status="ok", factor_source=factor_source)
            work = df.copy()
            for col in ("open", "high", "low", "close"):
                work[col] = df[col].to_numpy() * factor
            work["price_floor"] = .01 * factor
            g = SIGNAL.compute_one(work, common_warmup=args.warmup)
            raw = df.set_index("date")
            raw_lookup = {r.date: r for r in df.itertuples(index=False)}
            # Daily historical ST is audit metadata, not a selection filter.
            st_path = OLD / "baostock" / f"{code}.json"
            st = {}
            if st_path.exists():
                st_rows = archive.json(st_path)
                st = {r["date"]: r.get("isST", "unknown") for r in st_rows}
                # One common day per overlapping file verifies units/provenance.
                candidates = [r for r in reversed(st_rows) if r["date"] <= args.end
                              and r["date"] in raw.index and r.get("tradestatus") == "1"]
                if candidates and source == "TDX_raw":
                    r = candidates[0]
                    max_error = max(abs(raw.loc[r["date"], k] - float(r[k])) for k in ("open", "high", "low", "close"))
                    vol_diff = abs(float(raw.loc[r["date"], "volume"]) - float(r["volume"]))
                    source_checks.append(dict(code=code, date=r["date"], max_ohlc_abs_error=max_error,
                                              volume_abs_error_shares=vol_diff))
                    if max_error > .011:
                        raise ValueError("TDX/BaoStock raw OHLC cross-check failed")
                    if vol_diff > max(100., float(r["volume"]) * .001):
                        raise ValueError("TDX/BaoStock volume cross-check failed")
            outcome_cache = {}
            for strategy in STRATEGIES:
                for t in np.flatnonzero(g[strategy].to_numpy()):
                    day = str(g.loc[t, "date"])
                    segment_end = "2021-12-31" if day < "2022" else "2024-12-31" if day < "2025" else args.end
                    p = position[day]
                    # Same signal cohort for all horizons; cannot cross segment end.
                    cohort_complete = p + 21 < len(calendar) and calendar[p + 21] <= segment_end
                    base = dict(code=code, name=meta["code_name"], strategy=strategy, signal_date=day,
                                signal_year=int(day[:4]), segment=segment(day),
                                in_current_snapshot=code in current_codes,
                                delisted_by_cutoff=cov["delisted_by_cutoff"],
                                historical_isST=st.get(day, "unknown"), raw_source=source,
                                signal_raw_close=float(raw_lookup[day].close),
                                signal_factor=float(factor[t]), pioneer=float(g.at[t, "先锋"]),
                                mid=float(g.at[t, "中期"]), cohort_complete=cohort_complete)
                    if day not in outcome_cache:
                        outcome_cache[day] = [scheduled_outcome(raw_lookup, calendar, position, day, h, change_dates, benchmark_lookup)
                                              for h in SIGNAL.HORIZONS]
                    outcomes = outcome_cache[day]
                    for outcome in outcomes:
                        outcome["horizon_cohort_complete"] = outcome.get("sell_date", "9999") <= segment_end
                    common_clean = cohort_complete and all(o["status"] == "observable_raw_endpoints" for o in outcomes)
                    base["common_clean"] = common_clean
                    signal_rows.append(base)
                    event_rows.extend({**base, **o} for o in outcomes)
                    cov["signals"] += 1
        except (ValueError, KeyError, OSError) as exc:
            # Data defects must not silently leak signals into the result.
            if cov["signals"]:
                raise
            cov["error"] = f"{type(exc).__name__}: {exc}"
            if cov["status"] == "ok":
                cov["status"] = "invalid_data"
                factored = False
        coverage.append(cov)
        if signal_rows:
            signal_chunks.append(pd.DataFrame(signal_rows))
            event_chunks.append(pd.DataFrame(event_rows))
            total_signals += len(signal_rows)
        for year in range(2020, int(args.end[:4]) + 1):
            eligible = calendar[(calendar >= max(f"{year}-01-01", meta["ipoDate"] or "1900-01-01"))
                                & (calendar <= min(f"{year}-12-31", meta["outDate"] or args.end, args.end))]
            if not len(eligible):
                continue
            observed = df.date.isin(eligible) if len(df) else pd.Series([], dtype=bool)
            annual.append(dict(year=year, code=code, expected_stock_sessions=len(eligible),
                               raw_stock_sessions=int(observed.sum()),
                               adjusted_stock_sessions=int(observed.sum()) if factored else 0,
                               warmed_stock_sessions=int((observed & (np.arange(len(df)) >= args.warmup)).sum()) if factored else 0,
                               delisted_by_cutoff=cov["delisted_by_cutoff"],
                               has_raw=bool(observed.any()), has_factor=bool(factored and observed.any())))
        if j % 500 == 0 or j == len(selected):
            print(json.dumps(dict(processed=j, total=len(selected), signals=total_signals,
                                  usable_stocks=sum(c["status"] == "ok" for c in coverage),
                                  seconds=round(time.time() - started, 1))), flush=True)
    cov = pd.DataFrame(coverage)
    annual = pd.DataFrame(annual)
    signals = pd.concat(signal_chunks, ignore_index=True) if signal_chunks else pd.DataFrame()
    events = pd.concat(event_chunks, ignore_index=True) if event_chunks else pd.DataFrame()
    del signal_chunks, event_chunks
    if signals.empty:
        raise ValueError("no signals in selected local data")
    for col in ("gross_return", "csi300_return", "csi1000_return"):
        if col not in events:
            events[col] = np.nan
    events["net_return_fixed30bps"] = events.gross_return - .003
    events["net_excess_csi300"] = events.net_return_fixed30bps - events.csi300_return
    cov.to_csv(out / "股票覆盖审计.csv", index=False, encoding="utf-8-sig")
    year_coverage = annual.groupby("year").agg(
        expected_stocks=("code", "size"), raw_stocks=("has_raw", "sum"), factored_stocks=("has_factor", "sum"),
        expected_stock_sessions=("expected_stock_sessions", "sum"), raw_stock_sessions=("raw_stock_sessions", "sum"),
        adjusted_stock_sessions=("adjusted_stock_sessions", "sum"), warmed_stock_sessions=("warmed_stock_sessions", "sum"))
    for k in ("raw", "adjusted", "warmed"):
        year_coverage[k + "_session_coverage"] = year_coverage[k + "_stock_sessions"] / year_coverage.expected_stock_sessions
    year_coverage.to_csv(out / "逐年覆盖.csv", encoding="utf-8-sig")
    annual.groupby(["year", "delisted_by_cutoff"]).sum(numeric_only=True).to_csv(out / "退市分组覆盖.csv", encoding="utf-8-sig")
    signals.to_csv(out / "信号审计.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    events.to_csv(out / "逐笔事件.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    summary, usable = summarize(events, signals)
    summary.to_csv(out / "分段与年度统计.csv", index=False, encoding="utf-8-sig")
    paired_comparison(usable).to_csv(out / "同日比较.csv", index=False, encoding="utf-8-sig")
    events.groupby(["strategy", "horizon", "status", "cohort_complete", "common_clean"]).size().rename("events").to_csv(
        out / "事件排除原因.csv", encoding="utf-8-sig")
    signals.groupby(["strategy", "historical_isST"]).size().rename("signals").to_csv(out / "历史ST标签覆盖.csv", encoding="utf-8-sig")
    pd.DataFrame(source_checks).to_csv(out / "跨来源核对.csv", index=False, encoding="utf-8-sig")
    digest = hashlib.sha256(json.dumps(archive.manifest, sort_keys=True).encode()).hexdigest()
    metadata = dict(version="pioneer-archival-event-v1", start=str(calendar[0]), end=str(calendar[-1]),
                    signal_start=signals.signal_date.min(), last_common_signal_date=signals.loc[signals.cohort_complete, "signal_date"].max(),
                    warmup=args.warmup, cost_scenarios_roundtrip_bps=[30, 60], horizons=list(SIGNAL.HORIZONS),
                    description="T close confirmation, next market-session open to open H market sessions later; raw returns; corporate crossings excluded",
                    common_sample="all four horizons observed, no factor changes while holding, no crossing segment boundary",
                    segment_design="descriptive frozen-rule retrospective periods; not independent train/validation/test",
                    master_stocks=len(master), processed_stocks=len(selected), status_counts=cov.status.value_counts().to_dict(),
                    known_delisted_by_cutoff=int(cov.delisted_by_cutoff.sum()),
                    delisted_usable=int((cov.delisted_by_cutoff & cov.status.eq("ok")).sum()),
                    local_codes_absent_from_basic=local_unknown, benchmark="TDX CSI300/CSI1000 price indices, same raw open dates",
                    signal_counts=signals.groupby("strategy").size().to_dict(),
                    common_counts=signals.loc[signals.common_clean].groupby("strategy").size().to_dict(),
                    factor_policy="immutable archival replay to saved coverage_end; no live TTL refresh, no raw fallback for signals",
                    raw_provenance_checks=len(source_checks),
                    data_and_code_sha256=digest, input_files=len(archive.manifest),
                    python=PYTHON, pandas=pd.__version__, numpy=np.__version__, seconds=round(time.time() - started, 2))
    (out / "运行元数据.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out / "输入指纹.json").write_text(json.dumps(archive.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print(summary.loc[(summary.scope == "all") & summary.strategy.isin(CORE),
                      ["strategy", "horizon", "events", "net30_mean", "excess300_mean", "excess1000_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
