"""Frozen three-year OBV study: prediction, matched alpha, and raw execution."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.strategies.obv_resonance_factor import features
from src.utils.config_loader import ConfigLoader
from src.utils.price_adjustment import _factor_frame, assert_raw_prices
from src.utils.tdx_provider import TdxProvider
from scripts.backtest_turning_points import daily_bars, in_scope

START, END, TEST_START = "2023-10-01", "2026-09-30", "2025-10-01"
HORIZONS = (1, 5, 10, 20)
TRADE_HORIZONS = (5, 10, 20)
VERSION = "obv-current-week-alpha-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generator_hashes():
    paths = (Path(__file__).resolve(), ROOT / "src/strategies/obv_resonance_factor.py",
             ROOT / "src/utils/price_adjustment.py", ROOT / "src/utils/tdx_provider.py",
             ROOT / "scripts/backtest_turning_points.py", ROOT / "src/utils/config_loader.py",
             ROOT / "src/utils/indicators.py")
    return {str(p): sha(p) for p in paths}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def build_bars(work):
    result = daily_bars(work)
    for row in work.itertuples():
        b = result[str(row.dt.date())]
        b.update(adjustment_factor=float(row.adjustment_factor),
                 obv_resonance_factor=int(row.obv_resonance_factor))
        # Opening validation uses only the prior reference and today's open.
        b["open_price_reference_valid"] = (
            b.get("previous_close") is not None
            and b["down_limit"] - .011 <= b["open"] <= b["up_limit"] + .011
        )
    return result


def labeled_panel(work: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Labels use MARKET dates, never shift over missing stock bars."""
    aligned = work.set_index("dt").reindex(calendar)
    c = aligned.signal_close
    result = pd.DataFrame(index=calendar)
    result["code"] = work.code.iloc[0]
    result["factor"] = aligned.obv_resonance_factor.fillna(0).astype("int8")
    result["trigger"] = aligned.resonance_trigger.fillna(False).astype(bool)
    result["daily_condition"] = aligned.daily_above_maobv.fillna(False).astype(bool)
    result["weekly_condition"] = (
        aligned.weekly_rising_two.fillna(False).astype(bool)
        & aligned.weekly_above_maobv.fillna(False).astype(bool))
    history = pd.Series(np.arange(len(work)), index=work.dt).reindex(calendar)
    result["eligible"] = (aligned.history_ready.fillna(False).astype(bool)
                          & history.ge(120) & aligned.vol.gt(0))
    for n in (10, 20, 60):
        # Historical features count valid stock bars, as in the chart; future
        # endpoints below use market sessions. The two conventions are explicit.
        momentum = work.signal_close.pct_change(n, fill_method=None) * 100
        result[f"mom{n}"] = pd.Series(momentum.to_numpy(), index=work.dt).reindex(calendar)
    volatility = work.signal_close.pct_change(fill_method=None).rolling(20).std() * 100
    result["vol20"] = pd.Series(volatility.to_numpy(), index=work.dt).reindex(calendar)
    liquidity = np.log(work.amount.rolling(20).mean().where(lambda s: s > 0))
    result["log_amount20"] = pd.Series(liquidity.to_numpy(), index=work.dt).reindex(calendar)
    for h in HORIZONS:
        result[f"fwd_close_{h}"] = (c.shift(-h) / c - 1) * 100
        result[f"label_end_{h}"] = pd.Series(calendar, index=calendar).shift(-h)
    # A separate executable timing label: next open -> close h market sessions
    # AFTER entry. It is a price return, without fees or cash accounting.
    opening = aligned.open * aligned.signal_multiplier
    for h in TRADE_HORIZONS:
        result[f"fwd_open_{h}"] = (c.shift(-(h + 1)) / opening.shift(-1) - 1) * 100
    return result.loc[result.index >= pd.Timestamp(START)].rename_axis("dt").reset_index()


def frozen_design():
    return dict(version=VERSION, start=START, intended_end=END,
                test_start=TEST_START, ma_period=18, current_week=True,
                horizons=HORIZONS, primary_horizon=10, trade_horizons=TRADE_HORIZONS,
                universe="current mainboard non-ST/non-delisting name list; survivors only",
                factor_fixed=True, tuning=False, split_label_purge=True,
                primary_comparison="same-date matched control and conditional coefficient",
                matching="mom20 quintile x vol20 tertile; minimum 3 per side/cell",
                controls=["mom10", "mom20", "mom60", "vol20", "log_amount20"],
                inference="date-level HAC maxlag=h+1 and circular date-block bootstrap",
                bootstrap_draws=2000, bootstrap_block_days=40, seed=20261010,
                entry="signal close then next market-day open; no delayed entry",
                trade_exit="close h market sessions after entry; unfillable -> <=5 later opens",
                trade_capital=100000, lot=100, t_plus_one=True,
                commission=.00025, min_commission=5., sell_stamp=.0005,
                transfer_per_side=.00001, slippage_per_side=.001,
                cost_stress_slippage=.003,
                alpha_gate="last-year primary matched and conditional lower 95% HAC AND bootstrap bounds >0; net raw trials positive",
                boundaries=["no historical ST or delisted universe", "no industry/size controls",
                            "corporate-action cash/share accounting absent; trade outcomes censored",
                            "daily bars cannot verify queue or exact opening/closing fills",
                            "independent trades are not a shared-account portfolio"])


def load_dataset(out: Path, max_stocks=None):
    cfg_path = ROOT / "config.json"
    list_path = ROOT / "data/stock_list.csv"
    manifest = {str(p): sha(p) for p in (cfg_path, list_path)}
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["data_provider"]["local_cache_enabled"] = False
    stocks = pd.read_csv(list_path, dtype=str)
    stocks = stocks[stocks.apply(lambda r: in_scope(r.code, r["name"]), axis=1)]
    if max_stocks is not None:
        stocks = stocks.head(max_stocks)
    frames, raw_frames, coverage = [], {}, []
    started = time.monotonic()
    with ConfigLoader.runtime_snapshot(cfg):
        provider = TdxProvider()
        provider._cache_enabled = False
        index_path = Path(provider._symbol_file_hints("399300.SZ")[0])
        if not index_path.is_file():
            raise ValueError("local CSI300 calendar file missing")
        manifest[str(index_path)] = sha(index_path)
        index = provider._normalize_daily_df(provider._reader_daily("sz399300"), "399300.SZ")
        index = index[index.dt.between("2022-10-01", END)].reset_index(drop=True)
        assert_raw_prices(provider, "tdx", index, cfg)
        calendar = pd.DatetimeIndex(index.dt)
        if len(calendar) < 900 or calendar.max() < pd.Timestamp("2026-09-01"):
            raise ValueError("index calendar does not cover the planned study")
        for j, stock in enumerate(stocks.itertuples()):
            record = dict(code=stock.code, name=stock.name, status="ok")
            try:
                raw_path = Path(provider._symbol_file_hints(stock.code)[0])
                cache_path = ROOT / f"data/history/cache/adjustment_factors_v1_{stock.code[-2:].lower()}_{stock.code[:6]}.json"
                if not raw_path.is_file() or not cache_path.is_file():
                    raise ValueError("local raw data or adjustment snapshot missing")
                manifest[str(raw_path)] = sha(raw_path)
                manifest[str(cache_path)] = sha(cache_path)
                raw = provider._normalize_daily_df(provider._reader_daily(provider._raw_symbol(stock.code)), stock.code)
                raw = raw[raw.dt.between("2022-10-01", calendar.max())].reset_index(drop=True)
                record.update(bars=len(raw), first_date=str(raw.dt.min().date()),
                              last_date=str(raw.dt.max().date()))
                if len(raw) < 120:
                    record["status"] = "short_history"
                    coverage.append(record)
                    continue
                if not raw.dt.isin(calendar).all():
                    raise ValueError("stock dates outside verified index calendar")
                assert_raw_prices(provider, "tdx", raw, cfg)
                raw.attrs["raw_price_verified_external"] = True
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                market_code = f"{stock.code[-2:].lower()}.{stock.code[:6]}"
                if (cached.get("version") != 1 or cached.get("code") != market_code
                        or cached.get("source") != "baostock.backAdjustFactor"
                        or pd.Timestamp(cached["coverage_end"]) < raw.dt.max()):
                    raise ValueError("unverified adjustment snapshot contract")
                adjustment = _factor_frame(cached["rows"], cached["coverage_end"])
                work = features(raw, adjustment)
                panel = labeled_panel(work, calendar)
                record.update(eligible_days=int(panel.eligible.sum()),
                              signal_days=int((panel.eligible & panel.factor.eq(1)).sum()),
                              activations=int((panel.eligible & panel.trigger).sum()),
                              factor_fetched_at=cached["fetched_at"],
                              factor_coverage_end=cached["coverage_end"])
                frames.append(panel)
                raw_frames[stock.code] = work[["code", "dt", "open", "high", "low", "close",
                                               "vol", "amount", "adjustment_factor", "obv_resonance_factor"]].copy()
                # One true as-of midweek check per stock, including all original
                # factor columns, before committing this stock to the study.
                midweek = work.index[(work.dt >= TEST_START) & work.dt.dt.dayofweek.eq(2)]
                if len(midweek):
                    end = int(midweek[0]) + 1
                    prefix = features(raw.iloc[:end], adjustment)
                    pd.testing.assert_frame_equal(work.iloc[:end].reset_index(drop=True), prefix)
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            coverage.append(record)
            if (j + 1) % 100 == 0 or j + 1 == len(stocks):
                print(json.dumps(dict(stage="load", stocks=j+1, total=len(stocks),
                                      errors=sum(r["status"] == "error" for r in coverage),
                                      elapsed_sec=round(time.monotonic()-started, 1)), ensure_ascii=False), flush=True)
    pd.DataFrame(coverage).to_csv(out / "data-coverage.csv", index=False, encoding="utf-8-sig")
    errors = [r for r in coverage if r["status"] == "error"]
    if errors:
        raise ValueError(f"{len(errors)} invalid/unverified symbols: inspect data-coverage.csv")
    for path, expected in manifest.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"input changed during load: {Path(path).name}")
    if not frames:
        raise ValueError("empty usable universe")
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    dataset = dict(panel=pd.concat(frames, ignore_index=True), raw_frames=raw_frames,
                   index=index, calendar=calendar, coverage=coverage, fingerprint=fingerprint)
    with (out / "dataset.pkl").open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(out / "input-manifest.json", dict(fingerprint=fingerprint, files=manifest,
               generator_sha256=generator_hashes(), dataset_sha256=sha(out / "dataset.pkl")))
    return dataset


def reuse_dataset(out: Path):
    manifest = json.loads((out / "input-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("generator_sha256") != generator_hashes():
        raise ValueError("dataset generating code changed or is unbound; reload pinned inputs")
    if manifest.get("dataset_sha256") != sha(out / "dataset.pkl"):
        raise ValueError("serialized dataset checksum mismatch")
    for path, expected in manifest["files"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned input changed: {Path(path).name}")
    # Only our own local artifact is accepted, after complete input validation.
    with (out / "dataset.pkl").open("rb") as handle:
        dataset = pickle.load(handle)
    if dataset["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("dataset fingerprint mismatch")
    return dataset


def periods(end):
    return {"full": (START, end), "discovery": (START, "2025-09-30"),
            "recent_year": (TEST_START, end), "2023_Q4": (START, "2023-12-31"),
            "2024": ("2024-01-01", "2024-12-31"),
            "2025": ("2025-01-01", "2025-12-31"), "2026_YTD": ("2026-01-01", end)}


def statistical_study(dataset, out):
    from scripts.obv_alpha_statistics import daily_effects, mean_inference
    panel, calendar = dataset["panel"], dataset["calendar"]
    end = str(calendar.max().date())
    summaries, daily_parts = [], []
    idx = dataset["index"].set_index("dt")
    for variant in ("resonance", "daily_only", "weekly_only", "activation"):
        source = panel.copy()
        if variant == "daily_only":
            source["factor"] = source.daily_condition.astype("int8")
        elif variant == "weekly_only":
            source["factor"] = source.weekly_condition.astype("int8")
        elif variant == "activation":
            source["eligible"] &= source.factor.eq(0) | source.trigger
            source["factor"] = source.trigger.astype("int8")
        # Cross-sectional effects depend only on this date, not the enclosing
        # report period. Compute once; purge labels when slicing date series.
        all_daily = daily_effects(source, horizons=HORIZONS).reset_index().rename(
            columns={"all_selected": "selected_count", "all_control": "control_count"})
        date_end = pd.DataFrame({"dt": calendar})
        for h in HORIZONS:
            date_end[f"label_end_{h}"] = date_end.dt.shift(-h)
        all_daily = all_daily.merge(date_end, on="dt", how="left", validate="many_to_one")
        for period, (start, finish) in periods(end).items():
            for h in HORIZONS:
                # A sample belongs to this period only if its complete label
                # also ends inside it. In particular, discovery cannot see OOS.
                daily = all_daily.loc[all_daily.horizon.eq(h) & all_daily.dt.between(start, finish)
                                     & all_daily[f"label_end_{h}"].le(pd.Timestamp(finish))].copy()
                daily["variant"], daily["period"] = variant, period
                benchmark = (idx.close.shift(-h) / idx.close - 1) * 100
                if not daily.empty:
                    daily["benchmark_pct"] = pd.to_datetime(daily["dt"]).map(benchmark)
                    daily["selected_excess_csi300_pct"] = daily.selected_mean - daily.benchmark_pct
                daily_parts.append(daily)
                period_calendar = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(finish))]
                cols = ("spread_pct", "matched_spread_pct", "conditional_beta_pct", "selected_excess_csi300_pct")
                for metric in cols:
                    series = daily.set_index("dt")[metric].reindex(period_calendar) if metric in daily else pd.Series(np.nan, index=period_calendar)
                    stats = mean_inference(series.to_numpy(), maxlags=h+1, block_length=40,
                                           draws=2000, seed=20261010)
                    summaries.append(dict(variant=variant, period=period, horizon=h, metric=metric,
                                          **stats, selected_obs=int(daily.selected_count.sum()) if not daily.empty else 0,
                                          control_obs=int(daily.control_count.sum()) if not daily.empty else 0,
                                          matched_coverage_mean=float(daily.matched_selected_coverage.mean()) if not daily.empty else np.nan))
            print(json.dumps(dict(stage="statistics", variant=variant, period=period)), flush=True)
    daily_all, stats_all = pd.concat(daily_parts, ignore_index=True), pd.DataFrame(summaries)
    daily_all.to_csv(out / "daily-effects.csv", index=False, encoding="utf-8-sig")
    stats_all.to_csv(out / "alpha-summary.csv", index=False, encoding="utf-8-sig")
    signal_rows = panel.loc[panel.eligible & panel.factor.eq(1)]
    signal_rows.to_csv(out / "signal-labels.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    return stats_all, daily_all


def execution_study(dataset, out):
    from scripts.obv_execution_replay import replay_trade
    panel, raw_frames = dataset["panel"], dataset["raw_frames"]
    calendar = dataset["calendar"].strftime("%Y-%m-%d").tolist()
    index_bars = dataset["index"].set_index(dataset["index"].dt.dt.strftime("%Y-%m-%d"))
    records, started = [], time.monotonic()
    event_days = {code: g.dt.tolist() for code, g in panel.loc[panel.eligible & panel.trigger].groupby("code")}
    for j, (code, raw) in enumerate(raw_frames.items()):
        bars = build_bars(raw)
        events = event_days.get(code, [])
        for dt in events:
            signal_day = str(dt.date())
            for rule in ("hold5", "hold10", "hold20", "factor_loss", "hold10_stress"):
                horizon = {"hold5": 5, "hold10": 10, "hold20": 20, "factor_loss": 20, "hold10_stress": 10}[rule]
                result = replay_trade(code, signal_day, bars, calendar, horizon,
                                      slippage=.003 if rule.endswith("stress") else .001,
                                      exit_on_factor_loss=rule == "factor_loss")
                result["rule"] = rule
                if result["status"] == "closed":
                    entry, exit_ = index_bars.loc[result["entry_date"]], index_bars.loc[result["exit_date"]]
                    denominator = entry.open
                    endpoint = exit_.open if "open" in result["exit_reason"] else exit_.close
                    result["csi300_return_pct"] = (endpoint / denominator - 1) * 100
                    result["excess_csi300_pct"] = result["net_return_pct"] - result["csi300_return_pct"]
                records.append(result)
        if (j + 1) % 100 == 0 or j + 1 == len(raw_frames):
            print(json.dumps(dict(stage="replay", stocks=j+1, total=len(raw_frames), trials=len(records),
                                  elapsed_sec=round(time.monotonic()-started, 1))), flush=True)
    trials = pd.DataFrame(records)
    trials.to_csv(out / "trade-trials.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    summary = []
    for period, (start, finish) in periods(calendar[-1]).items():
        for rule, group in trials[trials.signal_date.between(start, finish)].groupby("rule"):
            # Closed observations crossing a subperiod end are separately
            # censored for this subperiod, never imported from the next year.
            closed = group[group.status.eq("closed") & group.exit_date.le(finish)]
            effective_status = group.status.mask(group.status.eq("closed") & group.exit_date.gt(finish), "censored_period_end")
            summary.append(dict(period=period, rule=rule, trials=len(group), closed=len(closed),
                                no_entry=int(group.status.eq("no_entry").sum()),
                                censored=len(group)-len(closed)-int(group.status.eq("no_entry").sum()),
                                mean_net_pct=closed.net_return_pct.mean(), median_net_pct=closed.net_return_pct.median(),
                                win_rate_pct=closed.net_return_pct.gt(0).mean()*100,
                                mean_excess_csi300_pct=closed.excess_csi300_pct.mean(),
                                mean_fees=closed.fees.mean(), delay_count=int(closed.delayed_sell_days.gt(0).sum()),
                                status_counts=json.dumps(Counter(effective_status), ensure_ascii=False)))
    result = pd.DataFrame(summary)
    result.to_csv(out / "execution-summary.csv", index=False, encoding="utf-8-sig")
    return result, trials


def report(dataset, stats, execution, trials, out):
    primary = stats[(stats.variant == "resonance") & (stats.period == "recent_year") & (stats.horizon == 10)]
    rows = primary.set_index("metric")
    evidence = []
    for metric in ("matched_spread_pct", "conditional_beta_pct"):
        row = rows.loc[metric]
        evidence.append(row.hac_ci_low_pct > 0 and row.bootstrap_ci_low_pct > 0)
    trade = execution[(execution.period == "recent_year") & (execution.rule == "hold10")].iloc[0]
    passed = all(evidence) and trade.mean_net_pct > 0
    verdict = "出现增量预测力证据，仍需真实历史股票池和组合验证" if passed else "本轮未通过稳定增量 alpha 门槛"
    coverage = dataset["coverage"]
    summary = dict(version=VERSION, verdict=verdict, alpha_gate_passed=bool(passed),
                   stocks=len(dataset["raw_frames"]), requested_stocks=len(coverage),
                   short_history=sum(r["status"] == "short_history" for r in coverage),
                   actual_end=str(dataset["calendar"].max().date()), study_start=START,
                   split=TEST_START, dataset_fingerprint=dataset["fingerprint"],
                   eligible_rows=int(dataset["panel"].eligible.sum()),
                   signal_rows=int((dataset["panel"].eligible & dataset["panel"].factor.eq(1)).sum()),
                   activation_rows=int((dataset["panel"].eligible & dataset["panel"].trigger).sum()),
                   independent_trials=len(trials), trial_status_counts=dict(Counter(trials.status)),
                   primary_recent_year=primary.to_dict("records"), recent_year_hold10=trade.to_dict(),
                   factor_code_sha256=sha(ROOT / "src/strategies/obv_resonance_factor.py"),
                   limitations=frozen_design()["boundaries"])
    write_json(out / "summary.json", summary)
    def table(frame):
        return frame.to_markdown(index=False, floatfmt=".3f")
    focus = stats[(stats.variant == "resonance") & (stats.horizon == 10) & stats.metric.isin(["matched_spread_pct", "conditional_beta_pct"])]
    focus = focus[["period", "metric", "n", "mean_pct", "hac_ci_low_pct", "hac_ci_high_pct", "bootstrap_ci_low_pct", "bootstrap_ci_high_pct"]]
    execution_focus = execution[execution.rule.eq("hold10")][["period", "trials", "closed", "censored", "mean_net_pct", "mean_excess_csi300_pct", "win_rate_pct"]]
    text = f"""# OBV 日周共振：近三年价值检验

结论：**{verdict}**。

固定 MAOBV18，包含本周但逐日只使用当时数据。研究期 {START} 至 {summary['actual_end']}，
{summary['stocks']} 只当前主板非 ST 股票，{summary['eligible_rows']:,} 个合格股票日，
{summary['signal_rows']:,} 个共振状态日、{summary['activation_rows']:,} 次从0到1。
前两年用于历史描述；最近一年 {TEST_START} 起为预先锁定的检验区间。
所有子区间都要求收益标签/交易退出落在该区间内，防止跨界取用后续收益。

## 因子额外信息：主要10日检验

收益标签为信号日收盘至10个市场交易日后收盘的复权价格收益，单位百分点。
同日动量20分五组、波动率20分三组匹配；另做逐日横截面回归，控制10/20/60日动量、
20日波动率和20日平均成交金额对数。逐日系数平均采用HAC及40日整日期分块bootstrap。
95%区间跨0表示未排除没有额外信息；1/5/20日和拆分条件均为探索性检验。

{table(focus)}

## 下一交易日开盘买入：固定10日后收盘退出

每次独立10万元现金试验，不累加为账户收益。100股整数手、T+1，
佣金万2.5/最低5元、卖出印花税0.05%、双边过户费0.001%、双边滑点0.1%。
开盘涨停/停牌等不买，不能卖出则最多等待后5个市场日开盘。
持仓跨公司行动或无法验证价格参考时保留入场记录，将收益标为缺失。
沪深300基准使用相同入场和实际退出时点的价格收益，不计基准交易费。

{table(execution_focus)}

## 证据范围

当前股票名单时间为2026-08-05，缺少历史ST和已退市股票，存在生存者偏差；
缺行业/市值控制，不能把残余系数当作完整风险模型alpha。
复权价格标签不是含分红现金流水的真实总收益；原价现金交易则拒绝跨公司行动收益结论。
独立交易区间会重叠，不能把交易次数视为独立样本、把均值当作组合年化。
日K撮合没有验证开盘/收盘排队、实际冲击成本。

## 文件

- `alpha-summary.csv`：全部区间、期限、条件拆分及两种置信区间。
- `daily-effects.csv`：逐日效应、回归系数、匹配覆盖率与基准。
- `signal-labels.csv.gz`：每个共振状态日及未来标签。
- `trade-trials.csv.gz`：首次触发逐笔现金交易，含拒绝/缺失原因。
- `execution-summary.csv`：5/10/20日、共振失效退出与滑点压力结果。
- `predeclared-design.json`、`input-manifest.json`、`data-coverage.csv`：预先规则、完整输入指纹及覆盖。

方法/费用来源：[Newey-West官方实现说明](https://www.statsmodels.org/dev/generated/statsmodels.stats.sandwich_covariance.cov_hac.html)、
[上交所过户费说明](https://one.sse.com.cn/onething/gptz/)、
[2023年8月28日印花税减半通知](https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20230827_5725662.shtml)。
"""
    (out / "研究结果.md").write_text(text, encoding="utf-8")
    # A portable table report; every displayed cell comes from exported CSVs.
    tables = [stats[(stats.variant == "resonance") & (stats.horizon == 10)].to_html(index=False, float_format=lambda x: f"{x:.3f}"), execution.to_html(index=False, float_format=lambda x: f"{x:.3f}")]
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OBV日周共振检验</title><style>body{{margin:0;background:#f3f5f8;color:#172238;font:15px/1.7 system-ui,"Microsoft YaHei"}}main{{max-width:1400px;margin:auto;padding:30px}}section{{background:white;padding:24px;margin:20px 0;border-radius:14px}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px}}td,th{{padding:8px;border-bottom:1px solid #dce2ed;text-align:right}}th{{background:#edf1f8}}.scroll{{overflow:auto}}a{{color:#1565c0}}h1{{margin-bottom:6px}}</style><main><h1>OBV 日周共振 · 价值检验</h1><p>{html.escape(verdict)}</p><p>{START} — {summary['actual_end']} · {summary['stocks']}只当前主板正常股票 · 固定18周期</p><section><h2>主要10日预测与超额检验</h2><p>均值和置信区间单位为百分点；正收益不等于独立alpha。逐日HAC和整日期分块bootstrap处理重叠。</p><div class="scroll">{tables[0]}</div></section><section><h2>执行检验</h2><p>首次触发后下一市场日开盘买入，含费用、滑点、T+1、停牌与涨跌停拒绝。独立10万元试验，不能累加为组合收益。</p><div class="scroll">{tables[1]}</div></section><section><h2>数据与文件</h2><p>当前名单有生存者和历史ST偏差；缺行业/市值控制；持仓公司行动核算缺失；日K不证明盘口成交。</p><a href="研究结果.md">完整说明</a> · <a href="alpha-summary.csv">全部统计</a> · <a href="daily-effects.csv">逐日系数</a> · <a href="trade-trials.csv.gz">逐笔交易</a> · <a href="signal-labels.csv.gz">信号标签</a> · <a href="summary.json">运行记录</a></section></main></html>'''
    (out / "因子价值报告.html").write_text(page, encoding="utf-8")
    print(json.dumps(clean(summary), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/obv-alpha-2026-10-10")
    parser.add_argument("--max-stocks", type=int)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    args = parser.parse_args()
    if args.max_stocks is not None and args.max_stocks < 1:
        parser.error("--max-stocks must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    design = frozen_design()
    if args.max_stocks is not None:
        design["pilot_max_stocks"] = args.max_stocks
    design_path = out / "predeclared-design.json"
    if design_path.exists():
        if json.loads(design_path.read_text(encoding="utf-8")) != clean(design):
            raise ValueError("existing predeclared design differs; use a new output directory")
    else:
        write_json(design_path, design)
    dataset = reuse_dataset(out) if args.reuse else load_dataset(out, args.max_stocks)
    if args.load_only:
        print(json.dumps(dict(stage="loaded", stocks=len(dataset["raw_frames"]),
                              rows=len(dataset["panel"]), fingerprint=dataset["fingerprint"])), flush=True)
        return
    stats, daily = statistical_study(dataset, out)
    execution, trials = execution_study(dataset, out)
    report(dataset, stats, execution, trials, out)


if __name__ == "__main__":
    main()
