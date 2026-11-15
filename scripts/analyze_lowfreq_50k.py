"""Supplementary economic-evidence summary for a frozen 50k cash-account study."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports/lowfreq-50k-2026-10-10"
UPSTREAM = ROOT / "reports/obv-alpha-2026-10-10"
SEED = 20261010
BOOTSTRAPS = 2000
BLOCK = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def load_trusted_index() -> tuple[pd.DataFrame, dict]:
    """Load only the local study pickle after verifying its original manifest hash."""
    manifest_path = UPSTREAM / "input-manifest.json"
    data_path = UPSTREAM / "dataset.pkl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = sha256(data_path)
    expected = manifest.get("dataset_sha256")
    if not expected or actual != expected:
        raise ValueError(f"upstream dataset SHA-256 mismatch: expected {expected}, got {actual}")
    # Trusted local artifact produced by this repository's frozen OBV study.
    with data_path.open("rb") as f:
        original = pickle.load(f)
    if original.get("fingerprint") != manifest.get("fingerprint"):
        raise ValueError("upstream dataset fingerprint does not match its original manifest")
    index = original.get("index")
    if not isinstance(index, pd.DataFrame) or not {"dt", "close"}.issubset(index.columns):
        raise ValueError("upstream index must be a DataFrame with dt and close columns")
    frame = index[["dt", "close"]].copy()
    frame["date"] = pd.to_datetime(frame.pop("dt"), errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty or (frame["close"] <= 0).any():
        raise ValueError("CSI300 index data is empty or contains nonpositive closes")
    return frame.set_index("date")["close"].rename("csi300_close").to_frame(), {
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "dataset": str(data_path), "dataset_sha256": actual,
        "dataset_fingerprint": manifest["fingerprint"], "rows": len(frame),
        "first_date": frame["date"].min().strftime("%Y-%m-%d"),
        "last_date": frame["date"].max().strftime("%Y-%m-%d"),
        "benchmark_definition": "CSI300 daily close price index from original['index']; price return only, no costs or distributions",
    }


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def account_trade_metrics(trades: pd.DataFrame) -> dict:
    if "pnl" not in trades:
        raise ValueError("trades.csv is missing pnl")
    pnl = _numeric(trades["pnl"]).dropna()
    positives = pnl[pnl > 0].sort_values(ascending=False)
    negatives = pnl[pnl < 0]
    avg_win = float(positives.mean()) if len(positives) else np.nan
    avg_loss = float(negatives.mean()) if len(negatives) else np.nan
    payoff = avg_win / abs(avg_loss) if np.isfinite(avg_loss) and avg_loss != 0 else np.nan
    holding = _numeric(trades.get("holding_sessions", pd.Series(dtype=float))).dropna()
    positive_sum = float(positives.sum()) if len(positives) else 0.0
    top1 = float(positives.head(1).sum() / positive_sum) if positive_sum > 0 else np.nan
    top5 = float(positives.head(5).sum() / positive_sum) if positive_sum > 0 else np.nan
    def example(row):
        result = {}
        for column in ("code", "entry_date", "exit_date", "pnl", "holding_sessions", "reason"):
            if column in row.index:
                result[column] = row[column]
        return json_safe(result)
    sorted_rows = trades.assign(_pnl=_numeric(trades["pnl"])).dropna(subset=["_pnl"])
    largest_win = example(sorted_rows.loc[sorted_rows._pnl.idxmax()]) if len(sorted_rows) and sorted_rows._pnl.max() > 0 else None
    largest_loss = example(sorted_rows.loc[sorted_rows._pnl.idxmin()]) if len(sorted_rows) and sorted_rows._pnl.min() < 0 else None
    return {
        "closed_trades_with_pnl": int(len(pnl)),
        "mean_holding_sessions": float(holding.mean()) if len(holding) else np.nan,
        "median_holding_sessions": float(holding.median()) if len(holding) else np.nan,
        "win_rate_positive_pnl": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "average_winner_pnl": avg_win, "average_loser_pnl": avg_loss,
        "payoff_ratio_mean_winner_to_absolute_mean_loser": payoff,
        "positive_pnl_sum_denominator_cny": positive_sum,
        "top1_positive_trade_share_of_all_positive_pnl": top1,
        "top5_positive_trades_share_of_all_positive_pnl": top5,
        "largest_winner": largest_win, "largest_loser": largest_loss,
    }


def account_activity(nav: pd.DataFrame, orders: pd.DataFrame) -> dict:
    dates = pd.to_datetime(nav.get("date", pd.Series(dtype=str)), errors="coerce").dropna().dt.normalize()
    if dates.empty:
        raise ValueError("nav.csv has no valid dates")
    iso = dates.dt.isocalendar()
    all_keys = pd.MultiIndex.from_arrays([iso.year.to_numpy(), iso.week.to_numpy()]).unique()
    ops = pd.Series(0, index=all_keys, dtype=int)
    if not orders.empty:
        if not {"date", "direction"}.issubset(orders.columns):
            raise ValueError("orders.csv must contain date and direction")
        order_dates = pd.to_datetime(orders["date"], errors="coerce")
        order_iso = order_dates.dt.isocalendar()
        order_keys = pd.MultiIndex.from_arrays([order_iso.year.to_numpy(), order_iso.week.to_numpy()])
        counts = pd.Series(1, index=order_keys).groupby(level=[0, 1]).sum()
        ops.loc[ops.index.intersection(counts.index)] = counts.reindex(ops.index.intersection(counts.index)).astype(int)
    nav_value = _numeric(nav.get("nav", pd.Series(dtype=float)))
    holdings = _numeric(nav.get("holdings", pd.Series(dtype=float)))
    invested = (holdings / nav_value).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "all_weeks_in_nav_calendar": int(len(ops)),
        "active_weeks_with_at_least_one_filled_order": int((ops > 0).sum()),
        "filled_operations_max_per_week": int(ops.max()) if len(ops) else 0,
        "filled_operations_mean_per_week_including_zero_weeks": float(ops.mean()) if len(ops) else np.nan,
        "mean_daily_invested_fraction_nav_weighted": float(invested.mean()) if len(invested) else np.nan,
        "investment_fraction_definition": "daily holdings divided by daily NAV, arithmetic mean over finite rows",
    }


def ols_and_block_bootstrap(strategy_returns: pd.Series, benchmark_returns: pd.Series,
                            seed: int = SEED, n_boot: int = BOOTSTRAPS, block: int = BLOCK) -> dict:
    pair = pd.concat([strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    y = pair.strategy.to_numpy(float)
    x = pair.benchmark.to_numpy(float)
    if len(y) < 30:
        return {"valid": False, "reason": "fewer than 30 aligned daily return observations", "observations": int(len(y))}
    def fit(indices):
        design = np.column_stack([np.ones(len(indices)), x[indices]])
        return np.linalg.lstsq(design, y[indices], rcond=None)[0]
    beta0, beta = fit(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    starts_max = len(y)
    estimates = np.empty(n_boot, dtype=float)
    blocks_needed = int(np.ceil(len(y) / block))
    for i in range(n_boot):
        starts = rng.integers(0, starts_max, size=blocks_needed)
        indices = np.concatenate([(np.arange(s, s + block) % len(y)) for s in starts])[:len(y)]
        estimates[i] = fit(indices)[0] * 252.0
    low, high = np.quantile(estimates, [.025, .975])
    return {
        "valid": True, "observations": int(len(y)),
        "daily_ols_intercept_arithmetic_annualized_252": float(beta0 * 252.0),
        "daily_ols_csi300_beta": float(beta),
        "moving_circular_block_bootstrap_block_sessions": block,
        "bootstrap_resamples": n_boot, "bootstrap_seed": seed,
        "annualized_intercept_95pct_conditional_interval": [float(low), float(high)],
        "interpretation": "conditional historical regression only; no causal interpretation and no untouched prospective test",
    }


def build_curve(nav: pd.DataFrame, benchmark: pd.DataFrame, initial: float = 50000.0) -> pd.DataFrame:
    frame = nav.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["nav"] = _numeric(frame["nav"])
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    aligned = frame[["nav"]].join(benchmark, how="inner").dropna()
    if aligned.empty:
        return pd.DataFrame(columns=["strategy_nav", "benchmark_nav", "strategy_drawdown", "benchmark_drawdown"])
    aligned["strategy_nav"] = aligned.nav
    prior = benchmark.loc[benchmark.index < aligned.index[0], "csi300_close"]
    base_close = float(prior.iloc[-1]) if len(prior) else float(aligned.csi300_close.iloc[0])
    aligned["benchmark_nav"] = initial * aligned.csi300_close / base_close
    strategy_path = np.r_[initial, aligned.strategy_nav.to_numpy(float)]
    benchmark_path = np.r_[initial, aligned.benchmark_nav.to_numpy(float)]
    aligned["strategy_drawdown"] = (strategy_path[1:] / np.maximum.accumulate(strategy_path)[1:]) - 1
    aligned["benchmark_drawdown"] = (benchmark_path[1:] / np.maximum.accumulate(benchmark_path)[1:]) - 1
    return aligned[["strategy_nav", "benchmark_nav", "strategy_drawdown", "benchmark_drawdown"]]


def save_plot(curves: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False, constrained_layout=True)
    for label, curve in curves.items():
        if curve.empty:
            continue
        axes[0].plot(curve.index, curve.strategy_nav, label=f"{label} 策略")
        axes[0].plot(curve.index, curve.benchmark_nav, linestyle="--", alpha=.8, label=f"{label} 沪深300价格指数")
        axes[1].plot(curve.index, curve.strategy_drawdown, label=f"{label} 策略")
        axes[1].plot(curve.index, curve.benchmark_drawdown, linestyle="--", alpha=.8, label=f"{label} 沪深300价格指数")
    axes[0].set_title("冻结规则账户净值与沪深300价格指数（各自起点对齐至5万元）")
    axes[0].set_ylabel("元")
    axes[1].set_title("按每日收盘估值的回撤")
    axes[1].set_ylabel("回撤")
    axes[1].set_xlabel("日期")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(ncol=2, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize_account(out: Path, label: str, spec_id: str, benchmark: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    folder = out / "accounts" / f"{label}-{spec_id}"
    required = [folder / n for n in ("summary.json", "nav.csv", "orders.csv", "trades.csv", "skips.csv")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("required account artifacts missing: " + ", ".join(missing))
    summary = json.loads(required[0].read_text(encoding="utf-8"))
    nav, orders, trades = (pd.read_csv(p) for p in required[1:4])
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce").dt.normalize()
    nav = nav.dropna(subset=["date"]).sort_values("date")
    curve = build_curve(nav, benchmark)
    # The first account-day return uses initial capital as the preceding NAV.
    strategy_returns = _numeric(nav.set_index("date")["nav"]).pct_change()
    if len(strategy_returns):
        strategy_returns.iloc[0] = _numeric(nav["nav"]).iloc[0] / 50000.0 - 1.0
    benchmark_returns = benchmark["csi300_close"].pct_change()
    if len(benchmark_returns):
        benchmark_returns.iloc[0] = np.nan
    complete = bool(summary.get("valid", summary.get("completed", False)))
    period = {
        "requested_start": summary.get("requested_start"),
        "requested_end": summary.get("requested_end"),
        "actual_end": summary.get("actual_end"),
        "account_valid_complete": complete,
        "complete_period_cagr": summary.get("cagr") if complete else None,
        "incomplete_period_reason": None if complete else "upstream summary marks account incomplete; complete-period CAGR and annualized intercept suppressed",
        "end_nav_cny": summary.get("end_nav", summary.get("nav")),
        "total_return_to_actual_end": summary.get("total_return"),
    }
    details = {
        "account_label": label, "strategy_id": spec_id, "source_folder": str(folder),
        "period": period,
        "trade_statistics": account_trade_metrics(trades),
        "account_activity": account_activity(nav, orders),
        "benchmark_comparison": {
            "benchmark": "CSI300 daily close price index, simple price return; no costs or total returns",
            "aligned_curve_observations": int(len(curve)),
            "aligned_first_date": curve.index.min().strftime("%Y-%m-%d") if len(curve) else None,
            "aligned_last_date": curve.index.max().strftime("%Y-%m-%d") if len(curve) else None,
            "benchmark_total_price_return": float(curve.benchmark_nav.iloc[-1] / 50000.0 - 1.0) if len(curve) else None,
            "benchmark_max_drawdown": float(curve.benchmark_drawdown.min()) if len(curve) else None,
            "complete_period_annualized_regression_alpha": None,
            "complete_period_regression": ols_and_block_bootstrap(strategy_returns, benchmark_returns) if complete else {
                "valid": False, "reason": "account incomplete; complete-period annualized intercept suppressed",
            },
        },
    }
    # More precise only when the frozen account is complete: use its daily OLS intercept.
    if complete:
        details["benchmark_comparison"]["complete_period_annualized_regression_alpha"] = details["benchmark_comparison"]["complete_period_regression"].get("daily_ols_intercept_arithmetic_annualized_252")
    return json_safe(details), curve


def write_explanation(path: Path, frozen: dict, accounts: dict) -> None:
    spec = frozen.get("selected_spec") or {}
    spec_id = frozen.get("selected_id") or spec.get("id") or "未冻结"
    family = spec.get("family", "未知")
    rules = spec.get("rules", spec)
    lines = [
        "# 5万元低频账户补充分析", "",
        f"本补充只分析冻结规则 **{spec_id}**（类别：{family}），规则参数来自冻结选择文件：", "",
        "```json", json.dumps(json_safe(rules), ensure_ascii=False, indent=2), "```", "",
        "不重选候选、不改阈值。OHLCV和成交额字段只能作为价格、成交活跃度代理，不能证明真实资金流、买卖主体、消息或基本面原因。经济机制是研究假设，不由本账户的历史收益倒推。", "",
        "规则解释（以冻结 ID 对应的固定脚本规则为准）：breakout20/60 越过前20/60日高点、收盘高于MA60、MA60走升且量比至少1.2；pullback3/5 前3/5日跌幅超过3%、前一日60日动量超过10%，其后高于MA60并收复前日高点、量缩；reversal3/5 前3/5日分别跌超6%/8%、前一日60日动量为正且高于MA120，再收复前日高点；compression20/40 前20/40日区间振幅低于20%后突破、量比至少1.5；rotation20/60 每周筛选，收盘高于MA60且对应动量超过5%/10%。成交额门槛只是流动性代理。这些条件不证明新信息、暂时抛售或真实资金流。", "",
        "## 交易与操作集中度", "",
        "持仓期按已平仓交易文件的 holding_sessions 统计；胜率定义为已平仓 pnl>0 比例；盈亏比为平均盈利额/平均亏损绝对额。盈利集中度的分母明确为全部正PnL之和，分别报告最大一笔与前五笔正收益之和占比。", "",
        "每周操作按 orders.csv 中实际成交订单行数统计，买入、卖出分别计一次；活跃周是至少有一笔成交的周。所有周数来自账户净值交易日历，均值分母包含无操作周。平均投入比例是日持仓市值/日净值的有效观测算术均值。", "",
    ]
    for label, item in accounts.items():
        trade, act, period = item["trade_statistics"], item["account_activity"], item["period"]
        lines += [f"### {label}", "",
                  f"账户完整：{period['account_valid_complete']}；请求区间 {period['requested_start']} 至 {period['requested_end']}，实际运行至 {period['actual_end']}。期末净值 {period['end_nav_cny']} 元。",
                  f"已平仓 {trade['closed_trades_with_pnl']} 笔；持仓期均值/中位数 {trade['mean_holding_sessions']} / {trade['median_holding_sessions']} 个交易日；正PnL胜率 {trade['win_rate_positive_pnl']}；平均盈亏比 {trade['payoff_ratio_mean_winner_to_absolute_mean_loser']}。",
                  f"全部正PnL合计 {trade['positive_pnl_sum_denominator_cny']} 元；最大一笔占比 {trade['top1_positive_trade_share_of_all_positive_pnl']}；前五笔占比 {trade['top5_positive_trades_share_of_all_positive_pnl']}。最大盈利例：`{json.dumps(trade['largest_winner'], ensure_ascii=False)}`；最大亏损例：`{json.dumps(trade['largest_loser'], ensure_ascii=False)}`。",
                  f"活跃周 {act['active_weeks_with_at_least_one_filled_order']}/{act['all_weeks_in_nav_calendar']}；单周最多操作 {act['filled_operations_max_per_week']}；含零周每周均值 {act['filled_operations_mean_per_week_including_zero_weeks']}；日均投入比例 {act['mean_daily_invested_fraction_nav_weighted']}。", ""]
        regression = item["benchmark_comparison"]["complete_period_regression"]
        if regression.get("valid"):
            lines.append(f"沪深300价格指数对照：日OLS beta={regression['daily_ols_csi300_beta']:.4f}；截距按252交易日年化（算术）={regression['daily_ols_intercept_arithmetic_annualized_252']:.2%}；5交易日圆形移动块bootstrap 2,000次95%区间={regression['annualized_intercept_95pct_conditional_interval']}。这是条件历史回归，不作因果解释，也不是独立前瞻检验。")
        else:
            lines.append("账户未完整：不计算完整期年化截距/CAGR；以上净值比较仅显示已观察到的部分区间。")
        lines.append("")
    lines += [
        "## 机制假设与证据边界", "",
        "各类机制均为可反驳的研究假设：突破或周度强势可能来自信息逐步进入价格及后续趋势需求；回调恢复可能是中期需求未消失而短期获利卖压被吸收；短期反转可能是急售造成的暂时价格压力消退；窄幅突破可能表示供求状态变化。OHLCV无法辨认卖压是暂时的还是永久信息，也无法识别资金来源。",
        "风险暴露以沪深300日收益回归beta作单一粗略描述，未覆盖小盘、行业、风格、波动、涨跌停和流动性暴露；若市场基准相关性低或持仓高度集中，beta解释更弱。", 
        "冻结选择是在训练窗口内按预先规定的指标选出的，之后的账户表现仍会受到候选筛选、当前股票名单生存者/历史ST缺口、历史已被查看、日线成交与涨跌停撮合近似、公司行动数据历史版本不全等限制。不能把训练选择优势当作未来统计保证。日收益OLS截距是样本内风险暴露调整的描述性统计，不能证明超额收益是alpha或来自某一经济机制。",
        "比较基准使用同日沪深300收盘价指数做简单价格收益，未计费用、股息或总回报；每段单独把起点归一到5万元，只是理论参照，不是可投资产品。图中账户净值沿用逐日账户记录，含持仓按收盘价估值。", "",
        "![策略与沪深300价格指数净值及回撤](equity.png)", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="lowfreq study output directory")
    args = parser.parse_args()
    out = args.out.resolve()
    frozen_path = out / "frozen-selection.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(f"frozen selection file not found: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    spec = frozen.get("selected_spec")
    spec_id = frozen.get("selected_id") or (spec or {}).get("id")
    if not spec_id or not spec or spec.get("id") != spec_id:
        raise ValueError("frozen-selection.json has no internally consistent selected_id/selected_spec")
    benchmark, input_record = load_trusted_index()
    accounts, curves = {}, {}
    for label in ("full", "recent", "recent_cost_stress", "recent_entry_delay"):
        accounts[label], curves[label] = summarize_account(out, label, spec_id, benchmark)
    out.mkdir(parents=True, exist_ok=True)
    save_plot({label: curves[label] for label in ("full", "recent")}, out / "equity.png")
    write_explanation(out / "解释与收益来源.md", frozen, accounts)
    result = {
        "frozen_selection": {"selected_id": spec_id, "selected_spec": spec,
                             "selection_rule": frozen.get("selection_rule"),
                             "selection_uses_recent_year": frozen.get("selection_uses_recent_year")},
        "upstream_dataset_verification": input_record,
        "benchmark_method": "same-calendar CSI300 daily close price simple returns; theoretical 50,000 CNY starting reference; no cost/distribution/total return",
        "regression_method": {"model": "strategy daily return = intercept + beta * CSI300 daily price return + residual",
                              "annualization": "arithmetic intercept times 252; not CAGR",
                              "bootstrap": f"circular moving blocks of {BLOCK} sessions, {BOOTSTRAPS} resamples, fixed seed {SEED}",
                              "limitations": "historical conditional small-sample regression; no causality or untouched prospective test"},
        "accounts": accounts,
        "generated": ["supplemental-summary.json", "解释与收益来源.md", "equity.png"],
    }
    (out / "supplemental-summary.json").write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "selected_id": spec_id, "accounts": list(accounts), "verified_dataset_sha256": input_record["dataset_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
