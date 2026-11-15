"""Recent-year factor study: four-session close>3% versus intraday high>3%."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.strategies.low_reversal_factor import DEFAULT, features, parameter_grid, signal_mask
from src.utils.config_loader import ConfigLoader
from src.utils.price_adjustment import _read_cache, assert_raw_prices
from src.utils.runtime_params import parameter_overrides
from src.utils.tdx_provider import TdxProvider
from scripts.backtest_turning_points import COSTS, daily_bars, find_entry, in_scope, run_trial

VERSION = "low-reversal-four-session-v1"
PERIODS = {"train": ("2025-10-09", "2026-04-30"), "test": ("2026-05-01", "2026-10-09")}
MODELS = ("close", "intraday", "next_open")
NAMES = {"close": "收盘>3%同收盘价（理论）", "intraday": "盘中最高>3%突破价", "next_open": "收盘>3%确认后次日开盘"}
HORIZONS = (5, 10, 20)
BUFFER, CAPITAL = 30, 100000.
STUDY_COSTS = {**COSTS, "trading_cost.transfer_fee": .00002}
FIXED_HOLD = dict(kind="fixed", stop=1., take=10.)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def execution_bars(work):
    """Keep execution raw; reject unverified ex-day reference bands."""
    bars, prior = daily_bars(work), None
    for row in work.itertuples():
        b = bars[str(row.dt.date())]
        changed = prior is not None and not np.isclose(row.factor, prior, rtol=1e-10, atol=1e-12)
        b.update(factor=float(row.factor), factor_changed=bool(changed))
        if changed:
            b["price_reference_valid"] = False
        prior = row.factor
    return bars


def period_cohorts(calendar):
    result = {}
    for period, (start, end) in PERIODS.items():
        days = [d for d in calendar if d <= end]
        if len(days) <= BUFFER:
            raise ValueError("insufficient calendar for split")
        result[period] = dict(start=start, end=days[-1], last_signal=days[-BUFFER-1], calendar=days, buffer=BUFFER)
    result["full"] = dict(start=PERIODS["train"][0], end=result["test"]["end"],
                          last_signal=result["test"]["last_signal"], calendar=result["test"]["calendar"], buffer=BUFFER)
    return result


def load_dataset(out, max_stocks=None):
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cfg["data_provider"]["local_cache_enabled"] = False
    stocks = pd.read_csv(ROOT / "data/stock_list.csv", dtype=str)
    stocks = stocks[stocks.apply(lambda r: in_scope(r.code, r["name"]), axis=1)]
    if max_stocks:
        stocks = stocks.head(max_stocks)
    setups, stocks_bars, coverage = [], {}, []
    digest = hashlib.sha256((ROOT / "data/stock_list.csv").read_bytes())
    with ConfigLoader.runtime_snapshot(cfg):
        provider = TdxProvider()
        provider._cache_enabled = False
        index = provider._normalize_daily_df(provider._reader_daily("sz399300"), "399300.SZ")
        if index.empty:
            raise ValueError("local CSI300 index calendar missing")
        index = index[(index.dt >= "2024-09-01") & (index.dt <= PERIODS["test"][1])].reset_index(drop=True)
        calendar = [str(d.date()) for d in index.dt]
        if len(calendar) < 200:
            raise ValueError("index calendar too short")
        cohorts = period_cohorts(calendar)
        benchmark = {str(r.dt.date()): dict(open=float(r.open), close=float(r.close)) for r in index.itertuples()}
        digest.update(pd.util.hash_pandas_object(index, index=False).values.tobytes())
        for j, stock in enumerate(stocks.itertuples()):
            cov = dict(code=stock.code, name=stock.name, status="ok")
            try:
                day_path, _ = provider._symbol_file_hints(stock.code)
                if not Path(day_path).is_file():
                    raise ValueError("local daily file missing; no network fallback")
                raw = provider._normalize_daily_df(provider._reader_daily(provider._raw_symbol(stock.code)), stock.code)
                raw = raw[(raw.dt >= "2024-09-01") & (raw.dt <= PERIODS["test"][1])].reset_index(drop=True)
                assert_raw_prices(provider, "tdx", raw, cfg)
                if len(raw) < 121:
                    raise ValueError("less than 121 historical bars")
                key = f"{stock.code[-2:].lower()}_{stock.code[:6]}"
                factor_path = ROOT / f"data/history/cache/adjustment_factors_v1_{key}.json"
                factors = _read_cache(factor_path, key.replace("_", "."), raw.dt.max(), 21600)
                if factors is None:
                    raise ValueError("verified factor cache missing; no raw-price fallback")
                work = features(raw, factors)
                digest.update(stock.code.encode())
                digest.update(pd.util.hash_pandas_object(raw, index=False).values.tobytes())
                digest.update(factor_path.read_bytes())
                work["date"], work["period"] = work.dt.dt.strftime("%Y-%m-%d"), ""
                full = cohorts["full"]
                work.loc[work.date.between(full["start"],full["last_signal"]),"period"] = "full_only"
                for per, cohort in cohorts.items():
                    if per != "full":
                        work.loc[work.date.between(cohort["start"], cohort["last_signal"]), "period"] = per
                possible = (work.history_ready & work.high_pct.gt(6. + 1e-10) & work.close_pct.gt(1e-10)
                            & work.close_pct.lt(6. - 1e-10) & work.period.ne(""))
                columns = ["date", "period", "high_pct", "close_pct", "fade_pct", "range_pos20", "range_pos60",
                           "drawdown60_pct", "near_low20_pct", "peak_age60", "below_ma20_days", "ma5_turn", "history_ready"]
                selected = work.loc[possible, columns].copy()
                selected["code"], selected["name"] = stock.code, stock.name
                setups.append(selected)
                if not selected.empty:
                    stocks_bars[stock.code] = execution_bars(work)
                cov.update(bars=len(raw), first_date=str(raw.dt.min().date()), last_date=str(raw.dt.max().date()), candidate_events=len(selected),
                           unlabelled_default_tail=int((signal_mask(work, DEFAULT) & work.date.between(PERIODS["train"][0], calendar[-1]) & work.period.eq("")).sum()))
            except Exception as exc:
                cov.update(status="error", error=f"{type(exc).__name__}: {exc}")
            coverage.append(cov)
            if (j + 1) % 200 == 0 or j + 1 == len(stocks):
                print(json.dumps(dict(loaded=j+1, total=len(stocks), errors=sum(c["status"] != "ok" for c in coverage)), ensure_ascii=False), flush=True)
    pd.DataFrame(coverage).to_csv(out / "data-coverage.csv", index=False, encoding="utf-8-sig")
    # Short histories are explicitly skipped; unexpected missing/invalid inputs
    # fail closed rather than permitting a selected-data success claim.
    unexpected = [c for c in coverage if c["status"]=="error" and c.get("error")!="ValueError: less than 121 historical bars"]
    if unexpected:
        raise ValueError(f"{len(unexpected)} stocks have unverified/invalid input; inspect data-coverage.csv")
    if not setups:
        raise ValueError("no usable stocks")
    return pd.concat(setups, ignore_index=True), stocks_bars, cohorts, benchmark, cfg, digest.hexdigest(), coverage


class Evaluator:
    def __init__(self, setups, bars, cohorts, benchmark):
        self.setups, self.bars, self.cohorts, self.benchmark = setups, bars, cohorts, benchmark
        self.entries, self.outcomes = {}, {}

    def outcome(self, setup, model, profile, hold):
        code, day, period = setup["code"], setup["date"], setup["period"]
        key = (code, day, model, period)
        calendar = self.cohorts[period]["calendar"]
        if key not in self.entries:
            self.entries[key] = find_entry(code, day, model, self.bars[code], calendar, CAPITAL, threshold=3.)
        entry = self.entries[key]
        if entry["status"] != "entry_available":
            return dict(net=0., excess=0., entered=False, status=entry["status"], entry_date=None, exit_date=None,
                        model=model, holding_days=0)
        trial_key = (*key, json.dumps(profile, sort_keys=True), hold)
        if trial_key not in self.outcomes:
            trial = run_trial(code, model, entry, profile, self.bars[code], calendar, CAPITAL, max_hold=hold)
            if trial["cohort_return_pct"] is None or trial["status"]=="open_at_data_end":
                result = dict(net=None, excess=None, entered=True, status=trial["status"], entry_date=entry["entry_date"],
                              exit_date=None, model=model, holding_days=trial["holding_days"])
            else:
                net = float(trial["equity_return_pct"] if trial["status"] == "closed"
                            else trial["cohort_return_pct"] * trial["invested"] / CAPITAL)
                exit_day = trial["exit_date"] or calendar[-1]
                entry_base = self.benchmark[entry["entry_date"]]["close" if model=="close" else "open"]
                benchmark_return = (self.benchmark[exit_day]["close"] / entry_base - 1) * 100
                result = dict(net=net, excess=net-benchmark_return, entered=True, status=trial["status"],
                    entry_date=entry["entry_date"], exit_date=trial["exit_date"], model=model, holding_days=trial["holding_days"],
                    quantity=trial["quantity"], entry_price=trial["entry_price"], exit_price=trial["exit_price"],
                    realized_pnl=trial["realized_pnl"], delayed_sell_days=trial["delayed_sell_days"],
                    ambiguous_both_hit_days=trial["ambiguous_both_hit_days"])
            self.outcomes[trial_key] = result
        return self.outcomes[trial_key]

    def evaluate(self, mask, period, model, profile=FIXED_HOLD, hold=10, details=False):
        selected = self.setups.loc[mask & (self.setups.date.between(self.cohorts["full"]["start"],self.cohorts["full"]["last_signal"])
                                           if period=="full" else self.setups.period.eq(period))].copy()
        selected["period"] = period
        rows = [dict(code=s["code"], name=s["name"], signal_date=s["date"], **self.outcome(s, model, profile, hold))
                for s in selected.to_dict("records")]
        metric = summarize(rows)
        return (metric, rows) if details else metric


def summarize(rows, interval=False):
    valid = [r for r in rows if r["net"] is not None]
    fills = [r for r in valid if r["entered"]]
    dates = pd.DataFrame([dict(date=r["signal_date"], excess=r["excess"]) for r in valid])
    dated = dates.groupby("date").excess.mean() if not dates.empty else pd.Series(dtype=float)
    score = None if dated.empty else float(dated.mean() - .25*np.sqrt(np.mean(np.minimum(dated.to_numpy(),0)**2)))
    result = dict(signals=len(rows), labels=len(valid), censored=len(rows)-len(valid), trades=len(fills),
        no_entry=len(valid)-len(fills), stocks=len({r["code"] for r in fills}), days=len(dated),
        mean_signal_net_pct=None if not valid else float(np.mean([r["net"] for r in valid])),
        mean_trade_net_pct=None if not fills else float(np.mean([r["net"] for r in fills])),
        win_rate_pct=None if not fills else float(np.mean([r["net"]>0 for r in fills])*100),
        event_mean_excess_pct=None if not valid else float(np.mean([r["excess"] for r in valid])),
        date_mean_excess_pct=None if dated.empty else float(dated.mean()), score=score,
        open_at_end=sum(r["status"]=="open_at_data_end" for r in rows), status_counts=dict(Counter(r["status"] for r in rows)))
    if interval and len(dated) >= 15:
        clusters = dated.groupby(dated.index.str[:7]).agg(["sum", "count"])
        samples = np.random.default_rng(42).integers(0,len(clusters),size=(2000,len(clusters)))
        means = clusters["sum"].to_numpy()[samples].sum(axis=1)/clusters["count"].to_numpy()[samples].sum(axis=1)
        result["month_cluster_ci95_pct"] = [float(v) for v in np.quantile(means,[.025,.975])]
    return result


def exit_grid():
    result = [dict(profile=FIXED_HOLD, hold=h, label=f"固定观察{h}日") for h in HORIZONS]
    result += [dict(profile=dict(kind="fixed",stop=s/100,take=t/100),hold=h,label=f"止损{s}%/止盈{t}%/{h}日")
               for s in (4,8) for t in (8,15,20) for h in (10,20)]
    result += [dict(profile=dict(kind="trailing",stop=s/100,arm=.10,tight=t/100),hold=h,label=f"滑动{s}%盈利10%后收紧{t}%/{h}日")
               for s in (4,8) for t in (3,5) for h in (10,20) if t<=s]
    return result


def report_table(rows):
    cols = ["variant","entry_name","signals","trades","mean_trade_net_pct","win_rate_pct","date_mean_excess_pct","censored"]
    def fmt(v):
        return "—" if v is None else f"{v:.3f}" if isinstance(v,float) else str(v)
    return "| "+" | ".join(cols)+" |\n|"+"|".join("---" for _ in cols)+"|\n"+"\n".join("| "+" | ".join(fmt(r.get(k)) for k in cols)+" |" for r in rows)


def make_report(out, summary, comparisons):
    rows = [r for r in comparisons if r["period"]=="test" and (r["hold"]==10 or r["variant"]=="训练冻结调参")]
    text = f"""# 冲高回落 × 近期低位：4日内3%买入点对照

完成于2026-10-09。请求最近一年2025-10-09—2026-10-09，本地完整日K实际截至{summary['data_end']}。成功扫描{summary['stocks_ok']}只主板现名单非ST，失败{summary['stocks_failed']}只。

## 因子定义

信号1：最高涨幅=(H/REF(C,1)-1)×100；收盘涨幅=(C/REF(C,1)-1)×100。严格要求最高涨幅>A且A3<收盘涨幅<A2，默认7.2、4.8、1，等于边界不选。连续特征另输出冲高回落幅度=最高涨幅-收盘涨幅。

信号2是对截图的三种可检验解释，图片没有给出唯一公式：

- range：区间位置=(C-近N根最低L)/(近N根最高H-近N根最低L)≤Q。N=20/60，Q=0.2/0.35。
- pullback：收盘距此前60根最高H回撤≥D（8/15/25%）、收盘距近20根最低L≤B（10/18%）、此前最高点距今3—55根、信号前10根至少3根收盘低于MA20。默认D=8、B=18。
- turn：pullback再要求MA5近5根从非上升转为上升且当日仍上升。

因子只使用当日或以前的数据；此前60根高点不含信号日。使用核验缓存的BaoStock后复权因子按历史起始日归一，成交及现金用原价。截图箭头和未来走势未作为标签。

## 买入与退出

两个条件满足的日子为信号日。观察随后4个市场交易日，不含信号日；停牌或缺日线不延长窗口。观察日的涨幅相对该日前收，而非初始信号价。严格>3%按0.01元报价处理。

1. close：首个收盘涨幅>3%的观察日按该日收盘价模拟。依赖最终收盘数据，是理论对照。
2. intraday：4日内预置前收×1.03以上第一档报价的触发单；最高价达到就触发，跳空按开盘价。其最终收盘可以≤3%，所以会比close组增加一些交易。
3. next_open：收盘>3%确认后次日开盘成交。开盘不可成交则继续原4日观察；第4日确认允许第5日开盘。

涨停开盘、同收盘价组收盘涨停、滑点跨涨跌停和无量时拒绝。项目组件处理100股整手、费用预留和T+1，买入当天不卖出。基本组买入后第5/10/20个市场交易日收盘退出；跌停或停牌则延迟下一可成交开盘。末端未平仓记收益缺失并独立显示数量，不参与选参。止盈止损同日触发：开盘跳空优先，其余止损优先；滑动止损只用此前高点。

费用：佣金万2.5、最低5元，双边滑点0.1%，卖出印花税0.05%，过户费用双边0.002%保守常数。印花税规则依据[税务总局公告](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html)。普通主板10%价带由前收推算，缺历史逐日ST与官方价带；除权当天拒绝未知参考价成交，持仓跨复权因子变化收益记缺失。

每次信号是独立10万元试验，收益以初始资本为分母，未成交记0，不能累计为组合年化。沪深300基准使用对应日期毛收益；盘中入场仅以同日开盘近似指数入场，延迟开盘退出以同日收盘近似指数退出，盘中超额收益不精确。

## 调参与时间隔离

训练2025-10-09—2026-04-30，冻结测试2026-05-01—{summary['data_end']}。每段末尾保留30个市场交易日用于4日观察、1日确认后执行、20日持仓与5日退出延迟缓冲。实际可评价信号末日：训练{summary['cohorts']['train']['last_signal']}，测试{summary['cohorts']['test']['last_signal']}；近期未满持仓窗口的信号不用于结论。

阶段一：每种入场独立比较459组因子，固定10日观察退出。A=6/7.2/8.5，A2=3.5/4.8/6，A3=0/1/2，与17种低位解释组合。至少30次有效成交、15个有效信号日。

阶段二：只对各入场的训练第一名因子，搜索{summary['exit_candidates']}组退出（固定观察、固定止盈止损、滑动止损及10/20日上限），仍仅使用训练。并非因子与退出的全联合搜索。目标为日期等权超额均值减0.25×下行均方根。先保存冻结参数再评估测试；测试不用于重新选择参数或入场方式。

```json
{json.dumps(summary['frozen'],ensure_ascii=False,indent=2)}
```

## 冻结测试对照

{report_table(rows)}

收益单位均为%。默认低位为pullback。完整5/10/20日见comparisons.csv。paired-entry.csv只比较同一默认信号两种方式都成交且收益有效的配对。common-cohort.csv用三种方式、三个期限均有效的共同信号，检查缺失样本影响。按月份聚类的95%区间仅为描述统计，月份很少且未校正多重搜索。

## 结论与边界

**{summary['research_status']}**。各入场分别检查训练和测试分数均为正、测试至少30笔及15日，不以测试第一名宣布最佳入场点。

现名单来自2026-08-05股票快照，缺退市和历史ST变更，存在生存者与状态偏差。指数日历未独立逐日核验；普通主板10%价带和盘口成交是假设。公司行动缺现金/股数核算，收益缺失可能偏向某些股票/月份。部分2026行情曾用于项目其他研究；本轮保证训练搜索隔离，不能声称全新外部盲测。

实际调用TdxProvider、OrderMatcher、ExecutionEngine和AccountLedger，完成{summary['execution_trials']}个有入场的交易试验与{summary['entry_searches']}次缓存入场搜索。数据/代码指纹、保护文件哈希、覆盖见summary.json和data-coverage.csv。项目配置、私有配置和股票名单保持原哈希。

## 复现

```powershell
python scripts/research_low_reversal.py --out reports/low-reversal-factor-2026-10-09
```

因子实现src/strategies/low_reversal_factor.py，研究入口scripts/research_low_reversal.py。本轮仅新增独立文件，保留既有研究及运行配置。
"""
    (out/"因子研究与回测.md").write_text(text,encoding="utf-8")
    payload = json.dumps(comparisons,ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    doc = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>4日内3%买入点因子研究</title><style>body{background:#f4f6f8;color:#1d2939;font:15px/1.6 'Microsoft YaHei',sans-serif;margin:0}main{max-width:1300px;margin:auto;padding:28px}h1{font-size:28px}select{padding:8px;margin:8px}section{background:white;padding:20px;margin:18px 0;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;border-bottom:1px solid #e4e7ec;text-align:left}a{color:#175cd3}.scroll{overflow:auto}pre{white-space:pre-wrap;font-size:13px}.muted{color:#667085}</style><main><h1>冲高回落 × 低位：4日内3%入场</h1><p>__STATUS__ · 本地日K至__END__ · __STOCKS__只主板现名单非ST</p><section><b>前置信号</b>：最高涨幅&gt;A，A3&lt;收盘涨幅&lt;A2，默认7.2 / 4.8 / 1，加近期低位。<br><b>后续4日</b>：收盘&gt;3%按收盘买；最高涨幅&gt;3%按突破价买；补测收盘确认后次日开盘。<p class="muted">最近一年；训练至2026-04-30，之后冻结测试。每次信号为独立10万元试验，收益不是组合年化。459组因子和退出搜索仅用训练排名。</p></section><section><h2>对照结果</h2><label>时段<select id="period"><option value="test">冻结测试</option><option value="full">最近一年整体</option><option value="train">训练</option></select></label><label>观察期限<select id="hold"><option value="10">10日固定退出</option><option value="5">5日固定退出</option><option value="20">20日固定退出</option><option value="optimized">训练冻结调参</option></select></label><div class="scroll"><table><thead><tr><th>版本</th><th>入场</th><th>信号</th><th>成交</th><th>缺失</th><th>平均交易净收益%</th><th>胜率%</th><th>日期等权超额%</th></tr></thead><tbody id="rows"></tbody></table></div><p class="muted">同收盘价组依赖最终收盘数据，属理论对照。缺失含持仓跨除权；未成交记0。沪深300基准盘中时点仅作近似。</p></section><section><h2>冻结参数</h2><pre>__PARAMS__</pre><p>现名单缺历史ST和退市，有生存者偏差。价带按普通主板假设，日K不能确认盘口成交；每段末30个交易日留给观察、持仓与延迟退出。</p><p><a href="因子研究与回测.md">完整研究</a> · <a href="comparisons.csv">全部对照</a> · <a href="training-grid.csv">训练因子参数</a> · <a href="training-exit-grid.csv">退出参数</a> · <a href="frozen-parameters.json">冻结参数</a> · <a href="test-trades.csv">测试交易</a> · <a href="paired-entry.csv">同信号配对</a> · <a href="summary.json">运行记录</a></p></section></main><script id="data" type="application/json">__DATA__</script><script>const data=JSON.parse(document.getElementById('data').textContent);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=v=>v==null?'—':typeof v==='number'?v.toFixed(3):esc(v);function render(){const p=document.getElementById('period').value,h=document.getElementById('hold').value;document.getElementById('rows').innerHTML=data.filter(r=>r.period===p&&(h==='optimized'?r.variant==='训练冻结调参':r.variant!=='训练冻结调参'&&r.hold===Number(h))).map(r=>'<tr>'+[r.variant,r.entry_name,r.signals,r.trades,r.censored,r.mean_trade_net_pct,r.win_rate_pct,r.date_mean_excess_pct].map(v=>'<td>'+fmt(v)+'</td>').join('')+'</tr>').join('')}document.getElementById('period').onchange=render;document.getElementById('hold').onchange=render;render();</script></html>'''
    doc = doc.replace("__STATUS__",html.escape(summary["research_status"])).replace("__END__",summary["data_end"]).replace("__STOCKS__",str(summary["stocks_ok"])).replace("__PARAMS__",html.escape(json.dumps(summary["frozen"],ensure_ascii=False,indent=2))).replace("__DATA__",payload)
    (out/"因子回测报告.html").write_text(doc,encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"reports/low-reversal-factor-2026-10-09")
    parser.add_argument("--max-stocks",type=int,help="bounded smoke test only")
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    protected = [ROOT/p for p in ("config.json","config.private.json","data/stock_list.csv")]
    before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.exists()}
    started, grid, exits = time.monotonic(), parameter_grid(), exit_grid()
    write_json(out/"predeclared-design.json",dict(version=VERSION,periods=PERIODS,horizons=HORIZONS,observation_sessions=4,trigger_pct=3.,
        strict_threshold=True,entry_models=MODELS,factor_candidates=len(grid),exit_candidates=len(exits),costs=STUDY_COSTS,capital=CAPITAL,
        objective="training date_mean_excess - .25*date_downside_RMS",min_trades=30,min_days=15,purge_sessions=BUFFER))
    panel,bars,cohorts,benchmark,cfg,fingerprint,coverage = load_dataset(out,args.max_stocks)
    evaluator = Evaluator(panel,bars,cohorts,benchmark)
    minimum_trades,minimum_days = (5,5) if args.max_stocks else (30,15)
    train_grid,exit_rows,frozen = [],[],{}
    with ConfigLoader.runtime_snapshot(cfg), parameter_overrides(STUDY_COSTS):
        for model in MODELS:
            ranks=[]
            for i,params in enumerate(grid):
                m=evaluator.evaluate(signal_mask(panel,params),"train",model)
                # Formula-only controls must not replace the requested low condition.
                eligible=params["low_mode"]!="none" and m["trades"]>=minimum_trades and m["days"]>=minimum_days
                train_grid.append(dict(trial=i,model=model,eligible=eligible,**params,**m))
                if eligible: ranks.append((m["score"],i,params,m))
                if (i+1)%100==0:
                    print(json.dumps(dict(stage="factor_search",model=model,completed=i+1,total=len(grid),trials=len(evaluator.outcomes))),flush=True)
            if not ranks: raise ValueError(f"no training factor candidate meets sample gate: {model}")
            best=max(ranks,key=lambda r:(r[0],-r[1])); params=best[2]; mask=signal_mask(panel,params)
            exit_rank=[]
            for i,rule in enumerate(exits):
                m=evaluator.evaluate(mask,"train",model,rule["profile"],rule["hold"])
                eligible=m["trades"]>=minimum_trades and m["days"]>=minimum_days
                exit_rows.append(dict(model=model,trial=i,label=rule["label"],hold=rule["hold"],profile=json.dumps(rule["profile"]),eligible=eligible,**m))
                if eligible: exit_rank.append((m["score"],-i,rule,m))
            winner=max(exit_rank,key=lambda r:(r[0],r[1]))
            frozen[model]=dict(factor=params,exit=winner[2],training=winner[3])
        # Persist all entry-specific parameters before examining the test outcomes.
        write_json(out/"frozen-parameters.json",dict(selection="training_only_before_test",configs=frozen,dataset_fingerprint=fingerprint))
        pd.DataFrame(train_grid).to_csv(out/"training-grid.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(exit_rows).to_csv(out/"training-exit-grid.csv",index=False,encoding="utf-8-sig")
        comparisons,trials,default_trials,frozen_test=[],[],{},{}
        variants=[("信号1单独",dict(low_mode="none")),("默认+区间低位",dict(low_mode="range",window=60,range_max=.35)),
                  ("默认+回调低位",DEFAULT),("默认+初拐",dict(low_mode="turn"))]
        for label,params in variants:
            mask=signal_mask(panel,params)
            for per in ("full","train","test"):
                for model in MODELS:
                    for hold in HORIZONS:
                        m,rows=evaluator.evaluate(mask,per,model,hold=hold,details=True)
                        if per in ("full","test"): m=summarize(rows,interval=True)
                        comparisons.append(dict(variant=label,period=per,model=model,entry_name=NAMES[model],hold=hold,**m))
                        if per=="test": trials += [dict(variant=label,hold=hold,**r) for r in rows]
                        if label=="默认+回调低位" and per=="test": default_trials[(model,hold)]=rows
        for model,setting in frozen.items():
            mask=signal_mask(panel,setting["factor"])
            for per in PERIODS:
                rule=setting["exit"]
                m,rows=evaluator.evaluate(mask,per,model,rule["profile"],rule["hold"],details=True)
                if per=="test":
                    m=summarize(rows,interval=True); frozen_test[model]=m
                    trials += [dict(variant="训练冻结调参",hold=rule["hold"],**r) for r in rows]
                comparisons.append(dict(variant="训练冻结调参",period=per,model=model,entry_name=NAMES[model],hold=rule["hold"],**m))
    paired,common=[],[]
    for hold in HORIZONS:
        maps={model:{(r["code"],r["signal_date"]):r for r in default_trials[(model,hold)]} for model in MODELS}
        for key,c in maps["close"].items():
            p=maps["intraday"][key]
            if c["entered"] and p["entered"] and c["net"] is not None and p["net"] is not None:
                paired.append(dict(code=key[0],signal_date=key[1],hold=hold,close_net_pct=c["net"],intraday_net_pct=p["net"],
                              intraday_minus_close_pct=p["net"]-c["net"],close_entry_date=c["entry_date"],intraday_entry_date=p["entry_date"]))
    valid_keys=set.intersection(*[{(r["code"],r["signal_date"]) for r in rows if r["net"] is not None} for rows in default_trials.values()])
    for (model,hold),rows in default_trials.items():
        selected=[r for r in rows if (r["code"],r["signal_date"]) in valid_keys]
        common.append(dict(model=model,hold=hold,**summarize(selected)))
    for name,rows in [("comparisons",comparisons),("test-trades",trials),("paired-entry",paired),("common-cohort",common)]:
        pd.DataFrame(rows).to_csv(out/f"{name}.csv",index=False,encoding="utf-8-sig")
    panel.to_csv(out/"factor-events.csv",index=False,encoding="utf-8-sig")
    if not all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in before.items()):
        raise RuntimeError("protected project files changed")
    gates={model:(frozen[model]["training"]["score"]>0 and m["score"] is not None and m["score"]>0 and m["trades"]>=30 and m["days"]>=15)
           for model,m in frozen_test.items()}
    summary=dict(version=VERSION,periods=PERIODS,cohorts={k:{x:v for x,v in c.items() if x!="calendar"} for k,c in cohorts.items()},
        data_end=cohorts["test"]["end"],stocks_ok=sum(c["status"]=="ok" for c in coverage),stocks_failed=sum(c["status"]!="ok" for c in coverage),
        candidate_events=len(panel),factor_candidates=len(grid),exit_candidates=len(exits),frozen=frozen,frozen_test=frozen_test,research_gates=gates,
        research_status="部分入场方式支持继续验证" if any(gates.values()) and not args.max_stocks else "尚不支持稳定有效的结论",
        execution_trials=len(evaluator.outcomes),entry_searches=len(evaluator.entries),costs=STUDY_COSTS,
        unlabelled_default_tail=sum(c.get("unlabelled_default_tail",0) for c in coverage),dataset_fingerprint=fingerprint,
        code_fingerprint=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/"src/strategies/low_reversal_factor.py").read_bytes()).hexdigest(),
        protected_files_unchanged=True,elapsed_seconds=round(time.monotonic()-started,1),max_stocks=args.max_stocks)
    write_json(out/"summary.json",summary); make_report(out,summary,comparisons)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
