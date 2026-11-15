"""Predeclared cash-account search with fixed hypotheses and weekly turnover cap."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from scripts.lowfreq_signals import SPECS, candidate_signals, prepare_frame

OUT = ROOT / "reports/lowfreq-50k-2026-10-10"
UPSTREAM = ROOT / "reports/obv-alpha-2026-10-10"
START, TRAIN_END, TEST_START, END = "2023-10-01", "2025-09-30", "2025-10-01", "2026-09-30"
NAMES = {"breakout": "趋势突破", "pullback": "强势回调恢复", "reversal": "短期抛压反转",
         "compression": "盘整后突破", "rotation": "周度相对强势"}
LOGIC = {
    "breakout": "假设新信息和新增需求逐步反映在价格上，赚突破后持续买盘带来的延续行情。假突破和拥挤反转会造成亏损。",
    "pullback": "假设中期上涨需求未消失，短期获利卖盘使价格暂时回落，赚恢复趋势时的价格折让。坏消息导致的趋势破坏会使假设失效。",
    "reversal": "假设部分急跌来自暂时流动性需求，赚抛压消退后的恢复。永久负面信息和连续跌停会使假设失效，中国市场并非所有短期滞后都有反转。",
    "compression": "假设窄幅盘整后价格突破代表供求重新失衡，赚新趋势的延续。窄幅和量增只是代理，不能证明吸筹，假突破会使假设失效。",
    "rotation": "假设相对强势包含尚未完全反映的信息和持续需求，每周从可交易股票中选择强者。趋势切换、拥挤和开盘涨停会使假设失效。",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, np.integer)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def design():
    return dict(version="lowfreq-50k-fixed-hypotheses-v2", created_before_results=True,
                start=START, train_end=TRAIN_END, test_start=TEST_START, end=END,
                specs=SPECS, capital=50000, leverage=False, maximum_positions=2,
                position_budget=.45, reserve=.10, weekly_completed_operations=4,
                operations_definition="each completed buy or sell counts one; refused orders separately recorded",
                same_day_sale_proceeds_reused=False, top_candidates=20, liquidity_amount20=50000000,
                market_filter="signal-date CSI300 close above its MA60", entry="next market open, no postponed entry",
                exit="completed-close stop/MA/take/trail then next open; max_hold preplanned market open",
                ranking="highest signal score, code tie-break; daily top20 before opening tradability checks",
                commission=.00025, minimum_commission=5, sell_stamp=.0005, transfer_per_side=.00001,
                slippage_per_side=.001, stress_slippage=.003, delay_stress_market_days=1,
                corporate_action_policy="only formally announced events visible by prior close block entry/force next-open exit before exdate; unknown action halts whole account",
                selection="max training net CAGR among complete accounts; minimum20 closed trades; no recent-year selection",
                target_gate="selected frozen strategy complete full3years AND recentyear net CAGR >=100%; week cap respected; stress results separately assessed",
                candidate_count=len(SPECS), max_dividend_fetch_pairs=1200, max_resolution_rounds=32,
                dividend_coverage="query operated events for every provisional bought stock, each holding calendar year plus next year; replay until no uncovered buys; failed coverage makes entire account invalid",
                limitations=["current survivor universe lacks historical ST/delisting state", "historical recent year already viewed in other research; no untouched prospective sample",
                             "OHLCV proxies do not identify real investor flows or news", "daily bars cannot prove queues or impact", "corporate-action accounting absent; avoidance uses historical formal announcement dates, without archived announcement vintages"])


def compile_hashes():
    return {str(p): digest(p) for p in (Path(__file__), ROOT / "scripts/lowfreq_signals.py")}


def archive_run_sources(out, label):
    paths = [Path(__file__), ROOT / "scripts/lowfreq_signals.py",
             ROOT / "scripts/lowfreq_portfolio.py", ROOT / "scripts/lowfreq_dividend_data.py",
             ROOT / "src/core/execution_engine.py", ROOT / "src/services/order_matcher.py",
             ROOT / "src/services/account_ledger.py", ROOT / "src/services/risk_audit_log.py",
             ROOT / "src/utils/runtime_params.py", ROOT / "src/utils/constants.py",
             ROOT / "src/utils/config_loader.py"]
    snapshot = out / "executed-source" / label
    hashes = {}
    for source in paths:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[str(source)] = digest(target)
    write_json(snapshot / "manifest.json", dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), files=hashes))


def load_upstream(out):
    manifest = json.loads((UPSTREAM / "input-manifest.json").read_text(encoding="utf-8"))
    assert digest(UPSTREAM / "dataset.pkl") == manifest["dataset_sha256"], "upstream dataset changed"
    snapshot = UPSTREAM / "executed-source"
    for path, expected in manifest["generator_sha256"].items():
        archived = snapshot / Path(path).relative_to(ROOT)
        assert digest(archived) == expected, "upstream executed code changed"
    for path, expected in manifest["files"].items():
        assert digest(path) == expected, f"upstream input changed: {Path(path).name}"
    # This is our own checksummed artifact, not an arbitrary external pickle.
    with (UPSTREAM / "dataset.pkl").open("rb") as stream:
        original = pickle.load(stream)
    assert original["fingerprint"] == manifest["fingerprint"]
    write_json(out / "upstream-input-manifest.json", manifest)
    return original, manifest


def compile_matrix(out):
    original, upstream = load_upstream(out)
    dates = original["calendar"].strftime("%Y-%m-%d").tolist()
    codes = sorted(original["raw_frames"])
    rows, columns = len(dates), len(codes)
    data = dict(dates=dates, codes=codes)
    for field in ("open", "close", "vol", "up_limit", "down_limit"):
        data[field] = np.full((rows, columns), np.nan, dtype=np.float64)
    for field in ("factor_changed", "open_valid", "day_valid"):
        data[field] = np.zeros((rows, columns), dtype=bool)
    for field in ("entry", "exit"):
        data[field] = {spec["id"]: np.zeros((rows, columns), dtype=bool) for spec in SPECS}
    data["score"] = {spec["id"]: np.full((rows, columns), np.nan, dtype=np.float64) for spec in SPECS}
    lookup = {day: i for i, day in enumerate(dates)}
    index = original["index"].set_index(original["index"].dt.dt.strftime("%Y-%m-%d"))
    aligned = index.reindex(dates)
    data["market_ok"] = aligned.close.gt(aligned.close.rolling(60).mean()).to_numpy()
    weeks = pd.DatetimeIndex(dates).isocalendar()
    keys = list(zip(weeks.year, weeks.week))
    data["week_last"] = np.array([i == len(dates)-1 or keys[i] != keys[i+1] for i in range(len(dates))])
    started = time.monotonic()
    for j, code in enumerate(codes):
        raw = original["raw_frames"][code]
        prepared = prepare_frame(raw)
        positions = np.array([lookup[str(dt.date())] for dt in prepared.dt], dtype=int)
        for field in ("open", "close", "vol"):
            data[field][positions, j] = prepared[field].to_numpy(float)
        prior_factor = prepared.adjustment_factor.shift(1)
        changed = ~np.isclose(prepared.adjustment_factor, prior_factor, rtol=1e-12, atol=1e-12)
        data["factor_changed"][positions, j] = changed
        reference = prepared.close.shift(1) * prior_factor / prepared.adjustment_factor
        cents = np.floor(reference.to_numpy() * 100 + .5)
        upper = np.floor(cents * 1.10 + .5) / 100
        lower = np.floor(cents * .90 + .5) / 100
        data["up_limit"][positions, j], data["down_limit"][positions, j] = upper, lower
        data["open_valid"][positions, j] = (prepared.open.ge(lower-.0100001) & prepared.open.le(upper+.0100001)).to_numpy()
        data["day_valid"][positions, j] = (prepared.low.ge(lower-.0100001) & prepared.high.le(upper+.0100001)).to_numpy()
        for spec in SPECS:
            signals = candidate_signals(prepared, spec)
            for field, source in (("entry", "entry"), ("exit", "exit"), ("score", "score")):
                data[field][spec["id"]][positions, j] = signals[source].to_numpy()
        if (j+1) % 300 == 0 or j+1 == len(codes):
            print(json.dumps(dict(stage="compile", stocks=j+1, total=len(codes), elapsed=round(time.monotonic()-started, 1))), flush=True)
    with (out / "matrix.pkl").open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(out / "matrix-manifest.json", dict(upstream_fingerprint=upstream["fingerprint"],
               upstream_dataset_sha256=upstream["dataset_sha256"], matrix_sha256=digest(out / "matrix.pkl"),
               generator_sha256=compile_hashes(), stocks=len(codes), sessions=len(dates)))
    return data


def reuse_matrix(out):
    manifest = json.loads((out / "matrix-manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator_sha256"] == compile_hashes(), "matrix-generating code changed"
    assert manifest["matrix_sha256"] == digest(out / "matrix.pkl"), "matrix checksum mismatch"
    with (out / "matrix.pkl").open("rb") as stream:
        return pickle.load(stream)


def save_run(out, key, result):
    target = out / "accounts" / key
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "summary.json", result["summary"])
    write_json(target / "blockers.json", result.get("blockers", []))
    for table in ("nav", "orders", "trades", "skips"):
        result[table].to_csv(target / f"{table}.csv", index=False, encoding="utf-8-sig")


def resolved_runs(data, specs, start, end, out, label, slippage=.001, entry_delay=0):
    from scripts.lowfreq_dividend_data import fetch_pairs, load_events, cache_path, _read_valid_cache
    from scripts.lowfreq_portfolio import simulate_portfolio
    cache = out / "dividend-cache"
    fetched = set()
    logs = []
    final = {}
    for round_no in range(32):
        events = load_events(cache)
        needs = set()
        for spec in specs:
            result = simulate_portfolio(data, spec, start, end, events=events, slippage=slippage, entry_delay=entry_delay)
            required = set()
            intervals = [(row.code, row.entry_date, row.exit_date) for row in result["trades"].itertuples()]
            intervals += [(pos["code"], pos["entry_date"], result["summary"]["actual_end"])
                          for pos in result["summary"]["open_positions"]]
            for code, entered, exited in intervals:
                for year in range(int(str(entered)[:4]), int(str(exited)[:4]) + 2):
                    required.add((code, year))
            missing = {pair for pair in required if _read_valid_cache(cache_path(cache, *pair), *pair) is None}
            if missing:
                result["summary"].update(valid=False, completed=False, cagr=np.nan, year_performance=np.nan)
                result["blockers"].extend(dict(code=code, year=year, date=str(start), reason="corporate_action_data_missing")
                                           for code, year in sorted(missing))
            final[spec["id"]] = result
            for blocker in result.get("blockers", []):
                if "corporate" in str(blocker.get("reason", "")):
                    pair = (blocker["code"], int(blocker.get("year", str(blocker["date"])[:4])))
                    if pair not in fetched:
                        needs.add(pair)
        print(json.dumps(dict(stage=label, round=round_no+1, completed=sum(bool(r["summary"].get("valid")) for r in final.values()),
                              candidates=len(specs), event_queries_needed=len(needs))), flush=True)
        if not needs:
            break
        if len(fetched) + len(needs) > 1200:
            logs.append(dict(error="predeclared event-query cap reached", pending=sorted(needs)))
            break
        ordered = sorted(needs)
        for offset in range(0, len(ordered), 25):
            batch = ordered[offset:offset+25]
            statuses = fetch_pairs(batch, cache, timeout=60)
            logs.append(dict(round=round_no+1, pairs=batch, statuses=statuses))
            fetched.update(batch)
            print(json.dumps(dict(stage="event_source", batch=len(batch), fetched=len(fetched)), ensure_ascii=False), flush=True)
    write_json(out / f"{label}-event-resolution.json", logs)
    for spec_id, result in final.items():
        save_run(out, f"{label}-{spec_id}", result)
    return final


def summary_table(results):
    return pd.DataFrame([{**result["summary"], "id": key} for key, result in results.items()])


def readable_table(frame):
    fields = ["id", "valid", "actual_end", "end_nav", "total_return", "cagr",
              "max_drawdown", "trades_closed", "buy_count", "sell_count", "max_weekly_operations"]
    result = frame[[c for c in fields if c in frame]].copy()
    for column in ("total_return", "cagr", "max_drawdown"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").map(lambda value: f"{value:.2%}" if pd.notna(value) else "未完整完成")
    return result.rename(columns={"id":"候选/验证", "valid":"完整", "actual_end":"运行至", "end_nav":"期末净值/元",
                                  "total_return":"累计收益", "cagr":"年化收益", "max_drawdown":"最大回撤", "trades_closed":"完成交易",
                                  "buy_count":"买入次数", "sell_count":"卖出次数", "max_weekly_operations":"单周最多操作"})


def train(data, out):
    archive_run_sources(out, "train")
    results = resolved_runs(data, SPECS, START, TRAIN_END, out, "train")
    table = summary_table(results)
    table.to_csv(out / "training-candidates.csv", index=False, encoding="utf-8-sig")
    eligible = table[table.valid.eq(True) & table.trades_closed.ge(20) & table.cagr.notna()]
    selected = eligible.sort_values(["cagr", "id"], ascending=[False, True]).iloc[0].id if len(eligible) else None
    frozen = dict(selected_id=selected, selection_uses_recent_year=False, selection_rule=design()["selection"],
                  frozen_at_utc=datetime.now(timezone.utc).isoformat(), training_table_sha256=digest(out / "training-candidates.csv"),
                  selected_spec=next((s for s in SPECS if s["id"] == selected), None))
    write_json(out / "frozen-selection.json", frozen)
    return results, frozen


def annual_nav(nav):
    if nav.empty:
        return []
    frame = nav.copy()
    date_column = "date" if "date" in frame else "dt"
    frame["year"] = pd.to_datetime(frame[date_column]).dt.year
    nav_column = "nav" if "nav" in frame else "equity"
    previous = 50000.
    result = []
    for year, group in frame.groupby("year", sort=True):
        ending = float(group[nav_column].iloc[-1])
        result.append(dict(year=int(year), start_nav=previous, end_nav=ending, return_pct=(ending/previous-1)*100,
                           observations=len(group), first_day=str(group[date_column].iloc[0]), last_day=str(group[date_column].iloc[-1])))
        previous = ending
    return result


def evaluate(data, frozen, out):
    selected = frozen["selected_spec"]
    if selected is None:
        return {}
    archive_run_sources(out, "evaluate")
    checks = {}
    for label, start, slip, delay in (("recent", TEST_START, .001, 0), ("full", START, .001, 0),
                                     ("recent_cost_stress", TEST_START, .003, 0), ("recent_entry_delay", TEST_START, .001, 1)):
        result = resolved_runs(data, [selected], start, END, out, label, slippage=slip, entry_delay=delay)[selected["id"]]
        checks[label] = result
    table = summary_table(checks)
    table.to_csv(out / "selected-validation.csv", index=False, encoding="utf-8-sig")
    years = annual_nav(checks["full"]["nav"]) if checks["full"]["summary"]["valid"] else []
    pd.DataFrame(years).to_csv(out / "selected-annual.csv", index=False, encoding="utf-8-sig")
    return checks


def report(out, frozen, checks):
    training = pd.read_csv(out / "training-candidates.csv")
    selected = frozen["selected_spec"]
    gate = bool(selected and all(checks[k]["summary"]["valid"] and checks[k]["summary"]["cagr"] >= 1.0
                                and checks[k]["summary"]["max_weekly_operations"] <= 4 for k in ("full", "recent")))
    verdict = "历史数据达到100%门槛，尚不能承诺未来收益" if gate else "本轮没有找到经完整账户验证、达到年化100%门槛的策略"
    summaries = {key: value["summary"] for key, value in checks.items()}
    write_json(out / "study-summary.json", dict(verdict=verdict, target_gate_passed=gate, frozen_selection=frozen,
               validations=summaries, candidate_count=len(SPECS), limitations=design()["limitations"]))
    lines = ["# 5万元低频策略研究", "", f"**{verdict}。**", "",
             "每周最多4次已成交买卖，买卖各算一次；最多2只股票、每仓45%预算、保留10%现金，无杠杆。",
             "先在2023-10至2025-09的前两年选规则，写入冻结文件，再检验2025-10至2026-09；近期历史数据已用于其他研究，不能称为全新未见样本。", "",
             "## 先说明收益假设", ""]
    for family, name in NAMES.items():
        lines.append(f"- **{name}**：{LOGIC[family]}")
    lines += ["", "OHLCV只能构造代理，不能确认谁在买卖、是否有新信息。机制解释必须和回测证据分别判断。", "",
              "## 训练期10个固定候选", "", readable_table(training).to_markdown(index=False), ""]
    if selected:
        lines += [f"冻结候选：**{selected['id']} / {NAMES[selected['family']]}**。", "",
                  "```json", json.dumps(selected, ensure_ascii=False, indent=2), "```", "",
                  "## 冻结后的账户验证", "", readable_table(pd.read_csv(out / "selected-validation.csv")).to_markdown(index=False), "",
                  "## 连续账户年度收益", "", pd.read_csv(out / "selected-annual.csv").to_markdown(index=False) if (out/"selected-annual.csv").stat().st_size>5 else "连续账户未完整完成，年度收益不输出。", ""]
    lines += ["## 执行和证据范围", "",
              "收盘观察信号，下一市场日开盘买入；停止使用当天未来高低价决定开盘成交。止损、止盈、MA与追踪退出由已完成收盘观察，下一开盘执行，缺口损失可能超过止损比例。",
              "每仓费用内预算、100股整数手、T+1；涨停开盘不买、跌停开盘不卖；同日卖出资金不复用。未成交最多重试5个市场日，仍不能退出则整个账户测试不完整。",
              "正式实施公告在当时已可知后，避开将来的除权除息；未知公司行动或持仓原价参考异常终止账户，截断结果不能年化，也不能拼接其他股票收益。公告数据库没有历史版本归档，仍可能存在修订信息边界。",
              "期末净值按原价收盘估值，开放持仓和停牌陈旧报价单独保留；不模拟可保证的清仓成交。", "",
              "当前名单有生存者偏差及历史ST缺口，日K不能证明盘口成交，且没有真正未来样本。100%只是研究门槛，不是未来承诺。", "",
              "## 依据和反证", "",
              "[Jegadeesh-Titman中期动量研究](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1993.tb04702.x)主要检验美股3至12月的动量，不能直接证明A股20/60日突破或精确回调形态。",
              "[中国短期反转研究](https://doi.org/10.1016/j.frl.2022.103220)提示不同滞后日的动量/反转可能不同，不能把所有下跌一概视为流动性折让。",
              "[Zipline实际账本实现](https://github.com/stefan-jansen/zipline-reloaded/blob/main/src/zipline/finance/ledger.py)分别处理送转、应收权益和派息现金，支持本轮拒绝用单一复权因子假装现金入账的设计。", "",
              "账户逐日净值、全部买卖、完成交易和拒绝记录在 `accounts/`；原始公告数据在 `dividend-cache/`，输入和源码指纹在清单文件。"]
    (out / "研究结果.md").write_text("\n".join(lines), encoding="utf-8")
    # Detailed machine-friendly tables are linked; figures may be added separately.
    display = readable_table(training)
    validation = readable_table(pd.read_csv(out / "selected-validation.csv")) if selected else pd.DataFrame()
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5万元低频策略研究</title><style>body{{background:#f3f5f8;color:#172238;font:15px/1.8 system-ui,"Microsoft YaHei";margin:0}}main{{max-width:1350px;margin:auto;padding:24px}}section{{background:white;border-radius:12px;padding:24px;margin:18px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px}}td,th{{padding:8px;border-bottom:1px solid #dce2ed;text-align:right}}a{{color:#1565c0}}img{{max-width:100%}}@media(max-width:600px){{main{{padding:14px}}section{{padding:16px}}}}</style><main><h1>5万元 · 低频现金账户研究</h1><section><h2>{html.escape(verdict)}</h2><p>每周最多4次成交操作，最多2仓，无杠杆。100%是检验门槛，结果不承诺未来收益。</p><p>冻结候选：{html.escape(selected['id']) if selected else '没有完整有效候选'}。</p></section><section><h2>赚什么钱：五个明确假设</h2>{''.join('<p><b>'+NAMES[k]+'</b>：'+html.escape(v)+'</p>' for k,v in LOGIC.items())}</section><section><h2>前两年候选比较</h2><div class="scroll">{display.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div></section><section><h2>冻结验证</h2><div class="scroll">{validation.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div></section><section><h2>方法与明细</h2><p>当前名单有生存者/历史ST偏差；真实公司行动账务不支持，以当时可见正式公告避开；未知公司行动使整个测试不完整，不能输出完整年化。近期是复用的历史样本，尚无未来验证。</p><a href="研究结果.md">完整说明</a> · <a href="training-candidates.csv">训练候选</a> · <a href="frozen-selection.json">冻结规则</a> · <a href="study-summary.json">运行摘要</a></section></main></html>'''
    (out / "策略研究报告.html").write_text(page, encoding="utf-8")
    print(json.dumps(dict(verdict=verdict, selected=frozen["selected_id"], validations=clean(summaries)), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--reuse-matrix", action="store_true")
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frozen_design = out / "predeclared-design.json"
    if frozen_design.exists():
        assert json.loads(frozen_design.read_text(encoding="utf-8")) == clean(design()), "predeclared design changed"
    else:
        write_json(frozen_design, design())
    data = reuse_matrix(out) if args.reuse_matrix else compile_matrix(out)
    if args.compile_only:
        return
    if args.stage in ("train", "all"):
        _, frozen = train(data, out)
    else:
        frozen = json.loads((out / "frozen-selection.json").read_text(encoding="utf-8"))
        assert frozen["training_table_sha256"] == digest(out / "training-candidates.csv")
    if args.stage in ("evaluate", "all"):
        checks = evaluate(data, frozen, out)
        report(out, frozen, checks)


if __name__ == "__main__":
    main()
