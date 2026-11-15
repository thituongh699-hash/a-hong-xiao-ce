"""Present frozen CSV results in a readable local research report."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


PERIODS = {
    "full": "近三年",
    "discovery": "前两年",
    "2023_Q4": "2023年四季度",
    "2024": "2024年",
    "2025": "2025年",
    "2026_YTD": "2026年至9月",
    "recent_year": "最近一年",
}
RULES = {
    "hold5": "固定5日",
    "hold10": "固定10日",
    "hold20": "固定20日",
    "factor_loss": "共振失效退出 / 最长20日",
    "hold10_stress": "固定10日 / 单边滑点0.3%",
}


def number(value, digits=2):
    return "缺失" if pd.isna(value) else f"{value:+.{digits}f}"


def interval(row, method="hac"):
    return f"[{number(row[f'{method}_ci_low_pct'])}, {number(row[f'{method}_ci_high_pct'])}]"


def render(out: Path):
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    alpha = pd.read_csv(out / "alpha-summary.csv")
    execution = pd.read_csv(out / "execution-summary.csv")
    primary = alpha[alpha.variant.eq("resonance") & alpha.horizon.eq(10)]
    recent = primary[primary.period.eq("recent_year")].set_index("metric")
    full = primary[primary.period.eq("full")].set_index("metric")
    annual_rows = []
    for period in PERIODS:
        selected = primary[primary.period.eq(period)].set_index("metric")
        matched, conditional = selected.loc["matched_spread_pct"], selected.loc["conditional_beta_pct"]
        annual_rows.append({
            "区间": PERIODS[period],
            "有效日期": int(matched["n"]),
            "匹配收益差": number(matched.mean_pct),
            "匹配差95% HAC区间": interval(matched),
            "匹配差95%分块区间": interval(matched, "bootstrap"),
            "控制变量后系数": number(conditional.mean_pct),
            "系数95% HAC区间": interval(conditional),
        })
    trading_rows = []
    for period in ("full", "recent_year"):
        for rule in RULES:
            r = execution[execution.period.eq(period) & execution.rule.eq(rule)].iloc[0]
            trading_rows.append({
                "区间": PERIODS[period], "退出规则": RULES[rule],
                "触发试验": int(r.trials), "完成交易": int(r.closed),
                "未入场": int(r.no_entry), "收益不可评价": int(r.censored),
                "平均净收益%": number(r.mean_net_pct),
                "相对沪深300百分点": number(r.mean_excess_csi300_pct),
                "胜率%": f"{r.win_rate_pct:.1f}",
            })
    prediction_table = pd.DataFrame(annual_rows).to_html(index=False, border=0)
    trading_table = pd.DataFrame(trading_rows).to_html(index=False, border=0)
    verdict = html.escape(summary["verdict"])
    rm, rc = recent.loc["matched_spread_pct"], recent.loc["conditional_beta_pct"]
    fm = full.loc["matched_spread_pct"]
    direction = "高" if fm.mean_pct >= 0 else "低"
    interval_text = (
        "两项HAC区间都跨过0。"
        if all(row.hac_ci_low_pct <= 0 <= row.hac_ci_high_pct for row in (rm, rc))
        else "两项的置信区间见下表。"
    )
    evidence_text = (
        "最近一年主检验达到预先锁定的正向证据门槛，仍需真实历史股票池和组合验证。"
        if summary["alpha_gate_passed"]
        else "本次结果没有提供通过预先锁定门槛的正向独立alpha证据。"
    )
    gate_text = "原条件在本轮通过正向alpha门槛" if summary["alpha_gate_passed"] else "原条件尚未通过本轮正向alpha门槛"
    recent_trade = execution[execution.period.eq("recent_year") & execution.rule.eq("hold10")].iloc[0]
    verification_path = out / "verification.json"
    audit = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else None
    audit_text = (
        f"现金流水核对通过：{audit['closed']:,} 条完成交易的价差、费用和现金收益已重算；"
        f"输入指纹核对 {audit['input_files_checked']:,} 个文件。"
        if audit else "审计文件尚未生成。"
    )
    paired_text = ""
    if audit:
        paired_text = (
            f"在最近一年相同信号、两种成本下均完成的 {audit['recent_paired_stress_count']:,} 笔交易中，"
            f"单边滑点提高到0.3%后，平均净收益变化为 "
            f"{number(audit['recent_paired_cost_difference_pct'])} 个百分点。"
        )
    links = {
        "研究结果.md": "完整方法说明",
        "alpha-summary.csv": "全部统计与区间",
        "execution-summary.csv": "全部执行结果",
        "trade-trials.csv.gz": "逐笔交易与失败原因",
        "daily-effects.csv": "逐日效应",
        "signal-labels.csv.gz": "逐股信号与标签",
        "predeclared-design.json": "锁定的研究设计",
        "input-manifest.json": "输入与代码指纹",
        "summary.json": "运行摘要",
        "verification.json": "现金与来源核验",
        "executed-source/snapshot.json": "本轮源码指纹",
    }
    link_text = " · ".join(f'<a href="{file}">{label}</a>' for file, label in links.items() if (out / file).exists())
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OBV日周共振检验</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f8;color:#172238;font:15px/1.8 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1440px;margin:auto;padding:32px}}h1{{font-size:30px;margin:0}}h2{{font-size:21px;margin:0 0 12px}}p{{margin:10px 0}}
.sub,.note{{color:#586579}}section{{background:#fff;padding:24px;margin:20px 0;border-radius:14px;border:1px solid #e1e6ef}}
.verdict{{border-left:5px solid #ba7931;background:#fff9ec}}.metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}
.metric{{background:#f6f8fc;border-radius:10px;padding:16px}}.metric strong{{display:block;font-size:25px;font-variant-numeric:tabular-nums}}.metric small{{display:block;color:#586579}}
.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap;font-size:13px}}th,td{{padding:10px 12px;border-bottom:1px solid #e1e6ef;text-align:right}}th{{background:#edf1f8}}th:first-child,td:first-child{{text-align:left}}
a{{color:#1565c0;text-decoration:none}}a:hover{{text-decoration:underline}}img{{width:100%;height:auto}}.files{{line-height:2.3}}
@media(max-width:650px){{main{{padding:16px}}h1{{font-size:24px}}section{{padding:18px}}.metrics{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>OBV 日周共振 · 因子价值检验</h1>
<p class="sub">2023-10-01 — {summary['actual_end']} · {summary['stocks']:,}只当前沪深主板正常股票 · 固定MAOBV18 · 包含当周</p>
<section class="verdict"><h2>{verdict}</h2>
<p>近三年，共振股票未来10日收益比动量、波动率相近的股票平均{direction} {abs(fm.mean_pct):.2f} 个百分点。
最近一年匹配收益差为 {number(rm.mean_pct)} 个百分点，控制更多变量后系数为 {number(rc.mean_pct)} 个百分点；{interval_text}</p>
<p>{evidence_text}交易盈利应结合市场同期收益判断；本次没有根据结果改动参数，也没有进行未来样本验证。</p></section>
<section><div class="metrics">
<div class="metric"><small>近三年 · 10日匹配收益差</small><strong>{number(fm.mean_pct)} 个百分点</strong><small>95% HAC区间 {interval(fm)}</small></div>
<div class="metric"><small>最近一年 · 10日匹配收益差</small><strong>{number(rm.mean_pct)} 个百分点</strong><small>95% HAC区间 {interval(rm)}</small></div>
<div class="metric"><small>最近一年 · 次日开盘入场 / 固定10日</small><strong>{number(recent_trade.mean_net_pct)}%</strong><small>相对沪深300 {number(recent_trade.mean_excess_csi300_pct)} 个百分点</small></div>
</div></section>
<section><img src="study-overview.png" alt="按年增量预测力、最近一年持有期限和交易净收益对照"></section>
<section><h2>主要10日检验 · 年度与最近一年</h2>
<p>收益差和系数单位均为<strong>百分点</strong>。预测标签是信号日收盘至10个市场日后收盘。
匹配控制20日动量和波动率；回归同时控制10/20/60日动量、20日波动率与平均成交金额。
置信区间按日期处理重叠和共同市场冲击；区间跨0表示尚未排除没有额外信息。</p>
<div class="scroll">{prediction_table}</div>
<p class="note">最近一年为2025-10-01至2026-09-30。各子区间只计入结束前已经完成的收益标签。2023年仅有四季度，2026年仅到9月。</p></section>
<section><h2>首次触发交易 · 次日开盘入场</h2>
<p>每次从0变1独立使用10万元；固定h日退出指<strong>买入后h个市场日的收盘</strong>。
100股整数手、T+1、停牌和开盘涨跌停限制；无法卖出最多再尝试5个市场日开盘。
基础成本：佣金万2.5 / 最低5元、卖出印花税0.05%、双边过户费0.001%、单边滑点0.1%。</p>
<div class="scroll">{trading_table}</div>
<p>{paired_text}</p><p class="note">“收益不可评价”包括跨公司行动、原价参考无法核验、期末未退出等；交易明细保留原因和已经发生的入场。
净收益以10万元现金为分母；沪深300基准用相同入场、实际退出时点的指数价格收益，不计基准交易费。
“独立”指每次账户隔离；交易在统计上会相互依赖，触发笔数不能作为独立样本量做显著性推断。
执行均值为描述性结果，未计算假设交易彼此独立的置信区间，也不能累加为共享账户收益或组合年化。</p></section>
<section><h2>信号与证据范围</h2>
<p>信号：本周OBV &gt; 上周OBV &gt; 上上周OBV，同时本周OBV &gt; 周MAOBV18、日OBV &gt; 日MAOBV18。
每天收盘仅用截至当日的周累计成交量与价格重算本周，周中信号可能变化。</p>
<p>当前股票名单来自2026-08-05，缺历史ST和退市股票，有生存者偏差；未控制行业和市值。
复权预测标签不等同于含分红现金流水的真实总收益；持仓公司行动的现金与股数核算缺失。
日K撮合不能证明盘口排队和实际冲击成本。{gate_text}，亦不能据此认定反向策略有效。</p>
<p class="note">{audit_text}</p></section>
<section><h2>数据与可审计文件</h2><p class="files">{link_text}</p></section>
</main></body></html>'''
    target = out / "因子价值报告.html"
    target.write_text(page, encoding="utf-8")
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    render(args.out.resolve())


if __name__ == "__main__":
    main()
