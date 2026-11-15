"""Human-readable conclusions without selecting or changing a strategy."""
from pathlib import Path
import html
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_lowfreq_50k as base


def main():
    repaired=ROOT/"reports/lowfreq-pullback-repair-2026-10-10"
    selected=json.loads((repaired/"frozen-selection.json").read_text(encoding="utf-8"))["selected_id"]
    data=json.loads((repaired/"supplemental-summary.json").read_text(encoding="utf-8"))
    full,recent=[data["accounts"][key] for key in ("full","recent")]
    orders=pd.read_csv(repaired/"accounts"/f"recent-{selected}"/"orders.csv")
    fixed_cash=50000.
    for row in orders.itertuples():
        buy=row.direction=="BUY"
        price=row.price*(1.003/1.001 if buy else .997/.999)
        amount=price*row.qty
        fee=max(5.,amount*.00025)+amount*.00001+(0 if buy else amount*.0005)
        fixed_cash+=-amount-fee if buy else amount-fee
    attribution=dict(original_slippage_per_side=.001,new_slippage_per_side=.003,
                     frozen_original_order_dates_quantities_and_path=True,end_cash=fixed_cash,
                     return_pct=(fixed_cash/50000.-1)*100,
                     interpretation="cost attribution on fixed trades only, not an executable rebalanced account; no new strategy selection")
    base.write_json(repaired/"fixed-path-cost-attribution.json",attribution)
    sensitivity_note=("\n\n## 近期压力结果的解释\n\n"
                      "所有近期变体均为已看过近期历史后的回顾性敏感性回放。提高单边滑点会改变成交价、费用内整数手、止损触发价、持仓占用及后续候选路径，完成交易从38笔变为36笔；延迟一天也会改变路径。+26.03%或+36.71%不能证明提高费用或延迟有益，也不构成稳定性验证。"
                      f"固定原成交日期、股数与路径仅做费用归因，0.3%单边滑点的收益是{attribution['return_pct']:.2f}%，低于基线15.98%。固定路径归因不是新的可执行账户回测。\n")
    for filename in ("修正结果.md","解释与收益来源.md"):
        path=repaired/filename
        original=path.read_text(encoding="utf-8")
        if "## 近期压力结果的解释" not in original:
            path.write_text(original+sensitivity_note,encoding="utf-8")
    rows=pd.read_csv(repaired/"selected-validation.csv")
    annual=pd.read_csv(repaired/"selected-annual.csv")
    annual["期间"]=annual.year.map({2023:"2023年第四季度",2024:"2024年",2025:"2025年",2026:"2026年1—9月"})
    annual["收益"]=annual.return_pct.map(lambda x:f"{x:.2f}%")
    stat=recent["trade_statistics"]
    regression=full["benchmark_comparison"]["complete_period_regression"]
    ci=regression["annualized_intercept_95pct_conditional_interval"]
    text=["# 有逻辑的低频策略：研究结论","",
          "**本轮未找到年化100%的可信方案。修正后的强势股回调恢复可保留为研究候选，尚未确认稳定alpha。**","",
          "首轮在3035只当前主板股票的近三年数据上预设10组规则，用前两年选出pullback3后冻结。原版最近一年−14.02%，不通过。",
          "逐笔发现入场与MA10退出口径不一致：47次最近一年买入中23次的信号收盘已低于MA10。第二轮只增加信号收盘不低于MA10，并比较原来3/5日两种回调口径。最近一年已经看过，所以修正结果属于回顾性研究。","",
          "## 赚什么钱、怎么赚钱","",
          "赚的是一段中期上涨中，短期调整结束后的下一段上涨。研究假设是原有上涨需求仍在，而部分短期卖压只是暂时的；实际量价数据无法确认卖盘来源，也无法排除永久坏消息。",
          "中期强势、缩量回撤、收复前日高点和回到10日线上分别代理趋势仍在、回撤交易减弱及价格恢复。确认后下一开盘买入，不预先抄底。若恢复失败则退出，多次小亏换少量持续上涨的大盈利。","",
          "## 固定修正规则","",
          "- 市场：信号日沪深300收盘高于MA60；股票有120根此前日K，前20日平均成交额至少5000万元。",
          "- 前一日：近60日涨幅>10%，收盘>MA60；此前3日累计跌幅<-3%。",
          "- 信号日：此前3日均量<此前20日均量；收盘>前日最高价，日涨幅在0%至7%之间；新增收盘>=MA10。",
          "- 排序：按前一日60日涨幅从高到低、代码打破并列，先取20个候选，再检查开盘可成交性。",
          "- 账户：5万元，无杠杆，最多2仓；新仓含费预算为前收盘净值45%，保留10%现金，100股整手、T+1；同日卖出资金不用于买入；一买一卖分别算一次，每周成交操作<=4。",
          "- 退出：已完成收盘跌破实际买价的94%、此前收盘峰值的90%、或MA10，下一开盘退出；最长20个市场交易日。没有固定止盈；缺口损失可超过6%。",
          "- 公司行动：已可见正式实施公告触发退出并避开除权；未知事件使账户不完整。","",
          "## 账户结果","",base.readable_table(rows).to_markdown(index=False),"",
          "三年连续账户5万元到69500.99元，年化11.61%，最大回撤30.56%；最近一年单独5万元到57988.88元，收益15.98%，最大回撤20.24%。收益已含模拟手续费和滑点。","",
          annual[["期间","收益"]].to_markdown(index=False),"",
          f"最近一年38笔完成交易，胜率{stat['win_rate_positive_pnl']:.2%}，平均持有{stat['mean_holding_sessions']:.2f}个交易日，平均盈利/平均亏损额比{stat['payoff_ratio_mean_winner_to_absolute_mean_loser']:.2f}。每周平均1.49次成交，最多4次。",
          f"前五笔盈利占全部正利润{stat['top5_positive_trades_share_of_all_positive_pnl']:.2%}；603115.SH在2026-04-15至05-18一笔盈利11768.99元，超过全年净利润7988.88元。这是历史贡献，不是删掉交易后重算的反事实账户。","",
          "## 是否存在alpha","",
          f"三年同期沪深300价格指数累计+18.11%，策略+39.00%。以日收益对沪深300回归，beta约{regression['daily_ols_csi300_beta']:.3f}；截距乘252的算术年化约{regression['daily_ols_intercept_arithmetic_annualized_252']:.2%}，5日块bootstrap95%区间[{ci[0]:.2%}, {ci[1]:.2%}]，跨过零。这个描述性回归未调整小盘、行业、风格或多次研究选择，不能确认稳定alpha或证明经济机制。","",
          f"重新模拟0.3%单边滑点为+26.03%，延迟一天为+36.71%。费用/价格会改变整数手、止损触发、开盘可成交性和后续交易，因此路径不同，不能解释为成本增加有利。固定原始成交日期/股数/路径，只做成本归因时，0.3%单边滑点收益降至{attribution['return_pct']:.2f}%；该归因不是新的可执行账户回测。","",
          "## 核验与限制","",
          "108项针对性测试通过；原版14个与修正版6个账户均独立重建现金、费用、原价净值与交易盈亏，误差不到0.000001元，训练与三年连续账户前两年净值一致。",
          "完整指模型内数据与账务完成。股票池有生存者偏差，缺历史ST状态可能误用10%而非5%限价，日K不能证明真实成交队列，公告库没有历史版本归档；修正研究已看过近期数据且无未来样本。费率是研究假设，不构成当前券商报价或统一收费标准。",
          "本轮结论：逻辑得到明确表达，但实际收益仅支持继续验证这个候选，不能支持年化100%的实盘预期。","",
          "![修正版净值与回撤](../lowfreq-pullback-repair-2026-10-10/equity.png)"]
    (base.OUT/"逻辑与最终结论.md").write_text("\n".join(text),encoding="utf-8")
    comparisons=base.readable_table(rows).to_html(index=False)
    logic="<p>中期上涨需求仍在 → 短期回撤与交易减弱 → 收复前日高点并回到MA10 → 次日开盘进入 → 赚下一段上涨，承担恢复失败的亏损。</p><p>这是一项可反驳的供需假设。OHLCV无法识别消息、买卖主体或永久坏消息。</p>"
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5万元低频策略研究结论</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#eef2f7;color:#172338;font:16px/1.8 system-ui,"Microsoft YaHei"}}main{{max-width:1250px;margin:auto;padding:28px}}section{{background:white;padding:24px;border-radius:14px;margin:20px 0}}h1{{font-size:32px;line-height:1.4}}h2{{font-size:22px}}.tag{{color:#a04910;font-weight:700}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{background:#f0f5fc;padding:18px;border-radius:10px}}.number{{font-size:28px;font-weight:700}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #d9e1ef}}img{{display:block;width:100%;height:auto}}a{{color:#165cac}}@media(max-width:650px){{main{{padding:14px}}section{{padding:17px}}h1{{font-size:26px}}.cards{{grid-template-columns:1fr}}}}</style><main><h1>5万元低频策略 · 逻辑与回测结论</h1><p class="tag">未达到年化100%；修正版属于回顾性研究，未确认稳定alpha。</p><div class="cards"><div class="card">近三年年化<div class="number">11.61%</div>最大回撤30.56%</div><div class="card">最近一年收益<div class="number">15.98%</div>5万元 → 57,988.88元</div><div class="card">近期每周成交操作<div class="number">平均1.49次</div>最多4次；平均持有5.18日</div></div><section><h2>赚什么钱</h2>{logic}<p>规则：前一日60日涨幅&gt;10%、3日回调超过3%；缩量、收复前日高点，日涨幅0%—7%，信号收盘不低于MA10。沪深300在MA60上；按60日强度排序。止损6%、收盘追踪回撤10%、跌破MA10或最长20个交易日退出，收盘观察、下期开盘执行。</p></section><section><h2>预设规则、失败与修正</h2><p>首轮10组规则先在前两年选择，冻结后的原版最近一年−14.02%。47次买入里23次信号已低于MA10，入场与退出不一致。</p><p>第二轮只增加入场收盘不低于MA10，比较原有3/5日回调两种口径，前两年仍选3日版。修正时最近一年已经看过，结果不能称为独立验证。</p></section><section><h2>修正版完整账户回放</h2><div class="scroll">{comparisons}</div><p>0.3%滑点和延迟一天都重新撮合，持仓与交易路径变化，收益较高不代表提高成本有利。固定原始成交路径的成本归因：收益{attribution['return_pct']:.2f}%。</p></section><section><h2>依赖少数大行情</h2><p>近期胜率31.58%；前五笔赢家占全部正利润90.29%。最大一笔盈利11,768.99元，超过全年净利润7,988.88元。</p><p>三年市场回归截距算术年化13.41%，95%区间−18.97%至48.83%，跨过零；不能确认稳定alpha。</p><img src="../lowfreq-pullback-repair-2026-10-10/equity.png" alt="修正版近三年及最近一年净值与回撤，同期沪深300价格指数参照"></section><section><h2>核验与边界</h2><p>108项针对性测试；20个账户独立现金/原价净值审计通过。5万元共用账户，最多2仓，45%新仓预算、10%现金，100股整手、T+1，同日卖出资金不复用，每周买卖各计一次、最多4次。</p><p>当前名单生存者偏差、历史ST/5%限价缺口、日线队列近似、公告库修订和无未来样本仍影响可信度。费率为回测假设。历史逻辑和正收益不足以支持100%的实盘预期。</p><a href="逻辑与最终结论.md">逻辑、完整规则与最终结论</a> · <a href="研究结果.md">首轮10个候选</a> · <a href="../lowfreq-pullback-repair-2026-10-10/修正结果.md">修正结果</a> · <a href="../lowfreq-pullback-repair-2026-10-10/verification.json">修正审计</a> · <a href="../lowfreq-pullback-repair-2026-10-10/accounts/recent-pullback3_above_ma10/trades.csv">近期逐笔交易</a></section></main></html>'''
    (base.OUT/"策略研究报告.html").write_text(page,encoding="utf-8")
    print(json.dumps(attribution,ensure_ascii=False))


if __name__=="__main__":main()
