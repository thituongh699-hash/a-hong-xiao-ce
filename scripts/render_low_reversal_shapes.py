"""Render signal-date-only charts for a causal check of the low interpretation."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.strategies.low_reversal_factor import DEFAULT, signal_mask
from src.utils.config_loader import ConfigLoader
from src.utils.price_adjustment import _read_cache, AdjustmentPlan
from src.utils.tdx_provider import TdxProvider


def chart(frame, selected):
    w,h,left,top,bottom=1200,460,65,28,410
    n=len(frame)
    low,high=float(frame.low.min()),float(frame.high.max())
    span=max(high-low,.01); low-=span*.1; high+=span*.1
    x=lambda i:left+(i+.5)*(w-left-25)/n
    y=lambda p:top+(high-p)/(high-low)*(bottom-top)
    pieces=[f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="信号日以前的日K线">']
    for p in np.linspace(low,high,5):
        pieces.append(f'<line x1="{left}" y1="{y(p)}" x2="1175" y2="{y(p)}" stroke="#d0d5dd"/><text x="5" y="{y(p)+4}" fill="#667085" font-size="13">{p:.2f}</text>')
    for m,color in [(5,"#667085"),(10,"#16a34a"),(20,"#b83ac9"),(60,"#2563eb")]:
        pts=" ".join(f'{x(i):.2f},{y(v):.2f}' for i,v in enumerate(frame[f"ma{m}"]) if pd.notna(v))
        pieces.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    cw=max(2,(w-left-25)/n*.60)
    for i,r in enumerate(frame.itertuples()):
        color="#e5484d" if r.close>=r.open else "#0891b2"
        pieces.append(f'<line x1="{x(i)}" y1="{y(r.high)}" x2="{x(i)}" y2="{y(r.low)}" stroke="{color}"/><rect x="{x(i)-cw/2}" y="{y(max(r.open,r.close))}" width="{cw}" height="{max(1,abs(y(r.open)-y(r.close)))}" fill="{color}"><title>{r.dt.date()} 开{r.open:.2f} 高{r.high:.2f} 低{r.low:.2f} 收{r.close:.2f}</title></rect>')
        if str(r.dt.date())==selected:
            ay=y(r.low)+8
            pieces.append(f'<path d="M{x(i)-3},{ay}v9h-5l8,11 8,-11h-5v-9z" fill="#16a34a"/>')
        if i%20==0 or i==n-1:
            pieces.append(f'<text x="{x(i)}" y="445" text-anchor="middle" font-size="12" fill="#667085">{r.dt.strftime("%m-%d")}</text>')
    return "".join(pieces)+"</svg>"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"reports/low-reversal-factor-2026-10-09")
    args=parser.parse_args(); out=args.out.resolve()
    events=pd.read_csv(out/"factor-events.csv",dtype={"code":str})
    selected=events.loc[signal_mask(events,DEFAULT)&events.period.eq("test")].sort_values(["date","code"],ascending=[False,True]).drop_duplicates("code").head(16)
    cfg=json.loads((ROOT/"config.json").read_text(encoding="utf-8"));cfg["data_provider"]["local_cache_enabled"]=False
    samples=[]
    with ConfigLoader.runtime_snapshot(cfg):
        p=TdxProvider();p._cache_enabled=False
        for r in selected.itertuples():
            raw=p._normalize_daily_df(p._reader_daily(p._raw_symbol(r.code)),r.code)
            raw=raw[(raw.dt>="2024-09-01")&(raw.dt<=r.date)].reset_index(drop=True)
            key=f"{r.code[-2:].lower()}_{r.code[:6]}"
            factors=_read_cache(ROOT/f"data/history/cache/adjustment_factors_v1_{key}.json",key.replace("_","."),raw.dt.max(),21600)
            plan=AdjustmentPlan(raw,factors,"hfq")
            # Anchor historical adjusted chart at signal-day raw units. This
            # uses no factor effective after the selected signal date.
            ratio=plan.frame["__factor"].to_numpy()/plan.frame["__factor"].iloc[-1]
            for k in ("open","high","low","close"): raw[k]*=ratio
            for m in (5,10,20,60): raw[f"ma{m}"]=raw.close.rolling(m).mean()
            title=f"{r.code} {r.name} · {r.date}"
            detail=f"最高涨幅{r.high_pct:.2f}% / 收盘涨幅{r.close_pct:.2f}% / 距此前60根高点回撤{r.drawdown60_pct:.2f}% / 距近20根最低价{r.near_low20_pct:.2f}% / 高点距今{r.peak_age60:.0f}根"
            samples.append(dict(title=title,detail=detail,svg=chart(raw.tail(120),r.date)))
    payload=json.dumps(samples,ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>低位定义图形核对</title><style>body{font:15px/1.6 'Microsoft YaHei',sans-serif;margin:0;background:#f4f6f8;color:#1d2939}main{max-width:1250px;margin:auto;padding:25px}section{padding:20px;background:white;border-radius:10px}svg{width:100%}select{padding:10px;width:min(100%,550px)}a{color:#175cd3}p{color:#667085}</style><main><h1>近期低位定义 · 信号日图形核对</h1><p>默认7.2 / 4.8 / 1 + 回调低位：回撤≥8%、距20根低点≤18%、高点距今3—55根、前10根至少3根低于MA20。按日期取最近16只不同股票，未按后续收益筛选。</p><p>图仅显示到信号日。绿箭头为组合信号；灰/绿/紫/蓝是MA5/10/20/60。价格以当时因子归一，未使用后续因子。</p><section><select id="sample"></select><h2 id="title"></h2><div id="chart"></div><p id="detail"></p></section><p><a href="因子回测报告.html">返回回测对照报告</a> · <a href="factor-events.csv">全部参数候选事件</a></p></main><script id="data" type="application/json">__DATA__</script><script>const d=JSON.parse(document.getElementById('data').textContent),s=document.getElementById('sample');d.forEach((r,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=r.title;s.append(o)});function render(){const r=d[Number(s.value)];if(!r)return;document.getElementById('title').textContent=r.title;document.getElementById('chart').innerHTML=r.svg;document.getElementById('detail').textContent=r.detail}s.onchange=render;render();</script></html>'''
    (out/"低位形态核对.html").write_text(doc.replace("__DATA__",payload),encoding="utf-8")
    main_report=out/"因子回测报告.html"
    report=main_report.read_text(encoding="utf-8")
    summary=json.loads((out/"summary.json").read_text(encoding="utf-8"))
    comparisons=pd.read_csv(out/"comparisons.csv")
    friendly=[]
    for model,setting in summary["frozen"].items():
        factor=setting["factor"]
        low=(f"{factor['window']}日区间位置≤{factor['range_max']*100:g}%" if factor["low_mode"]=="range"
             else f"距60日高点回撤≥{factor['drawdown_min']:g}%，距20日低点≤{factor['near_low_max']:g}%")
        friendly.append(f"{model}：A={factor['a']:g}，A2={factor['a2']:g}，A3={factor['a3']:g}\n低位：{low}\n退出：{setting['exit']['label']}\n")
    original=html.escape(json.dumps(summary["frozen"],ensure_ascii=False,indent=2))
    report=report.replace(original,html.escape("\n".join(friendly)))
    report=report.replace("typeof v==='number'?v.toFixed(3):esc(v)","typeof v==='number'?(Number.isInteger(v)?String(v):v.toFixed(3)):esc(v)")
    # Give an explanation rather than an empty grid for full+optimized: the
    # overall period contains training, so optimized assessment stays split.
    target="const p=document.getElementById('period').value,h=document.getElementById('hold').value;"
    guard="if(p==='full'&&h==='optimized'){document.getElementById('rows').innerHTML='<tr><td colspan=8>调参结果按训练和冻结测试分别评价，请切换时段。</td></tr>';return;}"
    if guard not in report:
        report=report.replace(target,target+guard)
    base=comparisons[(comparisons.variant=="默认+回调低位")&(comparisons.hold==10)]
    pairs=pd.read_csv(out/"paired-entry.csv")
    pair=pairs[pairs.hold==10]
    details=pd.read_csv(out/"test-trades.csv",dtype={"code":str})
    d=details[(details.variant=="默认+回调低位")&(details.hold==10)]
    c=d[(d.model=="close")&d.entered&d.net.notna()]
    i=d[(d.model=="intraday")&d.entered&d.net.notna()]
    keys=set(zip(c.code,c.signal_date))
    extra=i[[(code,day) not in keys for code,day in zip(i.code,i.signal_date)]]
    result="""# 实测结果摘要

最近一年日K已回测，行情截至2026-09-30，完整信号评价截至2026-08-18。3035只主板现名单非ST，10只因不足121根历史跳过；727? 个试验以summary.json为准。

默认公式7.2/4.8/1，加回调低位定义，触发后4日内买入、买入后10日退出：

|时段|买入方式|有效成交|平均交易净收益|胜率|
|---|---|---:|---:|---:|
"""
    result=result.replace("727? 个试验以summary.json为准。",f"{summary['execution_trials']:,}个有入场交易试验。")
    names={"close":"收盘>3%按收盘（理论）","intraday":"最高涨幅>3%盘中突破","next_open":"收盘>3%后次日开盘"}
    for per,title in [("full","整体"),("test","冻结测试")]:
        for r in base[base.period==per].itertuples():
            result+=f"|{title}|{names[r.model]}|{r.trades}|{r.mean_trade_net_pct:.2f}%|{r.win_rate_pct:.2f}%|\n"
    result+=f"""
在冻结测试两种方式都成交的{len(pair)}个默认信号中，盘中平均净收益{pair.intraday_net_pct.mean():.2f}%，同收盘价平均{pair.close_net_pct.mean():.2f}%。但盘中组额外{len(extra)}笔有效交易平均{extra.net.mean():.2f}%，拉低总体表现。

这是事后配对解释，不能提前知道当天收盘是否>3%，所以不能把“盘中买入且最终收盘也>3%”的结果当作可直接执行的策略。收盘同价结果也是理论对照。

训练期选出的参数，冻结测试平均净收益仍为正，但三种入场的风险调整日期等权超额分数都为负，未通过预先设定的门槛。尚不支持稳定有效；不能用测试最高收益再挑参数。当前样本里额外MA5初拐条件的10日平均交易净收益为负，也没有支持把它加入默认条件。

训练所选参数：

"""+"\n".join(friendly)+"""
完整定义、5/10/20日结果、止盈止损调参和共同样本敏感性见因子研究与回测.md、comparisons.csv。现名单缺历史ST和退市、公司行动收益缺失、盘口成交和价带假设限制结论；独立10万元试验收益不等于组合收益。

核验：82项测试通过；test-trades.csv已平仓收益逐行与现金盈亏核对，误差见verification.json；保护配置、私有配置、股票名单哈希不变。
"""
    (out/"结果摘要.md").write_text(result,encoding="utf-8")
    result_link='<p><a href="结果摘要.md">实测结论摘要：整体、冻结测试和盘中额外交易</a></p>'
    if "结果摘要.md" not in report:
        report=report.replace("<section>",result_link+"<section>",1)
    link='<p><a href="低位形态核对.html">低位定义图形核对（16只股票，图仅至信号日）</a></p>'
    if "低位形态核对.html" not in report:
        report=report.replace("</main>",link+"</main>")
    main_report.write_text(report,encoding="utf-8")
    print(json.dumps(dict(samples=len(samples),file=str(out/"低位形态核对.html")),ensure_ascii=False))


if __name__=="__main__":main()
