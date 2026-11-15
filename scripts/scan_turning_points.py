"""Historical daily signal screening using the project's local TDX provider.

Only information available on each signal date is used for shape classification.
This is a candidate screen, not an order execution or return simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config_loader import ConfigLoader
from src.utils.tdx_provider import TdxProvider


EPS = 1e-10


def features(frame: pd.DataFrame, a=7.2, a2=4.8, a3=1.0) -> pd.DataFrame:
    if not a3 < a2 or not all(np.isfinite([a, a2, a3])):
        raise ValueError('parameters require finite numbers and A3 < A2')
    work = frame.sort_values('dt').reset_index(drop=True).copy()
    if work.dt.duplicated().any():
        raise ValueError('duplicate daily bars')
    ohlc = work[['open', 'high', 'low', 'close']].astype(float)
    valid = (np.isfinite(ohlc).all(axis=1) & ohlc.gt(0).all(axis=1)
             & work.high.ge(work[['open', 'close', 'low']].max(axis=1))
             & work.low.le(work[['open', 'close', 'high']].min(axis=1)))
    if not valid.all():
        raise ValueError('invalid OHLC; refusing to bridge invalid records')
    previous = work.close.shift(1)
    work['previous_close'] = previous
    work['high_pct'] = (work.high - previous) / previous * 100
    work['close_pct'] = (work.close - previous) / previous * 100
    work['signal'] = (work.high_pct.gt(a + EPS) & work.close_pct.lt(a2 - EPS)
                      & work.close_pct.gt(a3 + EPS) & previous.gt(0))
    for n in (5, 10, 20, 60, 120):
        work[f'ma{n}'] = work.close.rolling(n, min_periods=n).mean()
    slope = work.ma5.diff()
    work['ma5_upturn_today'] = slope.gt(EPS) & slope.shift(1).le(EPS)
    work['ma5_recent_turn'] = work.ma5_upturn_today.rolling(5, min_periods=1).max().fillna(0).astype(bool) & slope.gt(EPS)
    return work


def describe_shape(work: pd.DataFrame, index: int) -> dict:
    """Explicit first-pass interpretation of the user's example shapes."""
    row = work.iloc[index]
    before = work.iloc[max(0, index - 60):index]
    out = {'shape': '其他信号', 'shape_match': False, 'turn_confirmed': False,
           'drawdown_pct': None, 'peak_age_bars': None, 'peak_date': None,
           'distance_low20_pct': None, 'prior_rise_pct': None, 'shape_score': 0.0,
           'ma5_recent_turn': bool(row.ma5_recent_turn), 'near_ma60': False,
           'near_ma120': False, 'history_sufficient': len(before) == 60}
    if len(before) < 60:
        out['shape'] = '形态历史不足'
        return out
    peak_pos = int(before.high.idxmax())
    peak = float(work.loc[peak_pos, 'high'])
    peak_age = index - peak_pos
    dd = (1 - float(row.close) / peak) * 100
    low20 = float(work.iloc[max(0, index - 19):index + 1].low.min())
    near_low = (float(row.close) / low20 - 1) * 100
    prepeak = work.iloc[max(0, peak_pos - 40):peak_pos + 1]
    rise = (peak / float(prepeak.low.min()) - 1) * 100
    trend_history = work.iloc[max(0, index - 10):index]
    below_ma20_days = int((trend_history.close < trend_history.ma20).sum())
    pullback = (3 <= peak_age <= 55 and dd >= 8 and near_low <= 18
                and below_ma20_days >= 3)
    turn = pullback and bool(row.ma5_recent_turn) and (rise >= 15 or dd >= 20)
    if turn:
        shape = '深跌初拐' if dd >= 20 else '上涨回调初拐'
    elif pullback:
        shape = '回调待确认'
    else:
        shape = '其他信号'
    score = (min(dd, 40) * 0.7 + max(0, 18 - near_low) * 1.5
             + (25 if turn else 0) + min(rise, 60) * 0.15)
    out.update(shape=shape, shape_match=pullback, turn_confirmed=turn,
               drawdown_pct=dd, peak_age_bars=peak_age,
               peak_date=str(work.loc[peak_pos, 'dt'].date()),
               distance_low20_pct=near_low, prior_rise_pct=rise, shape_score=score)
    for n in (60, 120):
        ma = float(row[f'ma{n}'])
        out[f'near_ma{n}'] = bool(np.isfinite(ma) and ma > 0 and abs(row.close / ma - 1) <= 0.08)
    return out


def write_report(out: Path, events: pd.DataFrame, summary: dict, charts: list[dict]):
    # Keep the offline report responsive; CSVs contain the complete results.
    # Include distinct stocks by both similarity and recency, not just one rank.
    samples, seen = [], set()
    core = [r for r in charts if r['turn_confirmed']]
    pending = [r for r in charts if not r['turn_confirmed']]
    for candidates, limit in [(core, 80), (sorted(core, key=lambda r: (r['date'], r['shape_score']), reverse=True), 80), (sorted(pending, key=lambda r: (r['date'], r['shape_score']), reverse=True), 40)]:
        count, codes = 0, set()
        for event in candidates:
            key = (event['code'], event['date'])
            if event['code'] in codes:
                continue
            codes.add(event['code'])
            count += 1
            if key not in seen:
                seen.add(key)
                samples.append(event)
            if count == limit:
                break
    payload = json.dumps(samples, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    html = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>拐弯买入点历史筛选</title>
<style>body{margin:0;background:#11161f;color:#e8edf5;font:15px/1.65 "Microsoft YaHei",sans-serif}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:29px;margin:0}p{color:#aebbd0}a{color:#6db7ff}.stats{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}.stats div{background:#1c2432;padding:14px 24px;border-radius:8px}.stats strong{font-size:28px;display:block;color:#7fddb8}.panel{background:#1a2230;padding:20px;border-radius:10px;margin:20px 0}input,select{background:#101722;color:#e8edf5;border:1px solid #44516a;padding:9px;border-radius:5px;margin-right:12px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px;border-bottom:1px solid #2c3749;text-align:left}tr[data-id]{cursor:pointer}tr[data-id]:hover,tr.selected{background:#2b3b51}button{background:#224c74;border:0;border-radius:6px;padding:9px 18px;color:white;cursor:pointer;margin:10px 6px}svg{width:100%;background:#070b12}small{color:#9cacbf}.tablewrap{max-height:550px;overflow:auto}th{position:sticky;top:0;background:#1a2230}.rules{font-size:13px}</style>
<main><h1>拐弯买入点 · 历史候选</h1><p>日K｜区间：2026-04-08 — 2026-10-08｜实际本地末日：2026-09-29｜价格：不复权</p>
<div class="stats">__STATS__</div>
<div class="panel rules"><b>你的精确信号：</b>最高价涨幅 &gt; __A__%，且 __A3__% &lt; 收盘涨幅 &lt; __A2__%。数值等于边界时不选。<br>
<b>首版形态解释：</b>此前60根K线最高点距信号3—55根，收盘距高点回落≥8%，收盘距近20根最低价≤18%，信号前10根至少3根收盘低于MA20。<br>
<b>初拐确认：</b>再要求MA5于最近5根由非上升转为上升且信号当日仍上升，以及高点前40根至高点的涨幅≥15%或当前回落≥20%。回落≥20%归为“深跌初拐”。这些形态阈值是本轮为匹配图片所做的解释，可修改，不是用户提供的箭头公式。<br>
<b>数据边界：</b>股票池来自项目8月5日名单，未覆盖验证其后的新增/退市变更；本地行情缺9月30日及10月8日完整记录。复权因子查询超时，因此本轮采用原价；除权可能影响信号与形态。候选用于图形核对，未进行交易收益回测。全部分类只使用信号当日及以前的K线。</div>
<div class="panel"><h2 id="title">候选走势图</h2><div id="chart"></div><p id="detail"></p><small>灰 / 绿 / 紫 / 蓝 / 白分别为MA5 / MA10 / MA20 / MA60 / MA120；亮绿色箭头对应你的公式。图仅显示到选中信号日，避免后续走势影响核对。</small></div>
<div class="panel"><h2>形态候选图形核对</h2><p>本页提供__SAMPLES__个代表样本：按形态得分、最近日期各选取最多80条不同股票的初拐信号，再选取最多40条近期回调待确认信号，去掉重复事件。完整名单请下载下方CSV。</p><input id="search" placeholder="样本股票代码或名称"><select id="group"><option value="初拐">已出现初拐</option><option value="全部">全部图形样本</option><option value="回调待确认">回调待确认</option></select><span id="count"></span><div class="tablewrap"><table><thead><tr><th>股票</th><th>日期</th><th>形态</th><th>最高涨幅</th><th>收盘涨幅</th><th>距高点回落</th><th>距20日低点</th></tr></thead><tbody id="rows"></tbody></table></div></div>
<div class="panel"><h2>下载结果</h2><a href="初拐股票名单.csv">初拐股票名单（去重）</a> · <a href="初拐候选.csv">初拐信号明细</a> · <a href="全部形态候选.csv">全部形态候选</a> · <a href="全部公式信号.csv">全部公式信号</a> · <a href="股票汇总.csv">按股票汇总</a> · <a href="scan-summary.json">扫描记录</a><p>股票名单每只股票一行，保留最近一次初拐信号；信号明细每个信号单独一行，同一股票可以有多次信号。形态得分只用于候选排序，不是上涨概率或收益预测。</p></div></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), byId=new Map(data.map((r,i)=>[String(i),r]));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=n=>Number(n).toFixed(2);let active='';
function draw(id){const r=byId.get(String(id));if(!r)return;active=String(id);const b=r.bars,W=1200,H=440,L=60,R=30,T=25,B=55,n=b.length;const vals=b.flatMap(x=>[x.h,x.l,...x.ma.filter(x=>x!=null)]),lo=Math.min(...vals),hi=Math.max(...vals),pad=(hi-lo)*.13||1,ymn=lo-pad,ymx=hi+pad;const X=i=>L+(i+.5)*(W-L-R)/n,Y=p=>T+(ymx-p)/(ymx-ymn)*(H-T-B),cw=Math.max(2,(W-L-R)/n*.62);let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(r.name)}信号日K线">`;for(let j=0;j<5;j++){let p=ymn+(ymx-ymn)*j/4,y=Y(p);s+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="#253043"/><text x="8" y="${y+4}" fill="#8c9bad" font-size="12">${fmt(p)}</text>`}
const colors=['#8c949e','#1de868','#e437f0','#496eff','#ffffff'];for(let m=0;m<5;m++){const pts=b.map((v,i)=>v.ma[m]==null?null:`${X(i)},${Y(v.ma[m])}`).filter(Boolean).join(' ');s+=`<polyline points="${pts}" fill="none" stroke="${colors[m]}" stroke-width="1.6"/>`}
b.forEach((v,i)=>{let color=v.c>=v.o?'#ff525b':'#40d5d0',x=X(i),yt=Y(Math.max(v.o,v.c)),height=Math.max(1,Math.abs(Y(v.o)-Y(v.c)));s+=`<line x1="${x}" y1="${Y(v.h)}" x2="${x}" y2="${Y(v.l)}" stroke="${color}"/><rect x="${x-cw/2}" y="${yt}" width="${cw}" height="${height}" fill="${v.c>=v.o?'#070b12':color}" stroke="${color}"><title>${esc(v.d)} 开${v.o} 高${v.h} 低${v.l} 收${v.c}</title></rect>`;if(v.s){let ay=Y(v.l)+8;s+=`<path d="M${x-3},${ay}v8h-4l7,9 7,-9h-4v-8z" fill="#00ff56"/>`}if(i%Math.max(1,Math.floor(n/7))===0||i===n-1)s+=`<text x="${x}" y="${H-18}" text-anchor="middle" fill="#8c9bad" font-size="11">${v.d.slice(5)}</text>`});s+='</svg>';document.getElementById('chart').innerHTML=s;document.getElementById('title').textContent=`${r.code} ${r.name} · ${r.date} · ${r.shape}`;document.getElementById('detail').textContent=`最高涨幅 ${fmt(r.high_pct)}% / 收盘涨幅 ${fmt(r.close_pct)}% / 距此前高点回落 ${fmt(r.drawdown_pct)}% / 距近20根低点 ${fmt(r.distance_low20_pct)}% / 高点距今 ${r.peak_age_bars} 根K线`;document.querySelectorAll('tr[data-id]').forEach(t=>t.classList.toggle('selected',t.dataset.id===active))}
function render(){const q=document.getElementById('search').value.trim().toLowerCase(),g=document.getElementById('group').value;const items=data.map((r,i)=>({r,i})).filter(({r})=>(g==='全部'||(g==='初拐'?r.turn_confirmed:r.shape===g))&&(`${r.code}${r.name}`.toLowerCase().includes(q)));document.getElementById('count').textContent=`${items.length} 个信号`;document.getElementById('rows').innerHTML=items.map(({r,i})=>`<tr data-id="${i}" class="${String(i)===active?'selected':''}"><td>${esc(r.code)} ${esc(r.name)}</td><td>${r.date}</td><td>${r.shape}</td><td>${fmt(r.high_pct)}%</td><td>${fmt(r.close_pct)}%</td><td>${fmt(r.drawdown_pct)}%</td><td>${fmt(r.distance_low20_pct)}%</td></tr>`).join('')}
document.getElementById('rows').addEventListener('click',e=>{const t=e.target.closest('tr[data-id]');if(t)draw(t.dataset.id)});document.getElementById('search').addEventListener('input',render);document.getElementById('group').addEventListener('change',render);render();if(data.length)draw(0);
</script></html>'''
    stats = ''.join(f'<div><strong>{summary[k]:,}</strong>{label}</div>' for k, label in [
        ('stocks_scanned', '已扫描股票'), ('signal_events', '公式信号'),
        ('shape_stocks', '形态候选股票'), ('turn_stocks', '初拐候选股票')])
    html = html.replace('__STATS__', stats).replace('__DATA__', payload).replace('__SAMPLES__', str(len(samples)))
    for token, key in [('__A__', 'A'), ('__A2__', 'A2'), ('__A3__', 'A3')]:
        html = html.replace(token, str(summary['parameters'][key]))
    (out / '拐弯买入点筛选报告.html').write_text(html, encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', default='2026-04-08')
    p.add_argument('--end', default='2026-10-08')
    p.add_argument('--a', type=float, default=7.2)
    p.add_argument('--a2', type=float, default=4.8)
    p.add_argument('--a3', type=float, default=1.0)
    p.add_argument('--out', type=Path, default=ROOT / 'reports/turning-point-2026-10-08')
    args = p.parse_args()
    if not args.a3 < args.a2 or not all(np.isfinite([args.a, args.a2, args.a3])):
        p.error('finite parameters and A3 < A2 required')
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start > end:
        p.error('start must be <= end')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    protected = [ROOT / 'config.json', ROOT / 'config.private.json', ROOT / 'data/stock_list.csv']
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in protected if f.exists()}
    config['data_provider']['local_cache_enabled'] = False
    stocks = pd.read_csv(ROOT / 'data/stock_list.csv', dtype=str)
    all_events, coverage, charts = [], [], []
    t0 = time.monotonic()
    with ConfigLoader.runtime_snapshot(config):
        provider = TdxProvider()
        provider._cache_enabled = False
        for i, stock in stocks.iterrows():
            code, name = stock['code'], stock['name']
            cov = {'code': code, 'name': name, 'bars': 0, 'first_date': '', 'last_date': '', 'status': 'error', 'error': ''}
            try:
                day_path, _ = provider._symbol_file_hints(code)
                if not Path(day_path).is_file():
                    raise ValueError('missing local daily file; no network fallback')
                # Reuse the project's local reader and normalizer. Loading through
                # fetch_kline_data would also work, but incurs needless cache work.
                raw = provider._normalize_daily_df(provider._reader_daily(provider._raw_symbol(code)), code)
                if raw.empty:
                    raise ValueError(provider.last_error or 'empty local daily bars')
                raw = raw[(raw.dt >= start - pd.Timedelta(days=400)) & (raw.dt <= end)].reset_index(drop=True)
                work = features(raw, args.a, args.a2, args.a3)
                cov.update(bars=len(work), first_date=str(work.dt.min().date()), last_date=str(work.dt.max().date()), status='ok')
                selected = work.index[work.signal & work.dt.between(start, end)]
                for ix in selected:
                    row = work.loc[ix]
                    event = {'code': code, 'name': name, 'date': str(row['dt'].date()),
                             'previous_close': float(row.previous_close), 'open': float(row.open),
                             'high': float(row.high), 'low': float(row.low), 'close': float(row.close),
                             'high_pct': float(row.high_pct), 'close_pct': float(row.close_pct),
                             'price_mode': 'none', 'local_last_date': cov['last_date'],
                             **describe_shape(work, int(ix))}
                    all_events.append(event)
                    if event['shape_match']:
                        bars = []
                        for b in work.iloc[max(0, ix - 119):ix + 1].itertuples():
                            bars.append({'d': str(b.dt.date()), 'o': float(b.open), 'h': float(b.high),
                                         'l': float(b.low), 'c': float(b.close), 's': bool(b.signal),
                                         'ma': [None if not np.isfinite(getattr(b, f'ma{n}')) else float(getattr(b, f'ma{n}')) for n in (5, 10, 20, 60, 120)]})
                        charts.append({**event, 'bars': bars})
            except Exception as exc:
                cov['status'] = 'error'
                cov['error'] = f'{type(exc).__name__}: {exc}'
                if sum(c['status'] == 'error' for c in coverage) < 3:
                    print(json.dumps({'code': code, 'error': cov['error']}, ensure_ascii=False), flush=True)
            coverage.append(cov)
            if (i + 1) % 250 == 0 or i + 1 == len(stocks):
                print(json.dumps({'scanned': i + 1, 'total': len(stocks), 'signals': len(all_events), 'shape_candidates': len(charts), 'errors': sum(c['status'] == 'error' for c in coverage), 'elapsed_seconds': round(time.monotonic() - t0, 1)}, ensure_ascii=False), flush=True)
    if not all_events:
        raise RuntimeError('no events; inspect data before claiming an empty market result')
    events = pd.DataFrame(all_events).sort_values(['date', 'shape_score'], ascending=[False, False])
    matched = events[events.shape_match].sort_values(['turn_confirmed', 'shape_score', 'date'], ascending=[False, False, False])
    turns = matched[matched.turn_confirmed]
    for name, frame in [('全部公式信号.csv', events), ('全部形态候选.csv', matched), ('初拐候选.csv', turns)]:
        frame.to_csv(out / name, encoding='utf-8-sig', index=False, float_format='%.8f')
    turn_stocks = turns.sort_values(['date', 'shape_score'], ascending=[False, False]).drop_duplicates('code').copy()
    turn_stocks['turn_signal_count'] = turn_stocks.code.map(turns.groupby('code').size())
    turn_stocks.to_csv(out / '初拐股票名单.csv', encoding='utf-8-sig', index=False, float_format='%.8f')
    pd.DataFrame(coverage).to_csv(out / '扫描覆盖.csv', encoding='utf-8-sig', index=False)
    grouped = events.groupby(['code', 'name'], as_index=False).agg(signal_count=('date', 'count'), first_signal=('date', 'min'), latest_signal=('date', 'max'), shape_count=('shape_match', 'sum'), turn_count=('turn_confirmed', 'sum'))
    grouped = grouped.sort_values(['turn_count', 'shape_count', 'latest_signal'], ascending=[False, False, False])
    grouped.to_csv(out / '股票汇总.csv', encoding='utf-8-sig', index=False)
    unchanged = all(hashlib.sha256(Path(f).read_bytes()).hexdigest() == value for f, value in hashes.items())
    summary = {'requested_start': args.start, 'requested_end': args.end, 'timeframe': 'D',
               'parameters': {'A': args.a, 'A2': args.a2, 'A3': args.a3}, 'comparison': 'strict; numerical boundary tolerance 1e-10 percentage points',
               'signal_price_mode': 'none', 'configured_adjustment_mode': config['data_provider'].get('adjustment_mode'),
               'price_mode_reason': 'BaoStock factor probe timed out after 12s; explicitly labeled raw-price initial screen',
               'data_source': 'project TdxProvider local daily reader; no network fallback',
               'stock_list_date': '2026-08-05', 'stocks_requested': len(stocks),
               'stocks_scanned': sum(c['status'] == 'ok' for c in coverage), 'failures': [c for c in coverage if c['status'] != 'ok'],
               'actual_last_date': max(c['last_date'] for c in coverage),
               'last_date_counts': dict(Counter(c['last_date'] for c in coverage)),
               'signal_events': len(events), 'signal_stocks': int(events.code.nunique()),
               'shape_events': len(matched), 'shape_stocks': int(matched.code.nunique()),
               'turn_events': len(turns), 'turn_stocks': int(turns.code.nunique()),
               'shape_counts': events['shape'].value_counts().to_dict(),
               'protected_files_unchanged': unchanged, 'no_future_bars_used_for_classification': True,
               'no_trading_backtest_executed': True, 'elapsed_seconds': round(time.monotonic() - t0, 1),
               'shape_parameters': {'peak_lookback': 60, 'peak_age_min': 3, 'peak_age_max': 55, 'drawdown_min_pct': 8, 'low20_distance_max_pct': 18, 'below_ma20_in_prior_10_min': 3, 'ma5_turn_lookback': 5, 'prior_rise_min_pct': 15, 'deep_drawdown_pct': 20},
               'calendar_source': 'https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml'}
    assert unchanged, 'a protected configuration or stock list changed during the scan'
    (out / 'scan-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    charts.sort(key=lambda r: (r['turn_confirmed'], r['shape_score'], r['date']), reverse=True)
    (out / 'chart-data.json').write_text(json.dumps(charts, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    write_report(out, events, summary, charts)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
