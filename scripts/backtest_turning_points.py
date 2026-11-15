"""Compare daily turning-point entries using project execution and cash services.

Each historical setup is an independent 100,000 CNY trial. This is a matched
entry/exit experiment, not a combined portfolio or a forecast of future returns.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config_loader import ConfigLoader
from src.utils.tdx_provider import TdxProvider
from src.utils.runtime_params import parameter_overrides
from src.services.order_matcher import OrderMatcher
from src.services.account_ledger import AccountLedger
from src.services.risk_audit_log import RiskAuditLog
from src.core.execution_engine import ExecutionEngine


PROFILES = [
    dict(id='fixed15', label='固定止盈15%/止损8%', kind='fixed', stop=.08, take=.15),
    dict(id='fixed20', label='固定止盈20%/止损8%', kind='fixed', stop=.08, take=.20),
    dict(id='trailing', label='滑动8%，盈利10%后收紧至5%', kind='trailing', stop=.08, arm=.10, tight=.05),
    dict(id='drawdown', label='盈利10%后高点回撤8%退出', kind='drawdown', stop=.08, arm=.10, trail=.08),
    dict(id='ma10', label='MA10次日开盘退出/止损8%', kind='ma10', stop=.08),
]
MODEL_LABELS = {'close': '收盘>3%按收盘成交（理论）', 'intraday': '盘中>3%突破成交', 'next_open': '收盘确认后次日开盘（补充）'}
COSTS = {'execution.slippage': .001, 'trading_cost.commission_rate': .00025,
         'trading_cost.min_commission': 5., 'trading_cost.stamp_duty': .0005,
         'trading_cost.transfer_fee': 0.}


def in_scope(code, name):
    main = re.fullmatch(r'(?:60[0135]\d{3}\.SH|00[0123]\d{3}\.SZ)', str(code)) is not None
    return main and re.search(r'ST|退', str(name), flags=re.I) is None


def cents(value):
    return int(round(float(value) * 100))


def first_tick_above_pct(previous_close, percent=3):
    # Decimal arithmetic preserves strict inequalities for fractional thresholds.
    target = Decimal(cents(previous_close)) * (100 + Decimal(str(percent))) / 100
    return (int(target.to_integral_value(rounding=ROUND_FLOOR)) + 1) / 100


def entry_label(model, threshold=3):
    return MODEL_LABELS[model].replace('3%', f'{threshold:g}%')


def daily_bars(frame):
    frame = frame.sort_values('dt').reset_index(drop=True).copy()
    frame['ma10'] = frame.close.rolling(10).mean()
    bars, previous = {}, None
    for history_index, r in enumerate(frame.itertuples()):
        day = str(r.dt.date())
        b = dict(code=r.code, dt=r.dt, open=float(r.open), high=float(r.high), low=float(r.low),
                 close=float(r.close), vol=float(r.vol), amount=float(r.amount),
                 ma10=None if pd.isna(r.ma10) else float(r.ma10), history_sessions=history_index)
        if previous is not None:
            pc = cents(previous)
            up = max(pc + 1, (pc * 110 + 50) // 100)
            down = min(pc - 1, (pc * 90 + 50) // 100)
            b.update(previous_close=previous, up_limit=up / 100, down_limit=down / 100,
                     close_gt3=cents(b['close']) * 100 > pc * 103,
                     high_gt3=cents(b['high']) * 100 > pc * 103,
                     is_limit_up=cents(b['close']) >= up, is_limit_down=cents(b['close']) <= down,
                     price_reference_valid=cents(b['high']) <= up + 1 and cents(b['low']) >= down - 1)
        else:
            b.update(previous_close=None, price_reference_valid=False, close_gt3=False, high_gt3=False)
        previous = b['close']
        bars[day] = b
    return bars


def buy_order(code, model, bar, capital, threshold=3):
    mode = {'close': 'at_close', 'intraday': 'buy_stop', 'next_open': 'at_open'}[model]
    trigger = first_tick_above_pct(bar['previous_close'], threshold)
    price = bar['close'] if mode == 'at_close' else max(bar['open'], trigger) if mode == 'buy_stop' else bar['open']
    qty = int(capital / (price * (1 + COSTS['execution.slippage'])) // 100) * 100
    return dict(code=code, strategy_id='turning', direction='BUY', execution_mode=mode,
                price=price, trigger_price=trigger, qty=qty)


def entry_rejection(model, b):
    if b['vol'] <= 0:
        return 'suspended_or_no_volume'
    if not b['price_reference_valid']:
        return 'unverified_price_reference'
    if b.get('history_sessions', 60) < 60:
        return 'insufficient_prelisting_history'
    if model == 'close' and b.get('is_limit_up'):
        return 'close_limit_up'
    if model != 'close' and b['open'] >= b['up_limit'] - 1e-8:
        return 'open_at_limit_up'
    return None


def find_entry(code, signal_day, model, bars, calendar, capital=100000, threshold=3):
    if model not in MODEL_LABELS or not 0 < threshold < 10:
        raise ValueError('supported entry model and threshold between 0 and 10 required')
    signal_index = calendar.index(signal_day)
    attempts = []
    observed = 0
    matcher = OrderMatcher()
    for offset in range(1, 5):
        j = signal_index + offset
        if j >= len(calendar):
            break
        observed += 1
        day = calendar[j]
        b = bars.get(day)
        if not b:
            attempts.append(dict(date=day, reason='missing_stock_bar'))
            continue
        trigger_price = first_tick_above_pct(b['previous_close'], threshold) if b.get('previous_close') else None
        triggered = trigger_price is not None and cents(b['high'] if model == 'intraday' else b['close']) >= cents(trigger_price)
        if not triggered:
            continue
        reject = entry_rejection('close' if model == 'next_open' else model, b)
        attempts.append(dict(date=day, reason=reject or 'triggered', close_pct=(b['close']/b['previous_close']-1)*100,
                             high_pct=(b['high']/b['previous_close']-1)*100))
        if reject:
            continue
        trigger_day, execution_index = day, j
        if model == 'next_open':
            execution_index = j + 1
            if execution_index >= len(calendar):
                return dict(status='entry_pending_next_open', observed_days=observed, attempts=attempts)
            day = calendar[execution_index]
            b = bars.get(day)
            if not b:
                attempts.append(dict(date=day, reason='missing_next_open_bar'))
                continue
            reject = entry_rejection(model, b)
            if reject:
                attempts.append(dict(date=day, reason=reject))
                continue
        order = buy_order(code, model, b, capital, threshold)
        ok, fill = matcher.match_order(order, b)
        if not ok or order['qty'] < 100:
            attempts.append(dict(date=day, reason='matching_or_capital_rejection'))
            continue
        return dict(status='entry_available', entry_date=day, trigger_date=trigger_day,
                    entry_index=execution_index, observed_days=observed, entry_fill=fill,
                    order=order, attempts=attempts)
    return dict(status='no_entry' if observed == 4 else 'observation_incomplete', observed_days=observed, attempts=attempts)


def exit_levels(profile, entry_price, prior_peak):
    stop = entry_price * (1 - profile['stop'])
    take = entry_price * (1 + profile['take']) if profile['kind'] == 'fixed' else None
    if profile['kind'] == 'trailing':
        trail = profile['tight'] if prior_peak >= entry_price * (1 + profile['arm']) else profile['stop']
        stop = max(stop, prior_peak * (1 - trail))
    elif profile['kind'] == 'drawdown' and prior_peak >= entry_price * (1 + profile['arm']):
        stop = max(stop, prior_peak * (1 - profile['trail']))
    return math.floor(stop * 100 + 1e-8) / 100, None if take is None else math.ceil(take * 100 - 1e-8) / 100


def choose_exit(bar, stop, take):
    # Opening gaps are known first. Only ambiguous intraday paths use stop first.
    if bar['open'] <= stop:
        return 'sell_stop', stop, 'gap_stop', False
    if take is not None and bar['open'] >= take:
        return 'sell_limit', take, 'gap_take_profit', False
    stop_hit = bar['low'] <= stop
    take_hit = take is not None and bar['high'] >= take
    if stop_hit:
        return 'sell_stop', stop, 'stop_loss_or_trailing', bool(take_hit)
    if take_hit:
        return 'sell_limit', take, 'take_profit', False
    return None


def run_trial(code, model, entry, profile, bars, calendar, capital=100000, max_hold=40):
    account = AccountLedger(capital)
    audit = RiskAuditLog()
    engine = ExecutionEngine(account, OrderMatcher(), audit)
    entry_bar = bars[entry['entry_date']]
    assert engine.execute_order('turning', entry['order'], entry_bar, account=account)
    bought = account.transactions[-1]
    price, quantity = bought['price'], bought['quantity']
    assert abs(price - entry['entry_fill']) < 1e-8
    invested = bought['amount'] + bought['cost']
    prior_peak = max(price, entry_bar['high']) if model in {'intraday', 'next_open'} else price
    pending_reason = None
    blocked_sells = 0
    ambiguous = 0
    last_day = entry['entry_date']
    hold = 0
    result = dict(status='open_at_data_end', entry_date=last_day, entry_price=price, quantity=quantity,
                  entry_commission=bought['commission'], entry_cost=bought['cost'], invested=invested,
                  exit_date=None, exit_price=None, net_return_pct=None, equity_return_pct=None,
                  holding_days=0, delayed_sell_days=0, ambiguous_both_hit_days=0,
                  exit_reason=None, realized_pnl=None)
    account.update_daily_nav(entry_bar['dt'], quantity * entry_bar['close'])
    for j in range(entry['entry_index'] + 1, len(calendar)):
        day = calendar[j]
        hold = j - entry['entry_index']
        b = bars.get(day)
        if not b:
            continue
        last_day = day
        if not b['price_reference_valid']:
            # Without corporate-action/reference data, refuse a normal ROI claim.
            result.update(status='excluded_unverified_price_event', exit_reason='price_reference_outside_10pct_band')
            break
        selected = None
        if pending_reason:
            selected = ('at_open', b['open'], pending_reason, False)
        else:
            stop, take = exit_levels(profile, price, prior_peak)
            selected = choose_exit(b, stop, take)
            if not selected and hold >= max_hold:
                selected = ('at_close', b['close'], f'max_holding_{max_hold}d', False)
        if selected:
            mode, trigger, reason, both = selected
            ambiguous += int(both)
            sell = dict(strategy_id='turning', code=code, direction='SELL', qty=quantity,
                        price=trigger, trigger_price=trigger, execution_mode=mode)
            if engine.execute_order('turning', sell, b, account=account):
                sold = account.transactions[-1]
                pnl = account.cash - capital
                result.update(status='closed', exit_date=day, exit_price=sold['price'],
                              exit_reason=reason, realized_pnl=pnl,
                              net_return_pct=pnl / invested * 100, equity_return_pct=pnl / capital * 100,
                              exit_cost=sold['cost'], exit_commission=sold['commission'],
                              stamp_duty=sold['stamp_duty'], holding_days=hold,
                              total_fees=account.total_commission + account.total_stamp_duty + account.total_transfer_fee)
                account.update_daily_nav(b['dt'], 0)
                break
            blocked_sells += 1
            pending_reason = 'delayed_' + reason if not reason.startswith('delayed_') else reason
        # Today's new high can change only tomorrow's planned trailing stop.
        prior_peak = max(prior_peak, b['high'])
        account.update_daily_nav(b['dt'], quantity * b['close'])
        if not pending_reason and profile['kind'] == 'ma10' and b['ma10'] is not None and b['close'] < b['ma10']:
            pending_reason = 'ma10_next_open'
    result.update(last_observed_date=last_day, holding_days=hold, delayed_sell_days=blocked_sells,
                  ambiguous_both_hit_days=ambiguous)
    if result['status'] == 'open_at_data_end':
        final = bars[last_day]
        result['unrealized_return_pct'] = (account.cash + quantity * final['close'] - capital) / invested * 100
        estimated_cost, _, _, _ = account.calculate_cost(quantity * final['close'], 'SELL', final['close'], quantity)
        result['cohort_return_pct'] = (account.cash + quantity * final['close'] - estimated_cost - capital) / invested * 100
    elif result['status'] == 'closed':
        result['cohort_return_pct'] = result['net_return_pct']
    else:
        result['cohort_return_pct'] = None
    return result


def summarize_trials(trials):
    records = []
    for (model, threshold, profile), g in trials.groupby(['entry_model', 'entry_threshold_pct', 'exit_profile'], sort=False):
        closed = g[g.status == 'closed']
        returns = closed.net_return_pct.dropna()
        cohort = g.cohort_return_pct.dropna()
        records.append(dict(entry_model=model, entry_threshold_pct=threshold, entry_label=entry_label(model, threshold), exit_profile=profile,
                            exit_label=next(p['label'] for p in PROFILES if p['id'] == profile),
                            trials=len(g), closed=len(closed), open_at_end=int((g.status == 'open_at_data_end').sum()),
                            excluded=int((g.status == 'excluded_unverified_price_event').sum()),
                            mean_net_return_pct=None if returns.empty else float(returns.mean()),
                            median_net_return_pct=None if returns.empty else float(returns.median()),
                            win_rate_pct=None if returns.empty else float(returns.gt(0).mean()*100),
                            cohort_mean_return_pct=None if cohort.empty else float(cohort.mean()),
                            cohort_count=len(cohort),
                            mean_holding_days=None if closed.empty else float(closed.holding_days.mean()),
                            total_sell_delays=int(g.delayed_sell_days.sum()),
                            ambiguous_both_hit_days=int(g.ambiguous_both_hit_days.sum())))
    return pd.DataFrame(records)


def write_html(out, summary, entries, stats):
    def table(frame):
        return frame.to_html(index=False, border=0, float_format=lambda n: f'{n:.2f}')
    shown = stats[['entry_label', 'exit_label', 'trials', 'closed', 'open_at_end', 'excluded', 'mean_net_return_pct', 'median_net_return_pct', 'win_rate_pct', 'cohort_mean_return_pct', 'mean_holding_days']].copy()
    shown.columns = ['买入方式', '卖出方式', '买入次数', '已平仓', '期末未平仓', '异常数据剔除', '已平仓平均净收益%', '中位净收益%', '胜率%', '含未平仓估值均值%', '平均持有日数']
    if 'entry_date' not in entries:
        entries = entries.assign(entry_date=None)
    ent = entries.groupby(['entry_model','entry_threshold_pct'],sort=False).agg(已买入=('entry_date', lambda x:x.notna().sum()),信号数=('status','size')).reset_index()
    ent['买入方式'] = ent.apply(lambda r: entry_label(r.entry_model, r.entry_threshold_pct), axis=1)
    ent = ent[['买入方式','信号数','已买入']]
    rules = '<br>'.join(p['label'] for p in PROFILES)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>拐弯买入点：买入与止盈止损对比</title><style>body{{background:#111721;color:#e7edf5;font:15px/1.75 "Microsoft YaHei",sans-serif;margin:0}}main{{max-width:1450px;margin:auto;padding:28px}}h1{{font-size:28px}}section{{background:#1b2636;padding:22px;margin:20px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;border-bottom:1px solid #364357;text-align:left}}th{{color:#81c8f5}}a{{color:#80c3ff}}.scroll{{overflow:auto}}small,p{{color:#b7c5d8}}</style><main><h1>拐弯买入点 · 买入与止盈止损对比</h1><p>沪深主板 / 排除名单中的ST、*ST与退市名称 / 本地不复权日K / 截至{summary['actual_data_end']}</p><section><b>样本：</b>{summary['eligible_signals']}次信号，{summary['eligible_stocks']}只股票。来自前轮同时满足箭头公式及初拐形态的候选；股票名单日期2026-08-05，历史逐日ST状态未独立验证。<br><b>观察：</b>信号后的4个市场交易日，不含信号日；收盘严格大于3%，盘中分别严格大于3%、4%、4.5%、5%，相对触发日上一根收盘价计算，停牌不延长观察窗口。<br><b>收盘成交：</b>当日收盘涨幅&gt;3%，按当日收盘价模拟，收盘涨停一律拒绝；该模型需要尾盘或适用的盘后机制及成交量验证，是理论基准。<br><b>盘中突破：</b>预先设定前收盘价×(1+所选阈值)以上的第一个0.01元报价，最高价达到时模拟买入；跳空高于阈值按实际开盘价。一字板、涨停价开盘均拒绝。盘中低于涨停价买入后收盘封板可以成交，不用收盘封板倒推整天买不到。<br><b>次日开盘：</b>另列收盘确认后的次日开盘成交敏感性对照；仍跳过收盘封板的确认日。<br><b>成本：</b>每次独立投入上限10万元；佣金万2.5、最低5元，卖出印花税0.05%，双边滑点0.1%，过户费沿用项目配置为0。T+1和100股整手由项目执行/账本模块处理。<br><b>每日撮合：</b>移动止损只用前一天已知高点；一天内双触发按止损优先，开盘越过阈值时优先按开盘成交。跌停不能卖出时推迟到之后可成交的开盘。最长持有40个市场交易日。</section><section><h2>买入成交对比</h2>{table(ent)}</section><section><h2>五组卖出方式</h2>{rules}<p>已平仓均值及胜率仅计已卖出样本；另列含未平仓估值的均值，按期末收盘估值并扣估计卖出费。价格参考异常样本剔除，估值不代表能成交。每条信号为独立试验，股票或持仓区间可能重复；这些均值不是组合收益率、年化收益率或实盘成交保证。</p></section><section><h2>净收益对比</h2><div class="scroll">{table(shown)}</div></section><section><h2>配对比较</h2><p>paired-comparison.csv 只对同一信号、同一卖出方式下两种买入都已平仓的样本比较，以减少不同买入样本集合造成的偏差。</p><a href="paired-comparison.csv">配对结果</a> · <a href="entry-events.csv">全部买入观察记录</a> · <a href="entry-attempts.csv">逐日触发及拒绝原因</a> · <a href="trade-trials.csv">逐笔交易与止盈止损明细</a> · <a href="performance-summary.csv">汇总表</a> · <a href="input-setups.csv">本轮主板非ST候选</a></section><section><h2>边界与项目审计</h2><p>使用项目的TdxProvider、OrderMatcher、ExecutionEngine及AccountLedger实际逐笔执行；本轮采用专用回放器明确成交时点，避免原主引擎“收盘信号按同日开盘价成交”的问题。新增显式成交模式供本轮调用，旧调用路径保留；不代表已修复全部既有策略的主引擎时序问题。</p><p>行情最晚9月29日，缺后续完整记录；无独立公司行动/复权及逐日ST状态数据。10%涨跌停价按上一根原价收盘价四舍五入至0.01元推算；超过该价格带的记录标为参考价异常，持仓遇到时剔除普通收益结论。分红、送转和无法识别的小额除息仍未计入，结果为原价模型下的历史初测。日线不足以验证触发先后或排队成交。</p><a href="回测结果与规则审计.md">完整说明与规则审计</a> · <a href="backtest-summary.json">运行记录</a></section></main></html>'''
    (out / '买入与止盈止损对比.html').write_text(html, encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setups', type=Path, default=ROOT/'reports/turning-point-2026-10-08/初拐候选.csv')
    p.add_argument('--out', type=Path, default=ROOT/'reports/turning-point-backtest-2026-10-08')
    p.add_argument('--end', default='2026-09-29')
    p.add_argument('--capital', type=float, default=100000)
    p.add_argument('--max-hold', type=int, default=40)
    p.add_argument('--sample', type=int, default=0)
    p.add_argument('--intraday-thresholds', type=float, nargs='+', default=[3, 4, 4.5, 5])
    args = p.parse_args()
    if args.capital <= 0 or args.max_hold < 1:
        p.error('positive capital and holding period required')
    if any(not 0 < x < 10 for x in args.intraday_thresholds):
        p.error('intraday thresholds must be between 0 and 10')
    variants = [('close',3.),('next_open',3.)] + [('intraday',x) for x in dict.fromkeys(args.intraday_thresholds)]
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    config['data_provider']['local_cache_enabled'] = False
    protected = [ROOT/'config.json', ROOT/'config.private.json', ROOT/'data/stock_list.csv']
    before = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in protected if f.exists()}
    inputs = pd.read_csv(args.setups, dtype={'code':str})
    mask = inputs.apply(lambda r: in_scope(r['code'], r['name']), axis=1)
    excluded = inputs[~mask].copy(); setups = inputs[mask].copy()
    if args.sample:
        setups = setups.head(args.sample)
    setups.to_csv(out/'input-setups.csv', encoding='utf-8-sig', index=False)
    excluded.to_csv(out/'excluded-scope.csv', encoding='utf-8-sig', index=False)
    entries, attempts, trials, data_failures = [], [], [], []
    t0 = time.monotonic()
    with ConfigLoader.runtime_snapshot(config), parameter_overrides(COSTS):
        provider = TdxProvider(); provider._cache_enabled = False
        reference = provider._normalize_daily_df(provider._reader_daily('000001'), '000001.SZ')
        reference = reference[(reference.dt >= '2025-01-01') & (reference.dt <= args.end)]
        calendar = [str(x.date()) for x in reference.dt]
        for i, (code, signals) in enumerate(setups.groupby('code', sort=False)):
            try:
                frame = provider._normalize_daily_df(provider._reader_daily(provider._raw_symbol(code)), code)
                frame = frame[(frame.dt >= '2025-01-01') & (frame.dt <= args.end)]
                bars = daily_bars(frame)
                if not bars:
                    raise ValueError('empty bars')
                for setup in signals.to_dict('records'):
                    signal_day = setup['date']
                    if signal_day not in calendar or signal_day not in bars:
                        raise ValueError(f'missing setup date {signal_day}')
                    for model, threshold in variants:
                        entry = find_entry(code, signal_day, model, bars, calendar, args.capital, threshold)
                        base = dict(code=code, name=setup['name'], signal_date=signal_day,
                                    signal_shape=setup['shape'], entry_model=model, entry_threshold_pct=threshold,
                                    entry_variant=f'{model}_{threshold:g}')
                        entries.append({**base, **{k:v for k,v in entry.items() if k not in {'attempts','order','entry_index'}}})
                        attempts.extend({**base, **a} for a in entry['attempts'])
                        if entry['status'] != 'entry_available':
                            continue
                        for profile in PROFILES:
                            trial = run_trial(code, model, entry, profile, bars, calendar, args.capital, args.max_hold)
                            trials.append({**base, 'exit_profile':profile['id'], **trial})
            except Exception as exc:
                data_failures.append(dict(code=code, error=f'{type(exc).__name__}: {exc}'))
                print(json.dumps(data_failures[-1], ensure_ascii=False), flush=True)
            if (i+1)%50 == 0:
                print(json.dumps(dict(stocks=i+1, signals=len(entries)//len(variants), trials=len(trials), elapsed=round(time.monotonic()-t0,1))), flush=True)
    if data_failures:
        (out/'data-failures.json').write_text(json.dumps(data_failures,ensure_ascii=False,indent=2),encoding='utf-8')
        raise RuntimeError('data failure; do not interpret incomplete batch as completed')
    entries, attempts, trials = pd.DataFrame(entries), pd.DataFrame(attempts), pd.DataFrame(trials)
    stats = summarize_trials(trials)
    paired = []
    for profile in PROFILES:
        g = trials[(trials.exit_profile==profile['id']) & (trials.status=='closed')]
        pairs = g.pivot(index=['code','signal_date'], columns='entry_variant', values='net_return_pct')
        for threshold in args.intraday_thresholds:
            variant = f'intraday_{threshold:g}'
            if 'close_3' in pairs and variant in pairs:
                both = pairs[['close_3',variant]].dropna()
                paired.append(dict(exit_profile=profile['id'],exit_label=profile['label'],intraday_threshold_pct=threshold,pairs=len(both),
                                   mean_close_net_pct=float(both['close_3'].mean()), mean_intraday_net_pct=float(both[variant].mean()),
                                   mean_intraday_minus_close_pp=float((both[variant]-both['close_3']).mean()),
                                   intraday_better_pct=float((both[variant]>both['close_3']).mean()*100)))
    for filename, frame in [('entry-events.csv',entries),('entry-attempts.csv',attempts),('trade-trials.csv',trials),('performance-summary.csv',stats),('paired-comparison.csv',pd.DataFrame(paired))]:
        frame.to_csv(out/filename,encoding='utf-8-sig',index=False,float_format='%.8f')
    entry_summary = []
    for (model,threshold),g in entries.groupby(['entry_model','entry_threshold_pct'],sort=False):
        a = attempts[(attempts.entry_model==model)&(attempts.entry_threshold_pct==threshold)]
        entry_summary.append(dict(model=model,threshold_pct=threshold,label=entry_label(model,threshold),setup_count=len(g),
                                  entries=int((g.status=='entry_available').sum()),status_counts=g.status.value_counts().to_dict(),
                                  attempt_reason_counts=a.reason.value_counts().to_dict(),
                                  unique_signals_with_close_limit_up=int(a[a.reason=='close_limit_up'][['code','signal_date']].drop_duplicates().shape[0])))
    unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in before.items())
    assert unchanged
    summary=dict(eligible_signals=len(setups),eligible_stocks=int(setups.code.nunique()),original_signals=len(inputs),
                 excluded_scope_signals=len(excluded),actual_data_end=calendar[-1],observation_days=4,
                 include_signal_day=False,close_breakout_percent=3,intraday_thresholds=args.intraday_thresholds,capital_per_independent_trial=args.capital,max_hold=args.max_hold,
                 price_mode='none',stock_list_date='2026-08-05',historical_st_status_verified=False,
                 limit_price_source='normal_main_board_10pct_from_previous_raw_close_round_half_up',
                 shared_services=['TdxProvider','OrderMatcher','ExecutionEngine','AccountLedger'],
                 uses_main_backtest_engine=False,reason='explicit execution timing required; legacy main engine same-bar open lookahead is audited',
                 exit_profiles=PROFILES,cost_overrides=COSTS,project_original_stamp_duty=config.get('trading_cost',{}).get('stamp_duty'),
                 entry_summary=entry_summary,trials=len(trials),trial_status_counts=trials.status.value_counts().to_dict(),
                 protected_files_unchanged=unchanged,data_failures=data_failures,elapsed_seconds=round(time.monotonic()-t0,1))
    (out/'backtest-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_html(out,summary,entries,stats)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(stats.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
