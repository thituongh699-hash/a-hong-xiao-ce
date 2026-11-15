"""Training-only parameter search and frozen holdout for the turning setup.

The reusable search engine is independent of this first strategy adapter.
Daily-bar fills are modeled; none of these results is a portfolio return.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import pandas as pd

from .search import ParameterOptimizer
from scripts.backtest_turning_points import (
    COSTS, daily_bars, entry_label, find_entry, in_scope, run_trial,
)
from src.utils.config_loader import ConfigLoader
from src.utils.runtime_params import parameter_overrides
from src.utils.tdx_provider import TdxProvider

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_VERSION = 'turning-point-v3-bounded-cohort-union-calendar-research-acceptance'
CALENDAR_SOURCE = 'local_bar_union_unverified_exchange_calendar'


def default_config():
    return dict(train_start='2026-04-08', train_end='2026-07-31',
                validation_start='2026-08-03', end='2026-09-29',
                intraday_thresholds=[3,4,4.5,5], close_thresholds=[3],
                entry_models=['intraday','next_open'],
                exit_kinds=['fixed','trailing','drawdown','ma10'],
                stops_pct=[4,6,8,10], takes_pct=[8,12,15,20],
                arms_pct=[5,10,15], trails_pct=[5,8], tight_pct=[3,5],
                max_holds=[20,40], max_trials=1000, min_trades=30, seed=42,
                objective='downside_adjusted', capital=100000, risk_penalty=.25)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('参数设置必须是对象')
    result = default_config()
    unknown = set(config)-set(result)
    if unknown:
        raise ValueError('未知参数：'+', '.join(sorted(unknown)))
    result.update(config)
    for key in ('train_start','train_end','validation_start','end'):
        try:
            result[key] = date.fromisoformat(str(result[key])).isoformat()
        except ValueError as exc:
            raise ValueError(f'{key} 日期格式应为 YYYY-MM-DD') from exc
    if not result['train_start'] <= result['train_end'] < result['validation_start'] <= result['end']:
        raise ValueError('训练期和验证期必须按时间先后分开')
    lists = dict(intraday_thresholds=(0,10),close_thresholds=(0,10),stops_pct=(0,30),
                 takes_pct=(0,100),arms_pct=(0,50),trails_pct=(0,30),tight_pct=(0,30),max_holds=(1,120))
    for key,(low,high) in lists.items():
        values = result[key]
        if not isinstance(values,list) or not values or len(values)>20:
            raise ValueError(f'{key} 需要 1 至 20 个数值')
        clean=[]
        for x in values:
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
                raise ValueError(f'{key} 含无效数值')
            if key=='max_holds':
                if int(x)!=x or not low<=x<=high:
                    raise ValueError('持仓天数应为 1 至 120 的整数')
                x=int(x)
            elif not low<x<high:
                raise ValueError(f'{key} 数值必须大于 {low} 且小于 {high}')
            if x not in clean: clean.append(x)
        result[key]=clean
    for key,allowed in [('entry_models',{'close','intraday','next_open'}),('exit_kinds',{'fixed','trailing','drawdown','ma10'})]:
        values=result[key]
        if not isinstance(values,list) or not values or any(x not in allowed for x in values):
            raise ValueError(f'{key} 含不支持的方式')
        result[key]=list(dict.fromkeys(values))
    for key,low,high in [('max_trials',1,5000),('min_trades',1,5000),('seed',0,2**31-1)]:
        x=result[key]
        if isinstance(x,bool) or not isinstance(x,int) or not low<=x<=high:
            raise ValueError(f'{key} 必须是 {low} 至 {high} 的整数')
    for key,low,high in [('capital',1000,100000000),('risk_penalty',0,10)]:
        x=result[key]
        if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or not low<=x<=high:
            raise ValueError(f'{key} 数值范围为 {low} 至 {high}')
    if result['objective'] not in {'downside_adjusted','mean_signal_return_pct'}:
        raise ValueError('不支持的优化目标')
    return result


def parameter_space(config):
    exits=[]
    for stop in config['stops_pct']:
        if 'fixed' in config['exit_kinds']:
            exits.extend(dict(kind='fixed',stop=stop/100,take=take/100) for take in config['takes_pct'])
        if 'trailing' in config['exit_kinds']:
            exits.extend(dict(kind='trailing',stop=stop/100,arm=arm/100,tight=tight/100)
                         for arm in config['arms_pct'] for tight in config['tight_pct'] if tight<=stop)
        if 'drawdown' in config['exit_kinds']:
            exits.extend(dict(kind='drawdown',stop=stop/100,arm=arm/100,trail=trail/100)
                         for arm in config['arms_pct'] for trail in config['trails_pct'])
        if 'ma10' in config['exit_kinds']:
            exits.append(dict(kind='ma10',stop=stop/100))
    result=[]
    for model in config['entry_models']:
        thresholds=config['intraday_thresholds'] if model=='intraday' else config['close_thresholds']
        for threshold in thresholds:
            for profile in exits:
                for hold in config['max_holds']:
                    result.append(dict(entry_model=model,threshold_pct=threshold,exit=profile,max_hold=hold))
                    if len(result)>100000:
                        raise ValueError('参数组合超过 100000，请缩小范围')
    if not result:
        raise ValueError('没有有效参数组合')
    return result


def profile_label(profile):
    kind=profile['kind']; stop=profile['stop']*100
    if kind=='fixed': return f"固定止盈{profile['take']*100:g}% / 止损{stop:g}%"
    if kind=='trailing': return f"滑动{stop:g}%，盈利{profile['arm']*100:g}%后收紧{profile['tight']*100:g}%"
    if kind=='drawdown': return f"止损{stop:g}%，盈利{profile['arm']*100:g}%后回撤{profile['trail']*100:g}%"
    return f'MA10次日开盘退出 / 止损{stop:g}%'


@dataclass
class TurningDataset:
    setups: pd.DataFrame
    calendar: list
    bars: dict
    fingerprint: str
    source_config: dict
    actual_end: str


def load_dataset(end, setups_path=None):
    path=Path(setups_path or ROOT/'reports/turning-point-2026-10-08/初拐候选.csv')
    setups=pd.read_csv(path,dtype={'code':str})
    setups=setups[setups.apply(lambda r:in_scope(r.code,r['name']),axis=1)].copy()
    config=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    config['data_provider']['local_cache_enabled']=False
    digest=hashlib.sha256(path.read_bytes()); bars={}; calendar_days=set()
    with ConfigLoader.runtime_snapshot(config):
        provider=TdxProvider(); provider._cache_enabled=False
        for code in setups.code.unique():
            frame=provider._normalize_daily_df(provider._reader_daily(provider._raw_symbol(code)),code)
            frame=frame[(frame.dt>='2025-01-01')&(frame.dt<=end)]
            if frame.empty: raise ValueError(f'{code} 本地行情为空')
            digest.update(code.encode())
            digest.update(pd.util.hash_pandas_object(frame,index=False).values.tobytes())
            bars[code]=daily_bars(frame)
            calendar_days.update(bars[code])
    calendar=sorted(calendar_days)
    if not calendar: raise ValueError('候选股票本地行情日历为空')
    digest.update(CALENDAR_SOURCE.encode())
    digest.update('|'.join(calendar).encode())
    return TurningDataset(setups,calendar,bars,digest.hexdigest(),config,calendar[-1])


class TurningEvaluator:
    """A fixed cohort: incomplete windows and abnormal raw-price events excluded.

    No-entry signals contribute zero. Open positions use the period-end mark
    less estimated exit fees. Price-event exclusions are identical across all
    configurations evaluated in this period, avoiding closure-only selection.
    """
    def __init__(self,dataset,start,end,capital=100000,risk_penalty=.25,execution_buffer=1,max_hold_horizon=40):
        self.dataset=dataset; self.calendar=[d for d in dataset.calendar if d<=end]
        self.capital=capital; self.risk_penalty=risk_penalty; self.entries={}
        self.common_horizon=4+execution_buffer+max_hold_horizon
        candidates=dataset.setups[(dataset.setups.date>=start)&(dataset.setups.date<=end)]
        self.setups=[]; self.excluded=[]
        indices={d:i for i,d in enumerate(self.calendar)}
        for setup in candidates.to_dict('records'):
            code,day=setup['code'],setup['date']; reason=None
            index=indices.get(day)
            if index is None or day not in dataset.bars[code]:
                raise ValueError(f'{code} 信号日数据缺失 {day}')
            if index+4+execution_buffer>=len(self.calendar): reason='incomplete_observation_window'
            elif dataset.bars[code][day].get('history_sessions',0)<60: reason='less_than_60_history_sessions'
            elif any(not b['price_reference_valid'] for d,b in dataset.bars[code].items()
                     if day<d<=self.calendar[min(index+self.common_horizon,len(self.calendar)-1)]):
                reason='unverified_raw_price_event_after_signal'
            if reason: self.excluded.append({**setup,'reason':reason})
            else: self.setups.append(setup)
        if not self.setups: raise ValueError('区间内没有可用于比较的完整信号')

    def evaluate(self,params,details=False):
        model=params['entry_model']; threshold=params['threshold_pct']; profile=params['exit']
        returns=[]; trades=[]; statuses={}; attempts={}; observations=[]
        for setup in self.setups:
            code,day=setup['code'],setup['date']; bars=self.dataset.bars[code]
            key=(code,day,model,threshold)
            if key not in self.entries:
                self.entries[key]=find_entry(code,day,model,bars,self.calendar,self.capital,threshold)
            entry=self.entries[key]
            statuses[entry['status']]=statuses.get(entry['status'],0)+1
            for attempt in entry['attempts']:
                attempts[attempt['reason']]=attempts.get(attempt['reason'],0)+1
            observations.append(dict(code=code,name=setup['name'],signal_date=day,
                                     **{k:v for k,v in entry.items() if k not in {'attempts','order','entry_index'}}))
            if entry['status']!='entry_available':
                returns.append(0.); continue
            trial=run_trial(code,model,entry,profile,bars,self.calendar,self.capital,params['max_hold'])
            if trial['cohort_return_pct'] is None:
                raise ValueError('固定比较样本遇到未验证价格事件，禁止排名')
            returns.append(float(trial['cohort_return_pct']))
            trades.append(dict(code=code,name=setup['name'],signal_date=day,**trial))
        series=pd.Series(returns,dtype=float)
        closed=[x for x in trades if x['status']=='closed']
        open_trades=[x for x in trades if x['status']=='open_at_data_end']
        stock_counts=Counter(x['code'] for x in trades)
        week_counts=Counter(tuple(date.fromisoformat(x['signal_date']).isocalendar()[:2]) for x in trades)
        downside=float((series.clip(upper=0).pow(2).mean())**.5)
        metrics=dict(signals=len(self.setups),trades=len(trades),closed=len(closed),open_at_end=len(open_trades),
                     mean_signal_return_pct=float(series.mean()),median_signal_return_pct=float(series.median()),
                     mean_trade_cohort_return_pct=None if not trades else sum(x['cohort_return_pct'] for x in trades)/len(trades),
                     closed_mean_net_return_pct=None if not closed else sum(x['net_return_pct'] for x in closed)/len(closed),
                     closed_win_rate_pct=None if not closed else sum(x['net_return_pct']>0 for x in closed)/len(closed)*100,
                     downside_rms_pct=downside,downside_adjusted=float(series.mean())-self.risk_penalty*downside,
                     worst_signal_return_pct=float(series.min()),entry_status_counts=statuses,
                     attempt_reason_counts=attempts,excluded_fixed_cohort=len(self.excluded),
                      eligible_fraction_pct=len(trades)/len(self.setups)*100,
                      distinct_signal_stocks=len({x['code'] for x in self.setups}),
                      distinct_trade_stocks=len(stock_counts),distinct_trade_weeks=len(week_counts),
                      max_trades_per_stock=max(stock_counts.values(),default=0),
                      max_trades_per_week=max(week_counts.values(),default=0))
        return (metrics,trades,observations) if details else metrics


def research_acceptance(best, validation_metrics, config):
    """Descriptive gates only; they do not establish predictive efficacy."""
    objective=config['objective']
    train_score=float(best['score'])
    holdout_score=float(validation_metrics[objective])
    min_samples=config['min_trades']
    gates=dict(training_above_cash=train_score>0,holdout_above_cash=holdout_score>0,
               holdout_min_samples=validation_metrics['trades']>=min_samples)
    return dict(status='research_support_for_followup' if all(gates.values()) else 'unsupported_hypothesis',
                baseline='cash_0_pct_per_independent_signal',objective=objective,
                training_score=train_score,holdout_score=holdout_score,
                holdout_trades=validation_metrics['trades'],min_samples=min_samples,
                gates=gates,efficacy_claim=False)


def training_one_axis_sensitivity(ranked,best):
    """Inspect evaluated training candidates that change one parameter axis."""
    chosen=best['params']; fields=('entry_model','threshold_pct','exit','max_hold')
    output={}
    for field in fields:
        peers=[record for record in ranked if all(record['params'][key]==chosen[key] for key in fields if key!=field)]
        scores=[record['score'] for record in peers]
        output[field]=dict(evaluated=len(peers),positive_vs_cash=sum(score>0 for score in scores),
                           score_min=min(scores,default=None),score_max=max(scores,default=None),
                           alternatives=[dict(value=record['params'][field],score=record['score'])
                                         for record in peers[:5]])
    return output


def _write_json(path,payload):
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run_optimization(config,output_dir,progress=None,setups_path=None):
    cfg=validate_config(config); out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    protected=[ROOT/'config.json',ROOT/'config.private.json',ROOT/'data/stock_list.csv']
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.exists()}
    _write_json(out/'parameters.json',cfg)
    t0=time.monotonic(); dataset=load_dataset(cfg['end'],setups_path)
    if cfg['train_end']>=dataset.actual_end or cfg['validation_start']>dataset.actual_end:
        raise ValueError('训练/验证分段超出可用行情')
    execution_buffer=int('next_open' in cfg['entry_models'])
    max_hold_horizon=max(cfg['max_holds'])
    train=TurningEvaluator(dataset,cfg['train_start'],cfg['train_end'],cfg['capital'],cfg['risk_penalty'],execution_buffer,max_hold_horizon)
    validation=TurningEvaluator(dataset,cfg['validation_start'],dataset.actual_end,cfg['capital'],cfg['risk_penalty'],execution_buffer,max_hold_horizon)
    space=parameter_space(cfg)
    code_paths=[Path(__file__),ROOT/'scripts/backtest_turning_points.py',ROOT/'src/services/order_matcher.py',ROOT/'src/core/execution_engine.py',ROOT/'src/services/account_ledger.py',ROOT/'src/optimization/search.py']
    code_fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in code_paths)).hexdigest()
    config_fingerprint=hashlib.sha256(json.dumps(dataset.source_config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    context=dict(adapter=ADAPTER_VERSION,dataset=dataset.fingerprint,code=code_fingerprint,project_config=config_fingerprint,settings=cfg,costs=COSTS,
                  calendar_source=CALENDAR_SOURCE,common_exclusion_horizon_sessions=train.common_horizon,
                 training_calendar_end=train.calendar[-1],training_signals=len(train.setups),
                  selection='training_only; no_entry_zero; bounded_fixed_cohort; censored_mark_less_estimated_exit_fees')
    with ConfigLoader.runtime_snapshot(dataset.source_config), parameter_overrides(COSTS):
        result=ParameterOptimizer().run(space,train.evaluate,objective=cfg['objective'],
            max_trials=cfg['max_trials'],seed=cfg['seed'],checkpoint=out/'checkpoint.jsonl',context=context,
            evaluator_version=f'{ADAPTER_VERSION}:{code_fingerprint}',
            eligibility_version=f"min_trades={cfg['min_trades']};min_fill_pct=20;v1",
            eligible=lambda m:m['trades']>=cfg['min_trades'] and m['eligible_fraction_pct']>=20,
            progress=progress)
        if result.metadata['errors']:
            _write_json(out/'optimization-errors.json',[r for r in result.records if r['status']=='error'])
            raise RuntimeError('存在参数试验错误，停止最佳参数结论；详见 optimization-errors.json')
        if result.best is None:
            raise ValueError('没有参数组达到最少成交样本数')
        # Freeze before any holdout objective is evaluated.
        chosen=json.loads(json.dumps(result.best))
        _write_json(out/'frozen-best-training.json',chosen)
        val_metrics,val_trades,val_entries=validation.evaluate(chosen['params'],details=True)
        best_by_entry=[]
        seen=set()
        for record in result.ranked:
            key=(record['params']['entry_model'],record['params']['threshold_pct'])
            if key in seen: continue
            seen.add(key)
            metrics=validation.evaluate(record['params'])
            best_by_entry.append(dict(entry_model=key[0],threshold_pct=key[1],label=entry_label(*key),
                                      params=record['params'],training=record['metrics'],validation=metrics,
                                      selection='best_for_this_entry_on_training_only'))
        # Top 10 remains in training order; never promote validation winners.
        top_validation=[]
        for rank,record in enumerate(result.ranked[:10],1):
            top_validation.append(dict(training_rank=rank,trial_id=record['trial_id'],params=record['params'],
                                       training_score=record['score'],validation=validation.evaluate(record['params'])))
    records=[]
    for r in result.records:
        p=r['params']; records.append(dict(trial_id=r['trial_id'],status=r['status'],score=r['score'],error=r['error'],
            entry_model=p['entry_model'],threshold_pct=p['threshold_pct'],exit_label=profile_label(p['exit']),
            max_hold=p['max_hold'],params_json=json.dumps(p,ensure_ascii=False),**{k:v for k,v in r['metrics'].items() if not isinstance(v,dict)}))
    pd.DataFrame(records).to_csv(out/'training-grid.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(val_trades).to_csv(out/'validation-trades.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(val_entries).to_csv(out/'validation-entries.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(train.excluded).to_csv(out/'training-exclusions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(validation.excluded).to_csv(out/'validation-exclusions.csv',index=False,encoding='utf-8-sig')
    _write_json(out/'top-training-with-validation.json',top_validation)
    summary=dict(adapter=ADAPTER_VERSION,settings=cfg,metadata=result.metadata,actual_data_end=dataset.actual_end,
        calendar_source=CALENDAR_SOURCE,common_exclusion_horizon_sessions=train.common_horizon,
        exclusion_lookahead_caveat='fixed common horizon still uses post-signal prices; corporate actions and exchange calendars unverified',
        train_period=[cfg['train_start'],train.calendar[-1]],validation_period=[cfg['validation_start'],dataset.actual_end],
        training_signals=len(train.setups),validation_signals=len(validation.setups),
        training_exclusions=len(train.excluded),validation_exclusions=len(validation.excluded),
        total_parameter_combinations=len(space),best_training=chosen,validation=val_metrics,
        research_acceptance=research_acceptance(chosen,val_metrics,cfg),
        training_one_axis_sensitivity=training_one_axis_sensitivity(result.ranked,chosen),
        selected_entry_model_status='theoretical_close_fill' if chosen['params']['entry_model']=='close' else 'daily_bar_executable_model_unverified',
        close_entry_opt_in_theoretical='close' in cfg['entry_models'],
        best_by_entry=best_by_entry,elapsed_seconds=round(time.monotonic()-t0,2),
        candidate_universe_signals=len(dataset.setups),candidate_universe_stocks=int(dataset.setups.code.nunique()),
        price_mode='none',stock_list_date='2026-08-05',historical_st_status_verified=False,
        limit_bands_verified=False,corporate_actions_accounted=False,uses_main_backtest_engine=False,
        cost_overrides=COSTS,common_cohort_execution_buffer_days=execution_buffer,performance_unit='independent_signal_experiment_not_portfolio',
        validation_used_to_select=False,
        objective_description=f"每条完整信号等权；未买入=0；未平仓按期末收盘估值扣估计卖出费；分数={'平均收益' if cfg['objective']=='mean_signal_return_pct' else '平均收益-'+str(cfg['risk_penalty'])+'×亏损均方根'}",
        protected_files_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in before.items()))
    if not summary['protected_files_unchanged']:
        raise RuntimeError('运行期间项目配置或股票名单发生变化，需重新验证结果')
    _write_json(out/'optimization-summary.json',summary)
    write_report(out,summary,records,top_validation)
    return summary


def write_report(out,summary,records,top_validation):
    from html import escape
    best=summary['best_training']; p=best['params']; val=summary['validation']
    acceptance=summary['research_acceptance']
    acceptance_label=('可进入下一轮研究' if acceptance['status']=='research_support_for_followup'
                      else '当前假设未获支持')
    gate_names={'training_above_cash':'训练得分高于现金基准',
                'holdout_above_cash':'验证得分高于现金基准',
                'holdout_min_samples':'验证成交样本达到预设下限'}
    gates='；'.join(f"{gate_names[key]}：{'通过' if value else '未通过'}"
                   for key,value in acceptance['gates'].items())
    sensitivity=pd.DataFrame([
        dict(参数轴=key,训练邻近参数数=value['evaluated'],高于现金基准=value['positive_vs_cash'],
             最低训练得分=value['score_min'],最高训练得分=value['score_max'])
        for key,value in summary['training_one_axis_sensitivity'].items()
    ]).to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    research_section=f'<section><h2>研究验收：{acceptance_label}</h2><p>{escape(gates)}。现金基准为每信号0%；比较所选目标得分。即使全部通过，也只支持下一轮研究，不构成策略有效或可实盘的结论。</p><p>验证股票数 {val["distinct_trade_stocks"]}，单只股票成交占比最高 {(val["max_trades_per_stock"]/max(1,val["trades"])*100):.2f}%；信号周分组最高成交占比 {(val["max_trades_per_week"]/max(1,val["trades"])*100):.2f}%。集中样本需要额外验证。</p><h3>训练参数单轴敏感性</h3><div class="scroll">{sensitivity}</div><p>只比较训练期已评估的参数；验证期结果不用于选参。</p></section>'
    display=[]
    for r in summary['best_by_entry']:
        tr,va=r['training'],r['validation']
        display.append(dict(买入=r['label'],训练选定卖出=profile_label(r['params']['exit']),最长持有=r['params']['max_hold'],
            训练平均每信号收益=tr['mean_signal_return_pct'],验证平均每信号收益=va['mean_signal_return_pct'],
            验证每笔收益含估值=va['mean_trade_cohort_return_pct'],验证成交=va['trades'],验证已平仓=va['closed'],验证未平仓=va['open_at_end']))
    grid=pd.DataFrame(records).sort_values('score',ascending=False,na_position='last')
    names=['trial_id','status','entry_model','threshold_pct','exit_label','max_hold','trades','closed','open_at_end','mean_signal_return_pct','downside_adjusted']
    table=pd.DataFrame(display).to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    rows=grid[names].to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>拐弯买入点参数优化结果</title><style>body{{margin:0;background:#101722;color:#e2ebf5;font:15px/1.75 "Microsoft YaHei",sans-serif}}main{{max-width:1450px;margin:auto;padding:28px}}section{{background:#1b2738;padding:22px;margin:20px 0;border-radius:12px}}h1{{font-size:28px}}p{{color:#b9c8da}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;border-bottom:1px solid #35445a;text-align:left}}th{{color:#8fcfff}}a{{color:#8fcfff}}.scroll{{overflow:auto;max-height:650px}}input{{padding:9px;background:#101722;color:white;border:1px solid #526179;border-radius:6px;width:300px}}.cards{{display:flex;gap:24px;flex-wrap:wrap}}.metric{{font-size:25px;color:#95d7e8}}</style><main><h1>拐弯买入点 · 参数优化</h1><p>沪深主板，按2026-08-05股票名单剔除ST与退市名称 · 原价日K · 数据截至{summary['actual_data_end']}</p><section><div class="cards"><div>参数组合<div class="metric">{summary['metadata']['completed']}</div></div><div>训练信号<div class="metric">{summary['training_signals']}</div></div><div>验证信号<div class="metric">{summary['validation_signals']}</div></div><div>验证每信号均值<div class="metric">{val['mean_signal_return_pct']:.3f}%</div></div></div><p>训练期 {' 至 '.join(summary['train_period'])}；验证期 {' 至 '.join(summary['validation_period'])}。训练期内持仓只估值到训练截止日；选定参数写入冻结文件后才运行验证。</p></section><section><h2>训练选定的参数</h2><b>{escape(entry_label(p['entry_model'],p['threshold_pct']))} · {escape(profile_label(p['exit']))} · 最长持有{p['max_hold']}个交易日</b><p>训练得分 {best['score']:.3f}，训练每信号均值 {best['metrics']['mean_signal_return_pct']:.3f}%；验证每信号均值 {val['mean_signal_return_pct']:.3f}%，验证成交 {val['trades']} 次，已平仓 {val['closed']} 次，期末未平仓 {val['open_at_end']} 次。验证结果用于检验，未据此重新挑选参数。</p><p>{escape(summary['objective_description'])}。这是独立信号实验，信号和持仓可以重叠，不能把均值相加作为组合收益或年化收益。低于最少{summary['settings']['min_trades']}笔成交或20%成交比例的参数组不参与排名。</p></section>{research_section}<section><h2>各买入阈值：先在训练期选卖出参数，再验证</h2><div class="scroll">{table}</div><p>各行卖出参数可能不同；这张表比较整套规则。相同卖出方式的阈值对照请看上一份买入与止盈止损报告。</p></section><section><h2>全部训练参数组</h2><label for="search">筛选买入方式或卖出条件</label><br><input id="search" placeholder="例如 intraday、4.5、止损6%"><div class="scroll" id="grid">{rows}</div></section><section><h2>数据与成交规则</h2><p>箭头公式保持最高涨幅&gt;7.2%、1%&lt;收盘涨幅&lt;4.8%；初拐形态沿用前轮的可解释筛选定义。观察信号后4个本地行情交易日；日历取所有候选股票行情日期的并集，尚未经交易所日历核验。默认只搜索盘中突破和次日开盘；同日收盘成交需主动开启，属于理论对照。收盘模型拒绝封涨停买入；盘中阈值严格大于设定百分比，跳空按开盘价；涨停开盘拒绝。执行模块负责T+1、整手、佣金与印花税。滑动退出只用此前已知高点；同日两种退出同时触发时止损优先。</p><p>未平仓估值并不等于可卖出。筛选名单没有逐日ST历史；10%价格带由前收盘原价推算，无法完整识别历史ST、除权参考价与特殊涨跌停期。本轮要求至少60根此前行情并在预先固定的{summary['common_exclusion_horizon_sessions']}个交易日共用窗口内，一致剔除原价价格带异常信号；窗口由4日观察期、执行缓冲和最长持有期确定。此剔除仍使用信号后行情，存在事后筛选偏差；训练剔除{summary['training_exclusions']}个，验证剔除{summary['validation_exclusions']}个（也含末端观察期不足）。未计分红送转，结果为初步模型验证。</p></section><section><h2>下载与复核</h2><a href="training-grid.csv">全部训练参数表</a> · <a href="validation-trades.csv">选定参数验证逐笔交易</a> · <a href="validation-entries.csv">验证买入与拒绝记录</a> · <a href="frozen-best-training.json">训练冻结参数</a> · <a href="optimization-summary.json">运行记录</a> · <a href="training-exclusions.csv">训练剔除</a> · <a href="validation-exclusions.csv">验证剔除</a><p>通用模块支持网格组合、固定种子抽样、条件限制、失败隔离、带数据、代码及评估/筛选/约束回调版本指纹的断点恢复。当前首个成交时点明确的适配器为本策略；其他策略须提供自己的评估函数，主回测引擎已将收盘信号推迟到下一根K线开盘执行；本适配器仍属于独立信号实验，尚未与主引擎的共享资金组合回放统一。</p></section></main><script>document.getElementById('search').addEventListener('input',e=>{{let q=e.target.value.toLowerCase();document.querySelectorAll('#grid tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));}});</script></html>'''
    (out/'参数优化结果.html').write_text(page,encoding='utf-8')
