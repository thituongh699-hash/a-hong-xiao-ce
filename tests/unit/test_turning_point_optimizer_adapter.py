import json
from contextlib import nullcontext

import pandas as pd
import pytest

from src.optimization.turning_point import (
    TurningDataset, TurningEvaluator, default_config, load_dataset, parameter_space,
    research_acceptance, training_one_axis_sensitivity, validate_config,
)
from scripts.backtest_turning_points import COSTS, daily_bars
from src.utils.runtime_params import parameter_overrides


def dataset():
    days=[f'2026-08-{x:02d}' for x in [3,4,5,6,7,10,11,12,13,14]]
    bars={}
    for code in ['000001.SZ','600001.SH']:
        frame=pd.DataFrame([dict(code=code,dt=pd.Timestamp(day),open=10.,high=10.2,low=9.9,
                                 close=10.,vol=100000.,amount=1000000.) for day in days])
        bars[code]=daily_bars(frame)
        for b in bars[code].values(): b['history_sessions']=100
    bars['000001.SZ'][days[1]].update(high=10.5,close=10.2)
    setups=pd.DataFrame([dict(code=c,name='普通公司',date=days[0]) for c in bars])
    return TurningDataset(setups,days,bars,'test',{},days[-1])


def params():
    return dict(entry_model='intraday',threshold_pct=3,exit=dict(kind='fixed',stop=.08,take=.20),max_hold=40)


def test_default_parameter_space_is_bounded_and_contains_requested_thresholds():
    cfg=validate_config({})
    space=parameter_space(cfg)
    assert len(space)==650
    assert {p['entry_model'] for p in space}=={'intraday','next_open'}
    assert len(parameter_space(validate_config({'entry_models':['close']})))==130
    assert {p['threshold_pct'] for p in space if p['entry_model']=='intraday'}=={3,4,4.5,5}
    assert all(p['exit']['tight']<=p['exit']['stop'] for p in space if p['exit']['kind']=='trailing')
    assert json.loads(json.dumps(space))==space


@pytest.mark.parametrize('changes',[
    {'validation_start':'2026-07-01'}, {'intraday_thresholds':[10]},
    {'max_holds':[1.5]}, {'min_trades':False}, {'capital':float('nan')},
    {'shell':'bad'}, {'entry_models':['unknown']},
])
def test_reject_invalid_or_unknown_parameters(changes):
    with pytest.raises(ValueError): validate_config(changes)


def test_training_never_uses_future_prices_and_marks_open_positions():
    data=dataset(); cutoff=data.calendar[6]
    with parameter_overrides(COSTS):
        evaluator=TurningEvaluator(data,data.calendar[0],cutoff)
        before,trades,entries=evaluator.evaluate(params(),details=True)
        data.bars['000001.SZ'][data.calendar[7]].update(high=99.,close=99.,price_reference_valid=False)
        after=TurningEvaluator(data,data.calendar[0],cutoff).evaluate(params())
    assert before==after
    assert before['trades']==1 and before['open_at_end']==1 and before['closed']==0
    assert trades[0]['net_return_pct'] is None
    assert before['mean_signal_return_pct']==pytest.approx(before['mean_trade_cohort_return_pct']/2)
    assert len(entries)==2


def test_fixed_cohort_excludes_abnormal_reference_for_all_exit_profiles():
    data=dataset(); data.bars['000001.SZ'][data.calendar[-1]]['price_reference_valid']=False
    evaluator=TurningEvaluator(data,data.calendar[0],data.calendar[-1])
    assert [r['code'] for r in evaluator.setups]==['600001.SH']
    assert len(evaluator.excluded)==1
    with parameter_overrides(COSTS):
        metrics=evaluator.evaluate(params())
    assert metrics['signals']==1 and metrics['trades']==0 and metrics['mean_signal_return_pct']==0


def test_period_end_incomplete_signal_not_in_selection_cohort():
    data=dataset(); data.setups.loc[1,'date']=data.calendar[-2]
    evaluator=TurningEvaluator(data,data.calendar[0],data.calendar[-1])
    assert len(evaluator.setups)==1
    assert evaluator.excluded[0]['reason']=='incomplete_observation_window'


def test_common_cohort_allows_fifth_day_for_next_open_execution():
    data=dataset()
    # First signal needs observation sessions 1..4 plus execution session 5.
    with pytest.raises(ValueError,match='没有可用于比较'):
        TurningEvaluator(data,data.calendar[0],data.calendar[4])


def test_history_count_excludes_current_bar():
    frame=pd.DataFrame([dict(code='000001.SZ',dt=pd.Timestamp('2026-01-01')+pd.Timedelta(days=i),
                            open=10.,high=10.,low=10.,close=10.,vol=100.,amount=1000.) for i in range(61)])
    bars=daily_bars(frame)
    assert list(bars.values())[-2]['history_sessions']==59
    assert list(bars.values())[-1]['history_sessions']==60


def test_local_calendar_uses_union_when_one_candidate_lacks_a_session(tmp_path,monkeypatch):
    import src.optimization.turning_point as adapter
    (tmp_path/'config.json').write_text(json.dumps({'data_provider':{'local_cache_enabled':True}}),encoding='utf-8')
    setups=tmp_path/'setups.csv'
    pd.DataFrame([dict(code='000001.SZ',name='普通公司',date='2026-08-03'),
                  dict(code='600001.SH',name='普通公司',date='2026-08-03')]).to_csv(setups,index=False)
    def frame(code,days):
        return pd.DataFrame([dict(code=code,dt=pd.Timestamp(day),open=10.,high=10.2,
                                  low=9.9,close=10.,vol=100000.,amount=1000000.) for day in days])
    frames={'000001.SZ':frame('000001.SZ',['2026-08-03','2026-08-05']),
            '600001.SH':frame('600001.SH',['2026-08-03','2026-08-04','2026-08-05'])}
    class FakeProvider:
        def _raw_symbol(self,code): return code
        def _reader_daily(self,code): return frames[code]
        def _normalize_daily_df(self,raw,code): return raw
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    monkeypatch.setattr(adapter,'TdxProvider',FakeProvider)
    monkeypatch.setattr(adapter.ConfigLoader,'runtime_snapshot',lambda _:nullcontext())
    data=load_dataset('2026-08-05',setups)
    assert data.calendar==['2026-08-03','2026-08-04','2026-08-05']
    assert data.actual_end=='2026-08-05'


def test_abnormal_price_exclusion_stops_at_predeclared_common_horizon():
    data=dataset()
    data.bars['000001.SZ'][data.calendar[-1]]['price_reference_valid']=False
    bounded=TurningEvaluator(data,data.calendar[0],data.calendar[-1],execution_buffer=0,max_hold_horizon=1)
    assert {r['code'] for r in bounded.setups}=={'000001.SZ','600001.SH'}
    wider=TurningEvaluator(data,data.calendar[0],data.calendar[-1],execution_buffer=0,max_hold_horizon=40)
    assert {r['code'] for r in wider.setups}=={'600001.SH'}


def test_research_acceptance_requires_positive_holdout_and_minimum_fills():
    config=validate_config({'min_trades':3})
    best={'score':1.5}
    validation={'downside_adjusted':-0.2,'trades':4}
    rejected=research_acceptance(best,validation,config)
    assert rejected['status']=='unsupported_hypothesis'
    assert rejected['gates']['holdout_above_cash'] is False
    validation.update(downside_adjusted=0.3,trades=2)
    assert research_acceptance(best,validation,config)['status']=='unsupported_hypothesis'
    validation['trades']=3
    accepted=research_acceptance(best,validation,config)
    assert accepted['status']=='research_support_for_followup'
    assert accepted['efficacy_claim'] is False


def test_one_axis_sensitivity_uses_only_ranked_training_peers():
    base=dict(entry_model='intraday',threshold_pct=3,exit=dict(kind='fixed',stop=.08,take=.15),max_hold=20)
    best=dict(params=base,score=2.0)
    adjacent=dict(params={**base,'threshold_pct':4},score=-1.0)
    other=dict(params={**base,'threshold_pct':4,'max_hold':40},score=5.0)
    summary=training_one_axis_sensitivity([other,best,adjacent],best)
    assert summary['threshold_pct']['evaluated']==2
    assert summary['threshold_pct']['score_min']==-1.0
    assert summary['threshold_pct']['positive_vs_cash']==1
