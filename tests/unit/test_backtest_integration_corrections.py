"""Causal aggregation, exchange metadata and final settlement invariants."""
import asyncio
from datetime import datetime

import pandas as pd
import pytest

import src.core.backtest_engine as module
from src.core.backtest_engine import BacktestEngine
from src.services.account_ledger import AccountLedger
from src.core.execution_engine import ExecutionEngine
from src.services.order_matcher import OrderMatcher
from src.services.risk_audit_log import RiskAuditLog
from src.strategies.implemented_strategies import BaseImplementedStrategy
from src.utils.bar_aggregation import aggregate_completed_bars
from src.utils.runtime_params import parameter_overrides


CODE = '000001.SZ'
CONFIG = {
    'execution.slippage': 0, 'execution.force_close_on_backtest_end': False,
    'risk_control.max_drawdown_pct': 0, 'risk_control.max_pos_per_stock': 1,
    'risk_control.max_total_pos': 1, 'risk_control.max_stop_loss_pct': 1,
    'risk_control.max_daily_loss_pct': 1,
    'trading_cost.min_commission': 0, 'trading_cost.commission_rate': 0,
    'trading_cost.stamp_duty': 0, 'trading_cost.transfer_fee': 0,
}


def bars(dates, prices):
    return pd.DataFrame([dict(code=CODE, dt=pd.Timestamp(dt), open=p, high=p+.1,
        low=p-.1, close=p, vol=10000., amount=p*10000) for dt,p in zip(dates,prices)])


class Probe(BaseImplementedStrategy):
    supports_dynamic_qfq = True
    def __init__(self, timeframe='D', threshold=None, stop=None):
        super().__init__('probe', 'probe', timeframe)
        self.seen = []
        self.sent = False
        self.threshold = threshold
        self.stop = stop
    def on_bar(self, bar):
        self.seen.append(dict(bar))
        if self.sent or (self.threshold is not None and bar['close'] < self.threshold):
            return None
        self.sent = True
        return dict(strategy_id=self.id, code=bar['code'], dt=bar['dt'],
            direction='BUY', price=bar['close'], qty=100, stop_loss=self.stop)


class Provider:
    price_mode = 'none'
    def __init__(self, frame): self.frame, self.calls = frame, []
    def check_connectivity(self, code): return True, 'in-memory'
    def fetch_minute_data(self, *args):
        self.calls.append('1min')
        return self.frame.copy()
    def fetch_kline_data(self, code, start, end, timeframe):
        self.calls.append(timeframe)
        assert timeframe == 'D', 'engine must derive higher timeframes from its base feed'
        return self.frame.copy()


def run_engine(monkeypatch, frame, strategy, mode='none', factors=None, force=False):
    cfg = dict(CONFIG, **{'execution.force_close_on_backtest_end': force})
    monkeypatch.setattr(module, 'create_strategies', lambda **kwargs: [strategy])
    monkeypatch.setattr(module.ConfigLoader, 'reload', lambda: cfg)
    events = []
    async def emit(kind, data): events.append((kind,data))
    provider = Provider(frame)
    engine = BacktestEngine(CODE, strategy_ids=['probe'], initial_capital=100000,
        provider_override=provider, provider_source_override='tdx', adjustment_mode=mode,
        adjustment_factors=factors, event_callback=emit)
    async def no_cache(*args): pass
    engine._persist_backtest_cache_to_db = no_cache
    with parameter_overrides(cfg):
        asyncio.run(engine.run(frame.dt.min().to_pydatetime(),frame.dt.max().to_pydatetime()))
    failures = [data for kind,data in events if kind=='backtest_failed']
    assert not failures, failures
    return engine, events, provider


def test_future_minute_cannot_change_preceding_fill(monkeypatch):
    frame=bars(pd.date_range('2026-06-01 09:31',periods=12,freq='min'),[10.]*12)
    frame.loc[4,['open','high','low','close']]=[10.4,10.5,10.3,10.4]
    first=Probe('5min',threshold=10.3)
    engine, _, provider=run_engine(monkeypatch,frame,first)
    assert first.seen[0]['dt']==pd.Timestamp('2026-06-01 09:35')
    original=engine.strategy_accounts['probe'].transactions[0]
    assert original['dt']==pd.Timestamp('2026-06-01 09:36')
    assert original['price']==10
    assert provider.calls==['1min']
    frame.loc[8,['open','high','low','close']]=[50.,50.1,49.9,50.]
    engine, _, _=run_engine(monkeypatch,frame,Probe('5min',threshold=10.3))
    changed=engine.strategy_accounts['probe'].transactions[0]
    assert (changed['dt'],changed['price'],changed['quantity'])==(original['dt'],original['price'],original['quantity'])


def test_session_aggregation_drops_gaps_and_partial_windows():
    dates=list(pd.date_range('2026-06-01 09:31',periods=120,freq='min'))
    dates+=list(pd.date_range('2026-06-01 13:01',periods=120,freq='min'))
    frame=bars(dates,[10.]*240)
    hourly=aggregate_completed_bars(frame,'60min')
    assert hourly.dt.dt.strftime('%H:%M').tolist()==['10:30','11:30','14:00','15:00']
    assert hourly.vol.tolist()==[600000.]*4
    daily=aggregate_completed_bars(frame,'D')
    assert daily.dt.tolist()==[pd.Timestamp('2026-06-01 15:00')]
    assert aggregate_completed_bars(frame.iloc[:-1],'D').empty
    gapped=frame[frame.dt!=pd.Timestamp('2026-06-01 09:33')]
    assert pd.Timestamp('2026-06-01 09:35') not in set(aggregate_completed_bars(gapped,'5min').dt)
    assert aggregate_completed_bars(gapped,'D').empty


@pytest.mark.parametrize('mode',['qfq','hfq'])
def test_pending_signal_cannot_cross_exdate_with_stale_protection(monkeypatch,mode):
    frame=bars(pd.bdate_range('2026-06-01',periods=3),[10.,5.,5.1])
    factors=pd.DataFrame(dict(dt=pd.to_datetime(['1990-01-01','2026-06-02']),factor=[1.,2.]))
    factors.attrs.update(source='verified-test',verified=True,coverage_start='1990-01-01',coverage_end='2026-06-30')
    engine,events,_=run_engine(monkeypatch,frame,Probe(stop=9),mode,factors)
    assert engine.strategy_accounts['probe'].transactions==[]
    assert any('取消 1 条跨除权日' in data.get('msg','') for kind,data in events)


def test_declared_price_limit_survives_main_engine(monkeypatch):
    frame=bars(pd.bdate_range('2026-06-01',periods=2),[10.,10.5])
    frame.loc[1,['open','high','low']]=[11.,11.,10.]
    frame['up_limit'],frame['down_limit']=11.,9.
    engine,_,_=run_engine(monkeypatch,frame,Probe())
    assert engine.strategy_accounts['probe'].transactions==[]


def test_final_card_nav_and_settlement_reconcile_without_sale(monkeypatch):
    frame=bars(pd.bdate_range('2026-01-05',periods=100),[10.]*99+[11.])
    engine,events,_=run_engine(monkeypatch,frame,Probe())
    account=engine.strategy_accounts['probe']
    snapshot=next(data for kind,data in reversed(events) if kind=='account')
    report=next(data for kind,data in events if kind=='backtest_strategy_report')
    assert snapshot['assets']==account.daily_nav[-1]['nav']==report['end_capital']==100100
    assert report['execution_count']==1
    assert report['settlement']['state']=='open'
    assert report['settlement']['unrealized_pnl']==100
    assert report['settlement']['reconciliation_delta']==pytest.approx(0)


def test_force_close_reason_and_cash_are_in_final_report(monkeypatch):
    frame=bars(pd.bdate_range('2026-06-01',periods=3),[10.,10.,11.])
    engine,events,_=run_engine(monkeypatch,frame,Probe(),force=True)
    report=next(data for kind,data in events if kind=='backtest_strategy_report')
    assert report['force_close_count']==1
    assert report['settlement']['state']=='flat'
    assert report['settlement']['cash']==report['end_capital']==100100


def test_intraday_peak_survives_daily_nav_compaction():
    engine=BacktestEngine.__new__(BacktestEngine)
    engine.initial_capital=100000
    account=AccountLedger(100000,nav_frequency='daily')
    engine.strategy_accounts={'s':account}
    engine.execution_engine=ExecutionEngine(account,OrderMatcher(),RiskAuditLog())
    engine.aggregate_nav=[]
    account.cash=110000
    engine._record_nav(dict(code=CODE,dt=datetime(2026,6,1,10),close=10))
    account.cash=100000
    engine._record_nav(dict(code=CODE,dt=datetime(2026,6,1,11),close=10))
    assert len(account.daily_nav)==len(engine.aggregate_nav)==1
    assert engine._compute_current_drawdown_ratio(100000)==pytest.approx(1-100000/110000)
