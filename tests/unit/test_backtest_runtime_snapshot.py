"""Accepted backtest configuration is fixed for the entire async task."""
import asyncio
from copy import deepcopy

import server
from src.utils.config_loader import ConfigLoader, RuntimeConfigSnapshot
from src.utils.runtime_params import get_value


def test_snapshot_handles_nested_and_legacy_dotted_config():
    assert RuntimeConfigSnapshot({'risk_control':{'max_pos_per_stock':.1}}).get('risk_control.max_pos_per_stock')==.1
    assert RuntimeConfigSnapshot({'risk_control.max_pos_per_stock':.1}).get('risk_control.max_pos_per_stock')==.1


def test_server_backtest_uses_accepted_snapshot_without_per_bar_disk_loads(monkeypatch):
    disk={'risk_control':{'max_pos_per_stock':.1},'system':{'initial_capital':100000}}
    loads=[]
    def load(self,path):
        loads.append(path)
        self._config=deepcopy(disk)
    monkeypatch.setattr(ConfigLoader,'load_config',load)
    async def prefetch(**kwargs): pass
    monkeypatch.setattr(server,'_maybe_prefetch_fundamental_before_backtest',prefetch)
    monkeypatch.setattr(server,'apply_backtest_baseline',lambda **kwargs:{'applied':False,'runtime_overrides':{}})
    monkeypatch.setattr(server,'_is_active_backtest_report',lambda _:False)
    observed=[]
    class Engine:
        def __init__(self,**kwargs):
            self.baseline_meta={}
            observed.append(get_value('risk_control.max_pos_per_stock'))
        async def run(self,**kwargs):
            disk['risk_control']['max_pos_per_stock']=.9
            for _ in range(100): observed.append(get_value('risk_control.max_pos_per_stock'))
            assert await asyncio.to_thread(get_value,'risk_control.max_pos_per_stock')==.1
    monkeypatch.setattr(server,'BacktestEngine',Engine)
    asyncio.run(server.run_backtest_task('000001.SZ','s',provider_override=object(),realtime_push=False))
    assert observed==[.1]*101
    assert len(loads)==1
