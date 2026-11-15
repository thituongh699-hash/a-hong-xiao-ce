"""Backend startup gate regressions, independent of frontend prechecks."""
import asyncio

import pytest

import server


def ready(code):
    return {'stock_code': code, 'source': 'tdx', 'status': 'success', 'ok': True,
            'checks': [{'id': 'raw_prices', 'ok': True}, {'id': 'minute', 'ok': True}],
            'price_policy': {'execution_price_mode': 'none', 'raw_data_verified': True}}


def config():
    return server._DottedPathDict({'data_provider': {'source': 'tdx', 'adjustment_mode': 'none'},
                                  'system': {'mode': 'live', 'initial_capital': 100000}})


def test_gate_checks_all_symbols_with_two_probe_loops(monkeypatch):
    active = peak = 0
    checked = []

    async def probe(cfg, source, code, **kwargs):
        nonlocal active, peak
        assert kwargs == {'require_minute': True}
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.005)
        checked.append(code)
        active -= 1
        return ready(code)

    monkeypatch.setattr(server, '_run_price_readiness_probe', probe)
    codes = [f'00000{number}.SZ' for number in range(1, 7)]
    result = asyncio.run(server._check_live_start_price_readiness(codes, config()))
    assert peak == 2
    assert sorted(checked) == codes
    assert [item['stock_code'] for item in result] == codes
    assert all(item['ok'] for item in result)


@pytest.mark.parametrize('failure', ['minute', 'raw_prices', 'factors'])
def test_second_symbol_failure_preserves_running_tasks_and_starts_none(monkeypatch, failure):
    called = []
    class RunningTask:
        def done(self):
            return False
        def cancel(self):
            pytest.fail('failed gate cancelled an existing task')

    async def probe(cfg, source, code, **kwargs):
        called.append(code)
        if code == '000002.SZ':
            return {**ready(code), 'ok': False, 'status': 'error',
                'checks': [{'id': failure, 'ok': False, 'severity': 'error', 'title': failure, 'detail': 'failed'}]}
        return ready(code)

    def forbidden(*args, **kwargs):
        pytest.fail('failed gate reached startup mutations')
    monkeypatch.setattr(server, 'is_live_enabled', lambda: True)
    monkeypatch.setattr(server, '_build_runtime_test_config', lambda: config())
    monkeypatch.setattr(server, '_run_price_readiness_probe', probe)
    monkeypatch.setattr(server, 'engine_task', RunningTask())
    monkeypatch.setattr(server, '_clear_live_last_error', forbidden)
    monkeypatch.setattr(server, '_stop_live_tasks', forbidden)
    monkeypatch.setattr(server, 'run_engine_task', forbidden)
    result = asyncio.run(server.api_start_live(server.LiveRequest(stock_codes=['000001.SZ', '000002.SZ'], replace_existing=True)))
    assert result['status'] == 'error'
    assert called == ['000001.SZ', '000002.SZ']
    assert [item['stock_code'] for item in result['readiness_failures']] == ['000002.SZ']


def test_gate_timeout_has_one_budget_and_cancels_active_probes(monkeypatch):
    started, cancelled, budgets = [], [], []
    real_timeout = asyncio.timeout
    def bounded_timeout(seconds):
        budgets.append(seconds)
        return real_timeout(.01)
    async def probe(cfg, source, code, **kwargs):
        started.append(code)
        try:
            await asyncio.Future()
        finally:
            cancelled.append(code)
    monkeypatch.setattr(server.asyncio, 'timeout', bounded_timeout)
    monkeypatch.setattr(server, '_run_price_readiness_probe', probe)
    codes = ['000001.SZ', '000002.SZ', '000003.SZ']
    result = asyncio.run(server._check_live_start_price_readiness(codes, config()))
    assert budgets == [30]
    assert started == cancelled == codes[:2]
    assert all(not item['ok'] for item in result)
    assert [item['stock_code'] for item in result] == codes
    assert all(item['checks'][0]['id'] == 'timeout' for item in result)


def test_external_cancellation_propagates_and_cleans_probe_loops(monkeypatch):
    active = cancelled = 0
    async def probe(*args, **kwargs):
        nonlocal active, cancelled
        active += 1
        try:
            await asyncio.Future()
        finally:
            cancelled += 1
    monkeypatch.setattr(server, '_run_price_readiness_probe', probe)
    async def cancel_gate():
        task = asyncio.create_task(server._check_live_start_price_readiness(['000001.SZ', '000002.SZ'], config()))
        while active < 2:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(cancel_gate())
    assert cancelled == 2


def test_success_without_minute_evidence_cannot_bypass_gate(monkeypatch):
    async def probe(cfg, source, code, **kwargs):
        return {**ready(code), 'checks': []}
    monkeypatch.setattr(server, '_run_price_readiness_probe', probe)
    result = asyncio.run(server._check_live_start_price_readiness(['000001.SZ'], config()))
    assert result[0]['ok'] is False
    assert result[0]['checks'][0]['id'] == 'minute'


@pytest.mark.parametrize('change', ['stop', 'config', 'mode', 'new_start'])
def test_pending_start_is_invalidated_before_any_startup_mutation(monkeypatch, change):
    current = config()
    async def gate(codes, cfg):
        if change in {'stop', 'new_start'}:
            server._invalidate_pending_live_starts()
        elif change == 'config':
            current['data_provider']['adjustment_mode'] = 'hfq'
        else:
            current['system']['mode'] = 'backtest'
        return [ready(code) for code in codes]
    monkeypatch.setattr(server, 'is_live_enabled', lambda: current.get('system.mode') == 'live')
    monkeypatch.setattr(server, '_build_runtime_test_config', lambda: server._DottedPathDict(__import__('copy').deepcopy(current)))
    monkeypatch.setattr(server, '_check_live_start_price_readiness', gate)
    monkeypatch.setattr(server, '_clear_live_last_error', lambda: pytest.fail('invalidated start mutated state'))
    result = asyncio.run(server.api_start_live(server.LiveRequest(stock_code='000001.SZ')))
    assert result['status'] == 'error'


def test_unverified_engine_path_gates_and_releases_without_constructing_engine(monkeypatch):
    async def gate(codes, cfg):
        return [{**ready(code), 'ok': False} for code in codes]
    async def broadcast(*args):
        pass
    monkeypatch.setattr(server, '_build_runtime_test_config', config)
    monkeypatch.setattr(server, '_check_live_start_price_readiness', gate)
    monkeypatch.setattr(server.manager, 'broadcast', broadcast)
    monkeypatch.setattr(server, 'LiveTradingEngine', lambda **kw: pytest.fail('failed gate constructed engine'))
    monkeypatch.setattr(server, 'live_tasks', {})
    async def run():
        task = asyncio.create_task(server.run_engine_task('000001.SZ'))
        server.live_tasks['000001.SZ'] = task
        await task
    asyncio.run(run())
    assert server.live_tasks == {}


def test_engine_uses_accepted_snapshot_and_cannot_remove_replacement_task(monkeypatch):
    from src.utils.config_loader import ConfigLoader
    observed = []
    replacement = object()
    class Engine:
        def __init__(self, **kwargs):
            observed.append((kwargs['provider_type'], ConfigLoader.reload().get('data_provider.adjustment_mode')))
        async def run_live(self):
            observed.append(await asyncio.to_thread(lambda: ConfigLoader.reload().get('data_provider.source')))
            server.live_tasks['000001.SZ'] = replacement
        def _persist_virtual_fund_pool(self):
            pass
    monkeypatch.setattr(server, 'is_live_enabled', lambda: True)
    monkeypatch.setattr(server, '_build_runtime_test_config', config)
    monkeypatch.setattr(server, 'LiveTradingEngine', Engine)
    monkeypatch.setattr(server, 'live_tasks', {})
    monkeypatch.setattr(server, 'live_engines', {})
    monkeypatch.setattr(server, 'live_strategy_profiles', {})
    monkeypatch.setattr(server, 'live_capital_profiles', {})
    async def run():
        task = asyncio.create_task(server.run_engine_task('000001.SZ', config(), True, server.live_start_generation))
        server.live_tasks['000001.SZ'] = task
        await task
    asyncio.run(run())
    assert observed == [('tdx', 'none'), 'tdx']
    assert server.live_tasks['000001.SZ'] is replacement


def test_snapshot_is_independent_restores_context_and_isolates_tasks():
    from src.utils.config_loader import ConfigLoader, _runtime_config_snapshot
    original = config()
    async def one(source):
        data = config()
        data['data_provider']['source'] = source
        with ConfigLoader.runtime_snapshot(data) as snapshot:
            data['data_provider']['source'] = 'changed'
            returned = snapshot.get('data_provider')
            returned['source'] = 'changed-again'
            snapshot.to_dict_with_secrets()['data_provider']['source'] = 'changed-third'
            await asyncio.sleep(0)
            assert ConfigLoader.reload().get('data_provider.source') == source
            assert await asyncio.to_thread(lambda: ConfigLoader.reload().get('data_provider.source')) == source
        assert _runtime_config_snapshot.get() is None
    async def run():
        await asyncio.gather(one('tdx'), one('mysql'))
    asyncio.run(run())
    with pytest.raises(RuntimeError):
        with ConfigLoader.runtime_snapshot(original):
            raise RuntimeError('test')
    assert _runtime_config_snapshot.get() is None


@pytest.mark.parametrize('gate_ok', [True, False])
def test_cli_checks_effective_environment_config_and_awaits_engine(monkeypatch, gate_ok):
    import run_live
    from src.utils.config_loader import ConfigLoader, RuntimeConfigSnapshot
    seen = []
    async def gate(code, snapshot):
        assert snapshot['data_provider']['source'] == 'tushare'
        assert snapshot['data_provider']['tushare_token'] == 'test-environment-token'
        seen.append('gate')
        return [{**ready(code), 'ok': gate_ok}]
    class Engine:
        def __init__(self, *args, **kwargs):
            assert gate_ok
            seen.append('engine')
        async def run_live(self):
            seen.append('awaited')
    monkeypatch.setenv('DATA_PROVIDER', 'tushare')
    monkeypatch.setenv('TUSHARE_TOKEN', 'test-environment-token')
    monkeypatch.setattr(run_live, '_check_start_prices', gate)
    monkeypatch.setattr(run_live, 'LiveTradingEngine', Engine)
    with ConfigLoader.runtime_snapshot(config()):
        result = asyncio.run(run_live.run_live())
    assert result == (0 if gate_ok else 1)
    assert seen == (['gate', 'engine', 'awaited'] if gate_ok else ['gate'])


def test_cli_does_not_start_when_system_mode_is_backtest(monkeypatch):
    import run_live
    from src.utils.config_loader import ConfigLoader
    cfg = config()
    cfg['system']['mode'] = 'backtest'
    monkeypatch.setattr(run_live, 'LiveTradingEngine', lambda *a, **k: pytest.fail('mode gate bypassed'))
    with ConfigLoader.runtime_snapshot(cfg):
        assert asyncio.run(run_live.run_live()) == 1


def test_auto_start_uses_batch_gate_without_mutating_profiles_on_failure(monkeypatch):
    cfg = config()
    cfg['targets'] = ['000001.SZ', '000002.SZ']
    cfg['system']['live_auto_start_strategy_ids'] = ['test']
    called = []
    async def probe(config, source, code, **kwargs):
        called.append(code)
        return {**ready(code), 'ok': code != '000002.SZ'}
    async def broadcast(*args):
        pass
    monkeypatch.setattr(server, 'is_live_enabled', lambda: True)
    monkeypatch.setattr(server, '_build_runtime_test_config', lambda: cfg)
    monkeypatch.setattr(server, '_run_price_readiness_probe', probe)
    monkeypatch.setattr(server, 'live_strategy_profiles', {})
    monkeypatch.setattr(server.manager, 'broadcast', broadcast)
    monkeypatch.setattr(server, '_clear_live_last_error', lambda: pytest.fail('failed auto gate mutated startup'))
    with pytest.raises(RuntimeError, match='000002.SZ'):
        asyncio.run(server._auto_start_live_from_config(cfg))
    assert called == cfg['targets']
    assert not server.live_strategy_profiles
