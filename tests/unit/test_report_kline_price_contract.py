import asyncio
import pandas as pd
import pytest
import server


def setup_chart(monkeypatch, source='mysql', stored_mode='none'):
    cfg = server._DottedPathDict({'data_provider': {'source': source, 'stored_price_mode': stored_mode}})
    frame = pd.DataFrame([dict(dt=pd.Timestamp('2026-06-01'), open=10., high=11., low=9., close=10., vol=100)])
    class Provider:
        price_mode = 'none' if source == 'tdx' else None
        def fetch_kline_data(self, *args):
            return frame.copy()
    monkeypatch.setattr(server, '_build_runtime_test_config', lambda: cfg)
    monkeypatch.setattr(server, '_build_provider_by_source', lambda source, cfg: Provider())
    monkeypatch.setattr(server, 'load_report_history', lambda: None)
    monkeypatch.setattr(server, 'report_history', [{'report_id': 'report', 'stock_code': '000001.SZ',
        'strategy_reports': {'strategy': {'start_date': '2026-06-01', 'end_date': '2026-06-02'}}}])
    monkeypatch.setattr(server, 'report_strategy_kline_cache', {})
    monkeypatch.setattr(server, '_strategy_period_label', lambda *a, **k: '日线')
    return cfg, frame


@pytest.mark.parametrize('stored_mode', ['qfq', 'hfq', 'unknown'])
def test_adjusted_or_unknown_storage_cannot_be_labeled_raw(monkeypatch, stored_mode):
    setup_chart(monkeypatch, stored_mode=stored_mode)
    payload = asyncio.run(server.api_report_strategy_kline_data('report', 'strategy'))
    assert payload['status'] == 'error'
    assert not server.report_strategy_kline_cache


def test_source_and_declaration_changes_do_not_reuse_chart_cache(monkeypatch):
    cfg, frame = setup_chart(monkeypatch)
    first = asyncio.run(server.api_report_strategy_kline_data('report', 'strategy'))
    assert first['status'] == 'success'
    assert first['provider_source'] == 'mysql'
    assert first['chart_price_mode'] == 'none'
    assert first['chart_data_scope'] == 'current_source'
    cfg['data_provider']['source'] = 'duckdb'
    frame['close'] = 20.
    second = asyncio.run(server.api_report_strategy_kline_data('report', 'strategy'))
    assert second['provider_source'] == 'duckdb'
    assert second['candles'][0]['close'] == 20.
    assert len(server.report_strategy_kline_cache) == 2
    cfg['data_provider']['stored_price_mode'] = 'qfq'
    rejected = asyncio.run(server.api_report_strategy_kline_data('report', 'strategy'))
    assert rejected['status'] == 'error'


def test_provider_adjusted_frame_is_rejected_despite_raw_storage_declaration(monkeypatch):
    _, frame = setup_chart(monkeypatch)
    frame.attrs['price_mode'] = 'qfq'
    result = asyncio.run(server.api_report_strategy_kline_data('report', 'strategy'))
    assert result['status'] == 'error'
