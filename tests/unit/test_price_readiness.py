from datetime import datetime, timedelta

import pandas as pd
import pytest

import src.utils.price_readiness as readiness
from src.utils.price_adjustment import PriceAdjustmentError


class Config(dict):
    def get(self, key, default=None):
        value = self
        for part in key.split('.'):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


class Provider:
    price_mode = 'none'

    def fetch_kline_data(self, code, start, end, interval):
        if interval != 'D':
            return pd.DataFrame()
        return pd.DataFrame([dict(dt=datetime.now()-timedelta(days=1), open=10., high=11., low=9., close=10., vol=1000., amount=10000.)])


def no_factors(*args, **kwargs):
    raise PriceAdjustmentError('factor source unavailable')


@pytest.mark.parametrize('mode, ok', [('none', True), ('qfq', False), ('hfq', False)])
def test_missing_factors_never_silently_claim_adjustment(monkeypatch, mode, ok):
    monkeypatch.setattr(readiness, 'load_adjustment_factors', no_factors)
    config = Config(data_provider={'adjustment_mode': mode})
    result = readiness.check_price_readiness('tdx', '000001.SZ', config, lambda *args: Provider())
    assert result['ok'] is ok
    assert result['price_policy']['factor_available'] is False
    assert result['price_policy']['execution_price_mode'] == 'none'
    assert any(item['id'] == 'minute' and not item['ok'] for item in result['checks'])
    assert result['price_policy']['corporate_action_accounting'] == 'unsupported'


def test_unknown_database_price_mode_is_rejected(monkeypatch):
    monkeypatch.setattr(readiness, 'load_adjustment_factors', no_factors)
    result = readiness.check_price_readiness('duckdb', '000001.SZ', Config(), lambda *args: Provider())
    assert result['ok'] is False
    assert result['price_policy']['raw_data_verified'] is False


def test_draft_config_is_passed_without_persistence(monkeypatch):
    monkeypatch.setattr(readiness, 'load_adjustment_factors', no_factors)
    draft = Config(data_provider={'adjustment_mode': 'none', 'stored_price_mode': 'none', 'tdxdir': 'draft/path'})
    passed = []
    def factory(source, config):
        passed.append(config)
        return Provider()
    result = readiness.check_price_readiness('duckdb', '000001.SZ', draft, factory)
    assert passed == [draft]
    assert result['ok'] is True
    assert 'draft/path' not in result['fingerprint']
    assert draft['data_provider']['tdxdir'] == 'draft/path'


def test_provider_exception_does_not_echo_secret(monkeypatch):
    def factory(*args):
        raise RuntimeError('URL contains super-secret-token')
    result = readiness.check_price_readiness('default', '000001.SZ', Config(), factory)
    assert 'super-secret-token' not in str(result)


def test_tdx_checks_local_minute_update_without_network_or_old_cache(monkeypatch):
    monkeypatch.setattr(readiness, 'load_adjustment_factors', no_factors)
    class LocalProvider(Provider):
        _cache_enabled = True
        def _raw_symbol(self, code):
            return '000001'
        def _reader_minute(self, symbol):
            assert symbol == '000001'
            return pd.DataFrame([dict(dt=pd.Timestamp('2025-01-02 15:00'), open=10., high=11., low=9., close=10., vol=1000., amount=10000.)])
        def _normalize_ohlcv_df(self, frame, code):
            return frame
        def fetch_kline_data(self, code, start, end, interval):
            assert not self._cache_enabled
            assert interval == 'D', 'TDX minute readiness must inspect the local file'
            return super().fetch_kline_data(code, start, end, interval)
    result = readiness.check_price_readiness('tdx', '000001.SZ', Config(), lambda *args: LocalProvider())
    assert result['latest_minute_date'] == '2025-01-02T15:00:00'
    assert any(row['id'] == 'minute' and not row['ok'] and '本地' in row['detail'] for row in result['checks'])


def test_fingerprint_changes_for_credentials_and_does_not_echo_them(monkeypatch):
    monkeypatch.setattr(readiness, 'load_adjustment_factors', no_factors)
    results = [readiness.check_price_readiness('tdx', '000001.SZ', Config(data_provider={'tushare_token': token}), lambda *args: Provider()) for token in ('secret-one', 'secret-two')]
    assert results[0]['fingerprint'] != results[1]['fingerprint']
    assert 'secret-one' not in str(results[0])


@pytest.mark.parametrize('minute_date, expected', [
    ('2026-09-23 15:00', False),  # Historical data is not realtime-ready.
    ('2026-09-28 15:00', False),  # Recent, but behind the latest daily session.
    ('2026-09-29 15:00', True),
    ('2026-10-01 15:00', False),  # Future timestamps must not qualify.
])
def test_minute_readiness_requires_current_daily_coverage(monkeypatch, minute_date, expected):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 30, 19)
    class LocalProvider(Provider):
        def fetch_kline_data(self, *args):
            return pd.DataFrame([dict(dt=pd.Timestamp('2026-09-29'), open=10., high=11., low=9., close=10., vol=1000., amount=10000.)])
        def _raw_symbol(self, code):
            return '000001'
        def _reader_minute(self, symbol):
            return pd.DataFrame([dict(dt=pd.Timestamp(minute_date), open=10., high=11., low=9., close=10., vol=1000., amount=10000.)])
        def _normalize_ohlcv_df(self, frame, code):
            return frame
    monkeypatch.setattr(readiness, 'datetime', FixedDatetime)
    monkeypatch.setattr(readiness, 'load_adjustment_factors', no_factors)
    result = readiness.check_price_readiness('tdx', '000001.SZ', Config(), lambda *args: LocalProvider())
    minute = next(item for item in result['checks'] if item['id'] == 'minute')
    assert minute['ok'] is expected
    assert result['ok'] is True  # Historical daily backtests remain available.


def test_backtest_baseline_preview_does_not_write(monkeypatch):
    import src.utils.backtest_baseline as baseline
    class Loader:
        def to_dict(self):
            return {'data_provider': {'adjustment_mode': 'qfq'}, 'global_backtest_baseline': {'enabled': True, 'profiles': {'test': {'adjustment_mode': 'hfq', 'cost_model': {'slippage': .001}}}}}
    monkeypatch.setattr(baseline.ConfigLoader, 'reload', lambda: Loader())
    monkeypatch.setattr(baseline, '_resolve_profile_name', lambda *args, **kwargs: ('test', 'A股', []))
    def forbidden():
        pytest.fail('preview accessed disk config for writing')
    monkeypatch.setattr(baseline, '_load_base_config', forbidden)
    result = baseline.apply_backtest_baseline('000001.SZ', persist=False)
    assert result['applied'] is True
    assert result['reason'] == 'runtime_config_preserved'
    assert result['adjustment_mode'] == 'qfq'
    assert result['runtime_overrides'] == {'execution.slippage': .001}


@pytest.mark.parametrize('mode, expected', [('none', True), ('qfq', False), ('hfq', False)])
def test_historical_only_source_uses_requested_range_without_current_freshness(monkeypatch, mode, expected):
    calls = []

    class HistoricalProvider(Provider):
        def fetch_kline_data(self, code, start, end, interval):
            calls.append((start, end, interval))
            assert interval == 'D', 'historical daily readiness must not probe current minutes'
            assert start == datetime(2020, 1, 1)
            assert end == datetime(2020, 12, 31, 23, 59, 59, 999999)
            return pd.DataFrame([dict(dt=datetime(2020, 6, 1), open=10., high=11., low=9., close=10., vol=1000., amount=10000.)])

    def historical_factors(code, start, end, config, **kwargs):
        assert start.year == end.year == 2020
        return no_factors()

    monkeypatch.setattr(readiness, 'load_adjustment_factors', historical_factors)
    result = readiness.check_price_readiness('tdx', '000001.SZ',
        Config(data_provider={'adjustment_mode': mode}), lambda *args: HistoricalProvider(),
        start_date='2020-01-01', end_date='2020-12-31')
    assert result['ok'] is expected
    assert result['probe_scope'] == 'historical'
    assert result['requested_end'] == '2020-12-31'
    assert len(calls) == 1
    assert not any(item['id'] in {'minute', 'daily_freshness'} for item in result['checks'])


@pytest.mark.parametrize('start, end, minute', [
    ('2020-01-01', None, False), (None, '2020-12-31', False),
    ('bad-date', '2020-12-31', False), ('2020-12-31', '2020-01-01', False),
    ('2020-01-01', '2099-01-01', False), ('2020-01-01', '2020-12-31', True),
])
def test_invalid_or_live_historical_range_does_not_open_provider(start, end, minute):
    def forbidden(*args):
        pytest.fail('invalid dates reached provider')
    result = readiness.check_price_readiness('tdx', '000001.SZ', Config(), forbidden,
        start_date=start, end_date=end, require_minute=minute)
    assert result['ok'] is False
    assert any(item['severity'] == 'error' for item in result['checks'])


def test_historical_probe_rejects_provider_bars_outside_requested_range(monkeypatch):
    monkeypatch.setattr(readiness, 'load_adjustment_factors', no_factors)
    result = readiness.check_price_readiness('tdx', '000001.SZ', Config(), lambda *args: Provider(),
        start_date='2020-01-01', end_date='2020-12-31')
    assert result['ok'] is False
    assert '请求历史区间没有有效日线' in str(result['checks'])


@pytest.mark.parametrize('minute_date, expected', [
    ('2026-09-23 15:00', False), ('2026-09-28 15:00', False),
    ('2026-09-29 15:00', True), ('2026-10-01 15:00', False),
])
def test_require_minute_turns_freshness_failure_into_start_block(monkeypatch, minute_date, expected):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 30, 19)
    class MinuteProvider(Provider):
        def fetch_kline_data(self, code, start, end, interval):
            if interval == 'D':
                return pd.DataFrame([dict(dt=datetime(2026, 9, 29), open=10., high=11., low=9., close=10., vol=1000., amount=10000.)])
            return pd.DataFrame([dict(dt=pd.Timestamp(minute_date), open=10., high=11., low=9., close=10., vol=1000., amount=10000.)])
    monkeypatch.setattr(readiness, 'datetime', FixedDatetime)
    monkeypatch.setattr(readiness, 'load_adjustment_factors', no_factors)
    result = readiness.check_price_readiness('tdx', '000001.SZ', Config(), lambda *args: MinuteProvider(), require_minute=True)
    assert result['ok'] is expected
    minute = next(item for item in result['checks'] if item['id'] == 'minute')
    assert minute['severity'] == 'error'
    assert result['price_policy']['factor_available'] is False  # none mode retains its factor warning.


def test_require_minute_rejects_missing_minutes(monkeypatch):
    monkeypatch.setattr(readiness, 'load_adjustment_factors', no_factors)
    result = readiness.check_price_readiness('tdx', '000001.SZ', Config(), lambda *args: Provider(), require_minute=True)
    assert result['ok'] is False
    assert any(item['id'] == 'minute' and item['severity'] == 'error' for item in result['checks'])
