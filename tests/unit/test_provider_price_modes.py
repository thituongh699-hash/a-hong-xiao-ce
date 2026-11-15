"""Offline regressions for raw OHLC provider requests and BaoStock cursors."""

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

from src.utils import akshare_provider, baostock_provider, jqdata_provider


START = datetime(2024, 1, 2, 9, 30)
END = datetime(2024, 1, 2, 10, 30)


@pytest.fixture
def providers(monkeypatch, tmp_path):
    cfg = {
        "data_provider.local_cache_enabled": False,
        "data_provider.local_cache_dir": str(tmp_path),
    }
    for module in (akshare_provider, baostock_provider, jqdata_provider):
        monkeypatch.setattr(module.ConfigLoader, "reload", lambda: cfg)
    ak = akshare_provider.AkshareProvider()
    bs = baostock_provider.BaostockProvider()
    jq = jqdata_provider.JqdataProvider()
    jq._client = True
    return ak, bs, jq


def _ak_frame():
    return pd.DataFrame([{
        "时间": "2024-01-02 09:35:00", "开盘": 10, "最高": 11,
        "最低": 9, "收盘": 10.5, "成交量": 100, "成交额": 1050,
    }])


def _jq_frame():
    return pd.DataFrame([{
        "open": 10, "high": 11, "low": 9, "close": 10.5,
        "volume": 100, "money": 1050,
    }], index=pd.DatetimeIndex([START + timedelta(minutes=5)], name="time"))


def test_akshare_requests_raw_prices_for_history_and_latest(providers, monkeypatch):
    ak, _, _ = providers
    calls = []

    def get_minutes(**kwargs):
        calls.append(kwargs)
        return _ak_frame()

    monkeypatch.setattr(akshare_provider.ak, "stock_zh_a_hist_min_em", get_minutes)
    history = ak.fetch_minute_data("600000.SH", START, END)
    latest = ak.get_latest_bar("600000.SH")
    assert not history.empty and latest["close"] == 10.5
    assert len(calls) == 2 and all(call["adjust"] == "" for call in calls)
    assert ak.price_mode == "none"


def test_jqdata_requests_raw_prices_on_every_path(providers, monkeypatch):
    _, _, jq = providers
    calls = []

    def get_price(*args, **kwargs):
        calls.append(kwargs)
        return _jq_frame()

    monkeypatch.setitem(sys.modules, "jqdatasdk", SimpleNamespace(get_price=get_price))
    assert not jq.fetch_minute_data("600000.SH", START, END).empty
    assert not jq.fetch_daily_data("600000.SH", START, END).empty
    assert not jq.fetch_kline_data("600000.SH", START, END, "15min").empty
    assert jq.get_latest_bar("600000.SH") is not None
    assert len(calls) == 4 and all(call["fq"] is None for call in calls)
    assert jq.price_mode == "none"


def test_jqdata_latest_daily_fallback_is_raw(providers, monkeypatch):
    _, _, jq = providers
    calls = []

    def get_price(*args, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame() if len(calls) == 1 else _jq_frame()

    monkeypatch.setitem(sys.modules, "jqdatasdk", SimpleNamespace(get_price=get_price))
    assert jq.get_latest_bar("600000.SH") is not None
    assert len(calls) == 2 and all(call["fq"] is None for call in calls)


class _Cursor:
    def __init__(self, fields, rows, events):
        self.fields = fields.split(",")
        self.rows = list(rows)
        self.events = events
        self.error_code = "0"
        self.error_msg = ""
        self.index = -1

    def next(self):
        self.events.append("next")
        self.index += 1
        return self.index < len(self.rows)

    def get_row_data(self):
        self.events.append("row")
        return self.rows[self.index]


def test_baostock_minute_dates_time_and_logout(providers, monkeypatch):
    _, bs, _ = providers
    events = []
    calls = []

    def query(code, fields, **kwargs):
        calls.append(kwargs)
        return _Cursor(fields, [[
            "2024-01-02", "20240102093500123", "sh.600000",
            "10", "11", "9", "10.5", "100", "1050",
        ]], events)

    monkeypatch.setattr(baostock_provider.bs, "login", lambda: SimpleNamespace(error_code="0"))
    monkeypatch.setattr(baostock_provider.bs, "logout", lambda: events.append("logout"))
    monkeypatch.setattr(baostock_provider.bs, "query_history_k_data_plus", query)
    out = bs.fetch_kline_data("600000.SH", START, END, "5min")
    assert len(out) == 1
    assert out.iloc[0]["dt"] == pd.Timestamp("2024-01-02 09:35:00.123")
    assert calls[0] == {
        "start_date": "2024-01-02", "end_date": "2024-01-02",
        "frequency": "5", "adjustflag": "3",
    }
    assert events[-1] == "logout" and events.index("row") < events.index("logout")
    assert bs.price_mode == "none"


@pytest.mark.parametrize("interval,frequency", [
    ("D", "d"), ("5min", "5"),
    ("15min", "15"), ("30min", "30"), ("60min", "60"),
])
def test_baostock_backtest_intervals_are_raw(providers, monkeypatch, interval, frequency):
    _, bs, _ = providers
    calls = []
    monkeypatch.setattr(baostock_provider.bs, "login", lambda: SimpleNamespace(error_code="0"))
    monkeypatch.setattr(baostock_provider.bs, "logout", lambda: None)

    def query(code, fields, **kwargs):
        calls.append(kwargs)
        values = {
            "date": "2024-01-02", "time": "20240102093500000",
            "code": "sh.600000", "open": "10", "high": "11",
            "low": "9", "close": "10.5", "volume": "100", "amount": "1050",
        }
        return _Cursor(fields, [[values[field] for field in fields.split(",")]], [])

    monkeypatch.setattr(baostock_provider.bs, "query_history_k_data_plus", query)
    out = bs.fetch_kline_data("600000.SH", START, END, interval)
    assert len(out) == 1
    assert calls[0]["frequency"] == frequency
    assert calls[0]["adjustflag"] == "3"
    assert calls[0]["start_date"] == "2024-01-02"


def test_baostock_rejects_1min_without_querying_5min(providers, monkeypatch):
    _, bs, _ = providers
    monkeypatch.setattr(
        baostock_provider.bs, "query_history_k_data_plus",
        lambda *args, **kwargs: pytest.fail("1min must not query 5min bars"),
    )
    assert bs.minimum_interval == "5min"
    assert bs.fetch_minute_data("600000.SH", START, END).empty
    assert "5min or longer" in bs.last_error
    assert bs.fetch_kline_data("600000.SH", START, END, "1min").empty
    assert "provider with native 1min" in bs.last_error


def test_baostock_latest_uses_raw_cursor(providers, monkeypatch):
    _, bs, _ = providers
    events = []
    calls = []
    def query(code, fields, **kwargs):
        calls.append(kwargs)
        row = ["2024-01-02", "20240102093500000",
               "sh.600000", "10", "11", "9", "10.5", "100", "1050"]
        return _Cursor(fields, [row], events)

    monkeypatch.setattr(baostock_provider.bs, "login", lambda: SimpleNamespace(error_code="0"))
    monkeypatch.setattr(baostock_provider.bs, "logout", lambda: events.append("logout"))
    monkeypatch.setattr(baostock_provider.bs, "query_history_k_data_plus", query)
    bar = bs.get_latest_bar("600000.SH")
    assert bar["close"] == 10.5
    assert bar["dt"] == pd.Timestamp("2024-01-02 09:35:00")
    assert calls[0]["adjustflag"] == "3"
    assert events[-1] == "logout"


def test_raw_cache_names_do_not_load_legacy_adjusted_files(providers, tmp_path):
    for provider, prefix, interval in zip(
        providers, ("akshare", "baostock", "jqdata"), ("1min", "5min", "1min")
    ):
        provider._cache_enabled = True
        legacy = tmp_path / f"{prefix}_600000_SH_{interval}.csv"
        legacy.write_text("code,dt,open,high,low,close,vol,amount\n"
                          "600000.SH,2024-01-02 09:30:00,1,1,1,1,1,1\n")
        assert Path(provider._cache_file_path("600000.SH", interval)).name == (
            f"{prefix}_raw_v2_600000_SH_{interval}.csv"
        )
        cached, hit = provider._load_cached_minute_data("600000.SH", START, END)
        assert cached.empty and hit is False
