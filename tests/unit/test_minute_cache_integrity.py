from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils.akshare_provider import AkshareProvider
from src.utils.baostock_provider import BaostockProvider
from src.utils.cache_coverage import minute_cache_covers_range
from src.utils.tdx_provider import TdxProvider
from src.utils.tushare_provider import TushareProvider


def _frame(times, price=10.0) -> pd.DataFrame:
    return pd.DataFrame([
        {"code": "000001.SZ", "dt": timestamp, "open": price, "high": price,
         "low": price, "close": price, "vol": 1, "amount": price}
        for timestamp in times
    ])


def _provider(cls, cache_dir: Path):
    provider = cls.__new__(cls)
    provider._cache_enabled = True
    provider._cache_dir = str(cache_dir)
    return provider


def test_cache_endpoints_do_not_hide_an_interior_gap() -> None:
    times = pd.to_datetime(["2024-01-02 09:31", "2024-01-02 09:33"])
    assert not minute_cache_covers_range(
        times, pd.Timestamp("2024-01-02 09:31"), pd.Timestamp("2024-01-02 09:33"),
        interval_minutes=1,
    )


def test_lunch_and_overnight_are_not_cache_gaps() -> None:
    morning = pd.date_range("2024-01-02 09:31", "2024-01-02 11:30", freq="1min")
    afternoon = pd.date_range("2024-01-02 13:01", "2024-01-02 15:00", freq="1min")
    next_open = pd.date_range("2024-01-03 09:31", "2024-01-03 09:33", freq="1min")
    times = morning.append(afternoon).append(next_open)
    assert minute_cache_covers_range(
        times, pd.Timestamp("2024-01-02 09:31"), pd.Timestamp("2024-01-03 09:33"),
        interval_minutes=1,
    )


def test_five_minute_provider_uses_five_minute_slots() -> None:
    times = pd.date_range("2024-01-02 09:35", "2024-01-02 11:30", freq="5min").append(
        pd.date_range("2024-01-02 13:05", "2024-01-02 15:00", freq="5min")
    )
    assert minute_cache_covers_range(
        times, pd.Timestamp("2024-01-02 09:35"), pd.Timestamp("2024-01-02 15:00"),
        interval_minutes=5,
    )
    assert not minute_cache_covers_range(
        times.delete(5), pd.Timestamp("2024-01-02 09:35"), pd.Timestamp("2024-01-02 15:00"),
        interval_minutes=5,
    )


@pytest.mark.parametrize(
    "provider_cls, interval",
    [
        (TdxProvider, "1min"),
        (AkshareProvider, "1min"),
        (TushareProvider, "1min"),
        (BaostockProvider, "5min"),
    ],
)
def test_new_cache_revision_overrides_old_value(tmp_path, provider_cls, interval) -> None:
    provider = _provider(provider_cls, tmp_path)
    timestamp = pd.Timestamp("2024-01-02 09:35" if interval == "5min" else "2024-01-02 09:31")
    provider._save_minute_cache("000001.SZ", _frame([timestamp], price=10))
    provider._save_minute_cache("000001.SZ", _frame([timestamp], price=99))
    cached = pd.read_csv(provider._cache_file_path("000001.SZ", interval))
    assert cached["close"].tolist() == [99]


@pytest.mark.parametrize(
    "provider_cls, interval, times",
    [
        (TdxProvider, "1min", ["2024-01-02 09:31", "2024-01-02 09:33"]),
        (AkshareProvider, "1min", ["2024-01-02 09:31", "2024-01-02 09:33"]),
        (TushareProvider, "1min", ["2024-01-02 09:31", "2024-01-02 09:33"]),
        (BaostockProvider, "5min", ["2024-01-02 09:35", "2024-01-02 09:45"]),
    ],
)
def test_provider_cache_loader_marks_gap_as_miss(tmp_path, provider_cls, interval, times) -> None:
    provider = _provider(provider_cls, tmp_path)
    frame = _frame(pd.to_datetime(times))
    frame.to_csv(provider._cache_file_path("000001.SZ", interval), index=False)
    loaded, hit = provider._load_cached_minute_data(
        "000001.SZ", pd.Timestamp(times[0]), pd.Timestamp(times[-1])
    )
    assert len(loaded) == 2
    assert hit is False


def test_tdx_reads_legacy_alias_without_moving_or_deleting_it(tmp_path) -> None:
    provider = _provider(TdxProvider, tmp_path)
    legacy = tmp_path / "tdx_raw_v2_000001_1min.csv"
    _frame(pd.to_datetime(["2024-01-02 09:31", "2024-01-02 09:32"])).to_csv(legacy, index=False)
    loaded, hit = provider._load_cached_minute_data(
        "000001", pd.Timestamp("2024-01-02 09:31"), pd.Timestamp("2024-01-02 09:32")
    )
    assert hit is True and len(loaded) == 2
    assert legacy.exists()
    assert not Path(provider._cache_file_path("000001", "1min")).exists()
