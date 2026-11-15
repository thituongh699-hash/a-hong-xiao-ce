from types import SimpleNamespace

import pandas as pd

from src.utils.stock_list_refresh import normalize_stock_list_df, refresh_stock_list, TushareStockListClient


def test_stock_list_refresh_retains_920_and_explicit_historical_bj_codes():
    result = normalize_stock_list_df(pd.DataFrame({
        "code": ["920001", "430047.BJ", "600036.SH", 1, "510300", "510300.SH", "bad"],
        "name": ["北交新号", "北交旧号", "主板公司", "深圳公司", "ETF", "ETF后缀", "未知"],
    }), source="fixture")
    assert set(result.code) == {"920001.BJ", "430047.BJ", "600036.SH", "000001.SZ"}
    assert dict(zip(result.code, result.market))["920001.BJ"] == "BJ"


def test_tushare_adapter_preserves_provider_exchange_suffix():
    client = TushareStockListClient.__new__(TushareStockListClient)
    client._provider = SimpleNamespace(pro=SimpleNamespace(stock_basic=lambda **k:
        pd.DataFrame({"ts_code": ["430047.BJ", "920001.BJ"], "name": ["旧号", "新号"]})))
    result = client.fetch_stock_list()
    assert result.code.tolist() == ["430047.BJ", "920001.BJ"]


def test_refresh_writes_beijing_metadata_to_only_requested_temp_file(tmp_path):
    path = tmp_path / "stocks.csv"
    client = SimpleNamespace(fetch_stock_list=lambda: pd.DataFrame({"code": ["920001"], "name": ["北交普通示例"]}))
    result = refresh_stock_list(path, provider="akshare", akshare_client=client)
    assert result["status"] == "success" and result["codes"] == 1
    saved = pd.read_csv(path)
    assert saved.code.tolist() == ["920001.BJ"]
    assert saved.market.tolist() == ["BJ"]
