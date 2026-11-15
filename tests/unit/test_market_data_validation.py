from __future__ import annotations

import pandas as pd
import pytest

from src.services.market_data_service import MarketDataService
from src.utils.market_data_validation import MarketDataValidationError, validate_market_data
from src.utils.price_adjustment import AdjustmentPlan, PriceAdjustmentError
from src.utils import price_readiness


def _bars() -> pd.DataFrame:
    return pd.DataFrame([
        {"code": "000001.SZ", "dt": "2024-01-02", "open": 10, "high": 11,
         "low": 9, "close": 10.5, "vol": 100, "amount": 1050},
        {"code": "000001.SZ", "dt": "2024-01-03T15:00:00+08:00", "open": 10.5,
         "high": 12, "low": 10, "close": 11, "vol": 0, "amount": 0},
    ])


def test_validator_accepts_mixed_iso_dates_and_quantity_alias() -> None:
    bars = _bars().rename(columns={"vol": "quantity"})
    result = validate_market_data(bars, sort=True)
    assert result["dt"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03 15:00")]
    assert result["vol"].tolist() == [100, 0]
    assert "quantity" not in result


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda frame: frame.assign(dt=["2024-01-02", "2024-01-02"]), "duplicate dt"),
        (lambda frame: frame.assign(close=[float("nan"), 11]), "close must be finite"),
        (lambda frame: frame.assign(high=[10, 12]), "OHLC envelope"),
        (lambda frame: frame.assign(vol=[-1, 0]), "vol must be finite"),
        (lambda frame: frame.assign(amount=[float("nan"), 0]), "amount must be finite"),
    ],
)
def test_validator_rejects_structurally_bad_bars(mutate, match) -> None:
    with pytest.raises(MarketDataValidationError, match=match):
        validate_market_data(mutate(_bars()), sort=True)


def test_market_data_service_fails_instead_of_filling_missing_bar() -> None:
    broken = _bars()
    broken.loc[1, ["open", "high", "low", "close", "vol", "amount"]] = None
    with pytest.raises(MarketDataValidationError):
        MarketDataService().clean_data(broken)


def test_adjustment_plan_allows_price_only_bars_but_checks_optional_fields() -> None:
    price_only = _bars().drop(columns=["code", "vol", "amount"])
    assert len(AdjustmentPlan(price_only, None, "none").frame) == 2
    with pytest.raises(PriceAdjustmentError, match="vol must be finite"):
        AdjustmentPlan(_bars().assign(vol=[float("nan"), 0]), None, "none")


def test_price_readiness_reports_malformed_ohlcv_as_error(monkeypatch) -> None:
    class Provider:
        price_mode = "none"

        def fetch_kline_data(self, code, start, end, interval):
            return _bars().assign(dt=["2024-01-02", "2024-01-02"])

    result = price_readiness.check_price_readiness(
        "tdx",
        "000001.SZ",
        {"data_provider.adjustment_mode": "none"},
        lambda *_: Provider(),
        start_date="2024-01-02",
        end_date="2024-01-03",
    )
    assert result["ok"] is False
    assert result["status"] == "error"
    assert any(item["id"] == "prices" and item["severity"] == "error" for item in result["checks"])
