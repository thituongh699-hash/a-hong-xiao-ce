from __future__ import annotations

from datetime import datetime
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lowfreq_dividend_data.py"
SPEC = importlib.util.spec_from_file_location("lowfreq_dividend_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dividends = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dividends)


FIELDS = ["code", "dividPreNoticeDate", "dividPlanDate", "dividRegistDate",
          "dividOperateDate", "dividCashPsBeforeTax"]


def _cache(tmp_path: Path, *, rows: list[list[str]], fields=FIELDS,
           code="000001.SZ", year=2025) -> Path:
    path = dividends.cache_path(tmp_path, code, year)
    path.write_text(json.dumps({
        "version": dividends.VERSION,
        "source": dividends.SOURCE,
        "code": code,
        "year": year,
        "year_type": "operate",
        "fetched_at_utc": "2026-10-10T01:02:03+00:00",
        "verified": True,
        "fields": fields,
        "rows": rows,
    }), encoding="utf-8")
    return path


def test_load_events_uses_formal_announcement_and_actual_ex_date(tmp_path: Path) -> None:
    path = _cache(tmp_path, rows=[
        ["sz.000001", "2025-03-01", "2025-05-10", "2025-06-03", "2025-06-04", "0.2"],
    ])

    event = dividends.load_events(tmp_path)["000001.SZ"][0]

    assert event == {
        "announcement_date": "2025-05-10",
        "ex_date": "2025-06-04",
        "record_date": "2025-06-03",
        "source_cache": str(path),
    }
    assert event["announcement_date"] != "2025-03-01"
    assert datetime.fromisoformat("2025-05-09") < datetime.fromisoformat(event["announcement_date"])
    assert datetime.fromisoformat("2025-05-10") >= datetime.fromisoformat(event["announcement_date"])


@pytest.mark.parametrize("plan,operate", [
    ("", "2025-06-04"),
    ("not-a-date", "2025-06-04"),
    ("2025-06-05", "2025-06-04"),
    ("2025-05-10", ""),
])
def test_load_events_rejects_unannounced_or_invalid_dates(tmp_path: Path,
                                                         plan: str, operate: str) -> None:
    _cache(tmp_path, rows=[
        ["sz.000001", "2025-03-01", plan, "2025-06-03", operate, "0.2"],
    ])
    assert dividends.load_events(tmp_path) == {}


def test_load_events_rejects_invalid_nonempty_registration_date(tmp_path: Path) -> None:
    _cache(tmp_path, rows=[
        ["sz.000001", "2025-03-01", "2025-05-10", "bad-date", "2025-06-04", "0.2"],
    ])
    assert dividends.load_events(tmp_path) == {}


def test_load_events_fails_closed_on_code_mismatch(tmp_path: Path) -> None:
    _cache(tmp_path, rows=[
        ["sh.600000", "2025-03-01", "2025-05-10", "2025-06-03", "2025-06-04", "0.2"],
    ])
    with pytest.raises(ValueError, match="row code|another stock"):
        dividends.load_events(tmp_path)


def test_schema_requires_formal_announcement_field() -> None:
    payload = {
        "ok": True, "code": "000001.SZ", "year": 2025, "year_type": "operate",
        "fields": ["code", "dividPreNoticeDate", "dividOperateDate"],
        "rows": [["sz.000001", "2025-03-01", "2025-06-04"]],
    }
    with pytest.raises(ValueError, match="schema"):
        dividends._validate_raw_result(payload, "000001.SZ", 2025)


def test_valid_empty_provider_response_is_not_fabricated() -> None:
    payload = {
        "ok": True, "code": "000001.SZ", "year": 2025, "year_type": "operate",
        "fields": ["code", "dividPlanDate", "dividOperateDate"], "rows": [],
    }
    fields, rows = dividends._validate_raw_result(payload, "000001.SZ", 2025)
    assert fields == payload["fields"]
    assert rows == []


def test_fetch_pairs_reuses_only_verified_cache(tmp_path: Path, monkeypatch) -> None:
    _cache(tmp_path, rows=[])
    monkeypatch.setattr(dividends.subprocess, "run",
                        lambda *args, **kwargs: pytest.fail("worker should not run"))
    statuses = dividends.fetch_pairs([("000001.SZ", 2025)], tmp_path)
    assert statuses["000001.SZ:2025"]["status"] == "cached"


def test_worker_batch_cap_and_pair_validation(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"pairs": [[f"{index:06d}.SZ", 2025]
                                                     for index in range(26)]}), encoding="utf-8")
    assert dividends._worker(request, 1) == 1
