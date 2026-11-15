"""Regression for event coverage that ordinary exits cannot expose."""
import pandas as pd

from scripts import research_lowfreq_50k as study
from scripts import lowfreq_dividend_data as source
from scripts import lowfreq_portfolio as portfolio


def _provisional(code):
    return dict(summary=dict(valid=True, completed=True, cagr=.5, year_performance=.5,
                             actual_end="2025-12-31", open_positions=[]),
                trades=pd.DataFrame([dict(code=code, entry_date="2025-01-06", exit_date="2025-01-07")]),
                blockers=[])


def test_dividend_coverage_reaches_fixed_point_for_new_blackout_replacements(monkeypatch, tmp_path):
    covered = set()
    calls = []
    monkeypatch.setattr(source, "_read_valid_cache", lambda path, code, year: {} if (code, year) in covered else None)
    monkeypatch.setattr(source, "load_events", lambda path: {"known_a": True} if ("000001.SZ", 2025) in covered else {})

    def simulate(data, spec, start, end, events, **kwargs):
        calls.append(events)
        return _provisional("000002.SZ" if events else "000001.SZ")

    def fetch(pairs, cache_root, timeout):
        covered.update(pairs)
        return {f"{code}:{year}": dict(status="fetched") for code, year in pairs}

    monkeypatch.setattr(portfolio, "simulate_portfolio", simulate)
    monkeypatch.setattr(source, "fetch_pairs", fetch)
    monkeypatch.setattr(study, "save_run", lambda *args: None)
    result = study.resolved_runs({}, [dict(id="x")], "2025-01-01", "2025-12-31", tmp_path, "coverage")
    assert len(calls) == 3
    assert covered == {(code, year) for code in ("000001.SZ", "000002.SZ") for year in (2025, 2026)}
    assert result["x"]["summary"]["valid"] is True
    assert result["x"]["trades"].iloc[0].code == "000002.SZ"


def test_failed_event_coverage_cannot_report_complete_cagr(monkeypatch, tmp_path):
    monkeypatch.setattr(source, "_read_valid_cache", lambda *args: None)
    monkeypatch.setattr(source, "load_events", lambda path: {})
    monkeypatch.setattr(portfolio, "simulate_portfolio", lambda *args, **kwargs: _provisional("000001.SZ"))
    monkeypatch.setattr(source, "fetch_pairs", lambda *args, **kwargs: {"000001.SZ:2025": dict(status="timeout")})
    monkeypatch.setattr(study, "save_run", lambda *args: None)
    result = study.resolved_runs({}, [dict(id="x")], "2025-01-01", "2025-12-31", tmp_path, "failure")
    assert result["x"]["summary"]["valid"] is False
    assert pd.isna(result["x"]["summary"]["cagr"])
    assert all(row["reason"] == "corporate_action_data_missing" for row in result["x"]["blockers"])
