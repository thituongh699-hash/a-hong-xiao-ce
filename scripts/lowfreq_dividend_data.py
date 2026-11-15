"""Fetch and validate BaoStock implemented-dividend announcements.

The provider owns a module-global socket.  Public callers therefore execute
queries in short-lived child processes, one login per batch, with a watchdog
that terminates a wedged process.  stdout is reserved for one JSON document;
BaoStock's login/logout banners are redirected to stderr.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from typing import Iterable


VERSION = 1
SOURCE = "baostock.query_dividend_data"
YEAR_TYPE = "operate"
MAX_BATCH = 25
PYTHON = Path(sys.executable).resolve()
_CODE = re.compile(r"^(\d{6})\.(SH|SZ)$", re.IGNORECASE)
_PROVIDER_CODE = re.compile(r"^(SH|SZ)\.(\d{6})$", re.IGNORECASE)
_REQUIRED_FIELDS = {"code", "dividPlanDate", "dividOperateDate"}
_RECORD_DATE_FIELDS = ("dividRegistDate", "dividRegistrationDate")


def _canonical_code(value: object) -> str:
    text = str(value).strip()
    match = _CODE.fullmatch(text)
    if match:
        return f"{match.group(1)}.{match.group(2).upper()}"
    provider_match = _PROVIDER_CODE.fullmatch(text)
    if provider_match:
        return f"{provider_match.group(2)}.{provider_match.group(1).upper()}"
    raise ValueError(f"unsupported stock code: {value!r}")


def _provider_code(value: object) -> str:
    code = _canonical_code(value)
    number, market = code.split(".")
    return f"{market.lower()}.{number}"


def _pair_key(code: str, year: int) -> str:
    return f"{code}:{year}"


def _normalize_pairs(pairs: Iterable[tuple[str, int]]) -> list[tuple[str, int]]:
    normalized: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for raw_code, raw_year in pairs:
        code = _canonical_code(raw_code)
        if isinstance(raw_year, bool):
            raise ValueError("year must be an integer")
        year = int(raw_year)
        if str(year) != str(raw_year).strip() or not 1990 <= year <= 2100:
            raise ValueError(f"invalid year: {raw_year!r}")
        pair = (code, year)
        if pair not in seen:
            normalized.append(pair)
            seen.add(pair)
    return normalized


def cache_path(cache_root: Path, code: str, year: int) -> Path:
    canonical = _canonical_code(code)
    return Path(cache_root) / f"dividend_v{VERSION}_{canonical.replace('.', '_')}_{year}.json"


def _validate_raw_result(payload: object, code: str, year: int) -> tuple[list[str], list[list[str]]]:
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        error = payload.get("error") if isinstance(payload, dict) else "non-object response"
        raise RuntimeError(f"BaoStock dividend query failed: {error}")
    if payload.get("code") != code or payload.get("year") != year:
        raise ValueError("worker response belongs to another code/year")
    if payload.get("year_type") != YEAR_TYPE:
        raise ValueError("worker response did not use operate yearType")
    fields = payload.get("fields")
    rows = payload.get("rows")
    if (not isinstance(fields, list) or not all(isinstance(field, str) and field for field in fields)
            or len(fields) != len(set(fields)) or not _REQUIRED_FIELDS.issubset(fields)):
        raise ValueError("unexpected dividend response schema")
    if not isinstance(rows, list):
        raise ValueError("unexpected dividend rows")
    code_index = fields.index("code")
    checked: list[list[str]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(fields) or not all(isinstance(item, str) for item in row):
            raise ValueError("invalid dividend cursor row")
        if _provider_code(row[code_index]) != _provider_code(code):
            raise ValueError("dividend response belongs to another stock")
        checked.append(row)
    return fields, checked


def _validate_cache(payload: object, expected_code: str | None = None,
                    expected_year: int | None = None) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("dividend cache must be a JSON object")
    code = _canonical_code(payload.get("code"))
    year = payload.get("year")
    if isinstance(year, bool) or not isinstance(year, int) or not 1990 <= year <= 2100:
        raise ValueError("invalid dividend cache year")
    if expected_code is not None and code != _canonical_code(expected_code):
        raise ValueError("dividend cache code does not match filename/request")
    if expected_year is not None and year != expected_year:
        raise ValueError("dividend cache year does not match filename/request")
    if (payload.get("version") != VERSION or payload.get("source") != SOURCE
            or payload.get("year_type") != YEAR_TYPE or payload.get("verified") is not True):
        raise ValueError("unverified or incompatible dividend cache")
    fetched = payload.get("fetched_at_utc")
    try:
        parsed = datetime.fromisoformat(str(fetched).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid dividend cache fetch timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("dividend cache timestamp is not UTC-aware")
    synthetic = {
        "ok": True,
        "code": code,
        "year": year,
        "year_type": payload.get("year_type"),
        "fields": payload.get("fields"),
        "rows": payload.get("rows"),
    }
    _validate_raw_result(synthetic, code, year)
    return payload


def _read_valid_cache(path: Path, code: str, year: int) -> dict | None:
    if not path.is_file():
        return None
    try:
        return _validate_cache(json.loads(path.read_text(encoding="utf-8")), code, year)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _query_batch(pairs: list[tuple[str, int]]) -> dict:
    results: list[dict] = []
    with contextlib.redirect_stdout(sys.stderr):
        import baostock as bs

        login = bs.login()
        if str(login.error_code) != "0":
            raise RuntimeError(f"BaoStock login {login.error_code}: {login.error_msg}")
        try:
            for code, year in pairs:
                query = bs.query_dividend_data(code=_provider_code(code), year=year,
                                               yearType=YEAR_TYPE)
                if str(query.error_code) != "0":
                    results.append({"ok": False, "code": code, "year": year,
                                    "year_type": YEAR_TYPE,
                                    "error": f"BaoStock {query.error_code}: {query.error_msg}"})
                    continue
                fields = list(query.fields)
                rows: list[list[str]] = []
                cursor_error = None
                while query.next():
                    row = query.get_row_data()
                    if str(query.error_code) != "0":
                        cursor_error = f"BaoStock cursor {query.error_code}: {query.error_msg}"
                        break
                    rows.append(row)
                if cursor_error is None and str(query.error_code) != "0":
                    cursor_error = f"BaoStock cursor {query.error_code}: {query.error_msg}"
                if cursor_error:
                    results.append({"ok": False, "code": code, "year": year,
                                    "year_type": YEAR_TYPE, "error": cursor_error})
                else:
                    results.append({"ok": True, "code": code, "year": year,
                                    "year_type": YEAR_TYPE, "fields": fields, "rows": rows})
        finally:
            bs.logout()
    return {"ok": True, "results": results}


def _worker(request_path: Path, timeout: float) -> int:
    watchdog: threading.Timer | None = None
    try:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        raw_pairs = request.get("pairs") if isinstance(request, dict) else None
        if not isinstance(raw_pairs, list):
            raise ValueError("request pairs must be a list")
        pairs = _normalize_pairs((item[0], item[1]) for item in raw_pairs
                                 if isinstance(item, list) and len(item) == 2)
        if len(pairs) != len(raw_pairs) or not pairs or len(pairs) > MAX_BATCH:
            raise ValueError(f"worker batch must contain 1..{MAX_BATCH} unique valid pairs")
        watchdog = threading.Timer(timeout, lambda: os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        payload = _query_batch(pairs)
    except Exception as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if watchdog is not None:
            watchdog.cancel()
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    return 0 if payload["ok"] else 1


def fetch_pairs(pairs: Iterable[tuple[str, int]], cache_root: Path,
                timeout: float = 60) -> dict[str, dict]:
    """Fetch pairs in isolated batches and write only validated raw responses."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    requested = _normalize_pairs(pairs)
    statuses: dict[str, dict] = {}
    pending: list[tuple[str, int]] = []
    for code, year in requested:
        path = cache_path(cache_root, code, year)
        if _read_valid_cache(path, code, year) is not None:
            statuses[_pair_key(code, year)] = {"status": "cached", "cache": str(path)}
        else:
            pending.append((code, year))

    for offset in range(0, len(pending), MAX_BATCH):
        batch = pending[offset:offset + MAX_BATCH]
        request = {"pairs": [[code, year] for code, year in batch]}
        request_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json",
                                             delete=False) as handle:
                json.dump(request, handle, ensure_ascii=False, allow_nan=False)
                request_path = Path(handle.name)
            command = [str(PYTHON), str(Path(__file__).resolve()),
                       "--worker-pairs", str(request_path), "--timeout", str(timeout)]
            completed = subprocess.run(command, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace",
                                       timeout=timeout + 5, check=False)
            if completed.returncode == 124:
                raise TimeoutError(f"BaoStock batch exceeded {timeout:g}s")
            try:
                response = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError("dividend worker returned invalid JSON") from exc
            if completed.returncode != 0 or not isinstance(response, dict) or response.get("ok") is not True:
                detail = response.get("error") if isinstance(response, dict) else completed.stderr.strip()
                raise RuntimeError(f"dividend worker failed: {detail}")
            results = response.get("results")
            if not isinstance(results, list) or len(results) != len(batch):
                raise ValueError("dividend worker returned an incomplete batch")
            by_pair = {(item.get("code"), item.get("year")): item
                       for item in results if isinstance(item, dict)}
            for code, year in batch:
                raw = by_pair.get((code, year))
                try:
                    fields, rows = _validate_raw_result(raw, code, year)
                    path = cache_path(cache_root, code, year)
                    cached = {
                        "version": VERSION,
                        "source": SOURCE,
                        "code": code,
                        "year": year,
                        "year_type": YEAR_TYPE,
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "verified": True,
                        "fields": fields,
                        "rows": rows,
                    }
                    _validate_cache(cached, code, year)
                    _write_cache(path, cached)
                    statuses[_pair_key(code, year)] = {"status": "fetched", "cache": str(path),
                                                        "row_count": len(rows)}
                except Exception as exc:
                    statuses[_pair_key(code, year)] = {"status": "error",
                                                        "error": f"{type(exc).__name__}: {exc}"}
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            for code, year in batch:
                statuses[_pair_key(code, year)] = {"status": "timeout", "error": str(exc)}
        except Exception as exc:
            for code, year in batch:
                statuses[_pair_key(code, year)] = {"status": "error",
                                                    "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if request_path is not None:
                request_path.unlink(missing_ok=True)
    return statuses


def _iso_date(value: object) -> date | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return None
    return parsed


def load_events(cache_root: Path) -> dict[str, list[dict]]:
    """Load usable implemented events; never infer missing announcement dates."""
    events: dict[str, list[dict]] = {}
    for path in sorted(Path(cache_root).glob(f"dividend_v{VERSION}_*.json")):
        try:
            payload = _validate_cache(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid dividend cache {path}: {exc}") from exc
        fields = payload["fields"]
        announcement_index = fields.index("dividPlanDate")
        ex_index = fields.index("dividOperateDate")
        record_field = next((field for field in _RECORD_DATE_FIELDS if field in fields), None)
        record_index = fields.index(record_field) if record_field else None
        code_index = fields.index("code")
        for row in payload["rows"]:
            code = _canonical_code(payload["code"])
            if _provider_code(row[code_index]) != _provider_code(code):
                raise ValueError(f"invalid dividend cache {path}: row code mismatch")
            announcement = _iso_date(row[announcement_index])
            ex_date = _iso_date(row[ex_index])
            record_raw = row[record_index].strip() if record_index is not None else ""
            record = _iso_date(record_raw) if record_raw else None
            if announcement is None or ex_date is None or announcement > ex_date:
                continue
            if record_raw and record is None:
                continue
            events.setdefault(code, []).append({
                "announcement_date": announcement.isoformat(),
                "ex_date": ex_date.isoformat(),
                "record_date": record.isoformat() if record is not None else None,
                "source_cache": str(path),
            })
    for code in events:
        events[code].sort(key=lambda item: (item["announcement_date"], item["ex_date"],
                                            item["source_cache"]))
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-pairs", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    if args.worker_pairs is None:
        parser.error("--worker-pairs is required")
    return _worker(args.worker_pairs, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
