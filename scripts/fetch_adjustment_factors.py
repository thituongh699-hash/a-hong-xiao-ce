"""Fetch a stock-list snapshot into the existing verified factor cache.

BaoStock has a process-global socket, so each subprocess owns one session.
Successful stocks are saved immediately; rerunning resumes from valid caches.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import price_adjustment as adjustment


def normalized_code(value):
    value = value.strip().lower()
    if re.fullmatch(r"\d{6}\.(sh|sz)", value):
        number, market = value.split(".")
        return f"{market}.{number}"
    if re.fullmatch(r"(sh|sz)\.\d{6}", value):
        return value
    raise ValueError(f"unsupported stock-list code: {value!r}")


def cache_path(cache_dir, code):
    return cache_dir / f"adjustment_factors_v{adjustment._CACHE_VERSION}_{code.replace('.', '_')}.json"


def fetch_in_session(bs, code, end, cache_dir):
    cached = adjustment._read_cache(cache_path(cache_dir, code), code, end, 21600)
    if cached is not None:
        return "cached"
    result = bs.query_adjust_factor(code=code, start_date="1990-01-01", end_date=end.date().isoformat())
    if str(result.error_code) != "0":
        raise RuntimeError(f"BaoStock {result.error_code}: {result.error_msg}")
    fields = list(result.fields)
    if (len(fields) != len(set(fields)) or
            not {"code", "dividOperateDate", "backAdjustFactor"}.issubset(fields)):
        raise ValueError("unexpected factor response schema")
    rows = []
    while result.next():
        raw = result.get_row_data()
        if str(result.error_code) != "0" or len(raw) != len(fields):
            raise ValueError("invalid factor cursor row")
        record = dict(zip(fields, raw))
        if str(record["code"]).lower() != code:
            raise ValueError("factor response belongs to another stock")
        rows.append({"dt": record["dividOperateDate"], "factor": record["backAdjustFactor"]})
    if str(result.error_code) != "0":
        raise RuntimeError(f"BaoStock cursor {result.error_code}: {result.error_msg}")
    frame = adjustment._factor_frame(rows, end)
    adjustment._write_cache(cache_path(cache_dir, code), {
        "version": adjustment._CACHE_VERSION,
        "code": code,
        "source": adjustment._FACTOR_SOURCE,
        "coverage_end": end.date().isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "rows": [{"dt": dt.date().isoformat(), "factor": float(factor)}
                 for dt, factor in zip(frame["dt"], frame["factor"])],
    })
    return "fetched"


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def worker():
    request = json.load(sys.stdin)
    cache_dir = Path(request["cache_dir"])
    end = adjustment._day(request["end_date"], "end date")
    timeout = request["timeout"]
    bs = None
    logged_in = False
    consecutive_errors = 0
    for code in request["codes"]:
        emit({"kind": "start", "code": code})
        watchdog = threading.Timer(timeout, lambda: os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if bs is None:
                    import baostock as bs
                if not logged_in:
                    login = bs.login()
                    if str(login.error_code) != "0":
                        raise RuntimeError(f"BaoStock login {login.error_code}: {login.error_msg}")
                    from baostock.common import context as bs_context
                    bs_context.default_socket.settimeout(max(1, timeout - 2))
                    logged_in = True
                status = fetch_in_session(bs, code, end, cache_dir)
            emit({"kind": "result", "code": code, "status": status, "error": ""})
            consecutive_errors = 0
        except Exception as exc:
            emit({"kind": "result", "code": code, "status": "failed",
                  "error": f"{type(exc).__name__}: {exc}"})
            with contextlib.suppress(Exception):
                from baostock.common import context as bs_context
                bs_context.default_socket.close()
            logged_in = False
            consecutive_errors += 1
        finally:
            watchdog.cancel()
        if consecutive_errors >= 10:
            return 1
    # Socket closure also covers a hung logout; no query data relies on logout.
    with contextlib.suppress(Exception):
        from baostock.common import context as bs_context
        bs_context.default_socket.close()
    return 0


def run_batch(codes, args, batch_id, record):
    remaining = list(codes)
    restarts = 0
    while remaining:
        request = {"codes": remaining, "cache_dir": str(args.cache_dir),
                   "end_date": args.end_date, "timeout": args.timeout}
        active = None
        completed = set()
        log_path = args.report_dir / f"worker-{batch_id:04d}.log"
        with log_path.open("a", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, encoding="utf-8", errors="replace", cwd=ROOT,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            process.stdin.write(json.dumps(request))
            process.stdin.close()
            try:
                for line in process.stdout:
                    event = json.loads(line)
                    if event["code"] not in remaining:
                        raise ValueError("unexpected worker stock code")
                    if event["kind"] == "start":
                        active = event["code"]
                    elif event["kind"] == "result":
                        record(event)
                        completed.add(event["code"])
                        active = None
                process.wait(timeout=5)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()
        if process.returncode == 124 and active is not None:
            record({"code": active, "status": "failed", "error": f"query timed out after {args.timeout:g}s"})
            completed.add(active)
        remaining = [code for code in remaining if code not in completed]
        if not remaining:
            return
        restarts += 1
        if process.returncode != 124 or restarts >= 3:
            for code in remaining:
                record({"code": code, "status": "unattempted", "error": f"worker stopped; see {log_path.name}"})
            return


def write_reports(args, stocks, results, started, input_count):
    columns = ["code", "name", "status", "policy_compatible", "event_count", "last_event_date", "last_factor", "error"]
    status_rows = []
    invalid = []
    factor_rows = 0
    with (args.report_dir / "factors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "name", "effective_date", "back_adjust_factor", "coverage_end", "source"])
        for code, stock in stocks.items():
            status = dict(results.get(code, {"status": "unattempted", "error": "not processed"}))
            status.update(code=stock["code"], name=stock.get("name", ""))
            try:
                adjustment._stock_code(code)
                status["policy_compatible"] = True
            except adjustment.PriceAdjustmentError:
                status["policy_compatible"] = False
            if status["status"] in {"fetched", "cached"}:
                frame = adjustment._read_cache(cache_path(args.cache_dir, code), code,
                                               adjustment._day(args.end_date, "end date"), 21600)
                if frame is None:
                    status.update(status="invalid", error="cache verification failed")
                    invalid.append(code)
                else:
                    status.update(event_count=len(frame) - 1,
                                  last_event_date=frame.iloc[-1]["dt"].date().isoformat(),
                                  last_factor=float(frame.iloc[-1]["factor"]))
                    for dt, factor in zip(frame["dt"], frame["factor"]):
                        writer.writerow([stock["code"], stock.get("name", ""), dt.date().isoformat(),
                                         float(factor), args.end_date, adjustment._FACTOR_SOURCE])
                        factor_rows += 1
            status_rows.append(status)
    with (args.report_dir / "stocks.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(status_rows)
    counts = {label: sum(row["status"] == label for row in status_rows)
              for label in ("fetched", "cached", "failed", "unattempted", "invalid")}
    summary = {"stock_list": str(args.stock_list), "stock_list_sha256": args.input_sha256,
               "stock_list_input_rows": input_count, "unique_stocks": len(stocks),
               "coverage_start": "1990-01-01", "coverage_end": args.end_date,
               "source": adjustment._FACTOR_SOURCE, "counts": counts,
               "cache_dir": str(args.cache_dir), "factor_rows_including_baselines": factor_rows,
               "policy_incompatible": [row["code"] for row in status_rows if not row["policy_compatible"]],
               "failed_codes": [row["code"] for row in status_rows if row["status"] not in {"fetched", "cached"}],
               "elapsed_seconds": round(time.monotonic() - started, 1),
               "completed_at": datetime.now(timezone.utc).isoformat()}
    adjustment._write_cache(args.report_dir / "summary.json", summary)
    emit(summary)
    return 0 if counts["fetched"] + counts["cached"] == len(stocks) else 1


def main():
    if "--worker" in sys.argv:
        return worker()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-list", type=Path, default=ROOT / "data/stock_list.csv")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/history/cache")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 0 <= args.retries <= 2 or not 3 <= args.timeout <= 60 or args.limit < 0:
        parser.error("workers must be 1..4, retries 0..2, timeout 3..60, limit nonnegative")
    end = adjustment._day(args.end_date, "end date")
    today = adjustment.pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    if end < adjustment._BASE_DATE or end > today:
        parser.error("end date outside available history")
    args.stock_list = args.stock_list.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.report_dir = args.report_dir.resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    args.input_sha256 = hashlib.sha256(args.stock_list.read_bytes()).hexdigest()
    with args.stock_list.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    stocks = {normalized_code(row["code"]): row for row in source_rows}
    if args.limit:
        stocks = dict(list(stocks.items())[:args.limit])
    results = {}
    lock = threading.Lock()
    started = time.monotonic()
    last_progress = started
    journal_path = args.report_dir / "events.jsonl"
    emit({"stocks": len(stocks), "workers": args.workers, "end_date": args.end_date, "report_dir": str(args.report_dir)})

    with journal_path.open("a", encoding="utf-8") as journal:
        def record(event):
            nonlocal last_progress
            with lock:
                code = event["code"]
                results[code] = event
                journal.write(json.dumps({**event, "recorded_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
                journal.flush()
                now = time.monotonic()
                if len(results) % 100 == 0 or now - last_progress >= 10:
                    emit({"processed": len(results), "total": len(stocks),
                          "success": sum(e["status"] in {"fetched", "cached"} for e in results.values()),
                          "elapsed_seconds": round(now - started, 1)})
                    last_progress = now

        pending = list(stocks)
        batch_id = 0
        for attempt in range(args.retries + 1):
            if not pending:
                break
            emit({"attempt": attempt + 1, "pending": len(pending)})
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {}
                for offset in range(0, len(pending), 100):
                    batch_id += 1
                    batch = pending[offset:offset + 100]
                    futures[pool.submit(run_batch, batch, args, batch_id, record)] = batch
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        for code in futures[future]:
                            if code not in results:
                                record({"code": code, "status": "unattempted", "error": f"batch failed: {exc}"})
            pending = [code for code in stocks if results.get(code, {}).get("status") not in {"fetched", "cached"}]
    return write_reports(args, stocks, results, started, len(source_rows))


if __name__ == "__main__":
    raise SystemExit(main())
