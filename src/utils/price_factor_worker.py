"""Isolated BaoStock factor query. stdout is one JSON document only.

BaoStock keeps a module-global socket. Running this file in a short-lived
process prevents a hung query or login from blocking a long-running backtest.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import threading


def query(code: str, end_date: str) -> dict:
    # BaoStock prints login/logout banners; keep the protocol channel clean.
    with contextlib.redirect_stdout(sys.stderr):
        import baostock as bs

        login = bs.login()
        if str(login.error_code) != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
        try:
            result = bs.query_adjust_factor(
                code=code, start_date="1990-01-01", end_date=end_date
            )
            if str(result.error_code) != "0":
                raise RuntimeError(f"BaoStock factor query failed: {result.error_msg}")
            fields = list(result.fields)
            rows = []
            while result.next():
                rows.append(result.get_row_data())
                if str(result.error_code) != "0":
                    raise RuntimeError(f"BaoStock factor cursor failed: {result.error_msg}")
            if str(result.error_code) != "0":
                raise RuntimeError(f"BaoStock factor cursor failed: {result.error_msg}")
            # Consume every row before logout; the cursor uses the active socket.
            return {"ok": True, "fields": fields, "rows": rows}
        finally:
            bs.logout()


def main() -> int:
    watchdog = None
    try:
        if len(sys.argv) not in {3, 4}:
            raise ValueError("expected code and end date")
        timeout = float(sys.argv[3]) if len(sys.argv) == 4 else 20.0
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('invalid timeout')
        # This deadline survives the caller being cancelled/killed. Otherwise
        # a nested BaoStock query could outlive a readiness worker indefinitely.
        watchdog = threading.Timer(timeout, lambda: os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        payload = query(sys.argv[1], sys.argv[2])
    except Exception as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if watchdog is not None:
            watchdog.cancel()
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
