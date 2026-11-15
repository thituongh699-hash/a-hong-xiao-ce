import argparse
from pathlib import Path

import pytest

import scripts.batch_backtest_runner as runner


SCOPE = {
    "boards": ["main_board", "chinext"],
    "stock_types": ["normal", "st", "delisting"],
}


@pytest.mark.parametrize("raw,expected", [
    ("920001", "920001.BJ"), ("920001.BJ", "920001.BJ"),
    ("BJ920001", "920001.BJ"), ("430047.BJ", "430047.BJ"),
    ("430047", "430047"), ("600036", "600036.SH"),
])
def test_csv_code_normalization_retains_beijing_identity(raw, expected):
    assert runner.规范股票代码(raw) == expected


def _task(**overrides):
    task = {
        "task_id": "T-SCOPE-1",
        "stock_code": "600000.SH",
        "strategy_id": "S1",
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "capital": "1000000",
        "max_retry": "0",
        "data_source": "default",
    }
    task.update(overrides)
    return task


def _run_single(task, scope=SCOPE):
    return runner.单任务执行(
        task=task,
        base_url="http://service.test",
        poll_seconds=0,
        status_log_seconds=999,
        max_wait_seconds=1,
        retry_sleep_seconds=0,
        trading_scope=scope,
    )


def test_scope_rejection_does_not_change_source_or_start_backtest(monkeypatch):
    calls = []

    def fake_request(base_url, method, path, body=None, timeout=30):
        calls.append((method, path, body))
        assert path == "/api/backtest/trading_scope/preview"
        return True, {
            "status": "success",
            "eligible_codes": [],
            "excluded": [{"stock_code": "600000.SH", "reason": "board_not_selected"}],
        }, ""

    monkeypatch.setattr(runner, "安全请求", fake_request)

    result = _run_single(_task())

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert "board_not_selected" in result["error_msg"]
    assert calls == [
        (
            "POST",
            "/api/backtest/trading_scope/preview",
            {"stock_codes": ["600000.SH"], "trading_scope": SCOPE},
        )
    ]


def test_scope_preview_network_failure_fails_closed(monkeypatch):
    paths = []

    def fake_request(base_url, method, path, body=None, timeout=30):
        paths.append(path)
        return False, {}, "connection refused"

    monkeypatch.setattr(runner, "安全请求", fake_request)

    result = _run_single(_task(data_source=""))

    assert result["status"] == "failed"
    assert "connection refused" in result["error_msg"]
    assert paths == ["/api/backtest/trading_scope/preview"]


def test_scope_preview_passes_scope_to_start_payload(monkeypatch):
    calls = []

    def fake_request(base_url, method, path, body=None, timeout=30):
        calls.append((path, body))
        if path == "/api/backtest/trading_scope/preview":
            return True, {"status": "success", "eligible_codes": ["600000.SH"], "excluded": []}, ""
        if path == "/api/control/start_backtest":
            return True, {"status": "success", "report_id": "R-SCOPE-1"}, ""
        raise AssertionError(f"unexpected request: {path}")

    monkeypatch.setattr(runner, "安全请求", fake_request)
    monkeypatch.setattr(runner, "等待完成", lambda *args, **kwargs: (True, {}, ""))
    monkeypatch.setattr(
        runner,
        "拉取报告",
        lambda *args, **kwargs: (True, {"strategies": [{"strategy_id": "S1"}]}, ""),
    )

    result = _run_single(_task(data_source=""))

    assert result["ok"] is True
    assert calls[0] == (
        "/api/backtest/trading_scope/preview",
        {"stock_codes": ["600000.SH"], "trading_scope": SCOPE},
    )
    assert calls[1][0] == "/api/control/start_backtest"
    assert calls[1][1]["trading_scope"] == SCOPE


def test_missing_scope_preserves_legacy_start_without_preview(monkeypatch):
    calls = []

    def fake_request(base_url, method, path, body=None, timeout=30):
        calls.append((path, body))
        if path == "/api/control/start_backtest":
            return True, {"status": "success", "report_id": "R-LEGACY-1"}, ""
        raise AssertionError(f"unexpected request: {path}")

    monkeypatch.setattr(runner, "安全请求", fake_request)
    monkeypatch.setattr(runner, "等待完成", lambda *args, **kwargs: (True, {}, ""))
    monkeypatch.setattr(
        runner,
        "拉取报告",
        lambda *args, **kwargs: (True, {"strategies": [{"strategy_id": "S1"}]}, ""),
    )

    result = _run_single(_task(data_source=""), scope=None)

    assert result["ok"] is True
    assert [path for path, _ in calls] == ["/api/control/start_backtest"]
    assert "trading_scope" not in calls[0][1]


@pytest.mark.parametrize(
    "raw",
    [
        "{bad json}",
        '{"boards":["main_board"]}',
        '{"boards":["main_board"],"stock_types":["normal"],"extra":true}',
        '{"boards":["main_board"],"boards":["chinext"],"stock_types":["normal"]}',
        '{"boards":["main_board"],"stock_types":[NaN]}',
    ],
)
def test_trading_scope_cli_rejects_invalid_json_or_scope(raw):
    parser = runner.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--trading-scope-json", raw])
    assert exc.value.code == 2


def test_trading_scope_cli_normalizes_and_default_is_compatible():
    parser = runner.build_parser()
    raw = '{"stock_types":["delisting","normal","st"],"boards":["chinext","main_board"]}'
    parsed = parser.parse_args(["--trading-scope-json", raw])
    assert parsed.trading_scope == SCOPE
    assert parser.parse_args([]).trading_scope is None


def test_parallel_scheduler_passes_scope_into_worker(monkeypatch, tmp_path):
    queues = []
    processes = []

    class FakeQueue:
        def __init__(self):
            self.items = []
            self.drive_process = False

        def put(self, item):
            self.items.append(item)

        def get(self, timeout=None):
            if not self.items and self.drive_process:
                process = processes[0]
                process.target(*process.args)
            if not self.items:
                raise AssertionError("queue unexpectedly empty")
            return self.items.pop(0)

    class FakeProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args
            processes.append(self)

        def start(self):
            return None

        def join(self):
            return None

    def make_queue():
        q = FakeQueue()
        queues.append(q)
        if len(queues) == 2:
            q.drive_process = True
        return q

    seen_scopes = []

    def fake_single(**kwargs):
        seen_scopes.append(kwargs["trading_scope"])
        return {
            "task_id": kwargs["task"]["task_id"],
            "ok": False,
            "status": "failed",
            "report_id": "",
            "error_msg": "expected test stop",
            "result_row": None,
        }

    monkeypatch.setattr(runner.multiprocessing, "Queue", make_queue)
    monkeypatch.setattr(runner.multiprocessing, "Process", FakeProcess)
    monkeypatch.setattr(runner, "单任务执行", fake_single)
    monkeypatch.setattr(runner, "写入CSV", lambda *args, **kwargs: None)

    task = _task()
    args = argparse.Namespace(
        base_url="http://one.test",
        base_urls="http://one.test,http://two.test",
        task_artifacts_dir=str(tmp_path / "artifacts"),
        parallel_workers=2,
        poll_seconds=0,
        status_log_seconds=999,
        max_wait_seconds=1,
        retry_sleep_seconds=0,
        rate_limit_interval_seconds=0,
        trading_scope=SCOPE,
    )
    runner.执行并发(
        candidates=[task],
        tasks_rows=[task],
        tasks_map={task["task_id"]: task},
        results_cache=[],
        args=args,
        tasks_path=Path(tmp_path / "tasks.csv"),
        results_path=Path(tmp_path / "results.csv"),
    )

    assert seen_scopes == [SCOPE]
    assert all(process.args[-1] == SCOPE for process in processes)
