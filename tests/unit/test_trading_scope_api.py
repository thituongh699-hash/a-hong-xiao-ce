import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import server
from src.core.backtest_engine import BacktestEngine


MAIN = {"boards": ["main_board"], "stock_types": ["normal"]}
RECORDS = [
    {"code": "600036.SH", "name": "招商银行"},
    {"code": "300750.SZ", "name": "宁德时代"},
    {"code": "688001.SH", "name": "华兴源创"},
    {"code": "920001.BJ", "name": "普通北交公司"},
    {"code": "600001.SH", "name": "*ST示例"},
    {"code": "600002.SH", "name": "示例退"},
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server.stock_manager, "stocks", RECORDS)
    monkeypatch.setattr(server.stock_manager, "ensure_loaded", lambda: None)
    monkeypatch.setattr(server.ConfigLoader, "reload", lambda: {"system.mode": "backtest"})
    return TestClient(server.app)


def test_preview_filters_main_board_and_current_name_types(client):
    response = client.post("/api/backtest/trading_scope/preview", json={
        "stock_codes": [record["code"] for record in RECORDS], "trading_scope": MAIN,
    })
    assert response.status_code == 200
    value = response.json()
    assert value["eligible_codes"] == ["600036.SH"]
    assert len(value["excluded"]) == 5
    assert value["metadata"]["historical_st_status_verified"] is False
    assert "现有股票名单" in value["metadata"]["historical_status_note"]


def test_preview_beijing_and_st_choices(client):
    payload = {"stock_codes": ["920001", "600001.SH", "600036.SH"],
               "trading_scope": {"boards": ["beijing", "main_board"], "stock_types": ["st"]}}
    assert client.post("/api/backtest/trading_scope/preview", json=payload).json()["eligible_codes"] == ["600001.SH"]
    payload["trading_scope"]["stock_types"] = ["normal", "st"]
    assert client.post("/api/backtest/trading_scope/preview", json=payload).json()["eligible_codes"] == ["920001.BJ", "600001.SH", "600036.SH"]


@pytest.mark.parametrize("scope", [
    {"boards": [], "stock_types": ["normal"]},
    {"boards": ["fake"], "stock_types": ["normal"]},
    {"boards": ["main_board"], "stock_types": []},
    {**MAIN, "include_all": True},
])
def test_invalid_scope_is_http_422_before_task_creation(client, monkeypatch, scope):
    monkeypatch.setattr(server, "start_new_backtest_report", lambda *a, **k: pytest.fail("no report may be written"))
    response = client.post("/api/control/start_backtest", json={"stock_code": "600036.SH", "trading_scope": scope})
    assert response.status_code == 422


def test_rejected_stock_does_not_cancel_current_task_or_write_report(client, monkeypatch):
    task = SimpleNamespace(done=lambda: False, cancel=lambda: pytest.fail("existing task must survive rejected scope"))
    monkeypatch.setattr(server, "engine_task", task)
    monkeypatch.setattr(server, "start_new_backtest_report", lambda *a, **k: pytest.fail("no report may be written"))
    monkeypatch.setattr(server, "_spawn_backtest_task", lambda *a, **k: pytest.fail("must not start"))
    response = client.post("/api/control/start_backtest", json={"stock_code": "300750.SZ", "trading_scope": MAIN})
    assert response.status_code == 200 and response.json()["status"] == "error"
    assert "所属板块未勾选" in response.json()["msg"]
    assert server.engine_task is task


def test_accepted_scope_is_canonicalized_and_saved_in_report_request(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(server, "engine_task", None)
    monkeypatch.setattr(server, "_live_running_codes", lambda: [])
    def report(code, strategy, params):
        captured["params"] = params
        return "scope-report"
    def spawn(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
    monkeypatch.setattr(server, "start_new_backtest_report", report)
    monkeypatch.setattr(server, "_spawn_backtest_task", spawn)
    value = client.post("/api/control/start_backtest", json={"stock_code": "600036", "trading_scope": MAIN}).json()
    assert value["status"] == "success"
    assert captured["args"][0] == "600036.SH"
    assert captured["kwargs"]["trading_scope"] == MAIN
    assert captured["params"]["trading_scope"] == MAIN
    assert captured["params"]["trading_scope_selection"]["metadata"]["historical_st_status_verified"] is False


def test_screener_scope_is_filtered_before_top_n_and_csv(client, monkeypatch, tmp_path):
    captured = {}
    task_path = tmp_path / "tasks.csv"
    monkeypatch.setattr(server, "_resolve_batch_tasks_path", lambda *a, **k: str(task_path))
    monkeypatch.setattr(server, "_project_rel_path", lambda value: value)
    async def run(req):
        captured["request"] = req
        return {"status": "success", "pid": 123}
    monkeypatch.setattr(server, "api_batch_run_start", run)
    monkeypatch.setattr(server.EVOLUTION_ANALYSIS_STORE, "save_analysis", lambda *a, **k: None)
    value = client.post("/api/screener/batch_backtest/start", json={
        "codes": ["300750.SZ", "600001.SH", "600036.SH"], "strategy_ids": ["S1"],
        "start_date": "2026-01-01", "end_date": "2026-02-01", "top_n": 1,
        "tasks_csv": str(task_path), "trading_scope": MAIN,
    }).json()
    assert value["status"] == "success"
    assert value["used_codes_count"] == 1
    assert len(value["trading_scope_selection"]["excluded"]) == 2
    assert captured["request"].trading_scope == MAIN
    text = task_path.read_text(encoding="utf-8-sig")
    assert "600036.SH" in text and "300750.SZ" not in text and "600001.SH" not in text


def test_search_scope_applies_before_ten_result_limit(client, monkeypatch):
    records = [{"code": f"300{i:03d}.SZ", "name": "同名"} for i in range(11)] + [{"code": "600036.SH", "name": "同名"}]
    monkeypatch.setattr(server.stock_manager, "stocks", records)
    value = client.get("/api/search", params={"q": "同名", "trading_scope": __import__("json").dumps(MAIN)}).json()
    assert [record["code"] for record in value["results"]] == ["600036.SH"]


def test_engine_enforces_scope_before_any_provider_initialization():
    with pytest.raises(ValueError, match="不在所选交易范围"):
        BacktestEngine("300750.SZ", trading_scope=MAIN, security_records=RECORDS)


def test_internal_runner_rejects_before_provider_precheck(client, monkeypatch):
    failures = []
    monkeypatch.setattr(server, "fail_current_backtest_report", failures.append)
    monkeypatch.setattr(server, "_run_backtest_provider_precheck", lambda **k: pytest.fail("must not read prices"))
    asyncio.run(server.run_backtest_task("300750.SZ", "all", trading_scope=MAIN))
    assert len(failures) == 1 and "交易范围" in failures[0]


def test_generic_batch_forwards_scope_as_one_subprocess_argument(client, monkeypatch, tmp_path):
    import json
    captured = {}
    monkeypatch.setattr(server, "batch_run_state", {})
    monkeypatch.setattr(server, "_is_batch_running", lambda: False)
    monkeypatch.setattr(server, "_resolve_batch_tasks_path", lambda *a, **k: str(tmp_path / "tasks.csv"))
    monkeypatch.setattr(server, "_project_rel_path", lambda value: value)
    def popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(pid=456)
    monkeypatch.setattr(server.subprocess, "Popen", popen)
    monkeypatch.setattr(server.threading, "Thread", lambda **k: SimpleNamespace(start=lambda: None))
    value = client.post("/api/batch/run/start", json={"trading_scope": MAIN}).json()
    assert value["status"] == "success"
    index = captured["cmd"].index("--trading-scope-json")
    assert json.loads(captured["cmd"][index + 1]) == MAIN
    assert server.batch_run_state["trading_scope"] == MAIN


@pytest.mark.parametrize("code,accepted", [("600036", True), ("300750.SZ", False)])
def test_websocket_entry_validates_scope_before_replacing_task(client, monkeypatch, code, accepted):
    import json
    commands = iter([json.dumps({"type": "start_backtest", "stock": code, "trading_scope": MAIN})])
    messages, spawned, cancelled, reports = [], [], [], []
    async def receive():
        try:
            return next(commands)
        except StopIteration:
            raise server.WebSocketDisconnect()
    async def connect(*a):
        pass
    async def broadcast(value):
        messages.append(value)
    monkeypatch.setattr(server.manager, "connect", connect)
    monkeypatch.setattr(server.manager, "broadcast", broadcast)
    monkeypatch.setattr(server.manager, "disconnect", lambda *a: None)
    task = SimpleNamespace(done=lambda: False, cancel=lambda: cancelled.append(True))
    monkeypatch.setattr(server, "engine_task", task)
    monkeypatch.setattr(server, "current_backtest_report", None)
    monkeypatch.setattr(server, "_live_running_codes", lambda: [])
    def report(*args):
        reports.append(args)
        return "ws-scope-report"
    monkeypatch.setattr(server, "start_new_backtest_report", report)
    monkeypatch.setattr(server, "_spawn_backtest_task", lambda *a, **k: spawned.append((a, k)))
    asyncio.run(server.websocket_endpoint(SimpleNamespace(receive_text=receive)))
    if accepted:
        assert len(spawned) == len(reports) == len(cancelled) == 1
        assert spawned[0][0][0] == "600036.SH"
        assert spawned[0][1]["trading_scope"] == MAIN
        assert reports[0][2]["trading_scope_selection"]["eligible_codes"] == ["600036.SH"]
    else:
        assert not spawned and not reports and not cancelled
        assert "交易范围" in messages[0]["data"]["msg"]
        assert server.engine_task is task
