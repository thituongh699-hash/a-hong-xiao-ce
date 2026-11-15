"""Process boundary regressions for the price readiness endpoint."""

import asyncio
import io
import json
import sys

import pytest

import server
from src.utils import price_readiness_worker as worker


class FakeProcess:
    def __init__(self, output=None, hang=False):
        self.returncode = None
        self.output = output
        self.hang = hang
        self.killed = False
        self.reaped = False
        self.input = None

    async def communicate(self, payload):
        self.input = payload
        if self.hang:
            await asyncio.Future()
        self.returncode = 0
        return self.output, b""

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        self.reaped = True
        return self.returncode


def _request():
    return server.DataSourceConnectivityTestRequest(
        source="tdx", stock_code="000001.SZ",
        config={"data_provider": {"tushare_token": "secret-in-stdin"}},
    )


def _setup(monkeypatch, process):
    monkeypatch.setattr(server, "_build_runtime_test_config", lambda config: server._DottedPathDict(config))
    command = []

    async def spawn(*args, **kwargs):
        command.extend(args)
        assert kwargs["stderr"] is asyncio.subprocess.DEVNULL
        assert kwargs['env']['PYTHONIOENCODING'] == 'utf-8'
        return process

    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", spawn)
    return command


def test_worker_protocol_hides_provider_chatter_and_errors(monkeypatch):
    request = {"source": "tdx", "stock_code": "000001.SZ", "config": {"token": "secret"}}
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(json.dumps(request)))
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr(worker.sys, "stdout", stdout)
    monkeypatch.setattr(worker.sys, "stderr", stderr)

    def chatter(_request):
        print("provider banner with secret")
        return {"status": "success", "ok": True, "source": "tdx", "stock_code": "000001.SZ", "checks": [],
                "detail": "secret"}

    monkeypatch.setattr(worker, "run", chatter)
    assert worker.main() == 0
    assert json.loads(stdout.getvalue())["ok"] is True
    assert "secret" not in stdout.getvalue()
    assert "secret" in stderr.getvalue()

    stdout.seek(0)
    stdout.truncate()
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(json.dumps(request)))
    monkeypatch.setattr(worker, "run", lambda _request: (_ for _ in ()).throw(RuntimeError("secret")))
    assert worker.main() == 1
    assert "secret" not in stdout.getvalue()


def test_endpoint_passes_secret_on_stdin_only_and_keeps_schema(monkeypatch):
    result = {"status": "success", "ok": True, "source": "tdx", "stock_code": "000001.SZ",
              "checks": [{"id": "daily", "ok": True}], "price_policy": {"execution_price_mode": "none"},
              "checked_at": "2026-09-30T00:00:00", "fingerprint": "abc"}
    process = FakeProcess(json.dumps(result).encode())
    command = _setup(monkeypatch, process)
    actual = asyncio.run(server.api_test_price_readiness(_request()))
    assert actual == result
    assert "secret-in-stdin" in process.input.decode()
    assert all("secret-in-stdin" not in arg for arg in command)
    assert len(command) == 2
    assert command[0] == sys.executable
    assert command[1].endswith("price_readiness_worker.py")


def test_endpoint_timeout_kills_and_reaps_worker(monkeypatch):
    process = FakeProcess(hang=True)
    _setup(monkeypatch, process)
    timeout = asyncio.timeout
    monkeypatch.setattr(server.asyncio, "timeout", lambda _seconds: timeout(0.01))
    result = asyncio.run(server.api_test_price_readiness(_request()))
    assert result["status"] == "error"
    assert result["checks"][0]["id"] == "timeout"
    assert process.killed and process.reaped


def test_endpoint_cancellation_kills_and_reaps_worker(monkeypatch):
    process = FakeProcess(hang=True)
    _setup(monkeypatch, process)

    async def cancel_request():
        task = asyncio.create_task(server.api_test_price_readiness(_request()))
        for _ in range(100):
            if process.input is not None:
                break
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_request())
    assert process.killed and process.reaped


def test_endpoint_rejects_malformed_child_output(monkeypatch):
    process = FakeProcess(b'{"status":"success","ok":true,"checks":[]}')
    _setup(monkeypatch, process)
    result = asyncio.run(server.api_test_price_readiness(_request()))
    assert result["status"] == "error"
    assert result["checks"][0]["id"] == "prices"


def test_real_hung_process_is_reaped(monkeypatch, tmp_path):
    worker_path = tmp_path / "src" / "utils" / "price_readiness_worker.py"
    worker_path.parent.mkdir(parents=True)
    worker_path.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    monkeypatch.setattr(server, "__file__", str(tmp_path / "server.py"))
    monkeypatch.setattr(server, "_build_runtime_test_config", lambda config: server._DottedPathDict(config))
    actual_spawn = asyncio.create_subprocess_exec
    actual_timeout = asyncio.timeout
    spawned = []

    async def capture_spawn(*args, **kwargs):
        process = await actual_spawn(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", capture_spawn)
    monkeypatch.setattr(server.asyncio, "timeout", lambda _seconds: actual_timeout(1))
    result = asyncio.run(server.api_test_price_readiness(_request()))
    assert result["checks"][0]["id"] == "timeout"
    assert len(spawned) == 1
    assert spawned[0].returncode is not None


def test_real_worker_protocol_roundtrips_chinese_config(monkeypatch, tmp_path):
    worker_path = tmp_path / 'src' / 'utils' / 'price_readiness_worker.py'
    worker_path.parent.mkdir(parents=True)
    worker_path.write_text('import json, sys\nr=json.load(sys.stdin)\nprint(json.dumps(dict(status="success", ok=True, source=r["source"], stock_code=r["stock_code"], price_policy={}, checked_at="2026-09-30", fingerprint="test", checks=[dict(detail=r["config"]["label"])]), ensure_ascii=False))\n', encoding='utf-8')
    monkeypatch.setattr(server, '__file__', str(tmp_path / 'server.py'))
    monkeypatch.setattr(server, '_build_runtime_test_config', lambda config: server._DottedPathDict(config))
    request = server.DataSourceConnectivityTestRequest(source='tdx', stock_code='000001.SZ', config={'label': '价格口径测试💰'})
    result = asyncio.run(server.api_test_price_readiness(request))
    assert result['ok'] is True
    assert result['checks'][0]['detail'] == '价格口径测试💰'


@pytest.mark.parametrize('options', [
    {'start_date': '2020-01-01', 'end_date': '2020-12-31'}, {'require_minute': True},
])
def test_endpoint_forwards_readiness_scope_on_stdin(monkeypatch, options):
    result = {'status': 'success', 'ok': True, 'source': 'tdx', 'stock_code': '000001.SZ',
              'checks': [], 'price_policy': {}, 'checked_at': '2026-09-30', 'fingerprint': 'scope'}
    process = FakeProcess(json.dumps(result).encode())
    _setup(monkeypatch, process)
    request = _request().model_copy(update=options)
    assert asyncio.run(server.api_test_price_readiness(request)) == result
    payload = json.loads(process.input)
    assert all(payload[key] == value for key, value in options.items())


def test_worker_passes_scope_to_readiness_without_loading_provider(monkeypatch):
    import src.utils.price_readiness as readiness
    called = []
    monkeypatch.setattr(readiness, 'check_price_readiness', lambda *args, **kwargs: called.append(kwargs) or {})
    worker.run({'source': 'tdx', 'stock_code': '000001.SZ', 'config': {},
                'start_date': '2020-01-01', 'end_date': '2020-12-31', 'require_minute': False})
    assert called == [{'start_date': '2020-01-01', 'end_date': '2020-12-31', 'require_minute': False}]


def test_simultaneous_config_probes_share_two_process_limit(monkeypatch):
    active = peak = 0
    monkeypatch.setattr(server, '_build_runtime_test_config', lambda config: server._DottedPathDict(config or {}))

    class ScheduledProcess(FakeProcess):
        async def communicate(self, payload):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(.005)
            request = json.loads(payload)
            active -= 1
            self.returncode = 0
            return json.dumps({'status': 'success', 'ok': True, 'source': request['source'],
                'stock_code': request['stock_code'], 'checks': [], 'price_policy': {},
                'checked_at': '2026-09-30', 'fingerprint': 'test'}).encode(), b''

    async def spawn(*args, **kwargs):
        return ScheduledProcess()
    monkeypatch.setattr(server.asyncio, 'create_subprocess_exec', spawn)

    async def check():
        return await asyncio.gather(*(server.api_test_price_readiness(_request()) for _ in range(6)))
    assert all(item['ok'] for item in asyncio.run(check()))
    assert peak == 2
