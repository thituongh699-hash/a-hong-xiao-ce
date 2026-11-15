"""Lifecycle checks across API workers and the durable optimization worker."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.optimization import api
from src.optimization import job_state
from src.optimization.job_state import file_lock, write_json


def _sharing_error(winerror=5):
    error = PermissionError("Windows sharing violation")
    error.winerror = winerror
    return error


def test_atomic_json_retries_transient_windows_sharing_errors(tmp_path, monkeypatch):
    if job_state.os.name != "nt":
        pytest.skip("Windows sharing semantics")
    path = tmp_path / "job.json"
    path.write_text('{"old":true}', encoding="utf-8")
    real_replace = job_state.os.replace
    calls = []
    def transient(src, dst):
        calls.append((src, dst))
        if len(calls) <= 2:
            raise _sharing_error(5 if len(calls) == 1 else 32)
        return real_replace(src, dst)
    monkeypatch.setattr(job_state.os, "replace", transient)
    write_json(path, {"status": "completed"})
    assert len(calls) == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "completed"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_json_permanent_windows_sharing_error_is_bounded_and_preserves_old_file(tmp_path, monkeypatch):
    if job_state.os.name != "nt":
        pytest.skip("Windows sharing semantics")
    path = tmp_path / "job.json"
    original = '{"old":true}'
    path.write_text(original, encoding="utf-8")
    calls = []
    def denied(src, dst):
        calls.append((src, dst))
        raise _sharing_error(33)
    monkeypatch.setattr(job_state.os, "replace", denied)
    started = time.monotonic()
    with pytest.raises(PermissionError, match="sharing violation"):
        write_json(path, {"status": "completed"})
    elapsed = time.monotonic() - started
    assert 0.65 <= elapsed < 1.5
    assert len(calls) > 2
    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_json_does_not_retry_unrelated_windows_error(tmp_path, monkeypatch):
    if job_state.os.name != "nt":
        pytest.skip("Windows sharing semantics")
    path = tmp_path / "job.json"
    path.write_text('{"old":true}', encoding="utf-8")
    calls = []
    def denied(src, dst):
        calls.append((src, dst))
        raise _sharing_error(87)
    monkeypatch.setattr(job_state.os, "replace", denied)
    with pytest.raises(PermissionError, match="sharing violation"):
        write_json(path, {"status": "completed"})
    assert len(calls) == 1
    assert path.read_text(encoding="utf-8") == '{"old":true}'
    assert list(tmp_path.glob("*.tmp")) == []


class FakeProcess:
    pid = 4242

    def __init__(self):
        self.code = None

    def poll(self):
        return self.code


def setup(tmp_path, monkeypatch):
    script = tmp_path / "optimize.py"
    script.write_text("# fake", encoding="utf-8")
    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(api, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api, "SCRIPT", script)
    monkeypatch.setattr(api, "PRECOMPUTED_DIR", tmp_path / "prior")
    monkeypatch.setattr(api, "_active_process", None)
    monkeypatch.setattr(api, "_active_id", None)
    monkeypatch.setattr(api, "_validate_config", lambda body: dict(body))
    monkeypatch.setattr(api.sys, "frozen", False, raising=False)
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


def test_cancel_then_resume_reuses_directory_and_config(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    launched = []
    def popen(args, **kwargs):
        process = FakeProcess()
        launched.append((args, process))
        return process
    monkeypatch.setattr(api.subprocess, "Popen", popen)
    first = client.post("/api/optimization/jobs", json={"max_trials": 3})
    assert first.status_code == 201
    job_id = first.json()["job_id"]
    directory = tmp_path / "jobs" / job_id
    before = (directory / "config.json").read_bytes()
    (directory / "checkpoint.jsonl").write_text('{"type":"header","fingerprint":"abc"}\n', encoding="utf-8")
    checkpoint = (directory / "checkpoint.jsonl").read_bytes()
    response = client.post(f"/api/optimization/jobs/{job_id}/cancel")
    assert response.status_code == 200
    assert response.json()["cancel_requested"] is True
    assert (directory / "cancel.request").exists()
    assert client.post(f"/api/optimization/jobs/{job_id}/resume").status_code == 409
    write_json(directory / "job.json", {**json.loads((directory / "job.json").read_text(encoding="utf-8")),
                                        "status": "cancelled", "error": "用户已取消"})
    launched[0][1].code = 2
    resumed = client.post(f"/api/optimization/jobs/{job_id}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["job_id"] == job_id
    assert resumed.json()["attempt"] == 2
    assert "--resume" in launched[1][0]
    assert launched[1][0][launched[1][0].index("--out") + 1] == str(directory.resolve())
    assert (directory / "config.json").read_bytes() == before
    assert (directory / "checkpoint.jsonl").read_bytes() == checkpoint
    assert not (directory / "cancel.request").exists()


def test_resume_waits_briefly_for_local_cancelled_worker_exit(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    class ExitingProcess(FakeProcess):
        def __init__(self):
            super().__init__()
            self.polls = 0
        def poll(self):
            self.polls += 1
            return 2 if self.polls >= 3 else None
    old = ExitingProcess()
    launched = []
    def popen(*args, **kwargs):
        process = old if not launched else FakeProcess()
        launched.append(process)
        return process
    monkeypatch.setattr(api.subprocess, "Popen", popen)
    job_id = client.post("/api/optimization/jobs", json={"max_trials": 2}).json()["job_id"]
    directory = tmp_path / "jobs" / job_id
    job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    write_json(directory / "job.json", {**job, "status": "cancelled"})
    result = client.post(f"/api/optimization/jobs/{job_id}/resume")
    assert result.status_code == 200
    assert result.json()["job_id"] == job_id
    assert len(launched) == 2


def test_resume_rejects_local_worker_still_finishing_without_mutation(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    old = FakeProcess()
    launched = []
    def popen(*args, **kwargs):
        launched.append(args)
        return old if len(launched) == 1 else FakeProcess()
    monkeypatch.setattr(api.subprocess, "Popen", popen)
    job_id = client.post("/api/optimization/jobs", json={"max_trials": 2}).json()["job_id"]
    directory = tmp_path / "jobs" / job_id
    job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    write_json(directory / "job.json", {**job, "status": "cancelled"})
    (directory / "cancel.request").write_text("requested", encoding="utf-8")
    result = client.post(f"/api/optimization/jobs/{job_id}/resume")
    assert result.status_code == 409
    assert "收尾" in result.json()["detail"]
    unchanged = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    assert unchanged["status"] == "cancelled"
    assert unchanged["attempt"] == 1
    assert (directory / "cancel.request").exists()
    assert len(launched) == 1
    old.code = 2
    assert client.post(f"/api/optimization/jobs/{job_id}/resume").status_code == 200


def test_other_api_worker_cannot_launch_while_terminal_lease_held(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    launched = []
    def popen(*args, **kwargs):
        launched.append(args)
        return FakeProcess()
    monkeypatch.setattr(api.subprocess, "Popen", popen)
    job_id = client.post("/api/optimization/jobs", json={"max_trials": 2}).json()["job_id"]
    directory = tmp_path / "jobs" / job_id
    job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    write_json(directory / "job.json", {**job, "status": "cancelled"})
    monkeypatch.setattr(api, "_active_process", None)
    monkeypatch.setattr(api, "_active_id", None)
    with file_lock(directory / "worker.lock"):
        resume = client.post(f"/api/optimization/jobs/{job_id}/resume")
        assert resume.status_code == 409
        assert "收尾" in resume.json()["detail"]
        assert client.post("/api/optimization/jobs", json={"max_trials": 3}).status_code == 409
        unchanged = json.loads((directory / "job.json").read_text(encoding="utf-8"))
        assert unchanged["status"] == "cancelled" and unchanged["attempt"] == 1
        assert len(launched) == 1
    assert client.post(f"/api/optimization/jobs/{job_id}/resume").status_code == 200


def test_other_api_worker_respects_live_lease_and_recovers_interrupted(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(api.subprocess, "Popen", lambda *a, **k: FakeProcess())
    job_id = client.post("/api/optimization/jobs", json={"max_trials": 2}).json()["job_id"]
    directory = tmp_path / "jobs" / job_id
    job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    job["started_at"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    write_json(directory / "job.json", job)
    monkeypatch.setattr(api, "_active_process", None)
    monkeypatch.setattr(api, "_active_id", None)
    # A missing acquisition record is uncertain even when the startup is old.
    assert client.get(f"/api/optimization/jobs/{job_id}").json()["status"] == "running"
    assert client.post("/api/optimization/jobs", json={"max_trials": 4}).status_code == 409
    job["lease_acquired_at"] = job["started_at"]
    write_json(directory / "job.json", job)
    with file_lock(directory / "worker.lock"):
        assert client.get(f"/api/optimization/jobs/{job_id}").json()["status"] == "running"
        assert client.post("/api/optimization/jobs", json={"max_trials": 4}).status_code == 409
    assert client.get(f"/api/optimization/jobs/{job_id}").json()["status"] == "interrupted"
    assert client.post(f"/api/optimization/jobs/{job_id}/resume").status_code == 200


def test_durable_terminal_state_and_rejections(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(api.subprocess, "Popen", lambda *a, **k: FakeProcess())
    job_id = client.post("/api/optimization/jobs", json={"max_trials": 2}).json()["job_id"]
    directory = tmp_path / "jobs" / job_id
    job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    write_json(directory / "job.json", {**job, "status": "completed", "finished_at": "2026-10-08T00:00:00+00:00"})
    (directory / "optimization-summary.json").write_text('{"research_acceptance":{"passed":false}}', encoding="utf-8")
    monkeypatch.setattr(api, "_active_process", None)
    monkeypatch.setattr(api, "_active_id", None)
    detail = client.get(f"/api/optimization/jobs/{job_id}").json()
    assert detail["status"] == "completed"
    assert detail["summary"]["research_acceptance"] == {"passed": False}
    assert client.post(f"/api/optimization/jobs/{job_id}/cancel").status_code == 409
    assert client.post(f"/api/optimization/jobs/{job_id}/resume").status_code == 409
    for suffix in ("cancel", "resume"):
        assert client.post(f"/api/optimization/jobs/not-an-id/{suffix}").status_code == 404
    monkeypatch.setattr(api.sys, "frozen", True, raising=False)
    assert client.post(f"/api/optimization/jobs/{job_id}/resume").status_code == 503


def test_zero_exit_without_durable_final_state_does_not_trust_stale_summary(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    process = FakeProcess()
    monkeypatch.setattr(api.subprocess, "Popen", lambda *a, **k: process)
    job_id = client.post("/api/optimization/jobs", json={"max_trials": 2}).json()["job_id"]
    directory = tmp_path / "jobs" / job_id
    (directory / "optimization-summary.json").write_text('{"old":true}', encoding="utf-8")
    process.code = 0
    job = client.get(f"/api/optimization/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "未写入最终状态" in job["error"]
    assert "summary" not in job


def test_worker_checks_cancel_each_callback_and_writes_final_state(tmp_path, monkeypatch):
    from scripts import optimize_turning_points as worker
    from src.optimization import turning_point

    directory = tmp_path / "job"
    directory.mkdir()
    config = directory / "config.json"
    config.write_text('{"max_trials":2}', encoding="utf-8")
    state = directory / "job.json"
    write_json(state, {"job_id": "a" * 32, "status": "running"})
    cancel = directory / "cancel.request"
    def fake_run(config, out, progress):
        progress({"status": "ok", "score": 1}, 1, 2)
        cancel.write_text("cancel", encoding="utf-8")
        progress({"status": "ok", "score": 2}, 2, 2)
        raise AssertionError("progress should have cancelled")
    monkeypatch.setattr(turning_point, "run_optimization", fake_run)
    monkeypatch.setattr(sys, "argv", ["optimize", "--config", str(config), "--out", str(directory),
                                  "--job-state", str(state), "--cancel-file", str(cancel),
                                  "--lease-file", str(directory / "worker.lock")])
    assert worker.main() == 2
    assert json.loads(state.read_text(encoding="utf-8"))["status"] == "cancelled"
    assert json.loads((directory / "progress.json").read_text(encoding="utf-8")) == {"completed": 2, "total": 2}


def test_worker_resume_mismatch_keeps_prior_files_and_reports_error(tmp_path, monkeypatch):
    from scripts import optimize_turning_points as worker
    from src.optimization import turning_point

    directory = tmp_path / "job"
    directory.mkdir()
    config = directory / "config.json"
    config.write_text('{"max_trials":2}', encoding="utf-8")
    state = directory / "job.json"
    write_json(state, {"job_id": "b" * 32, "status": "running"})
    (directory / "checkpoint.jsonl").write_text('prior checkpoint\n', encoding="utf-8")
    (directory / "parameters.json").write_text('prior parameters', encoding="utf-8")
    (directory / "optimization-summary.json").write_text('prior summary', encoding="utf-8")
    def mismatch(config, out, progress):
        (directory / "parameters.json").write_text('new parameters', encoding="utf-8")
        raise ValueError("checkpoint fingerprint mismatch or invalid header")
    monkeypatch.setattr(turning_point, "run_optimization", mismatch)
    monkeypatch.setattr(sys, "argv", ["optimize", "--config", str(config), "--out", str(directory), "--resume",
                                  "--job-state", str(state), "--cancel-file", str(directory / "cancel.request"),
                                  "--lease-file", str(directory / "worker.lock")])
    assert worker.main() == 1
    job = json.loads(state.read_text(encoding="utf-8"))
    assert job["status"] == "failed"
    assert "fingerprint mismatch" in job["error"]
    assert (directory / "checkpoint.jsonl").read_text(encoding="utf-8") == 'prior checkpoint\n'
    assert (directory / "parameters.json").read_text(encoding="utf-8") == 'prior parameters'
    assert (directory / "optimization-summary.json").read_text(encoding="utf-8") == 'prior summary'


def test_worker_writes_completed_state(tmp_path, monkeypatch):
    from scripts import optimize_turning_points as worker
    from src.optimization import turning_point

    directory = tmp_path / "job"
    directory.mkdir()
    config = directory / "config.json"
    config.write_text('{"max_trials":1}', encoding="utf-8")
    state = directory / "job.json"
    write_json(state, {"job_id": "c" * 32, "status": "running"})
    monkeypatch.setattr(turning_point, "run_optimization", lambda config, out, progress: {
        "best_training": {"score": 1}, "validation": {}, "metadata": {}, "elapsed_seconds": 0})
    monkeypatch.setattr(sys, "argv", ["optimize", "--config", str(config), "--out", str(directory),
                                  "--job-state", str(state), "--cancel-file", str(directory / "cancel.request"),
                                  "--lease-file", str(directory / "worker.lock")])
    assert worker.main() == 0
    job = json.loads(state.read_text(encoding="utf-8"))
    assert job["status"] == "completed"
    assert job["return_code"] == 0
    assert job["finished_at"]
