import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.optimization import api


class FakeProcess:
    pid = 4321

    def __init__(self):
        self.return_code = None

    def poll(self):
        return self.return_code


def client_for(tmp_path, monkeypatch):
    root = tmp_path
    script = root / "scripts" / "optimize_turning_points.py"
    script.parent.mkdir()
    script.write_text("# fake script\n", encoding="utf-8")
    page = root / "static" / "parameter-optimization.html"
    page.parent.mkdir()
    page.write_text("<html>test</html>", encoding="utf-8")
    monkeypatch.setattr(api, "ROOT", root)
    monkeypatch.setattr(api, "JOBS_ROOT", root / "reports" / "optimization-jobs")
    monkeypatch.setattr(api, "PRECOMPUTED_DIR", root / "reports" / "turning-point-optimization-2026-10-08")
    monkeypatch.setattr(api, "SCRIPT", script)
    monkeypatch.setattr(api, "PAGE", page)
    monkeypatch.setattr(api, "_active_process", None)
    monkeypatch.setattr(api, "_active_id", None)
    monkeypatch.setattr(api, "_validate_config", lambda body: {"max_trials": int(body["max_trials"])})
    monkeypatch.setattr(api, "_defaults", lambda: {"max_trials": 1000})
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app), root


def test_defaults_and_page(tmp_path, monkeypatch):
    client, _ = client_for(tmp_path, monkeypatch)
    assert client.get("/api/optimization/defaults").json() == {"max_trials": 1000}
    assert client.get("/optimization").status_code == 200


def test_launch_once_and_reject_second_active_job(tmp_path, monkeypatch):
    client, root = client_for(tmp_path, monkeypatch)
    calls = []
    process = FakeProcess()
    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        return process
    monkeypatch.setattr(api.subprocess, "Popen", fake_popen)
    first = client.post("/api/optimization/jobs", json={"max_trials": 5})
    assert first.status_code == 201
    job = first.json()
    assert job["status"] == "running"
    assert len(job["job_id"]) == 32
    assert client.post("/api/optimization/jobs", json={"max_trials": 7}).status_code == 409
    assert len(calls) == 1
    args, options = calls[0]
    directory = root / "reports" / "optimization-jobs" / job["job_id"]
    assert args == [api.sys.executable, str((root / "scripts" / "optimize_turning_points.py").resolve()),
                    "--config", str((directory / "config.json").resolve()), "--out", str(directory.resolve()),
                    "--job-state", str((directory / "job.json").resolve()),
                    "--cancel-file", str((directory / "cancel.request").resolve()),
                    "--lease-file", str((directory / "worker.lock").resolve())]
    assert options["shell"] is False
    assert options["cwd"] == str(root)
    assert options["env"]["PYTHONUTF8"] == "1"
    assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert options["env"].get("PATH") == api.os.environ.get("PATH")
    assert options["creationflags"] == (getattr(api.subprocess, "CREATE_NO_WINDOW", 0) if api.os.name == "nt" else 0)
    assert json.loads((directory / "config.json").read_text(encoding="utf-8")) == {"max_trials": 5}
    assert client.get(f"/api/optimization/jobs/{job['job_id']}").json()["status"] == "running"


def test_completed_job_summary_report_csv_and_list(tmp_path, monkeypatch):
    client, root = client_for(tmp_path, monkeypatch)
    process = FakeProcess()
    monkeypatch.setattr(api.subprocess, "Popen", lambda *args, **kwargs: process)
    job_id = client.post("/api/optimization/jobs", json={"max_trials": 2}).json()["job_id"]
    directory = root / "reports" / "optimization-jobs" / job_id
    (directory / "optimization-summary.json").write_text('{"best_training":{"score":2},"validation":{"score":1}}', encoding="utf-8")
    (directory / "optimization-report.html").write_text(
        '<html><a href="trials.csv">other</a><a href="training-grid.csv">grid</a>'
        '<a href="frozen-best-training.json">frozen</a><a href="optimization-summary.json">summary</a>'
        '<a href="config.json">unsafe</a></html>', encoding="utf-8")
    (directory / "trials.csv").write_text("trial,score\n1,2\n", encoding="utf-8")
    (directory / "training-grid.csv").write_text("trial,score\n1,2\n", encoding="utf-8")
    (directory / "frozen-best-training.json").write_text('{"score":2}', encoding="utf-8")
    (directory / "job.log").write_text('not json\n{"completed":1,"total":2}\n', encoding="utf-8")
    saved = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    api._write_json(directory / "job.json", {**saved, "status": "completed", "finished_at": api._now(), "return_code": 0})
    process.return_code = 0
    detail = client.get(f"/api/optimization/jobs/{job_id}").json()
    assert detail["status"] == "completed"
    assert detail["summary"]["best_training"]["score"] == 2
    assert detail["summary"]["validation"]["score"] == 1
    assert detail["report_url"].endswith("/report")
    assert detail["progress"] == {"completed": 1, "total": 2}
    report = client.get(detail["report_url"])
    assert report.status_code == 200
    assert f'href="/api/optimization/jobs/{job_id}/files/training-grid.csv"' in report.text
    assert f'href="/api/optimization/jobs/{job_id}/files/frozen-best-training.json"' in report.text
    assert f'href="/api/optimization/jobs/{job_id}/files/optimization-summary.json"' in report.text
    assert 'href="config.json"' in report.text
    assert client.get(f"/api/optimization/jobs/{job_id}/files/training-grid.csv").status_code == 200
    assert client.get(f"/api/optimization/jobs/{job_id}/files/frozen-best-training.json").status_code == 200
    assert client.get(f"/api/optimization/jobs/{job_id}/files/optimization-summary.json").status_code == 200
    assert client.get(f"/api/optimization/jobs/{job_id}/csv/training-grid.csv").status_code == 200
    assert client.get(f"/api/optimization/jobs/{job_id}/files/trials.csv").status_code == 404
    assert client.get(f"/api/optimization/jobs/{job_id}/files/config.json").status_code == 404
    assert client.get(f"/api/optimization/jobs/{job_id}/files/job.json").status_code == 404
    assert client.get(f"/api/optimization/jobs/{job_id}/files/job.log").status_code == 404
    assert client.get(f"/api/optimization/jobs/{job_id}/csv/..%2Fconfig.json").status_code == 404
    assert client.get(f"/api/optimization/jobs/{job_id}/csv/config.json").status_code == 404
    jobs = client.get("/api/optimization/jobs").json()["jobs"]
    assert any(item["job_id"] == job_id and item["status"] == "completed" for item in jobs)


def test_failed_process_and_validation_rejection(tmp_path, monkeypatch):
    client, root = client_for(tmp_path, monkeypatch)
    calls = []
    process = FakeProcess()
    def fake_popen(*args, **kwargs):
        calls.append(args)
        return process
    monkeypatch.setattr(api.subprocess, "Popen", fake_popen)
    job_id = client.post("/api/optimization/jobs", json={"max_trials": 1}).json()["job_id"]
    process.return_code = 2
    detail = client.get(f"/api/optimization/jobs/{job_id}").json()
    assert detail["status"] == "failed"
    assert "代码 2" in detail["error"]
    assert client.get(f"/api/optimization/jobs/{job_id}/report").status_code == 404
    monkeypatch.setattr(api, "_validate_config", lambda _: (_ for _ in ()).throw(ValueError("bad config")))
    rejected = client.post("/api/optimization/jobs", json={"max_trials": 2})
    assert rejected.status_code == 422
    assert len(calls) == 1
    assert len([path for path in (root / "reports" / "optimization-jobs").iterdir() if path.is_dir()]) == 1


def test_frozen_runtime_does_not_relaunch_desktop(tmp_path, monkeypatch):
    client, _ = client_for(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(api.sys, "frozen", True, raising=False)
    monkeypatch.setattr(api.subprocess, "Popen", lambda *args, **kwargs: calls.append(args))
    response = client.post("/api/optimization/jobs", json={"max_trials": 1})
    assert response.status_code == 503
    assert calls == []


def test_restart_orphan_is_interrupted_and_precomputed_is_fixed(tmp_path, monkeypatch):
    client, root = client_for(tmp_path, monkeypatch)
    job_id = "a" * 32
    directory = root / "reports" / "optimization-jobs" / job_id
    directory.mkdir(parents=True)
    (directory / "job.json").write_text(json.dumps({"job_id": job_id, "status": "running"}), encoding="utf-8")
    detail = client.get(f"/api/optimization/jobs/{job_id}").json()
    assert detail["status"] == "interrupted"
    assert client.get("/api/optimization/jobs/../config.json").status_code == 404
    assert client.get("/api/optimization/jobs/not-a-job").status_code == 404
    precomputed = root / "reports" / "turning-point-optimization-2026-10-08"
    precomputed.mkdir(parents=True)
    (precomputed / "optimization-summary.json").write_text('{"training":{"score":9}}', encoding="utf-8")
    (precomputed / "report.html").write_text("<html>prior report</html>", encoding="utf-8")
    listed = client.get("/api/optimization/jobs").json()["jobs"]
    assert any(job["job_id"] == api.PRECOMPUTED_ID and job["status"] == "completed" for job in listed)
    assert client.get(f"/api/optimization/jobs/{api.PRECOMPUTED_ID}/report").status_code == 200
