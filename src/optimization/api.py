"""Local FastAPI routes for bounded turning-point optimization jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
from uuid import uuid4
from .job_state import LockBusy, file_lock, read_json as _read_json, write_json as _write_json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse


ROOT = Path(__file__).resolve().parents[2]
JOBS_ROOT = ROOT / "reports" / "optimization-jobs"
PRECOMPUTED_ID = "turning-point-2026-10-08"
PRECOMPUTED_DIR = ROOT / "reports" / "turning-point-optimization-2026-10-08"
SCRIPT = ROOT / "scripts" / "optimize_turning_points.py"
PAGE = ROOT / "static" / "parameter-optimization.html"
SUMMARY_NAME = "optimization-summary.json"
_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_CSV_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.csv\Z", re.IGNORECASE)
_REPORT_LINK = re.compile(r"(\bhref\s*=\s*)([\"'])([^\"']+)\2", re.IGNORECASE)
ARTIFACT_NAMES = frozenset({
    "training-grid.csv", "validation-trades.csv", "validation-entries.csv",
    "training-exclusions.csv", "validation-exclusions.csv",
    "frozen-best-training.json", "optimization-summary.json",
    "top-training-with-validation.json", "parameters.json",
})
_lock = threading.Lock()
_active_process: subprocess.Popen[Any] | None = None
_active_id: str | None = None

router = APIRouter()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _defaults() -> dict[str, Any]:
    from .turning_point import default_config

    return default_config()


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    from .turning_point import validate_config

    return validate_config(config)


def _mutation_lock():
    return file_lock(JOBS_ROOT / ".mutation.lock", timeout=2.0)


def _direct_file(directory: Path, name: str) -> Path | None:
    path = directory / name
    return path if path.is_file() and path.resolve().parent == directory.resolve() else None


def _job_directory(job_id: str) -> Path:
    if job_id == PRECOMPUTED_ID:
        return PRECOMPUTED_DIR
    if not _ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="未找到该任务")
    directory = JOBS_ROOT / job_id
    if directory.resolve().parent != JOBS_ROOT.resolve():
        raise HTTPException(status_code=404, detail="未找到该任务")
    return directory


def _summary(directory: Path) -> dict[str, Any] | None:
    path = _direct_file(directory, SUMMARY_NAME)
    if path is None:
        return None
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _artifacts(directory: Path, job_id: str) -> dict[str, Any]:
    html_files = sorted(path for path in directory.glob("*.html") if path.is_file() and path.resolve().parent == directory.resolve())
    files = sorted(path for name in ARTIFACT_NAMES if (path := _direct_file(directory, name)) is not None)
    return {
        "report_url": f"/api/optimization/jobs/{job_id}/report" if html_files else None,
        "csv_files": [{"name": path.name, "url": f"/api/optimization/jobs/{job_id}/files/{path.name}"}
                      for path in files if path.suffix == ".csv"],
        "downloads": [{"name": path.name, "url": f"/api/optimization/jobs/{job_id}/files/{path.name}"}
                      for path in files],
    }


def _progress(directory: Path) -> dict[str, int] | None:
    progress_file = _direct_file(directory, "progress.json")
    if progress_file is not None:
        try:
            value = _read_json(progress_file)
            if (type(value.get("completed")) is int and type(value.get("total")) is int
                    and 0 <= value["completed"] <= value["total"] and value["total"] > 0):
                return {"completed": value["completed"], "total": value["total"]}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    path = _direct_file(directory, "job.log")
    if path is None:
        return None
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            start = max(0, size - 65_536)
            stream.seek(start)
            tail = stream.read(65_536)
        if start:
            tail = tail.partition(b"\n")[2]
        for raw in reversed(tail.splitlines()):
            try:
                item = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if (isinstance(item, dict) and type(item.get("completed")) is int
                    and type(item.get("total")) is int and 0 <= item["completed"] <= item["total"]
                    and item["total"] > 0):
                return {"completed": item["completed"], "total": item["total"]}
    except OSError:
        pass
    return None


def _precomputed() -> dict[str, Any] | None:
    summary = _summary(PRECOMPUTED_DIR)
    if summary is None:
        return None
    return {"job_id": PRECOMPUTED_ID, "status": "completed", "source": "precomputed",
            "summary": summary, **_artifacts(PRECOMPUTED_DIR, PRECOMPUTED_ID)}


def _load_job(job_id: str) -> dict[str, Any]:
    directory = _job_directory(job_id)
    if job_id == PRECOMPUTED_ID:
        job = _precomputed()
        if job is None:
            raise HTTPException(status_code=404, detail="未找到该任务")
        return job
    path = _direct_file(directory, "job.json")
    if path is None:
        raise HTTPException(status_code=404, detail="未找到该任务")
    try:
        job = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="任务状态文件损坏") from exc
    if job.get("job_id") != job_id:
        raise HTTPException(status_code=500, detail="任务状态文件不匹配")
    if job.get("status") == "running":
        try:
            with _mutation_lock():
                job = _reconcile_running(directory, _read_json(path))
        except LockBusy:
            pass
    job["progress"] = _progress(directory)
    job["cancel_requested"] = (directory / "cancel.request").exists() and job.get("status") == "running"
    if job.get("status") == "completed":
        job["summary"] = _summary(directory)
        job.update(_artifacts(directory, job_id))
    else:
        job.update(report_url=None, csv_files=[], downloads=[])
    return job


def _reconcile_running(directory: Path, job: dict[str, Any]) -> dict[str, Any]:
    """A released lease proves interruption only after worker acquisition was recorded."""
    if job.get("status") != "running":
        return job
    if _active_id == job.get("job_id") and _active_process is not None and _active_process.poll() is None:
        return job
    try:
        with file_lock(directory / "worker.lock"):
            # The worker writes its terminal state before releasing this lease.
            # Re-read here so a completed write cannot be overwritten as interrupted.
            job = _read_json(directory / "job.json")
            if job.get("status") != "running":
                return job
    except LockBusy:
        return job
    if not job.get("lease_acquired_at") and job.get("started_at"):
        return job  # Startup is uncertain; a new worker could still acquire the lease.
    job.update(status="interrupted", finished_at=_now(), error="优化进程已结束但未写入最终状态，可续跑。")
    _write_json(directory / "job.json", job)
    return job


def _refresh_active_locked() -> None:
    global _active_process, _active_id
    if _active_process is None or _active_id is None:
        return
    return_code = _active_process.poll()
    if return_code is None:
        return
    directory = _job_directory(_active_id)
    job = _read_json(directory / "job.json")
    if job.get("status") == "running":
        reason = f"优化进程退出，代码 {return_code}，但未写入最终状态。"
        job.update(status="failed", finished_at=_now(), return_code=return_code, error=reason)
        try:
            with _mutation_lock():
                latest = _read_json(directory / "job.json")
                if latest.get("status") == "running":
                    _write_json(directory / "job.json", job)
        except LockBusy:
            return
    _active_process = None
    _active_id = None


def _assert_idle_locked() -> None:
    if _active_process is not None and _active_process.poll() is None:
        raise HTTPException(status_code=409, detail="已有优化任务正在运行，请等待完成。")
    if JOBS_ROOT.is_dir():
        for directory in JOBS_ROOT.iterdir():
            if not directory.is_dir() or not _ID_PATTERN.fullmatch(directory.name):
                continue
            path = _direct_file(directory, "job.json")
            if path is None:
                continue
            try:
                job = _reconcile_running(directory, _read_json(path))
            except (OSError, ValueError, json.JSONDecodeError):
                raise HTTPException(status_code=409, detail="存在无法确认状态的优化任务，请检查任务目录。")
            if job.get("status") == "running":
                raise HTTPException(status_code=409, detail="已有优化任务正在运行，请等待完成。")
            if not _worker_lease_is_free(directory):
                raise HTTPException(status_code=409, detail="优化任务正在收尾，请稍后再试。")


def _worker_lease_is_free(directory: Path) -> bool:
    try:
        with file_lock(directory / "worker.lock"):
            return True
    except LockBusy:
        return False


def _wait_for_worker_cleanup(directory: Path, job_id: str, timeout: float = 0.5) -> None:
    """A durable terminal status can precede process exit and lease release."""
    deadline = time.monotonic() + timeout
    while True:
        local_alive = (_active_id == job_id and _active_process is not None
                       and _active_process.poll() is None)
        if not local_alive and _worker_lease_is_free(directory):
            return
        if time.monotonic() >= deadline:
            raise HTTPException(status_code=409, detail="任务正在收尾，请稍后续跑。")
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))


def _launch_job(directory: Path, job: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    global _active_process, _active_id
    args = [str(Path(sys.executable).resolve()), str(SCRIPT.resolve()), "--config", str((directory / "config.json").resolve()),
            "--out", str(directory.resolve()), "--job-state", str((directory / "job.json").resolve()),
            "--cancel-file", str((directory / "cancel.request").resolve()), "--lease-file", str((directory / "worker.lock").resolve())]
    if resume:
        args.append("--resume")
    try:
        child_env = os.environ.copy()
        child_env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        with (directory / "job.log").open("a" if resume else "w", encoding="utf-8") as log:
            process = subprocess.Popen(args, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, shell=False, env=child_env, creationflags=creationflags)
    except (OSError, ValueError) as exc:
        job.update(status="failed", finished_at=_now(), error=f"无法启动优化进程：{exc}")
        _write_json(directory / "job.json", job)
        raise HTTPException(status_code=500, detail=job["error"]) from exc
    _active_process, _active_id = process, job["job_id"]
    job["pid"] = process.pid  # Response only; avoid racing the worker's terminal write.
    return job


@router.get("/optimization", include_in_schema=False)
def optimization_page() -> FileResponse:
    if not PAGE.is_file():
        raise HTTPException(status_code=404, detail="页面未安装")
    return FileResponse(PAGE, media_type="text/html; charset=utf-8")


@router.get("/api/optimization/defaults")
def optimization_defaults() -> dict[str, Any]:
    return _defaults()


@router.get("/api/optimization/jobs")
def list_optimization_jobs() -> dict[str, list[dict[str, Any]]]:
    with _lock:
        _refresh_active_locked()
        jobs = []
        if JOBS_ROOT.is_dir():
            for directory in JOBS_ROOT.iterdir():
                if not directory.is_dir() or not _ID_PATTERN.fullmatch(directory.name):
                    continue
                try:
                    jobs.append(_load_job(directory.name))
                except HTTPException:
                    continue
        precomputed = _precomputed()
        if precomputed is not None:
            jobs.append(precomputed)
    jobs.sort(key=lambda job: job.get("created_at", ""), reverse=True)
    return {"jobs": jobs}


@router.post("/api/optimization/jobs", status_code=201)
def create_optimization_job(config: dict[str, Any]) -> dict[str, Any]:
    global _active_process, _active_id
    try:
        normalized = _validate_config(config)
        if not isinstance(normalized, dict):
            raise ValueError("配置校验必须返回对象")
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not SCRIPT.is_file():
        raise HTTPException(status_code=503, detail="优化脚本尚未安装")
    if getattr(sys, "frozen", False):
        raise HTTPException(status_code=503, detail="打包程序无法启动 Python 优化脚本，请使用源码运行。")
    with _lock:
        _refresh_active_locked()
        try:
            with _mutation_lock():
                _assert_idle_locked()
                job_id = uuid4().hex
                directory = JOBS_ROOT / job_id
                directory.mkdir(parents=True, exist_ok=False)
                _write_json(directory / "config.json", normalized)
                job = {"job_id": job_id, "status": "running", "source": "local",
                       "created_at": _now(), "started_at": _now(), "finished_at": None,
                       "error": None, "return_code": None, "attempt": 1}
                _write_json(directory / "job.json", job)
                return _launch_job(directory, job, resume=False)
        except LockBusy as exc:
            raise HTTPException(status_code=409, detail="优化任务状态正在更新，请稍后重试。") from exc


@router.post("/api/optimization/jobs/{job_id}/cancel")
def cancel_optimization_job(job_id: str) -> dict[str, Any]:
    directory = _job_directory(job_id)
    if job_id == PRECOMPUTED_ID:
        raise HTTPException(status_code=409, detail="已有实验结果不能取消")
    with _lock:
        _refresh_active_locked()
        try:
            with _mutation_lock():
                path = _direct_file(directory, "job.json")
                if path is None:
                    raise HTTPException(status_code=404, detail="未找到该任务")
                job = _reconcile_running(directory, _read_json(path))
                if job.get("status") != "running":
                    raise HTTPException(status_code=409, detail="任务未在运行")
                _write_json(directory / "cancel.request", {"requested_at": _now()})
                return _load_job_without_reconcile(directory, job)
        except LockBusy as exc:
            raise HTTPException(status_code=409, detail="优化任务状态正在更新，请稍后重试。") from exc


@router.post("/api/optimization/jobs/{job_id}/resume")
def resume_optimization_job(job_id: str) -> dict[str, Any]:
    directory = _job_directory(job_id)
    if job_id == PRECOMPUTED_ID:
        raise HTTPException(status_code=409, detail="已有实验结果不能续跑")
    if getattr(sys, "frozen", False):
        raise HTTPException(status_code=503, detail="打包程序无法启动 Python 优化脚本，请使用源码运行。")
    if not SCRIPT.is_file():
        raise HTTPException(status_code=503, detail="优化脚本尚未安装")
    with _lock:
        _refresh_active_locked()
        try:
            with _mutation_lock():
                path = _direct_file(directory, "job.json")
                if path is None:
                    raise HTTPException(status_code=404, detail="未找到该任务")
                job = _reconcile_running(directory, _read_json(path))
                if job.get("status") not in {"cancelled", "interrupted", "failed"}:
                    raise HTTPException(status_code=409, detail="只有已取消、中断或失败的任务可以续跑")
                if _direct_file(directory, "config.json") is None:
                    raise HTTPException(status_code=409, detail="任务配置缺失，无法续跑")
                _wait_for_worker_cleanup(directory, job_id)
                _assert_idle_locked()
                (directory / "cancel.request").unlink(missing_ok=True)
                job.update(status="running", started_at=_now(), finished_at=None,
                           error=None, return_code=None, attempt=int(job.get("attempt", 1))+1)
                job.pop("lease_acquired_at", None)
                _write_json(path, job)
                return _launch_job(directory, job, resume=True)
        except LockBusy as exc:
            raise HTTPException(status_code=409, detail="优化任务状态正在更新，请稍后重试。") from exc


def _load_job_without_reconcile(directory: Path, job: dict[str, Any]) -> dict[str, Any]:
    return {**job, "progress": _progress(directory), "cancel_requested": (directory / "cancel.request").exists(),
            "report_url": None, "csv_files": [], "downloads": []}


@router.get("/api/optimization/jobs/{job_id}")
def get_optimization_job(job_id: str) -> dict[str, Any]:
    with _lock:
        _refresh_active_locked()
        return _load_job(job_id)


@router.get("/api/optimization/jobs/{job_id}/report")
def get_optimization_report(job_id: str) -> HTMLResponse:
    with _lock:
        _refresh_active_locked()
        job = _load_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=404, detail="报告尚未生成")
    directory = _job_directory(job_id)
    preferred = directory / "optimization-report.html"
    reports = [preferred] if preferred.is_file() else sorted(directory.glob("*.html"))
    for path in reports:
        if path.is_file() and path.resolve().parent == directory.resolve():
            page = path.read_text(encoding="utf-8")
            def rewrite(match: re.Match[str]) -> str:
                name = match.group(3)
                if name in ARTIFACT_NAMES:
                    return f"{match.group(1)}{match.group(2)}/api/optimization/jobs/{job_id}/files/{name}{match.group(2)}"
                return match.group(0)
            return HTMLResponse(_REPORT_LINK.sub(rewrite, page))
    raise HTTPException(status_code=404, detail="未找到报告")


@router.get("/api/optimization/jobs/{job_id}/files/{name}")
def get_optimization_file(job_id: str, name: str) -> FileResponse:
    if name not in ARTIFACT_NAMES:
        raise HTTPException(status_code=404, detail="未找到文件")
    with _lock:
        _refresh_active_locked()
        job = _load_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=404, detail="文件尚未生成")
    directory = _job_directory(job_id)
    path = _direct_file(directory, name)
    if path is None:
        raise HTTPException(status_code=404, detail="未找到文件")
    media_type = "text/csv; charset=utf-8" if path.suffix == ".csv" else "application/json; charset=utf-8"
    return FileResponse(path, media_type=media_type, filename=name)


@router.get("/api/optimization/jobs/{job_id}/csv/{name}")
def get_optimization_csv(job_id: str, name: str) -> FileResponse:
    if not _CSV_PATTERN.fullmatch(name) or name not in ARTIFACT_NAMES:
        raise HTTPException(status_code=404, detail="未找到文件")
    return get_optimization_file(job_id, name)
