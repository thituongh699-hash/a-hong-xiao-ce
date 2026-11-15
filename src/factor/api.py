"""因子研究的本地 HTTP 接口。

与 ``src/optimization/api.py`` 保持同一套做法：独立 router、文件落盘、
不把长任务塞进主进程状态机。研究结果写进 ``reports/factor-studies/<id>/``，
报告页可直接用浏览器打开。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .dataset import load_bars, synthetic_bars
from .gates import GateConfig
from .panel import build_panel
from .pipeline import StudyConfig, run_study
from .preprocess import PreprocessConfig
from .report import dumps, render_html
from .spec import catalog

ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = ROOT / "reports" / "factor-studies"
PAGE = ROOT / "static" / "factor-lab.html"
_ID_PATTERN = re.compile(r"fa-\d{8}-\d{6}-[0-9a-f]{6}\Z")
MAX_CODES = 300

router = APIRouter()


class StudyRequest(BaseModel):
    codes: list[str] = Field(default_factory=list)
    start: str | None = None
    end: str | None = None
    source: str = "tdx"
    factor_ids: list[str] = Field(default_factory=lambda: ["vol_20"])
    horizon: int = Field(default=20, ge=1, le=120)
    quantiles: int = Field(default=5, ge=2, le=10)
    entry_lag: int = Field(default=1, ge=0, le=5)
    min_names: int = Field(default=20, ge=5, le=2000)
    oos_ratio: float = Field(default=0.3, ge=0.1, le=0.6)
    universe: str = "未命名股票池"
    survivorship_complete: bool = False
    industry: dict[str, str] | None = None
    combine_method: str = "icir"
    round_trip_cost_bps: float = Field(default=20.0, ge=0.0, le=200.0)
    cost_multiple: float = Field(default=2.0, ge=1.0, le=5.0)
    demo: bool = False
    persist: bool = True


def _study_dir(study_id: str) -> Path:
    if not _ID_PATTERN.match(study_id):
        raise HTTPException(status_code=400, detail="研究编号格式非法")
    directory = STUDIES_ROOT / study_id
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="研究记录不存在")
    return directory


@router.get("/factor", response_class=HTMLResponse, include_in_schema=False)
def factor_page() -> HTMLResponse:
    if not PAGE.is_file():
        raise HTTPException(status_code=404, detail="页面未安装")
    return HTMLResponse(PAGE.read_text(encoding="utf-8"))


@router.get("/api/factor/catalog")
def factor_catalog() -> dict[str, Any]:
    return {
        "factors": catalog(),
        "preprocess_defaults": PreprocessConfig().describe(),
        "gate_defaults": GateConfig().describe(),
        "horizon_note": "默认持有 20 日并按 20 日取样，避免重叠收益把 t 值抬高。",
        "cost_note": "扣费口径为换手率 × 双边费率 × 压力倍数，默认 20bp 双边再翻倍。",
    }


@router.post("/api/factor/study")
def create_study(payload: StudyRequest) -> dict[str, Any]:
    if not payload.demo:
        if not payload.codes:
            raise HTTPException(status_code=400, detail="非演示模式必须提供 codes")
        if len(payload.codes) > MAX_CODES:
            raise HTTPException(status_code=400, detail=f"单次最多 {MAX_CODES} 只标的")
        if not payload.start or not payload.end:
            raise HTTPException(status_code=400, detail="非演示模式必须提供 start 与 end")

    errors: dict[str, str] = {}
    if payload.demo:
        bars = synthetic_bars()
    else:
        bars, errors = load_bars(
            payload.codes, payload.start, payload.end, source=payload.source
        )
        if not bars:
            raise HTTPException(status_code=400, detail=f"没有取到任何行情：{errors}")

    try:
        build = build_panel(bars)
        config = StudyConfig(
            factor_ids=tuple(payload.factor_ids),
            horizon=payload.horizon,
            quantiles=payload.quantiles,
            entry_lag=payload.entry_lag,
            min_names=payload.min_names,
            start=payload.start,
            end=payload.end,
            universe=payload.universe if not payload.demo else "合成面板（演示）",
            # 合成面板按构造不存在退市偏差，因此演示模式才允许标记为完整样本。
            survivorship_complete=True if payload.demo else payload.survivorship_complete,
            industry=payload.industry,
            oos_ratio=payload.oos_ratio,
            combine_method=payload.combine_method,
            round_trip_cost_bps=payload.round_trip_cost_bps,
            cost_multiple=payload.cost_multiple,
            demo=payload.demo,
        )
        report = run_study(build, config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if errors:
        report["data_quality"]["fetch_errors"] = errors
        report["warnings"].append(f"{len(errors)} 只标的取数失败，已从样本中剔除。")

    response: dict[str, Any] = {
        "study_id": report["study_id"],
        "summary": report["summary"],
        "sample": report["sample"],
        "warnings": report["warnings"],
        "persisted": False,
        "report": report,
    }
    if payload.persist:
        directory = STUDIES_ROOT / str(report["study_id"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(dumps(report), encoding="utf-8")
        (directory / "index.html").write_text(render_html(report), encoding="utf-8")
        response["persisted"] = True
        response["report_url"] = f"/api/factor/study/{report['study_id']}/report"
        response["json_url"] = f"/api/factor/study/{report['study_id']}"
    return response


@router.get("/api/factor/studies")
def list_studies(limit: int = 20) -> dict[str, Any]:
    if not STUDIES_ROOT.is_dir():
        return {"studies": []}
    rows: list[dict[str, Any]] = []
    for directory in sorted(STUDIES_ROOT.iterdir(), reverse=True):
        if not directory.is_dir() or not _ID_PATTERN.match(directory.name):
            continue
        summary_path = directory / "report.json"
        if not summary_path.is_file():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({
            "study_id": data.get("study_id", directory.name),
            "universe": data.get("universe"),
            "generated_at": data.get("generated_at"),
            "headline": (data.get("summary") or {}).get("headline"),
        })
        if len(rows) >= max(1, min(int(limit), 100)):
            break
    return {"studies": rows}


@router.get("/api/factor/study/{study_id}")
def get_study(study_id: str) -> Any:
    directory = _study_dir(study_id)
    path = directory / "report.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="报告数据缺失")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/api/factor/study/{study_id}/report", response_class=HTMLResponse)
def get_study_report(study_id: str) -> HTMLResponse:
    directory = _study_dir(study_id)
    path = directory / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="报告页面缺失")
    return HTMLResponse(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
