"""Isolated HTTP fixture: real scope routes/dashboard; deterministic task stub.

Never starts the production lifespan, writes user configuration, or fetches prices.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import server

server.stock_manager.stocks = [
    {"code": "600036.SH", "name": "主板普通示例"},
    {"code": "300750.SZ", "name": "创业板普通示例"},
    {"code": "688001.SH", "name": "科创板普通示例"},
    {"code": "920001.BJ", "name": "北交所普通示例"},
    {"code": "600001.SH", "name": "*ST测试示例"},
    {"code": "600002.SH", "name": "测试示例退"},
]
server.stock_manager.ensure_loaded = lambda: None
server.ConfigLoader.reload = lambda: {"system.mode": "backtest"}
server.engine_task = None
server._live_running_codes = lambda: []
submissions = []
server.start_new_backtest_report = lambda *a: f"scope-qa-{len(submissions) + 1}"
server._spawn_backtest_task = lambda *a, **k: submissions.append({"code": a[0], "scope": k.get("trading_scope")})

app = FastAPI()
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.add_api_route("/api/backtest/trading_scope/options", server.api_trading_scope_options, methods=["GET"])
app.add_api_route("/api/backtest/trading_scope/preview", server.api_trading_scope_preview, methods=["POST"])
app.add_api_route("/api/search", server.search_stocks, methods=["GET"])
app.add_api_route("/api/control/start_backtest", server.api_start_backtest, methods=["POST"])


@app.get("/")
async def dashboard():
    return FileResponse(ROOT / "dashboard.html")


@app.get("/__qa__/submissions")
async def submitted():
    return submissions


@app.post("/__qa__/reset")
async def reset():
    submissions.clear()
    return {"status": "success"}


@app.post("/api/config/test_price_readiness")
async def readiness():
    return {"ok": True, "checks": [], "price_policy": {"signal_price_mode": "none", "execution_price_mode": "none"}}


@app.get("/api/config")
async def configuration():
    return {"status": "success", "config": {"system.mode": "backtest"}}


@app.get("/api/strategies")
async def strategies():
    return {"status": "success", "strategies": [{"id": "S-QA", "name": "验收示例", "timeframe": "D"}]}


@app.get("/api/status/light")
@app.get("/api/status")
async def status():
    return {"live_enabled": False, "is_running": False, "live_running_codes": []}


@app.websocket("/ws")
async def socket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass


@app.api_route("/api/{path:path}", methods=["GET", "POST"])
async def unrelated(path: str):
    return {"status": "error", "msg": "unrelated widget disabled in isolated scope QA", "results": [], "items": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9145)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
