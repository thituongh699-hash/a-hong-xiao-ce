# run_live.py
import asyncio
import os

from src.utils.dependency_bootstrap import ensure_project_dependencies

ensure_project_dependencies()

from src.core.live_trading_engine import LiveTradingEngine
from src.utils.config_loader import ConfigLoader


async def _check_start_prices(stock_code, snapshot):
    # Share the server's bounded subprocess gate without launching an HTTP server.
    from server import _check_live_start_price_readiness, _DottedPathDict
    return await _check_live_start_price_readiness([stock_code], _DottedPathDict(snapshot))


async def run_live():
    config = ConfigLoader.reload()
    if str(config.get("system.mode", "backtest") or "backtest").lower() != "live" or not config.get("system.enable_live", True):
        print("⛔ Live功能已关闭（需 system.mode=live）")
        return 1
    snapshot = config.to_dict_with_secrets()
    targets = config.get("targets", [])
    stock_code = str(targets[0] if targets else "601899.SH").strip().upper()
    provider = os.environ.get("DATA_PROVIDER", "").strip().lower() or config.get("data_provider.source", "akshare")
    token = os.environ.get("TUSHARE_TOKEN", "") or config.get("data_provider.tushare_token", "")
    if provider == 'tushare' and not token:
        print("⚠️ Tushare selected but no token found. Falling back to Akshare.")
        provider = 'akshare'
    snapshot.setdefault('data_provider', {}).update(source=provider, tushare_token=token)
    results = await _check_start_prices(stock_code, snapshot)
    if any(result.get('ok') is not True for result in results) or not results:
        print(f"⛔ 价格验收未通过，未启动 {stock_code}")
        return 1
    # Environment overrides participate in both the gate and all provider reads.
    with ConfigLoader.runtime_snapshot(snapshot) as accepted:
        print(f"🔌 Selected Data Provider: {provider}")
        print(f"🎯 Target Stock: {stock_code}")
        engine = LiveTradingEngine(stock_code,
            initial_capital=accepted.get("system.initial_capital", 1000000.0),
            provider_type=provider, tushare_token=token)
        await engine.run_live()
    return 0


def main():
    return asyncio.run(run_live())


if __name__ == "__main__":
    raise SystemExit(main())
