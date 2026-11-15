
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from time import perf_counter
from typing import Dict, List
from src.core.intent_planner import IntentPlanner
from src.core.strategy_engine import StrategyEngine
from src.core.risk_manager import RiskManager
from src.core.execution_engine import ExecutionEngine
from src.services.strategy_registry import StrategyRegistry
from src.services.account_ledger import AccountLedger
from src.services.performance_reporter import PerformanceReporter
from src.services.order_matcher import OrderMatcher
from src.services.risk_audit_log import RiskAuditLog
from src.services.market_data_service import MarketDataService
import src.strategies.strategy_factory as strategy_factory_module
from src.strategies.strategy_factory import create_strategies
from src.utils.data_provider import DataProvider
from src.utils.tushare_provider import TushareProvider
from src.utils.akshare_provider import AkshareProvider
from src.utils.mysql_provider import MysqlProvider
from src.utils.postgres_provider import PostgresProvider
from src.utils.duckdb_provider import DuckDbProvider
from src.utils.tdx_provider import TdxProvider
from src.utils.config_loader import ConfigLoader
from src.utils.trading_scope import normalize_trading_scope, select_trading_scope
from src.utils.indicators import Indicators
from src.utils.market_data_validation import MarketDataValidationError
from src.utils.baostock_provider import BaostockProvider
from src.utils.jqdata_provider import JqdataProvider
from src.utils.price_adjustment import (
    AdjustmentPlan, PriceAdjustmentError, assert_raw_prices, load_adjustment_factors, rebase_strategy_prices,
)

class BacktestEngine:
    _tf_cache = {}
    _tf_cache_limit = 24

    def __init__(
        self,
        stock_code,
        strategy_id='all',
        initial_capital=1000000.0,
        event_callback=None,
        strategy_mode=None,
        strategy_ids=None,
        combination_config=None,
        provider_override=None,
        provider_source_override=None,
        adjustment_mode=None,
        adjustment_factors=None,
        trading_scope=None,
        security_records=None,
    ):
        self.trading_scope = normalize_trading_scope(trading_scope)
        self.trading_scope_selection = None
        if self.trading_scope is not None:
            if security_records is None:
                from src.utils.stock_manager import stock_manager
                stock_manager.ensure_loaded()
                security_records = stock_manager.stocks
            self.trading_scope_selection = select_trading_scope([stock_code], self.trading_scope, security_records)
            if not self.trading_scope_selection['eligible_codes']:
                rejected = self.trading_scope_selection['excluded'][0]
                raise ValueError(f"标的不在所选交易范围：{rejected['stock_code']} ({rejected['reason']})")
            stock_code = self.trading_scope_selection['eligible_codes'][0]
        self.stock_code = stock_code
        self.strategy_id = strategy_id
        self.initial_capital = initial_capital
        self.event_callback = event_callback
        self.strategy_mode = strategy_mode
        self.strategy_ids = strategy_ids
        self.combination_config = self._normalize_combination_config(combination_config)
        self.provider_override = provider_override
        self.provider_source_override = str(provider_source_override or "").strip()
        self.config = ConfigLoader.reload()
        self.adjustment_mode = str(adjustment_mode if adjustment_mode is not None else self.config.get("data_provider.adjustment_mode", "none") or "none").lower()
        self.adjustment_factors = adjustment_factors
        # Cache raw data only within this run: provider changes and TDX file updates
        # must not reuse another run's prices.
        self._tf_cache = {}
        self.baseline_meta = {
            "profile_name": str(self.config.get("global_backtest_baseline.last_applied_profile", "") or ""),
            "market": str(self.config.get("global_backtest_baseline.last_applied_market", "") or ""),
            "adjustment_mode": self.adjustment_mode,
            "settlement_rule": str(self.config.get("execution.settlement_rule", "") or ""),
            "data_source": str(self.config.get("data_provider.source", "default") or "default")
        }
        
        if self.trading_scope_selection is not None:
            self.baseline_meta['trading_scope'] = self.trading_scope
            self.baseline_meta['trading_scope_metadata'] = self.trading_scope_selection['metadata']
        # Initialize Services
        self.strategy_registry = StrategyRegistry()
        self.account_ledger = AccountLedger(initial_capital)
        self.performance_reporter = PerformanceReporter()
        self.order_matcher = OrderMatcher()
        self.risk_audit_log = RiskAuditLog()
        self.market_data_service = MarketDataService()

        # Initialize Departments
        self.intent_planner = IntentPlanner()
        self.risk_manager = RiskManager(self.risk_audit_log)
        self.execution_engine = ExecutionEngine(self.account_ledger, self.order_matcher, self.risk_audit_log)

        # Initialize Strategies
        # Get the latest strategies every time we start a backtest
        explicit_single = str(self.strategy_id or "").strip() not in {"", "all"}
        use_active_filter = (not bool(self.strategy_ids)) and (not explicit_single)
        all_strategies = create_strategies(apply_active_filter=use_active_filter)
        if self.strategy_ids:
            allowed = {str(x).strip() for x in self.strategy_ids if str(x).strip()}
            self.strategies = [s for s in all_strategies if str(s.id).strip() in allowed]
        elif self.strategy_mode == 'top5':
            self.strategies = all_strategies[:5]
        elif self.strategy_id == 'all':
            self.strategies = all_strategies
        else:
            self.strategies = [s for s in all_strategies if s.id == self.strategy_id]
            
        self.strategy_engine = StrategyEngine(self.strategies)
        strategy_count = max(1, len(self.strategies))
        self.strategy_initial_capital = self.initial_capital / strategy_count
        self.strategy_accounts = {s.id: AccountLedger(self.strategy_initial_capital, nav_frequency='daily') for s in self.strategies}
        self.aggregate_nav = []
        self._portfolio_peak = float(self.initial_capital)
        self.drawdown_above_limit = False
        self.pending_signals = []
        self._risk_day = None
        self._session_start_nav = {}
        self._event_queue = None
        self._event_task = None
        self._last_yield_ts = perf_counter()
        self._coalesced_event_types = {"backtest_progress", "market", "module_tick"}
        self._coalesced_event_latest = {}
        self._coalesced_pending = set()
        self._flow_emit_stride = max(1, int(self.config.get("system.backtest_flow_emit_stride", 20) or 20))
        
        for s in self.strategies:
            self.strategy_registry.register_strategy(s)

    def _normalize_combination_config(self, cfg):
        if not isinstance(cfg, dict):
            return {"enabled": False, "mode": "or", "min_agree_count": 1, "weights": {}, "tie_policy": "skip"}
        mode = str(cfg.get("mode", "or") or "or").strip().lower()
        if mode not in {"and", "or", "vote"}:
            mode = "or"
        enabled = bool(cfg.get("enabled", True))
        min_agree_count = int(cfg.get("min_agree_count", 1) or 1)
        if min_agree_count < 1:
            min_agree_count = 1
        weights_raw = cfg.get("weights") if isinstance(cfg.get("weights"), dict) else {}
        weights = {}
        for k, v in weights_raw.items():
            sid = str(k or "").strip()
            if not sid:
                continue
            try:
                w = float(v)
            except Exception:
                w = 1.0
            weights[sid] = max(0.0, w)
        tie_policy = str(cfg.get("tie_policy", "skip") or "skip").strip().lower()
        if tie_policy not in {"skip", "buy", "sell"}:
            tie_policy = "skip"
        return {
            "enabled": enabled,
            "mode": mode,
            "min_agree_count": min_agree_count,
            "weights": weights,
            "tie_policy": tie_policy,
        }

    def _apply_signal_combination(self, signals, runnable_strategy_ids):
        cfg = self.combination_config if isinstance(self.combination_config, dict) else {}
        if not bool(cfg.get("enabled", False)):
            return list(signals or []), {"enabled": False, "mode": "or", "kept": len(signals or [])}
        mode = str(cfg.get("mode", "or") or "or").strip().lower()
        incoming = list(signals or [])
        if not incoming:
            return [], {"enabled": True, "mode": mode, "kept": 0, "reason": "no_signal"}
        by_sid = {}
        for sig in incoming:
            sid = str(sig.get("strategy_id", "")).strip()
            if sid and sid not in by_sid:
                by_sid[sid] = sig
        target_ids = [str(x).strip() for x in (runnable_strategy_ids or []) if str(x).strip()]
        if mode == "or":
            return incoming, {"enabled": True, "mode": "or", "kept": len(incoming)}
        if mode == "and":
            if not target_ids:
                return [], {"enabled": True, "mode": "and", "kept": 0, "reason": "no_target"}
            if any(sid not in by_sid for sid in target_ids):
                return [], {"enabled": True, "mode": "and", "kept": 0, "reason": "not_all_signaled"}
            dirs = {str(by_sid[sid].get("direction", "")).upper() for sid in target_ids}
            if len(dirs) != 1:
                return [], {"enabled": True, "mode": "and", "kept": 0, "reason": "direction_conflict"}
            agreed_direction = list(dirs)[0]
            kept = [by_sid[sid] for sid in target_ids if sid in by_sid and str(by_sid[sid].get("direction", "")).upper() == agreed_direction]
            min_agree = int(cfg.get("min_agree_count", 1) or 1)
            if len(kept) < min_agree:
                return [], {"enabled": True, "mode": "and", "kept": 0, "reason": "below_min_agree"}
            return kept, {"enabled": True, "mode": "and", "kept": len(kept), "direction": agreed_direction}
        if mode == "vote":
            weights = cfg.get("weights") if isinstance(cfg.get("weights"), dict) else {}
            buy_score = 0.0
            sell_score = 0.0
            for sid, sig in by_sid.items():
                w = float(weights.get(sid, 1.0) or 1.0)
                d = str(sig.get("direction", "")).upper()
                if d == "BUY":
                    buy_score += w
                elif d == "SELL":
                    sell_score += w
            tie_policy = str(cfg.get("tie_policy", "skip") or "skip").strip().lower()
            winner = ""
            if buy_score > sell_score:
                winner = "BUY"
            elif sell_score > buy_score:
                winner = "SELL"
            elif tie_policy == "buy":
                winner = "BUY"
            elif tie_policy == "sell":
                winner = "SELL"
            if not winner:
                return [], {
                    "enabled": True,
                    "mode": "vote",
                    "kept": 0,
                    "reason": "tie_skip",
                    "buy_score": round(buy_score, 4),
                    "sell_score": round(sell_score, 4),
                }
            kept = [sig for sig in by_sid.values() if str(sig.get("direction", "")).upper() == winner]
            min_agree = int(cfg.get("min_agree_count", 1) or 1)
            if len(kept) < min_agree:
                return [], {
                    "enabled": True,
                    "mode": "vote",
                    "kept": 0,
                    "reason": "below_min_agree",
                    "buy_score": round(buy_score, 4),
                    "sell_score": round(sell_score, 4),
                }
            return kept, {
                "enabled": True,
                "mode": "vote",
                "kept": len(kept),
                "winner": winner,
                "buy_score": round(buy_score, 4),
                "sell_score": round(sell_score, 4),
            }
        return incoming, {"enabled": True, "mode": "or", "kept": len(incoming)}

    def _cache_key(self, start_date, end_date, interval, provider_source):
        return (
            str(self.stock_code),
            str(provider_source),
            str(interval),
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

    def _cache_get(self, start_date, end_date, interval, provider_source):
        key = self._cache_key(start_date, end_date, interval, provider_source)
        df = self._tf_cache.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
        return pd.DataFrame()

    def _cache_set(self, start_date, end_date, interval, provider_source, df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        key = self._cache_key(start_date, end_date, interval, provider_source)
        self._tf_cache[key] = df.copy()
        if len(self._tf_cache) > self._tf_cache_limit:
            first_key = next(iter(self._tf_cache))
            if first_key != key:
                self._tf_cache.pop(first_key, None)

    def _normalize_trigger_tf(self, tf):
        x = str(tf or "1min").strip()
        if not x:
            return "1min"
        low = x.lower()
        if low in ("d", "1d", "day", "daily"):
            return "D"
        if low in {"1min", "5min", "10min", "15min", "30min", "60min"}:
            return low
        raise PriceAdjustmentError(f"unsupported strategy timeframe {x!r}; use intraday minutes or D")

    def _build_provider(self, source):
        if source == 'tushare':
            return TushareProvider(token=self.config.get("data_provider.tushare_token"))
        if source == 'akshare':
            return AkshareProvider()
        if source == 'mysql':
            return MysqlProvider()
        if source == 'postgresql':
            return PostgresProvider()
        if source == 'duckdb':
            return DuckDbProvider()
        if source == 'tdx':
            return TdxProvider()
        if source == 'baostock':
            return BaostockProvider()
        if source == 'jqdata':
            return JqdataProvider(username=self.config.get("data_provider.jqdata_username"), password=self.config.get("data_provider.jqdata_password"))
        return DataProvider()

    def _resolve_backtest_cache_db_source(self):
        target = str(self.config.get("data_provider.backtest_cache_db_source", "") or "").strip().lower()
        if target in {"mysql", "postgresql", "duckdb"}:
            return target
        return ""

    async def _persist_backtest_cache_to_db(self, df, interval, provider_source):
        if df is None or df.empty:
            return
        cache_db_source = self._resolve_backtest_cache_db_source()
        if not cache_db_source:
            return
        if provider_source == cache_db_source:
            return
        if self.config.get('data_provider.stored_price_mode', 'unknown') != 'none':
            await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'warning', 'msg': '缓存落库跳过：目标数据库未声明原价口径，避免混入不同复权价格。'})
            return
        writer = self._build_provider(cache_db_source)
        if not hasattr(writer, "upsert_kline_data"):
            await self._emit('backtest_flow', {
                'module': '行情与运维',
                'level': 'warning',
                'msg': f'缓存落库跳过：{cache_db_source} 不支持 upsert'
            })
            return
        written = await asyncio.to_thread(writer.upsert_kline_data, df, interval)
        if int(written or 0) <= 0 and getattr(writer, "last_error", ""):
            await self._emit('backtest_flow', {
                'module': '行情与运维',
                'level': 'warning',
                'msg': f'缓存落库失败：{cache_db_source} {writer.last_error}'
            })
            return
        await self._emit('backtest_flow', {
            'module': '行情与运维',
            'level': 'success',
            'msg': f'缓存落库完成：{cache_db_source} interval={interval} 写入 {int(written or 0)} 条'
        })

    def _check_provider_connectivity(self, provider, provider_source):
        try:
            if hasattr(provider, "check_connectivity"):
                ok, msg = provider.check_connectivity(self.stock_code)
                return bool(ok), str(msg or "")
            if provider_source == "tushare":
                pro = getattr(provider, "pro", None)
                if pro is None:
                    return False, "tushare_token 未配置"
                now = datetime.now()
                start_time = now - timedelta(days=3)
                pro.stk_mins(
                    ts_code=self.stock_code,
                    freq='1min',
                    start_date=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    end_date=now.strftime("%Y-%m-%d %H:%M:%S")
                )
                return True, "ok"
            if provider_source == "akshare":
                bar = provider.get_latest_bar(self.stock_code)
                if bar:
                    return True, "ok"
                return False, "akshare 连通性检查失败（未返回最新行情）"
            if provider_source == "default":
                if hasattr(provider, "_request_get") and hasattr(provider, "_code_variants"):
                    now = datetime.now()
                    start_time = now - timedelta(days=3)
                    for c in provider._code_variants(self.stock_code):
                        params = {
                            "code": c,
                            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "end_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                            "limit": 1
                        }
                        resp = provider._request_get("/market/minutes", params, timeout=10)
                        if resp is not None:
                            return True, "ok"
                    return False, getattr(provider, "last_error", "") or "default 数据源连通性检查失败"
            return False, f"未知数据源: {provider_source}"
        except Exception as e:
            return False, str(e)

    async def _fetch_minute_data_with_guard(self, provider, start_time, end_time, provider_source):
        return await self._fetch_kline_data_with_guard(provider, start_time, end_time, provider_source, interval="1min")

    async def _fetch_kline_data_with_guard(self, provider, start_time, end_time, provider_source, interval="1min"):
        timeout_raw = int(self.config.get("system.backtest_data_fetch_timeout_sec", 240) or 240)
        timeout_sec = max(30, min(timeout_raw, 3600))
        heartbeat_raw = int(self.config.get("system.backtest_data_fetch_heartbeat_sec", 10) or 10)
        heartbeat_sec = max(3, min(heartbeat_raw, 30))
        started_at = perf_counter()
        tf = self._normalize_trigger_tf(interval)
        if tf == "1min":
            task = asyncio.create_task(asyncio.to_thread(provider.fetch_minute_data, self.stock_code, start_time, end_time))
        else:
            task = asyncio.create_task(asyncio.to_thread(provider.fetch_kline_data, self.stock_code, start_time, end_time, tf))
        heartbeat_count = 0
        while True:
            try:
                df = await asyncio.wait_for(asyncio.shield(task), timeout=heartbeat_sec)
                used = int(perf_counter() - started_at)
                await self._emit('backtest_flow', {
                    'module': '行情与运维',
                    'level': 'system',
                    'msg': f'数据获取阶段：历史K线拉取完成 source={provider_source} interval={tf} 用时 {used}s'
                })
                return df, ""
            except asyncio.TimeoutError:
                waited = int(perf_counter() - started_at)
                heartbeat_count += 1
                progress = min(14, 6 + heartbeat_count)
                await self._emit('backtest_progress', {
                    'progress': progress,
                    'phase': 'data_fetch',
                    'phase_label': f'拉取历史K线（已等待 {waited}s）',
                    'current_date': f'{start_time.date()} ~ {end_time.date()}'
                })
                await self._emit('backtest_flow', {
                    'module': '行情与运维',
                    'level': 'system',
                    'msg': f'数据获取阶段：历史K线拉取进行中 source={provider_source} interval={tf} 已等待 {waited}s'
                })
                if waited >= timeout_sec:
                    if not task.done():
                        task.cancel()
                    return pd.DataFrame(), f'拉取超时（{timeout_sec}s）'

    def _compute_portfolio_snapshot(self, current_prices):
        holdings_value = 0.0
        cash_total = 0.0
        for sid, account in self.strategy_accounts.items():
            holdings_value += self.execution_engine.update_strategy_holdings_value(sid, current_prices)
            cash_total += float(account.cash)
        fund_value = cash_total + holdings_value
        return fund_value, cash_total, holdings_value

    def _compute_current_drawdown_ratio(self, fund_value):
        # Legacy synthetic callers may construct an engine without __init__.
        # Seed once; normal runs maintain this high-water mark incrementally.
        if not hasattr(self, '_portfolio_peak'):
            self._portfolio_peak = max([float(self.initial_capital)] +
                [float(x.get('nav', 0.0) or 0.0) for x in self.aggregate_nav])
        peak = self._portfolio_peak
        if peak <= 0:
            return 0.0
        return max(0.0, (float(peak) - float(fund_value)) / float(peak))

    def _start_risk_session(self, kline):
        day = pd.Timestamp(kline['dt']).date().isoformat()
        if day == getattr(self, '_risk_day', None):
            return
        self._risk_day = day
        self._session_start_nav = {}
        for sid, account in self.strategy_accounts.items():
            history = account.daily_nav
            self._session_start_nav[sid] = float(history[-1]['nav']) if history else float(account.initial_capital)
            self.risk_manager.start_session(sid, day)

    def _record_nav(self, kline):
        prices = {kline['code']: float(kline['close'])}
        value, _, _ = self._compute_portfolio_snapshot(prices)
        for sid, account in self.strategy_accounts.items():
            holdings = self.execution_engine.update_strategy_holdings_value(sid, prices)
            account.update_daily_nav(kline['dt'], holdings)
        self._portfolio_peak = max(getattr(self, '_portfolio_peak', self.initial_capital), value)
        if self.aggregate_nav and pd.Timestamp(self.aggregate_nav[-1]['dt']).date() == pd.Timestamp(kline['dt']).date():
            self.aggregate_nav.pop()
        self.aggregate_nav.append({'dt': kline['dt'], 'nav': value})

    def _record_close_outcome(self, sid, account):
        if account.transactions and account.transactions[-1]['direction'] == 'SELL':
            self.risk_manager.update_loss_count(sid, float(account.transactions[-1]['pnl']) < 0)

    async def _execute_pending_signals(self, kline):
        pending, self.pending_signals = self.pending_signals, []
        # Signals are known only after their source bar closes. Risk and fills
        # at the following open must not use this bar's eventual close.
        open_price = float(kline['open'])
        for original in pending:
            sid = original['strategy_id']
            account = self.strategy_accounts.get(sid)
            if account is None:
                continue
            signal = dict(original, price=open_price, dt=kline['dt'], execution_mode='at_open',
                          direction=str(original.get('direction', '')).strip().upper())
            signal['signal_dt'] = original['dt']
            holdings = self.execution_engine.update_strategy_holdings_value(sid, {kline['code']: open_price})
            value = float(account.cash) + holdings
            portfolio, _, _ = self._compute_portfolio_snapshot({kline['code']: open_price})
            drawdown = self._compute_current_drawdown_ratio(portfolio)
            max_dd = float(self.config.get('risk_control.max_drawdown_pct', 0) or 0)
            if signal['direction'] == 'BUY' and max_dd > 0 and drawdown > max_dd:
                approved, reason = False, '当前回撤超限，暂停开仓'
            else:
                daily_pnl = value - self._session_start_nav.get(sid, account.initial_capital)
                approved, reason = self.risk_manager.check_signal(
                    signal, value, self.execution_engine.positions.get(sid, {}), daily_pnl,
                    current_drawdown=drawdown,
                )
            await self._emit('risk_review', {'msg': '风控审核通过' if approved else '风控审核拒绝',
                            'decision': 'approved' if approved else 'rejected', 'strategy_id': sid,
                            'reason': reason, 'details': reason})
            if not approved:
                continue
            executed = self.execution_engine.execute_order(sid, signal, kline, account=account)
            if not executed:
                continue
            transaction = account.transactions[-1]
            qty = self.execution_engine.positions.get(sid, {}).get(signal['code'], {}).get('qty', 0)
            self.strategy_engine.update_strategy_state(sid, signal['code'], qty)
            self._record_close_outcome(sid, account)
            await self._emit('backtest_trade', {'dt': str(transaction['dt']), 'signal_dt': str(original['dt']),
                            'strategy': sid, 'code': signal['code'], 'dir': transaction['direction'],
                            'price': transaction['price'], 'qty': transaction['quantity'],
                            'execution_mode': 'at_open'})

    async def _force_close_positions_at_end(self, kline):
        if not bool(self.config.get("execution.force_close_on_backtest_end", True)):
            return
        closed_any = False
        for sid, stocks in list(self.execution_engine.positions.items()):
            account = self.strategy_accounts.get(sid)
            if account is None:
                continue
            for code, pos in list(stocks.items()):
                qty = int(pos.get("qty", 0) or 0)
                if qty <= 0:
                    continue
                order = {
                    "strategy_id": sid,
                    "code": code,
                    "dt": kline["dt"],
                    "direction": "SELL",
                    "qty": qty,
                    "price": float(kline.get("close", 0) or 0),
                    "execution_mode": "at_close",
                    "reason": "FORCE_CLOSE_END"
                }
                executed = self.execution_engine.execute_order(sid, order, kline, account=account)
                if not executed:
                    await self._emit('backtest_flow', {'module': '订单执行', 'level': 'warning',
                        'msg': f'期末平仓未成交: 策略 {sid} {code} 剩余 {qty} 股；按期末原价估值，T+1或成交边界拦截记录见风控日志'})
                    continue
                self.strategy_engine.update_strategy_state(sid, code, 0)
                self._record_close_outcome(sid, account)
                closed_any = True
                await self._emit('backtest_flow', {
                    'module': '订单执行',
                    'level': 'warning',
                    'msg': f"回测结束强制平仓: 策略 {sid} SELL {code} x {qty}"
                })
                await self._emit('backtest_trade', {
                    'dt': str(kline['dt']),
                    'strategy': sid,
                    'code': code,
                    'dir': 'SELL',
                    'price': account.transactions[-1]['price'],
                    'qty': account.transactions[-1]['quantity'],
                    'reason': 'FORCE_CLOSE_END'
                })
        if closed_any:
            await self._emit_account_snapshot(kline, active_strategy_id=None, compliance_status="PASS")

    async def _execute_protective_orders(self, kline):
        # Stops were set before this bar. Process them before creating signals
        # from its close, and report the actual matched price and quantity.
        for order in self.execution_engine.check_stops(kline):
            sid = order['strategy_id']
            account = self.strategy_accounts.get(sid)
            if account is None or not self.execution_engine.execute_order(sid, order, kline, account=account):
                continue
            transaction = account.transactions[-1]
            qty = self.execution_engine.positions.get(sid, {}).get(order['code'], {}).get('qty', 0)
            self.strategy_engine.update_strategy_state(sid, order['code'], qty)
            self._record_close_outcome(sid, account)
            await self._emit('backtest_trade', {'dt': str(transaction['dt']), 'strategy': sid,
                'code': order['code'], 'dir': 'SELL', 'price': transaction['price'],
                'qty': transaction['quantity'], 'reason': 'STOP',
                'execution_mode': order['execution_mode']})

    async def _enforce_drawdown_limit(self, kline):
        if not self.strategy_accounts:
            return False
        limit = float(self.config.get('risk_control.max_drawdown_pct', 0) or 0)
        if limit <= 0:
            return False
        value, _, _ = self._compute_portfolio_snapshot({kline['code']: kline['close']})
        if self._compute_current_drawdown_ratio(value) <= limit:
            self.drawdown_above_limit = False
            return False
        if not self.drawdown_above_limit:
            self._drawdown_signal_dt = kline['dt']
            await self._emit('backtest_flow', {'module': '风险审核', 'level': 'danger',
                'msg': '收盘净值触发回撤限制，暂停开仓并等待下一根K线开盘清仓'})
        self.drawdown_above_limit = True
        return True

    async def _execute_drawdown_orders(self, kline):
        if not self.drawdown_above_limit:
            return
        if pd.Timestamp(kline['dt']) <= pd.Timestamp(self._drawdown_signal_dt):
            return
        for sid, stocks in list(self.execution_engine.positions.items()):
            account = self.strategy_accounts.get(sid)
            if account is None:
                continue
            for code, pos in list(stocks.items()):
                qty = self.execution_engine._sellable_qty_t1(pos, pd.Timestamp(kline['dt']).date().isoformat())
                if qty <= 0:
                    continue
                order = dict(strategy_id=sid, code=code, dt=kline['dt'], direction='SELL',
                             qty=qty, price=kline['open'], execution_mode='at_open', reason='DRAWDOWN_LIMIT')
                if not self.execution_engine.execute_order(sid, order, kline, account=account):
                    continue
                transaction = account.transactions[-1]
                remaining = self.execution_engine.positions.get(sid, {}).get(code, {}).get('qty', 0)
                self.strategy_engine.update_strategy_state(sid, code, remaining)
                self._record_close_outcome(sid, account)
                await self._emit('backtest_trade', {'dt': str(transaction['dt']), 'strategy': sid,
                    'code': code, 'dir': 'SELL', 'price': transaction['price'],
                    'qty': transaction['quantity'], 'reason': 'DRAWDOWN_LIMIT', 'execution_mode': 'at_open'})

    async def _emit_loop(self):
        while True:
            item = await self._event_queue.get()
            if item is None:
                break
            event_type, data = item
            if event_type in self._coalesced_event_types:
                self._coalesced_pending.discard(event_type)
                latest = self._coalesced_event_latest.pop(event_type, None)
                if latest is not None:
                    data = latest
            if self.event_callback:
                await self.event_callback(event_type, data)

    async def _start_event_pump(self):
        if not self.event_callback:
            return
        self._event_queue = asyncio.Queue(maxsize=20000)
        self._event_task = asyncio.create_task(self._emit_loop())

    async def _stop_event_pump(self):
        if self._event_queue is None or self._event_task is None:
            return
        await self._event_queue.put(None)
        await self._event_task
        self._event_queue = None
        self._event_task = None
        self._coalesced_event_latest = {}
        self._coalesced_pending = set()

    async def _emit(self, event_type, data):
        if not self.event_callback:
            return
        if self._event_queue is None:
            await self.event_callback(event_type, data)
            return
        if event_type in self._coalesced_event_types:
            self._coalesced_event_latest[event_type] = data
            if event_type in self._coalesced_pending:
                await self._yield_control(0.01)
                return
            self._coalesced_pending.add(event_type)
        try:
            self._event_queue.put_nowait((event_type, data))
        except asyncio.QueueFull:
            await self._event_queue.put((event_type, data))
        await self._yield_control(0.01)

    async def _yield_control(self, min_interval_sec=0.02):
        now = perf_counter()
        if (now - self._last_yield_ts) >= float(min_interval_sec):
            await asyncio.sleep(0)
            self._last_yield_ts = perf_counter()

    async def _emit_account_snapshot(self, kline, active_strategy_id=None, compliance_status="PASS", *, is_final=False):
        current_prices = {kline['code']: kline['close']}
        fund_value, cash_total, holdings_value = self._compute_portfolio_snapshot(current_prices)
        self._record_nav(kline)
        pnl_pct = ((fund_value / self.initial_capital) - 1.0) * 100 if self.initial_capital else 0.0
        pos_ratio = (holdings_value / fund_value * 100) if fund_value > 0 else 0.0
        await self._emit('account', {
            'assets': round(fund_value, 2),
            'cash': round(cash_total, 2),
            'pnl': f"{pnl_pct:+.2f}%",
            'pos_ratio': f"{pos_ratio:.2f}%",
            'is_final': bool(is_final),
        })
        current_dd = self._compute_current_drawdown_ratio(fund_value) * 100
        await self._emit('module_tick', {
            'cash': round(cash_total, 2),
            'available_pos_pct': max(0.0, 100.0 - pos_ratio),
            'main_strategy': active_strategy_id,
            'drawdown_pct': round(current_dd, 2),
            'compliance_status': compliance_status
        })

    async def run(self, start_date=None, end_date=None):
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
        total_started_at = perf_counter()
        perf_data_fetch_ms = 0
        perf_period_build_ms = 0
        perf_main_loop_ms = 0
        perf_settlement_ms = 0
        await self._start_event_pump()
        try:
            if not self.strategies:
                requested = ",".join([str(x).strip() for x in (self.strategy_ids or []) if str(x).strip()])
                msg = "无可运行策略：未找到可实例化且启用的策略"
                if requested:
                    msg = f"{msg}（请求: {requested}）"
                await self._emit('system', {'msg': f"❌ {msg}"})
                await self._emit('backtest_failed', {
                    'msg': msg,
                    'stock': self.stock_code,
                    'period': f"{start_date.date()} - {end_date.date()}",
                    'provider_source': self.config.get("data_provider.source", "default")
                })
                return
            await self._emit('system', {'msg': f"开始回测 {self.stock_code} ({start_date.date()} - {end_date.date()})..."})
            await self._emit('backtest_flow', {'module': '意图规划', 'level': 'system', 'msg': f'校验标的与回测区间: {self.stock_code} {start_date.date()}~{end_date.date()}'})
            await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'system', 'msg': f'装载行情数据: {self.stock_code} {start_date.date()}~{end_date.date()}...'})
            await self._emit('backtest_progress', {
                'progress': 2,
                'phase': 'data_fetch',
                'phase_label': '初始化数据源',
                'current_date': f'{start_date.date()} ~ {end_date.date()}'
            })
            await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'system', 'msg': '数据获取阶段：初始化数据源'})
            stage_started_at = perf_counter()
            provider_source = self.provider_source_override or self.config.get("data_provider.source", "default")
            enable_fallback = bool(self.config.get("data_provider.enable_fallback", False)) and (self.provider_override is None)
            provider = self.provider_override if self.provider_override is not None else self._build_provider(provider_source)
            await self._emit('backtest_progress', {
                'progress': 4,
                'phase': 'data_fetch',
                'phase_label': '检查数据源连通性',
                'current_date': f'{start_date.date()} ~ {end_date.date()}'
            })
            await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'system', 'msg': f'数据获取阶段：连通性检查 {provider_source}'})
            ok, reason = await asyncio.to_thread(self._check_provider_connectivity, provider, provider_source)
            if not ok and enable_fallback:
                await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'warning', 'msg': f'主数据源连通性检查失败，开始回退。原因: {reason}'})
                fallback_sources = []
                token = self.config.get("data_provider.tushare_token")
                if provider_source != 'tushare' and token:
                    fallback_sources.append('tushare')
                if provider_source != 'akshare':
                    fallback_sources.append('akshare')
                for src in fallback_sources:
                    candidate = self._build_provider(src)
                    candidate_ok, candidate_reason = await asyncio.to_thread(self._check_provider_connectivity, candidate, src)
                    if candidate_ok:
                        provider = candidate
                        provider_source = src
                        ok = True
                        reason = "ok"
                        await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'warning', 'msg': f'连通性回退成功，切换到 {src}'})
                        break
                    await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'warning', 'msg': f'候选数据源 {src} 连通性失败: {candidate_reason}'})
            if not ok:
                fail_msg = f"数据源连通性检查失败，回测终止。来源={provider_source} 原因={reason}"
                await self._emit('system', {'msg': f"❌ {fail_msg}"})
                await self._emit('backtest_failed', {
                    'msg': fail_msg,
                    'stock': self.stock_code,
                    'period': f"{start_date.date()} - {end_date.date()}",
                    'provider_source': provider_source
                })
                return
            await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'success', 'msg': f'数据源连通性检查通过: {provider_source}'})
            strategy_trigger_tf = {s.id: self._normalize_trigger_tf(getattr(s, "trigger_timeframe", "1min")) for s in self.strategies}
            base_interval = "D" if strategy_trigger_tf and all(tf == "D" for tf in strategy_trigger_tf.values()) else "1min"
            minimum_interval = getattr(provider, 'minimum_interval', '1min')
            if base_interval != 'D' and minimum_interval == '5min':
                if any(tf == '1min' for tf in strategy_trigger_tf.values()):
                    await self._emit('backtest_failed', {'msg': '该数据源最小周期为 5 分钟，无法运行 1 分钟策略；请选择其他数据源。', 'stock': self.stock_code})
                    return
                base_interval = '5min'
            df = self._cache_get(start_date, end_date, base_interval, provider_source)
            if df.empty:
                await self._emit('backtest_progress', {
                    'progress': 6,
                    'phase': 'data_fetch',
                    'phase_label': '拉取历史K线',
                    'current_date': f'{start_date.date()} ~ {end_date.date()}'
                })
                await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'system', 'msg': f'数据获取阶段：拉取历史K线 {start_date.date()}~{end_date.date()} interval={base_interval}'})
                await self._yield_control(0.01)
                df, fetch_err = await self._fetch_kline_data_with_guard(provider, start_date, end_date, provider_source, interval=base_interval)
                if fetch_err:
                    await self._emit('backtest_flow', {
                        'module': '行情与运维',
                        'level': 'warning',
                        'msg': f'数据获取阶段：主源拉取异常 source={provider_source} {fetch_err}'
                    })
                if df.empty and enable_fallback:
                    token = self.config.get("data_provider.tushare_token")
                    if token and provider_source != 'tushare':
                        await self._emit('system', {'msg': f"主数据源无数据，尝试 Tushare 获取 {self.stock_code} 历史K线..."})
                        await self._emit('backtest_progress', {
                            'progress': 8,
                            'phase': 'data_fetch',
                            'phase_label': '主源无数据，尝试 Tushare',
                            'current_date': f'{start_date.date()} ~ {end_date.date()}'
                        })
                        await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'warning', 'msg': '数据获取阶段：主源无数据，切换 Tushare'})
                        await self._yield_control(0.01)
                        try:
                            candidate = TushareProvider(token=token)
                            df, _ = await self._fetch_kline_data_with_guard(candidate, start_date, end_date, 'tushare', interval=base_interval)
                            if not df.empty:
                                provider, provider_source = candidate, 'tushare'
                        except Exception:
                            df = pd.DataFrame()
                if df.empty and enable_fallback and provider_source != 'akshare':
                    await self._emit('system', {'msg': f"Tushare 无数据，尝试 Akshare 获取 {self.stock_code} 历史K线..."})
                    await self._emit('backtest_progress', {
                        'progress': 10,
                        'phase': 'data_fetch',
                        'phase_label': 'Tushare 无数据，尝试 Akshare',
                        'current_date': f'{start_date.date()} ~ {end_date.date()}'
                    })
                    await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'warning', 'msg': '数据获取阶段：Tushare 无数据，切换 Akshare'})
                    await self._yield_control(0.01)
                    try:
                        candidate = AkshareProvider()
                        df, _ = await self._fetch_kline_data_with_guard(candidate, start_date, end_date, 'akshare', interval=base_interval)
                        if not df.empty:
                            provider, provider_source = candidate, 'akshare'
                    except Exception:
                        df = pd.DataFrame()
                if df.empty:
                    provider_msg = ""
                    if hasattr(provider, "last_error") and provider.last_error:
                        provider_msg = f" 诊断: {provider.last_error}"
                    fail_msg = f"无法获取 {self.stock_code} 的历史数据，回测终止。{provider_msg}".strip()
                    await self._emit('system', {'msg': f"❌ {fail_msg}"})
                    await self._emit('backtest_failed', {
                        'msg': fail_msg,
                        'stock': self.stock_code,
                        'period': f"{start_date.date()} - {end_date.date()}",
                        'provider_source': provider_source
                    })
                    return
                df = await asyncio.to_thread(self.market_data_service.clean_data, df)
                await self._emit('backtest_progress', {
                    'progress': 15,
                    'phase': 'data_fetch',
                    'phase_label': '清洗数据',
                    'current_date': f'{start_date.date()} ~ {end_date.date()}'
                })
                await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'system', 'msg': '数据获取阶段：数据清洗完成，写入缓存'})
                self._cache_set(start_date, end_date, base_interval, provider_source, df)
            else:
                await self._emit('backtest_progress', {
                    'progress': 12,
                    'phase': 'data_fetch',
                    'phase_label': '命中缓存',
                    'current_date': f'{start_date.date()} ~ {end_date.date()}'
                })
                await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'system', 'msg': '数据获取阶段：命中缓存，跳过远程拉取'})
            await self._emit('backtest_progress', {
                'progress': 18,
                'phase': 'data_fetch',
                'phase_label': '数据就绪',
                'current_date': f'{start_date.date()} ~ {end_date.date()}'
            })
            await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'system', 'msg': '数据获取阶段：数据已就绪，进入策略初始化'})
            total_bars = len(df)
            try:
                assert_raw_prices(provider, provider_source, df, self.config)
                factors = self.adjustment_factors
                if factors is None and self.adjustment_mode != "none":
                    factors = await asyncio.to_thread(load_adjustment_factors, self.stock_code, start_date, end_date, self.config)
                price_plan = AdjustmentPlan(df, factors, self.adjustment_mode)
            except PriceAdjustmentError as exc:
                await self._emit('backtest_failed', {'msg': str(exc), 'stock': self.stock_code, 'provider_source': provider_source})
                return
            self.baseline_meta['execution_timing'] = 'next_bar_open'
            self.baseline_meta['aggregation_policy'] = 'completed_a_share_sessions'
            self.baseline_meta['minute_timestamp_convention'] = 'exchange_end'
            self.baseline_meta['nav_storage'] = 'last_observation_per_session'
            self.baseline_meta['data_source'] = provider_source
            self.baseline_meta['data_coverage'] = {
                'interval': base_interval, 'rows': len(df),
                'requested_start': pd.Timestamp(start_date).isoformat(),
                'requested_end': pd.Timestamp(end_date).isoformat(),
                'first_bar': pd.Timestamp(df['dt'].min()).isoformat(),
                'last_bar': pd.Timestamp(df['dt'].max()).isoformat(),
            }
            self.baseline_meta['price_policy'] = price_plan.metadata()
            self.baseline_meta['price_policy']['data_coverage'] = dict(self.baseline_meta['data_coverage'])
            if pd.Timestamp(end_date).normalize() - pd.Timestamp(df['dt'].max()).normalize() > pd.Timedelta(days=7):
                warning = f"行情仅到 {pd.Timestamp(df['dt'].max())}，未覆盖请求的结束日期，请检查数据源更新。"
                self.baseline_meta['price_policy']['limitations'].append(warning)
                await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'warning', 'msg': warning})
            self.baseline_meta['price_policy']['raw_data_verified'] = True
            self.baseline_meta['price_policy']['raw_price_evidence'] = 'provider_contract' if provider_source in {'tdx', 'tushare', 'akshare', 'baostock', 'jqdata'} else 'storage_declaration'
            self.baseline_meta['price_policy']['limitations'].append('未实现分红与送转入账；复权模式下持仓跨除权日会终止回测，避免输出失真的收益。')
            has_limits = all(column in df and pd.to_numeric(df[column], errors='coerce').gt(0).all()
                             for column in ('up_limit', 'down_limit'))
            self.baseline_meta['exchange_price_limits_available'] = bool(has_limits)
            if not has_limits:
                self.baseline_meta['price_policy']['limitations'].append(
                    '行情未提供完整逐日涨跌停边界；撮合仅使用已提供的边界，未验证历史ST状态和真实排队成交。')
            await self._emit('backtest_price_policy', dict(self.baseline_meta['price_policy']))
            await self._persist_backtest_cache_to_db(df, base_interval, provider_source)
            await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'warning', 'msg': '指标使用所选复权口径，成交及资产估值使用真实原价。分红、送转账户处理尚未支持。'})
            perf_data_fetch_ms = int((perf_counter() - stage_started_at) * 1000)
            await self._emit('system', {'msg': f"已获取 {total_bars} 条K线数据，正在初始化策略..."})
            await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'system', 'msg': f'数据清洗完成，共 {total_bars} 条分钟K线'})
            final_bar_dt = pd.to_datetime(df.iloc[-1]["dt"])
            for strategy in self.strategies:
                strategy.set_backtest_context(final_bar_dt=final_bar_dt)
            stage_started_at = perf_counter()
            needed_timeframes = sorted(set([tf for tf in strategy_trigger_tf.values() if tf != base_interval]))
            tf_dt_sets = {}
            tf_row_maps = {}
            for tf in needed_timeframes:
                # One validated feed, one timestamp convention. Provider aggregates
                # may be start-labelled or include an unfinished final window.
                tf_df = await asyncio.to_thread(Indicators.resample_completed, df, tf, base_interval)
                if tf_df.empty or "dt" not in tf_df.columns:
                    await self._emit('backtest_flow', {'module': '行情与运维', 'level': 'warning',
                        'msg': f'{tf} 周期没有完整交易时段K线，未向策略交付未完成窗口'})
                    continue
                tf_row_maps[tf] = {pd.Timestamp(row['dt']): row for row in tf_df.to_dict('records')}
                tf_dt_sets[tf] = set(tf_row_maps[tf])
            perf_period_build_ms = int((perf_counter() - stage_started_at) * 1000)
            if needed_timeframes:
                await self._emit('system', {'msg': f"策略周期映射已启用: {strategy_trigger_tf}"})
                await self._emit('backtest_flow', {'module': '策略引擎', 'level': 'system', 'msg': f'策略周期映射: {strategy_trigger_tf}'})
            report_interval = max(1, total_bars // 50)
            op_counter = 0
            close_series = pd.to_numeric(df["close"], errors="coerce").ffill().bfill()
            factor_series = pd.Series([price_plan.factor_at(dt) for dt in df['dt']], index=df.index)
            if self.adjustment_mode != 'none':
                close_series = close_series * factor_series / factor_series.iloc[0]
            ma5_series = Indicators.MA(close_series, 5).fillna(close_series)
            _, _, macd_series = Indicators.MACD(close_series)
            macd_series = macd_series.fillna(0.0)
            rsi_series = Indicators.RSI(close_series).fillna(50.0)
            col_idx = {c: i for i, c in enumerate(df.columns)}
            previous_factor = None
            stage_started_at = perf_counter()
            for i, row in enumerate(df.itertuples(index=False, name=None)):
                await self._yield_control(0.01)
                kline = dict(zip(df.columns, row))
                kline.update({
                    "code": row[col_idx.get("code")],
                    "dt": row[col_idx.get("dt")],
                    "open": row[col_idx.get("open")],
                    "high": row[col_idx.get("high")],
                    "low": row[col_idx.get("low")],
                    "close": row[col_idx.get("close")],
                    "vol": row[col_idx.get("vol")] if "vol" in col_idx else row[col_idx.get("volume")],
                    "amount": row[col_idx.get("amount")] if "amount" in col_idx else row[col_idx.get("turnover")]
                })
                if "volume" not in kline and "vol" in kline:
                    kline["volume"] = kline["vol"]
                if "vol" not in kline and "volume" in kline:
                    kline["vol"] = kline["volume"]
                if "turnover" not in kline and "amount" in kline:
                    kline["turnover"] = kline["amount"]
                if "amount" not in kline and "turnover" in kline:
                    kline["amount"] = kline["turnover"]
                current_dt = pd.to_datetime(kline["dt"])
                current_factor = float(factor_series.iloc[i])
                if previous_factor is not None and abs(current_factor / previous_factor - 1.0) > 1e-10:
                    has_holdings = any(int(position.get('qty', 0)) > 0 for positions in self.execution_engine.positions.values() for position in positions.values())
                    if has_holdings:
                        await self._emit('backtest_failed', {'msg': f'{current_dt.date()} 检测到除权事件且仍有持仓；分红、送转入账未实现，回测已终止。', 'stock': self.stock_code, 'price_policy': dict(self.baseline_meta['price_policy'])})
                        return
                    cancelled = [signal for signal in self.pending_signals if signal.get('code') == self.stock_code]
                    self.pending_signals = [signal for signal in self.pending_signals if signal.get('code') != self.stock_code]
                    if cancelled:
                        await self._emit('backtest_flow', {'module': '订单执行', 'level': 'warning',
                            'msg': f'{current_dt.date()} 因子变化，取消 {len(cancelled)} 条跨除权日待执行信号，等待策略重新生成'})
                    if self.adjustment_mode == 'qfq':
                        for strategy in self.strategies:
                            rebase_strategy_prices(strategy, self.stock_code, previous_factor / current_factor)
                previous_factor = current_factor
                self._start_risk_session(kline)
                await self._execute_drawdown_orders(kline)
                await self._execute_pending_signals(kline)
                await self._execute_protective_orders(kline)
                runnable_strategy_ids = []
                strategy_kline_map = {}
                for sid, tf in strategy_trigger_tf.items():
                    if tf == base_interval:
                        runnable_strategy_ids.append(sid)
                        strategy_kline_map[sid] = kline
                    else:
                        if current_dt in tf_dt_sets.get(tf, set()):
                            runnable_strategy_ids.append(sid)
                            strategy_kline_map[sid] = tf_row_maps[tf][current_dt]
                strategy_kline_map = {sid: price_plan.signal_bar(bar) for sid, bar in strategy_kline_map.items()}
                if i % report_interval == 0:
                    progress = int((i / total_bars) * 100)
                    await self._emit('backtest_progress', {'progress': progress, 'current_date': str(kline['dt'])})
                    runnable_tf = sorted(set([strategy_trigger_tf.get(s, "1min") for s in runnable_strategy_ids]))
                    await self._emit('market', {
                        'price': float(kline['close']),
                        'signal_price': float(price_plan.signal_bar(kline)['close']),
                        'signal_price_mode': self.adjustment_mode,
                        'ma5': float(ma5_series.iloc[i]) / (current_factor / factor_series.iloc[0] if self.adjustment_mode == 'qfq' else 1.0),
                        'macd': float(macd_series.iloc[i]) / (current_factor / factor_series.iloc[0] if self.adjustment_mode == 'qfq' else 1.0),
                        'rsi': float(rsi_series.iloc[i]),
                        'time': str(kline['dt']),
                        'kline_timeframe': f'{base_interval}驱动',
                        'kline_dt': str(kline['dt']),
                        'stock_code': str(self.stock_code or '').upper(),
                        'strategy_timeframes': strategy_trigger_tf,
                        'runnable_strategy_ids': runnable_strategy_ids,
                        'runnable_timeframes': runnable_tf
                    })
                    await self._emit_account_snapshot(kline, active_strategy_id=None, compliance_status="PASS")
                drawdown_limited = await self._enforce_drawdown_limit(kline)
                if drawdown_limited:
                    self._record_nav(kline)
                    continue
                strategy_context = {}
                for sid in runnable_strategy_ids:
                    account = self.strategy_accounts.get(sid)
                    if account is None:
                        continue
                    strategy_bar = strategy_kline_map.get(sid, kline) or kline
                    last_price = float(kline.get("close", 0.0) or 0.0)
                    holdings_value = self.execution_engine.update_strategy_holdings_value(
                        sid, {self.stock_code: last_price}
                    )
                    current_cash = float(account.cash)
                    strategy_context[sid] = {
                        "current_cash": current_cash,
                        "available_cash": current_cash,
                        "total_value": float(current_cash + holdings_value),
                        "last_price": last_price,
                        "signal_price": float(strategy_bar.get('close', last_price))
                    }
                signals = self.strategy_engine.generate_signals(
                    kline,
                    runnable_strategy_ids=runnable_strategy_ids,
                    strategy_context={"__by_strategy__": strategy_context, "__kline_by_strategy__": strategy_kline_map}
                )
                signals, combo_meta = self._apply_signal_combination(signals, runnable_strategy_ids)
                should_emit_detail_flow = (i % self._flow_emit_stride == 0)
                if signals and should_emit_detail_flow:
                    await self._emit('backtest_flow', {
                        'module': '策略引擎',
                        'level': 'warning',
                        'msg': f"{current_dt} 触发 {len(signals)} 条候选交易信号（策略: {','.join([s['strategy_id'] for s in signals])}）"
                    })
                if isinstance(combo_meta, dict) and bool(combo_meta.get("enabled", False)) and should_emit_detail_flow:
                    await self._emit('backtest_flow', {
                        'module': '策略引擎',
                        'level': 'system',
                        'msg': f"组合过滤 mode={combo_meta.get('mode')} kept={combo_meta.get('kept')} meta={combo_meta}"
                    })
                for signal in signals:
                    queued = dict(signal)
                    queued['dt'] = kline['dt']
                    self.pending_signals.append(queued)
                    await self._emit('strategy_signal', {
                        'msg': f"策略 {signal['strategy_id']} 生成信号，等待下一根K线开盘",
                        'details': f"> 标的: {signal['code']}<br>> 方向: {signal['direction']}",
                        'status': 'bg-trading-blue', 'execution_timing': 'next_bar_open',
                    })
                self._record_nav(kline)
            if self.pending_signals:
                await self._emit('backtest_flow', {'module': '订单执行', 'level': 'warning',
                    'msg': f'区间末尾 {len(self.pending_signals)} 条信号没有后续K线，未模拟成交'})
                self.pending_signals = []
            perf_main_loop_ms = int((perf_counter() - stage_started_at) * 1000)
            await self._emit('backtest_progress', {'progress': 100, 'current_date': 'Done'})
            stage_started_at = perf_counter()
            last_row = df.iloc[-1]
            final_kline = last_row.to_dict()
            final_kline.update({
                "dt": pd.to_datetime(last_row["dt"]),
                "code": self.stock_code,
                "open": float(last_row.get("open", last_row.get("close", 0)) or 0),
                "high": float(last_row.get("high", last_row.get("close", 0)) or 0),
                "low": float(last_row.get("low", last_row.get("close", 0)) or 0),
                "close": float(last_row.get("close", 0) or 0),
                "volume": float(last_row.get("volume", last_row.get("vol", 0)) or 0)
            })
            await self._force_close_positions_at_end(final_kline)
            # The final card and reports must reflect the same post-settlement NAV,
            # including runs with no sale or a T+1/limit-blocked final liquidation.
            await self._emit_account_snapshot(final_kline, is_final=True)
            reports = []
            for s in self.strategies:
                account = self.strategy_accounts.get(s.id)
                if account is None:
                    continue
                report = self.performance_reporter.generate_report(
                    s.id,
                    account,
                    self.risk_audit_log,
                    self.strategy_initial_capital,
                    start_date=start_date,
                    end_date=end_date
                )
                reports.append(report)
                strategy_transactions = list(account.transactions)
                formatted = self.performance_reporter.generate_backtest_report(
                    strategy_id=s.id,
                    transactions=strategy_transactions,
                    initial_capital=self.strategy_initial_capital,
                    start_date=start_date,
                    end_date=end_date,
                    summary_metrics=report,
                    nav_history=account.get_nav_history(),
                    settlement=self.execution_engine.get_strategy_settlement(
                        s.id, account, {self.stock_code: final_kline['close']}, as_of=final_kline['dt']),
                )
                tf = strategy_trigger_tf.get(s.id, "1min")
                formatted["kline_type"] = tf
                if tf == "D":
                    formatted["period_label"] = "日线"
                elif tf.endswith("min"):
                    formatted["period_label"] = tf.replace("min", "分钟")
                else:
                    formatted["period_label"] = tf
                formatted["baseline_meta"] = dict(self.baseline_meta)
                await self._emit('backtest_strategy_report', formatted)
            ranking = self.performance_reporter.generate_ranking(reports)
            ranking_dict = ranking.to_dict('records')
            perf_settlement_ms = int((perf_counter() - stage_started_at) * 1000)
            perf_total_ms = int((perf_counter() - total_started_at) * 1000)
            await self._emit('backtest_result', {
                'stock': self.stock_code,
                'period': f"{start_date.date()} - {end_date.date()}",
                'ranking': ranking_dict,
                'total_trades': sum(r['total_trades'] for r in reports),
                'baseline_meta': dict(self.baseline_meta),
                'perf_ms': {
                    'data_fetch': perf_data_fetch_ms,
                    'period_build': perf_period_build_ms,
                    'main_loop': perf_main_loop_ms,
                    'settlement': perf_settlement_ms,
                    'total': perf_total_ms
                }
            })
            await self._emit('backtest_flow', {
                'module': '市场监测',
                'level': 'system',
                'msg': f"耗时统计(ms): 数据获取={perf_data_fetch_ms} 周期构建={perf_period_build_ms} 主循环={perf_main_loop_ms} 结算={perf_settlement_ms} 总计={perf_total_ms}"
            })
            await self._emit('backtest_flow', {'module': '市场监测', 'level': 'success', 'msg': f'回测结算完成，总交易 {sum(r["total_trades"] for r in reports)} 笔'})
            await self._emit('system', {'msg': f"回测完成。"})
        except (PriceAdjustmentError, MarketDataValidationError) as exc:
            await self._emit('backtest_failed', {'msg': str(exc), 'stock': self.stock_code,
                                                'price_policy': self.baseline_meta.get('price_policy', {})})
        finally:
            await self._stop_event_pump()
