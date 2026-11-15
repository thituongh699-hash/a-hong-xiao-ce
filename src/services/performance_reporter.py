# src/services/performance_reporter.py
import pandas as pd
import numpy as np


class PerformanceReporter:
    """
    市场监测 (Rites): 生成每套策略独立业绩报表、排行榜
    """
    def _safe_float(self, value, default=0.0):
        try:
            v = float(value)
            if np.isnan(v) or np.isinf(v):
                return float(default)
            return v
        except Exception:
            return float(default)

    def _closed_trades(self, transactions):
        return [t for t in transactions if str(t.get("direction", "")).upper() == "SELL"]

    def _parse_nav_datetimes(self, values):
        """Parse heterogeneous ISO values without dropping malformed observations."""
        try:
            parsed = pd.to_datetime(values, format="mixed", errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("nav_history contains an invalid dt value") from exc
        invalid = pd.isna(parsed)
        if bool(np.asarray(invalid).any()):
            raise ValueError("nav_history contains a missing or invalid dt value")
        return parsed

    def _validated_settlement(self, settlement):
        if settlement is None:
            return None
        if not isinstance(settlement, dict):
            raise ValueError("settlement must be a dict or None")

        required = {
            "as_of", "state", "cash", "holdings_value", "open_position_count",
            "open_quantity", "realized_pnl", "unrealized_pnl", "fees_paid",
            "reconciliation_delta", "positions",
        }
        missing = sorted(required.difference(settlement))
        if missing:
            raise ValueError(f"settlement is missing required fields: {', '.join(missing)}")
        state = str(settlement.get("state", "")).strip().lower()
        if state not in {"flat", "open"}:
            raise ValueError("settlement state must be 'flat' or 'open'")
        if settlement.get("as_of") is None or str(settlement.get("as_of")).strip() == "":
            raise ValueError("settlement as_of is required")

        numeric_fields = (
            "cash", "holdings_value", "open_position_count", "open_quantity",
            "realized_pnl", "unrealized_pnl", "fees_paid", "reconciliation_delta",
        )
        for field in numeric_fields:
            try:
                value = float(settlement.get(field))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"settlement {field} must be numeric") from exc
            if not np.isfinite(value):
                raise ValueError(f"settlement {field} must be finite")
        open_position_count = float(settlement.get("open_position_count"))
        if open_position_count < 0 or not open_position_count.is_integer():
            raise ValueError("settlement open_position_count must be a non-negative integer")
        if float(settlement.get("open_quantity")) < 0:
            raise ValueError("settlement open_quantity must be non-negative")
        if not isinstance(settlement.get("positions"), list):
            raise ValueError("settlement positions must be a list")
        if any(not isinstance(position, dict) for position in settlement["positions"]):
            raise ValueError("settlement positions must contain dict items")

        # Keep the engine snapshot values intact; only copy the mutable containers.
        normalized = dict(settlement)
        normalized["state"] = state
        normalized["positions"] = [dict(position) for position in settlement["positions"]]
        return normalized

    def _daily_nav(self, nav_history):
        """Keep the last observation of each actual session, in date order."""
        if nav_history is None:
            return pd.Series(dtype=float)
        frame = nav_history.copy() if isinstance(nav_history, pd.DataFrame) else pd.DataFrame(nav_history)
        if frame.empty or not {"dt", "nav"}.issubset(frame.columns):
            return pd.Series(dtype=float)
        frame = frame[["dt", "nav"]].copy()
        frame["dt"] = self._parse_nav_datetimes(frame["dt"])
        frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
        frame = frame.dropna(subset=["nav"])
        frame = frame[np.isfinite(frame["nav"]) & (frame["nav"] >= 0)]
        if frame.empty:
            return pd.Series(dtype=float)
        frame = frame.sort_values("dt", kind="stable")
        frame["day"] = frame["dt"].dt.normalize()
        frame = frame.drop_duplicates("day", keep="last")
        return pd.Series(frame["nav"].to_numpy(dtype=float), index=pd.DatetimeIndex(frame["day"]), dtype=float)

    def _session_returns(self, equity, initial_capital):
        if equity is None or equity.empty:
            return pd.Series(dtype=float)
        initial_capital = self._safe_float(initial_capital)
        if initial_capital <= 0:
            return pd.Series(dtype=float)
        previous = equity.shift(1)
        previous.iloc[0] = initial_capital
        returns = (equity / previous.replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan)
        return returns.dropna()

    def _portfolio_metrics(self, equity, initial_capital, start_date=None, end_date=None):
        initial_capital = self._safe_float(initial_capital)
        ending_capital = self._safe_float(equity.iloc[-1], initial_capital) if len(equity) else initial_capital
        roi = ending_capital / initial_capital - 1 if initial_capital > 0 else 0.0
        if start_date is not None and end_date is not None:
            days = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days, 1)
        elif len(equity) > 1:
            days = max((equity.index[-1] - equity.index[0]).days, 1)
        else:
            days = 1
        annualized = -1.0 if roi <= -1 else (1 + roi) ** (365.25 / days) - 1
        # Include the initial capital as the starting peak, even when the first NAV loses money.
        peaks = equity.cummax().clip(lower=initial_capital)
        drawdowns = (equity / peaks.replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan)
        max_dd = abs(self._safe_float(drawdowns.min(), 0.0)) if len(equity) else 0.0
        returns = self._session_returns(equity, initial_capital)
        return ending_capital, roi, annualized, max_dd, self._compute_sharpe(returns=returns), days, returns

    def _profit_factor(self, closed_trades):
        wins = [self._safe_float(t.get("pnl")) for t in closed_trades if self._safe_float(t.get("pnl")) > 0]
        losses = [self._safe_float(t.get("pnl")) for t in closed_trades if self._safe_float(t.get("pnl")) < 0]
        # Undefined without a loss; null is JSON-safe and avoids a fictitious finite ratio.
        return sum(wins) / abs(sum(losses)) if losses else None

    def _compute_streaks(self, closed_trades):
        max_win = 0
        max_loss = 0
        cur_win = 0
        cur_loss = 0
        for t in sorted(closed_trades, key=lambda x: x.get("dt")):
            pnl = self._safe_float(t.get("pnl", 0.0))
            if pnl > 0:
                cur_win += 1
                cur_loss = 0
            elif pnl < 0:
                cur_loss += 1
                cur_win = 0
            else:
                cur_win = 0
                cur_loss = 0
            if cur_win > max_win:
                max_win = cur_win
            if cur_loss > max_loss:
                max_loss = cur_loss
        return max_win, max_loss

    def _compute_monthly_profit_ratio(self, closed_trades):
        if not closed_trades:
            return 0.0
        rows = []
        for t in closed_trades:
            dt = pd.to_datetime(t.get("dt"))
            if pd.isna(dt):
                continue
            rows.append({"month": dt.strftime("%Y-%m"), "pnl": self._safe_float(t.get("pnl", 0.0))})
        if not rows:
            return 0.0
        df = pd.DataFrame(rows)
        monthly = df.groupby("month", as_index=False)["pnl"].sum()
        if monthly.empty:
            return 0.0
        return float((monthly["pnl"] > 0).sum() / len(monthly))

    def _compute_equity_curve(self, closed_trades, initial_capital, start_date=None, end_date=None):
        init_cap = self._safe_float(initial_capital)
        if init_cap <= 0:
            init_cap = 1.0
        if start_date is not None and end_date is not None:
            start = pd.to_datetime(start_date).normalize()
            end = pd.to_datetime(end_date).normalize()
        elif closed_trades:
            dts = [pd.to_datetime(t.get("dt")) for t in closed_trades if not pd.isna(pd.to_datetime(t.get("dt")))]
            if dts:
                start = min(dts).normalize()
                end = max(dts).normalize()
            else:
                now = pd.Timestamp.now().normalize()
                start = now
                end = now
        else:
            now = pd.Timestamp.now().normalize()
            start = now
            end = now
        if pd.isna(start) or pd.isna(end) or end < start:
            end = start
        # Transaction-only compatibility path: keep observed dates, not fabricated weekends.
        trade_days = []
        for t in closed_trades:
            dt = pd.to_datetime(t.get("dt"), errors="coerce")
            if pd.isna(dt):
                continue
            key = dt.normalize()
            if start <= key <= end:
                trade_days.append((key, self._safe_float(t.get("pnl", 0.0))))
        if trade_days:
            daily_pnl = pd.DataFrame(trade_days, columns=["day", "pnl"]).groupby("day")["pnl"].sum().sort_index()
        else:
            daily_pnl = pd.Series([0.0], index=pd.DatetimeIndex([start]))
        equity = init_cap + daily_pnl.cumsum()
        return equity

    def _compute_sharpe(self, equity=None, rf=0.02, returns=None):
        if returns is not None:
            rets = returns
        elif equity is not None:
            rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        else:
            rets = pd.Series(dtype=float)
        if rets.empty:
            return 0.0
        daily_rf = (1 + rf) ** (1 / 252) - 1
        excess = rets - daily_rf
        vol = excess.std(ddof=0)
        if not np.isfinite(vol) or vol <= 1e-12:
            return 0.0
        return float((excess.mean() / vol) * np.sqrt(252))

    def _compute_regime_consistency_ratio(self, equity, initial_capital=None):
        if equity is None or len(equity) < 15:
            return 0.0
        ret = self._session_returns(equity, initial_capital) if initial_capital is not None else equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if ret.empty:
            return 0.0
        bull = ret[ret > 0.002]
        bear = ret[ret < -0.002]
        side = ret[(ret >= -0.002) & (ret <= 0.002)]
        vals = []
        for seg in [bull, bear, side]:
            if len(seg) == 0:
                continue
            vals.append(1.0 if seg.mean() > 0 else 0.0)
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def _compute_regime_layer_metrics(self, equity, initial_capital=None):
        if equity is None or len(equity) < 15:
            return {
                "trend_state_win_ratio": 0.0,
                "sideway_state_win_ratio": 0.0,
                "down_state_win_ratio": 0.0,
                "regime_layer_score": 0.0
            }
        ret = self._session_returns(equity, initial_capital) if initial_capital is not None else equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if ret.empty:
            return {
                "trend_state_win_ratio": 0.0,
                "sideway_state_win_ratio": 0.0,
                "down_state_win_ratio": 0.0,
                "regime_layer_score": 0.0
            }
        bull = ret[ret > 0.003]
        bear = ret[ret < -0.003]
        side = ret[(ret >= -0.003) & (ret <= 0.003)]
        bull_wr = float((bull > 0).sum() / len(bull)) if len(bull) > 0 else 0.5
        side_wr = float((side > 0).sum() / len(side)) if len(side) > 0 else 0.5
        bear_wr = float((bear > 0).sum() / len(bear)) if len(bear) > 0 else 0.5
        layer_score = float(np.mean([bull_wr, side_wr, bear_wr]))
        return {
            "trend_state_win_ratio": bull_wr,
            "sideway_state_win_ratio": side_wr,
            "down_state_win_ratio": bear_wr,
            "regime_layer_score": layer_score
        }

    def _score_piecewise(self, value, thresholds):
        v = self._safe_float(value)
        for cond, score in thresholds:
            if cond(v):
                return float(score)
        return 0.0

    def _grade_from_score(self, score):
        s = self._safe_float(score)
        if s >= 90:
            return "S", "可实盘、可加仓"
        if s >= 75:
            return "A", "可小资金实盘"
        if s >= 60:
            return "B", "继续优化"
        return "C", "放弃或重构"

    def _build_scorecard(self, metrics):
        annualized_roi = self._safe_float(metrics.get("annualized_roi", 0.0))
        total_return = self._safe_float(metrics.get("roi", 0.0))
        win_rate = self._safe_float(metrics.get("win_rate", 0.0))
        max_dd = self._safe_float(metrics.get("max_dd", 0.0))
        sharpe = self._safe_float(metrics.get("sharpe", 0.0))
        calmar = self._safe_float(metrics.get("calmar", 0.0))
        profit_ratio = self._safe_float(metrics.get("profit_factor", 0.0))
        max_loss_streak = int(metrics.get("max_loss_streak", 0) or 0)
        monthly_profit_ratio = self._safe_float(metrics.get("monthly_profit_ratio", 0.0))
        annual_trades = self._safe_float(metrics.get("annualized_trades", 0.0))
        avg_trade_amount_ratio = self._safe_float(metrics.get("avg_trade_amount_ratio", 0.0))
        monthly_stability = self._safe_float(metrics.get("monthly_stability_ratio", 0.0))
        regime_consistency = self._safe_float(metrics.get("regime_consistency_ratio", 0.0))
        regime_layer_score = self._safe_float(metrics.get("regime_layer_score", 0.0))
        capital_utilization_ratio = self._safe_float(metrics.get("capital_utilization_ratio", 0.0))
        holding_efficiency_index = self._safe_float(metrics.get("holding_efficiency_index", 0.0))
        trend_state_win_ratio = self._safe_float(metrics.get("trend_state_win_ratio", 0.0))
        sideway_state_win_ratio = self._safe_float(metrics.get("sideway_state_win_ratio", 0.0))
        down_state_win_ratio = self._safe_float(metrics.get("down_state_win_ratio", 0.0))

        score_ar = self._score_piecewise(annualized_roi, [
            (lambda x: x >= 0.30, 10),
            (lambda x: x >= 0.15, 7),
            (lambda x: x >= 0.08, 4),
            (lambda x: x > 0.00, 2),
            (lambda x: True, 0),
        ])
        score_tr = self._score_piecewise(total_return, [
            (lambda x: x >= 0.40, 10),
            (lambda x: x >= 0.20, 7),
            (lambda x: x >= 0.10, 4),
            (lambda x: x > 0.00, 2),
            (lambda x: True, 0),
        ])
        score_wr = self._score_piecewise(win_rate, [
            (lambda x: x >= 0.60, 10),
            (lambda x: x >= 0.50, 7),
            (lambda x: x >= 0.45, 4),
            (lambda x: True, 1),
        ])
        profitability = score_ar + score_tr + score_wr

        score_dd = self._score_piecewise(max_dd, [
            (lambda x: x <= 0.10, 15),
            (lambda x: x <= 0.15, 10),
            (lambda x: x <= 0.20, 5),
            (lambda x: True, 2),
        ])
        score_sharpe = self._score_piecewise(sharpe, [
            (lambda x: x >= 1.8, 10),
            (lambda x: x >= 1.2, 7),
            (lambda x: x >= 0.8, 4),
            (lambda x: True, 1),
        ])
        score_calmar = self._score_piecewise(calmar, [
            (lambda x: x >= 1.5, 10),
            (lambda x: x >= 1.0, 7),
            (lambda x: x >= 0.6, 4),
            (lambda x: True, 1),
        ])
        risk_control = score_dd + score_sharpe + score_calmar

        score_pl = self._score_piecewise(profit_ratio, [
            (lambda x: x >= 2.2, 10),
            (lambda x: x >= 1.8, 7),
            (lambda x: x >= 1.5, 4),
            (lambda x: True, 1),
        ]) if metrics.get("profit_factor") is not None else 0.0
        score_loss_streak = self._score_piecewise(max_loss_streak, [
            (lambda x: x <= 3, 5),
            (lambda x: x <= 5, 3),
            (lambda x: True, 1),
        ])
        score_monthly = self._score_piecewise(monthly_profit_ratio, [
            (lambda x: x >= 0.70, 5),
            (lambda x: x >= 0.55, 3),
            (lambda x: True, 1),
        ])
        pnl_quality = score_pl + score_loss_streak + score_monthly

        score_freq = self._score_piecewise(annual_trades, [
            (lambda x: 20 <= x <= 240, 5),
            (lambda x: 8 <= x < 20 or 240 < x <= 360, 3),
            (lambda x: True, 1),
        ])
        score_util = self._score_piecewise(capital_utilization_ratio, [
            (lambda x: x >= 0.45, 2.0),
            (lambda x: x >= 0.25, 1.5),
            (lambda x: x >= 0.12, 1.0),
            (lambda x: True, 0.5),
        ])
        score_hold_eff = self._score_piecewise(holding_efficiency_index, [
            (lambda x: x >= 0.18, 2.0),
            (lambda x: x >= 0.10, 1.5),
            (lambda x: x >= 0.04, 1.0),
            (lambda x: True, 0.5),
        ])
        score_capacity = score_util + score_hold_eff
        score_stability = self._score_piecewise(monthly_stability, [
            (lambda x: x >= 0.65, 3),
            (lambda x: x >= 0.50, 2),
            (lambda x: True, 1),
        ])
        regime_mix = (regime_consistency + regime_layer_score) / 2.0
        score_regime = self._score_piecewise(regime_mix, [
            (lambda x: x >= 0.66, 3),
            (lambda x: x >= 0.33, 2),
            (lambda x: True, 1),
        ])
        practicality = score_freq + score_capacity + score_stability + score_regime

        total = float(profitability + risk_control + pnl_quality + practicality)
        grade, conclusion = self._grade_from_score(total)
        return {
            "total_score": round(total, 2),
            "grade": grade,
            "conclusion": conclusion,
            "dimensions": {
                "profitability": {"name": "收益能力", "score": round(profitability, 2), "max_score": 30},
                "risk_control": {"name": "风险控制", "score": round(risk_control, 2), "max_score": 35},
                "pnl_quality": {"name": "盈亏质量", "score": round(pnl_quality, 2), "max_score": 20},
                "practicality": {"name": "实战可行性", "score": round(practicality, 2), "max_score": 15},
            },
            "metrics": {
                "annualized_roi": annualized_roi,
                "total_return": total_return,
                "win_rate": win_rate,
                "max_drawdown": max_dd,
                "sharpe": sharpe,
                "calmar": calmar,
                "profit_loss_ratio": profit_ratio,
                "max_loss_streak": max_loss_streak,
                "monthly_profit_ratio": monthly_profit_ratio,
                "annualized_trades": annual_trades,
                "avg_trade_amount_ratio": avg_trade_amount_ratio,
                "capital_utilization_ratio": capital_utilization_ratio,
                "holding_efficiency_index": holding_efficiency_index,
                "monthly_stability_ratio": monthly_stability,
                "regime_consistency_ratio": regime_consistency,
                "trend_state_win_ratio": trend_state_win_ratio,
                "sideway_state_win_ratio": sideway_state_win_ratio,
                "down_state_win_ratio": down_state_win_ratio,
                "regime_layer_score": regime_layer_score
            }
        }

    def generate_report(self, strategy_id, account_ledger, risk_audit_log, initial_capital, start_date=None, end_date=None):
        transactions = [t for t in account_ledger.transactions if t["strategy_id"] == strategy_id]
        closed = self._closed_trades(transactions)
        total_trades = len(closed)
        wins = len([t for t in closed if self._safe_float(t.get("pnl", 0.0)) > 0])
        losses = len([t for t in closed if self._safe_float(t.get("pnl", 0.0)) < 0])
        breakevens = len([t for t in closed if self._safe_float(t.get("pnl", 0.0)) == 0])
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        total_pnl = float(sum(self._safe_float(t.get("pnl", 0.0)) for t in closed))
        init_cap = self._safe_float(initial_capital, 1.0)
        nav_df = account_ledger.get_nav_history() if callable(getattr(account_ledger, "get_nav_history", None)) else None
        nav_equity = self._daily_nav(nav_df)
        has_nav = not nav_equity.empty
        equity = nav_equity if has_nav else self._compute_equity_curve(closed, init_cap, start_date=start_date, end_date=end_date)
        final_capital, roi, annualized_roi, max_dd_pct, sharpe, days, rets = self._portfolio_metrics(
            equity, init_cap, start_date=start_date, end_date=end_date
        )
        return_basis = "daily_nav" if has_nav else "closed_trade_pnl"

        avg_win = np.mean([self._safe_float(t.get("pnl", 0.0)) for t in closed if self._safe_float(t.get("pnl", 0.0)) > 0]) if wins > 0 else 0.0
        avg_loss_raw = np.mean([self._safe_float(t.get("pnl", 0.0)) for t in closed if self._safe_float(t.get("pnl", 0.0)) < 0]) if losses > 0 else 0.0
        avg_loss = abs(avg_loss_raw)
        profit_factor = self._profit_factor(closed)

        rejections = risk_audit_log.get_rejection_count(strategy_id)
        violations = risk_audit_log.get_violation_count(strategy_id)
        circuit_breaks = risk_audit_log.get_circuit_break_count(strategy_id)

        calmar = (annualized_roi / max_dd_pct) if max_dd_pct > 0 else 0.0
        max_win_streak, max_loss_streak = self._compute_streaks(closed)
        monthly_profit_ratio = self._compute_monthly_profit_ratio(closed)
        annualized_trades = total_trades * 365.25 / max(days, 1)
        trade_amounts = [abs(self._safe_float(t.get("amount", 0.0))) for t in transactions if self._safe_float(t.get("amount", 0.0)) > 0]
        avg_trade_amount = float(np.mean(trade_amounts)) if trade_amounts else 0.0
        avg_trade_amount_ratio = (avg_trade_amount / init_cap) if init_cap > 0 else 0.0
        capital_utilization_ratio = 0.0
        if isinstance(nav_df, pd.DataFrame) and (not nav_df.empty) and ("holdings" in nav_df.columns) and ("nav" in nav_df.columns):
            utilization_frame = nav_df.copy()
            if "dt" in utilization_frame.columns:
                utilization_frame["dt"] = self._parse_nav_datetimes(utilization_frame["dt"])
                utilization_frame = utilization_frame.sort_values("dt", kind="stable")
                utilization_frame["day"] = utilization_frame["dt"].dt.normalize()
                utilization_frame = utilization_frame.drop_duplicates("day", keep="last")
            nav = pd.to_numeric(utilization_frame["nav"], errors="coerce")
            holdings = pd.to_numeric(utilization_frame["holdings"], errors="coerce")
            util = (holdings / nav.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
            util = util.clip(lower=0.0, upper=1.5)
            if len(util) > 0:
                capital_utilization_ratio = float(util.mean())
        positive_day_ratio = float((rets > 0).sum() / len(rets)) if len(rets) > 0 else 0.0
        regime_consistency_ratio = self._compute_regime_consistency_ratio(equity, init_cap)
        regime_layer = self._compute_regime_layer_metrics(equity, init_cap)
        monthly_equity = equity.resample("ME").last() if len(equity) > 0 else pd.Series(dtype=float)
        monthly_ret = self._session_returns(monthly_equity, init_cap)
        monthly_stability_ratio = float((monthly_ret > 0).sum() / len(monthly_ret)) if len(monthly_ret) > 0 else 0.0
        util_weight = min(1.0, max(capital_utilization_ratio, 0.0) / 0.35)
        holding_efficiency_index = float(annualized_roi * (0.5 + 0.5 * util_weight))

        metrics = {
            "annualized_roi": annualized_roi,
            "roi": roi,
            "win_rate": win_rate,
            "max_dd": max_dd_pct,
            "sharpe": sharpe,
            "calmar": calmar,
            "profit_factor": profit_factor,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "monthly_profit_ratio": monthly_profit_ratio,
            "annualized_trades": annualized_trades,
            "avg_trade_amount_ratio": avg_trade_amount_ratio,
            "capital_utilization_ratio": capital_utilization_ratio,
            "holding_efficiency_index": holding_efficiency_index,
            "monthly_stability_ratio": monthly_stability_ratio,
            "regime_consistency_ratio": regime_consistency_ratio,
            "trend_state_win_ratio": regime_layer.get("trend_state_win_ratio", 0.0),
            "sideway_state_win_ratio": regime_layer.get("sideway_state_win_ratio", 0.0),
            "down_state_win_ratio": regime_layer.get("down_state_win_ratio", 0.0),
            "regime_layer_score": regime_layer.get("regime_layer_score", 0.0)
        }
        scorecard = self._build_scorecard(metrics)

        return {
            "strategy_id": strategy_id,
            "total_pnl": total_pnl,
            "realized_pnl": total_pnl,
            "return_basis": return_basis,
            "annualization_basis": "calendar_days_365.25",
            "sharpe_basis": "session_returns_252",
            "roi": roi,
            "annualized_roi": annualized_roi,
            "max_dd": max_dd_pct,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "trade_count": total_trades,
            "execution_count": len(transactions),
            "buy_count": len([t for t in transactions if str(t.get("direction", "")).upper() == "BUY"]),
            "sell_count": len([t for t in transactions if str(t.get("direction", "")).upper() == "SELL"]),
            "breakeven_num": breakevens,
            "rejections": rejections,
            "violations": violations,
            "circuit_breaks": circuit_breaks,
            "calmar": calmar,
            "sharpe": sharpe,
            "final_capital": final_capital,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "max_win_streak": int(max_win_streak),
            "max_loss_streak": int(max_loss_streak),
            "monthly_profit_ratio": float(monthly_profit_ratio),
            "annualized_trades": float(annualized_trades),
            "avg_trade_amount_ratio": float(avg_trade_amount_ratio),
            "capital_utilization_ratio": float(capital_utilization_ratio),
            "holding_efficiency_index": float(holding_efficiency_index),
            "monthly_stability_ratio": float(monthly_stability_ratio),
            "regime_consistency_ratio": float(regime_consistency_ratio),
            "trend_state_win_ratio": float(regime_layer.get("trend_state_win_ratio", 0.0)),
            "sideway_state_win_ratio": float(regime_layer.get("sideway_state_win_ratio", 0.0)),
            "down_state_win_ratio": float(regime_layer.get("down_state_win_ratio", 0.0)),
            "regime_layer_score": float(regime_layer.get("regime_layer_score", 0.0)),
            "positive_day_ratio": float(positive_day_ratio),
            "scorecard": scorecard,
            "score_total": float(scorecard.get("total_score", 0.0)),
            "rating": str(scorecard.get("grade", "C"))
        }

    def generate_ranking(self, reports):
        df = pd.DataFrame(reports)
        if df.empty:
            return df
        if "score_total" not in df.columns:
            df["score_total"] = 0.0
        if "rating" not in df.columns:
            df["rating"] = "C"
        for col in ["calmar", "annualized_roi"]:
            if col not in df.columns:
                df[col] = 0.0
        df = df.sort_values(by=["score_total", "calmar", "annualized_roi"], ascending=[False, False, False]).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df

    def generate_backtest_report(self, strategy_id, transactions, initial_capital, start_date=None, end_date=None, summary_metrics=None, nav_history=None, settlement=None):
        closed_trades = self._closed_trades(transactions)
        trade_count = len(closed_trades)
        win_trades = [t for t in closed_trades if self._safe_float(t.get("pnl", 0)) > 0]
        loss_trades = [t for t in closed_trades if self._safe_float(t.get("pnl", 0)) < 0]
        breakeven_trades = [t for t in closed_trades if self._safe_float(t.get("pnl", 0)) == 0]
        execution_count = len(transactions)
        buy_count = len([t for t in transactions if str(t.get("direction", "")).upper() == "BUY"])
        sell_count = len([t for t in transactions if str(t.get("direction", "")).upper() == "SELL"])

        win_num = len(win_trades)
        loss_num = len(loss_trades)
        win_rate = win_num / trade_count if trade_count > 0 else 0.0

        total_pnl = float(sum(self._safe_float(t.get("pnl")) for t in closed_trades))
        init_cap = self._safe_float(initial_capital)
        nav_equity = self._daily_nav(nav_history)
        has_nav = not nav_equity.empty
        equity_curve = nav_equity if has_nav else self._compute_equity_curve(closed_trades, init_cap, start_date=start_date, end_date=end_date)
        end_cap, total_return, annual_return, dd_abs, sharpe, days, _ = self._portfolio_metrics(
            equity_curve, init_cap, start_date=start_date, end_date=end_date
        )
        return_basis = "daily_nav" if has_nav else "closed_trade_pnl"
        if isinstance(summary_metrics, dict) and summary_metrics.get("return_basis") == "daily_nav":
            # The summary has already calculated the same portfolio metrics from the ledger.
            end_cap = self._safe_float(summary_metrics.get("final_capital"), end_cap)
            total_return = self._safe_float(summary_metrics.get("roi"), total_return)
            annual_return = self._safe_float(summary_metrics.get("annualized_roi"), annual_return)
            dd_abs = self._safe_float(summary_metrics.get("max_dd"), dd_abs)
            sharpe = self._safe_float(summary_metrics.get("sharpe"), sharpe)
            return_basis = "daily_nav"

        if start_date is not None and end_date is not None:
            start_txt = str(start_date)[:10]
            end_txt = str(end_date)[:10]
        elif has_nav:
            start_txt = str(nav_equity.index[0])[:10]
            end_txt = str(nav_equity.index[-1])[:10]
        elif closed_trades:
            trade_dates = [t["dt"] for t in closed_trades]
            sdt = min(trade_dates)
            edt = max(trade_dates)
            start_txt = str(sdt)[:10]
            end_txt = str(edt)[:10]
        else:
            start_txt = "--"
            end_txt = "--"
        # Keep the historical API sign for compatibility. Both report UIs render abs(value).
        max_drawdown = -dd_abs

        avg_win = float(np.mean([t.get("pnl", 0.0) for t in win_trades])) if win_trades else 0.0
        avg_loss = abs(float(np.mean([t.get("pnl", 0.0) for t in loss_trades]))) if loss_trades else 0.0
        profit_ratio = self._profit_factor(closed_trades)
        max_win_streak, max_loss_streak = self._compute_streaks(closed_trades)
        monthly_profit_ratio = self._compute_monthly_profit_ratio(closed_trades)
        annualized_trades = trade_count * 365.25 / max(days, 1)
        trade_amounts = [abs(self._safe_float(t.get("amount", 0.0))) for t in transactions if self._safe_float(t.get("amount", 0.0)) > 0]
        avg_trade_amount = float(np.mean(trade_amounts)) if trade_amounts else 0.0
        avg_trade_amount_ratio = (avg_trade_amount / init_cap) if init_cap > 0 else 0.0
        calmar = (annual_return / dd_abs) if dd_abs > 0 else 0.0
        monthly_equity = equity_curve.resample("ME").last() if len(equity_curve) > 0 else pd.Series(dtype=float)
        monthly_ret = self._session_returns(monthly_equity, init_cap)
        monthly_stability_ratio = float((monthly_ret > 0).sum() / len(monthly_ret)) if len(monthly_ret) > 0 else 0.0
        regime_consistency_ratio = self._compute_regime_consistency_ratio(equity_curve, init_cap)
        regime_layer = self._compute_regime_layer_metrics(equity_curve, init_cap)
        capital_utilization_ratio = self._safe_float(summary_metrics.get("capital_utilization_ratio"), min(1.0, max(0.0, avg_trade_amount_ratio * 1.8))) if isinstance(summary_metrics, dict) else min(1.0, max(0.0, avg_trade_amount_ratio * 1.8))
        holding_efficiency_index = self._safe_float(summary_metrics.get("holding_efficiency_index"), annual_return * (0.5 + 0.5 * min(1.0, max(capital_utilization_ratio, 0.0) / 0.35))) if isinstance(summary_metrics, dict) else annual_return * (0.5 + 0.5 * min(1.0, max(capital_utilization_ratio, 0.0) / 0.35))

        metrics = {
            "annualized_roi": self._safe_float(summary_metrics.get("annualized_roi")) if isinstance(summary_metrics, dict) else annual_return,
            "roi": self._safe_float(summary_metrics.get("roi")) if isinstance(summary_metrics, dict) else total_return,
            "win_rate": self._safe_float(summary_metrics.get("win_rate")) if isinstance(summary_metrics, dict) else win_rate,
            "max_dd": self._safe_float(summary_metrics.get("max_dd")) if isinstance(summary_metrics, dict) else dd_abs,
            "sharpe": self._safe_float(summary_metrics.get("sharpe")) if isinstance(summary_metrics, dict) else sharpe,
            "calmar": self._safe_float(summary_metrics.get("calmar")) if isinstance(summary_metrics, dict) else calmar,
            "profit_factor": summary_metrics.get("profit_factor") if isinstance(summary_metrics, dict) and "profit_factor" in summary_metrics else profit_ratio,
            "max_win_streak": int(self._safe_float(summary_metrics.get("max_win_streak"), max_win_streak)) if isinstance(summary_metrics, dict) else max_win_streak,
            "max_loss_streak": int(self._safe_float(summary_metrics.get("max_loss_streak"), max_loss_streak)) if isinstance(summary_metrics, dict) else max_loss_streak,
            "monthly_profit_ratio": self._safe_float(summary_metrics.get("monthly_profit_ratio"), monthly_profit_ratio) if isinstance(summary_metrics, dict) else monthly_profit_ratio,
            "annualized_trades": self._safe_float(summary_metrics.get("annualized_trades"), annualized_trades) if isinstance(summary_metrics, dict) else annualized_trades,
            "avg_trade_amount_ratio": self._safe_float(summary_metrics.get("avg_trade_amount_ratio"), avg_trade_amount_ratio) if isinstance(summary_metrics, dict) else avg_trade_amount_ratio,
            "capital_utilization_ratio": self._safe_float(summary_metrics.get("capital_utilization_ratio"), capital_utilization_ratio) if isinstance(summary_metrics, dict) else capital_utilization_ratio,
            "holding_efficiency_index": self._safe_float(summary_metrics.get("holding_efficiency_index"), holding_efficiency_index) if isinstance(summary_metrics, dict) else holding_efficiency_index,
            "monthly_stability_ratio": self._safe_float(summary_metrics.get("monthly_stability_ratio"), monthly_stability_ratio) if isinstance(summary_metrics, dict) else monthly_stability_ratio,
            "regime_consistency_ratio": self._safe_float(summary_metrics.get("regime_consistency_ratio"), regime_consistency_ratio) if isinstance(summary_metrics, dict) else regime_consistency_ratio,
            "trend_state_win_ratio": self._safe_float(summary_metrics.get("trend_state_win_ratio"), regime_layer.get("trend_state_win_ratio", 0.0)) if isinstance(summary_metrics, dict) else regime_layer.get("trend_state_win_ratio", 0.0),
            "sideway_state_win_ratio": self._safe_float(summary_metrics.get("sideway_state_win_ratio"), regime_layer.get("sideway_state_win_ratio", 0.0)) if isinstance(summary_metrics, dict) else regime_layer.get("sideway_state_win_ratio", 0.0),
            "down_state_win_ratio": self._safe_float(summary_metrics.get("down_state_win_ratio"), regime_layer.get("down_state_win_ratio", 0.0)) if isinstance(summary_metrics, dict) else regime_layer.get("down_state_win_ratio", 0.0),
            "regime_layer_score": self._safe_float(summary_metrics.get("regime_layer_score"), regime_layer.get("regime_layer_score", 0.0)) if isinstance(summary_metrics, dict) else regime_layer.get("regime_layer_score", 0.0)
        }
        scorecard = self._build_scorecard(metrics)

        print("\n" + "=" * 55)
        print("               策略回测报告")
        print("=" * 55)
        print(f"策略编号：{strategy_id}")
        print(f"回测周期：{start_txt} ~ {end_txt}")
        print(f"初始资金：{init_cap:.2f} 元")
        print(f"结束资金：{end_cap:.2f} 元")
        print(f"总收益：{total_return:.2%}")
        print(f"年化收益：{annual_return:.2%}")
        print(f"最大回撤：{max_drawdown:.2%}")
        print(f"成交次数：{execution_count}  | 已平仓次数：{trade_count}")
        print(f"盈利次数：{win_num}  | 亏损次数：{loss_num}  | 平手次数：{len(breakeven_trades)}")
        print(f"胜率：{win_rate:.2%}")
        print(f"平均盈利：{avg_win:.2f}  | 平均亏损：{avg_loss:.2f}")
        print(f"盈利因子：{profit_ratio:.2f}" if profit_ratio is not None else "盈利因子：--")
        print(f"评分卡总分：{scorecard['total_score']:.1f}/100 评级：{scorecard['grade']}（{scorecard['conclusion']}）")
        print("=" * 55 + "\n")
        trade_details = []
        force_close_count = 0
        for t in transactions:
            reason = str(t.get("reason", "") or "")
            if reason.strip().upper() == "FORCE_CLOSE_END":
                force_close_count += 1
            trade_details.append({
                "dt": str(t.get("dt", "")),
                "direction": str(t.get("direction", "")),
                "price": float(t.get("price", 0.0) or 0.0),
                "quantity": int(t.get("quantity", 0) or 0),
                "amount": float(t.get("amount", 0.0) or 0.0),
                "cost": float(t.get("cost", 0.0) or 0.0),
                "pnl": float(t.get("pnl", 0.0) or 0.0),
                "reason": reason
            })

        report_notes = []
        if force_close_count > 0:
            report_notes.append(f"触发回测结束强制平仓 {force_close_count} 次（FORCE_CLOSE_END）")

        return {
            "strategy_id": strategy_id,
            "start_date": start_txt,
            "end_date": end_txt,
            "init_capital": init_cap,
            "end_capital": end_cap,
            "realized_pnl": total_pnl,
            "return_basis": return_basis,
            "annualization_basis": "calendar_days_365.25",
            "sharpe_basis": "session_returns_252",
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": float(max_drawdown),
            "trade_count": trade_count,
            "execution_count": execution_count,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "win_num": win_num,
            "loss_num": loss_num,
            "breakeven_num": len(breakeven_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_ratio": profit_ratio,
            "max_win_streak": int(max_win_streak),
            "max_loss_streak": int(max_loss_streak),
            "monthly_profit_ratio": float(monthly_profit_ratio),
            "capital_utilization_ratio": float(self._safe_float(metrics.get("capital_utilization_ratio", 0.0))),
            "holding_efficiency_index": float(self._safe_float(metrics.get("holding_efficiency_index", 0.0))),
            "trend_state_win_ratio": float(self._safe_float(metrics.get("trend_state_win_ratio", 0.0))),
            "sideway_state_win_ratio": float(self._safe_float(metrics.get("sideway_state_win_ratio", 0.0))),
            "down_state_win_ratio": float(self._safe_float(metrics.get("down_state_win_ratio", 0.0))),
            "regime_layer_score": float(self._safe_float(metrics.get("regime_layer_score", 0.0))),
            "scorecard": scorecard,
            "trade_details": trade_details,
            "force_close_count": int(force_close_count),
            "report_notes": report_notes,
            "settlement_schema_version": 1,
            "settlement": self._validated_settlement(settlement),
        }
