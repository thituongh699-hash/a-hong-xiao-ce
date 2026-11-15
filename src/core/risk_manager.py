# src/core/risk_manager.py

from src.services.risk_audit_log import RiskAuditLog
from src.utils.constants import *
from src.utils.runtime_params import get_value

class RiskManager:
    """
    风险审核 (Risk Manager): 对每一个信号做强制风控审核，一票否决
    """
    def __init__(self, risk_audit_log: RiskAuditLog):
        self.risk_audit_log = risk_audit_log
        self.daily_losses = {} # Strategy ID -> Daily Loss
        self.consecutive_losses = {} # Strategy ID -> Consecutive Loss Count
        self._session_days = {} # Strategy ID -> Last trading day
        self.max_drawdowns = {} # Strategy ID -> Max Drawdown
        self.positions = {} # Strategy ID -> Current Positions

    def check_signal(self, signal, current_portfolio_value, current_positions, daily_pnl, current_drawdown=None):
        """
        Check if a signal passes risk control.
        Returns: (bool, reason)
        """
        max_stop_loss_pct = float(get_value("risk_control.max_stop_loss_pct", MAX_STOP_LOSS_PCT))
        max_pos_per_stock = float(get_value("risk_control.max_pos_per_stock", MAX_POS_PER_STOCK))
        max_total_pos = float(get_value("risk_control.max_total_pos", MAX_TOTAL_POS))
        max_daily_loss_pct = float(get_value("risk_control.max_daily_loss_pct", MAX_DAILY_LOSS_PCT))
        strategy_id = signal['strategy_id']
        direction = str(signal.get('direction', '') or '').strip().upper()
        if direction not in {'BUY', 'SELL'}:
            self.risk_audit_log.record_rejection(
                strategy_id,
                'R_DIRECTION_INVALID',
                f"Invalid direction: {direction}",
                signal.get('dt')
            )
            return False, f"非法交易方向: {direction or '<empty>'}"
        entry_price = signal['price']
        stop_loss_price = signal.get('stop_loss')
        portfolio_value = float(current_portfolio_value) if float(current_portfolio_value) > 0 else 1.0
        
        # Rule 1: Single trade stop loss <= 1%
        if stop_loss_price:
            sl_pct = abs(entry_price - stop_loss_price) / entry_price
            if sl_pct > max_stop_loss_pct:
                self.risk_audit_log.record_rejection(strategy_id, 'R1', f"Stop loss {sl_pct:.2%} > {max_stop_loss_pct:.2%}", signal['dt'])
                return False, f"止损幅度 {sl_pct:.2%} 超过限制 ({max_stop_loss_pct:.2%})"

        # Rule 2: Single stock max position 10%
        # Calculate position size based on signal amount/qty
        # Assuming signal['amount'] is the value or quantity * price
        # This requires context of current portfolio value.
        position_value = signal['qty'] * entry_price
        existing_stock_value = float(current_positions.get(signal['code'], {}).get('market_value', 0.0) or 0.0)
        projected_stock_ratio = (existing_stock_value + position_value) / portfolio_value
        if direction == 'BUY' and projected_stock_ratio > max_pos_per_stock:
              self.risk_audit_log.record_rejection(strategy_id, 'R2', f"Position size {projected_stock_ratio:.2%} > {max_pos_per_stock:.2%}", signal['dt'])
              return False, f"单票仓位 {projected_stock_ratio:.2%} 超过限制 ({max_pos_per_stock:.2%})"

        # Rule 3: Total position <= 50%
        current_total_pos_value = sum([p['market_value'] for p in current_positions.values()])
        if direction == 'BUY' and (current_total_pos_value + position_value) / portfolio_value > max_total_pos:
             self.risk_audit_log.record_rejection(strategy_id, 'R3', f"Total position {(current_total_pos_value + position_value)/portfolio_value:.2%} > {max_total_pos:.2%}", signal['dt'])
             return False, f"总仓位 {(current_total_pos_value + position_value)/portfolio_value:.2%} 超过上限 ({max_total_pos:.2%})"

        # Rule 4: Daily max loss 2% (Circuit Breaker)
        if direction == 'BUY' and daily_pnl / portfolio_value < -max_daily_loss_pct:
             self.risk_audit_log.record_circuit_break(strategy_id, f"Daily loss {daily_pnl/portfolio_value:.2%} < -{max_daily_loss_pct:.2%}", signal['dt'])
             return False, f"单日亏损 {daily_pnl/portfolio_value:.2%} 触发熔断 ({max_daily_loss_pct:.2%})"

        # Rule 5: Consecutive 3 losses -> Stop opening today
        # This state needs to be tracked externally or passed in.
        # Assuming we track it in strategy state or pass it here.
        if direction == 'BUY' and self.consecutive_losses.get(strategy_id, 0) >= CONSECUTIVE_LOSS_LIMIT:
             self.risk_audit_log.record_rejection(strategy_id, 'R5', f"Consecutive losses {self.consecutive_losses.get(strategy_id)} >= {CONSECUTIVE_LOSS_LIMIT}", signal['dt'])
             return False, f"连续亏损 {self.consecutive_losses.get(strategy_id)} 次，暂停开仓"

        # Rule 7: Banned stocks (ST, etc.) - Should be handled by Intent Planner, but double check here?
        # Rule 8: Limit Up/Down - Should be handled by Intent Planner.
        
        # Rule 9: 1-min single order per stock
        # This implies we shouldn't have multiple signals for the same stock in the same minute.
        
        return True, "批准执行"

    def start_session(self, strategy_id, day):
        """Begin a trading day, preserving losses accumulated within that day."""
        current_day = str(day)[:10]
        previous_day = self._session_days.get(strategy_id)
        if previous_day is not None and previous_day != current_day:
            self.consecutive_losses[strategy_id] = 0
        self._session_days[strategy_id] = current_day

    def update_loss_count(self, strategy_id, is_loss):
        if is_loss:
            self.consecutive_losses[strategy_id] = self.consecutive_losses.get(strategy_id, 0) + 1
        else:
            self.consecutive_losses[strategy_id] = 0
