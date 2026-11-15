# src/core/execution_engine.py

from src.services.account_ledger import AccountLedger
from src.services.order_matcher import OrderMatcher
from src.services.risk_audit_log import RiskAuditLog
from src.utils.constants import *
import math
import pandas as pd

class ExecutionEngine:
    """
    订单执行 (Execution Engine): 调度功能服务执行模拟成交、资金清算、持仓管理
    """
    def __init__(self, account_ledger: AccountLedger, order_matcher: OrderMatcher, risk_audit_log: RiskAuditLog):
        self.account_ledger = account_ledger
        self.order_matcher = order_matcher
        self.risk_audit_log = risk_audit_log
        self.positions = {} # Strategy ID -> {Stock Code -> Position Dict}

    def _trade_day(self, dt_value):
        x = pd.to_datetime(dt_value, errors='coerce')
        if pd.isna(x):
            return ""
        return x.strftime("%Y-%m-%d")

    def _normalize_protection_price(self, value):
        if value is None:
            return None
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        return price if math.isfinite(price) and price > 0 else None

    def _ensure_lots(self, pos):
        lots = pos.get('lots')
        if isinstance(lots, list):
            normalized = []
            for item in lots:
                if not isinstance(item, dict):
                    continue
                q = int(item.get('qty', 0) or 0)
                if q <= 0:
                    continue
                normalized.append({
                    'qty': q,
                    'buy_day': str(item.get('buy_day', '')).strip(),
                    'unit_cost': float(item.get('unit_cost', pos.get('avg_price', 0.0)) or 0.0)
                })
            pos['lots'] = normalized
            return pos['lots']
        legacy_qty = int(pos.get('qty', 0) or 0)
        if legacy_qty <= 0:
            pos['lots'] = []
            return pos['lots']
        pos['lots'] = [{
            'qty': legacy_qty,
            'buy_day': str(pos.get('last_buy_day', '')).strip(),
            'unit_cost': float(pos.get('avg_price', 0.0) or 0.0)
        }]
        return pos['lots']

    def _rebuild_position_from_lots(self, pos, mark_price):
        lots = self._ensure_lots(pos)
        total_qty = sum(int(x.get('qty', 0) or 0) for x in lots)
        if total_qty <= 0:
            pos['qty'] = 0
            pos['avg_price'] = 0.0
            pos['market_value'] = 0.0
            pos['last_buy_day'] = ''
            return
        total_cost = sum(float(x.get('unit_cost', 0.0) or 0.0) * int(x.get('qty', 0) or 0) for x in lots)
        pos['qty'] = total_qty
        pos['avg_price'] = total_cost / total_qty
        pos['market_value'] = float(mark_price) * total_qty
        buy_days = sorted([str(x.get('buy_day', '')).strip() for x in lots if str(x.get('buy_day', '')).strip()])
        pos['last_buy_day'] = buy_days[-1] if buy_days else ''

    def _sellable_qty_t1(self, pos, curr_day):
        lots = self._ensure_lots(pos)
        return sum(int(x.get('qty', 0) or 0) for x in lots if str(x.get('buy_day', '')).strip() != str(curr_day or '').strip())

    def _consume_lots_fifo(self, pos, sell_qty, curr_day):
        lots = self._ensure_lots(pos)
        need = int(sell_qty or 0)
        consumed = []
        for lot in lots:
            if need <= 0:
                break
            lot_day = str(lot.get('buy_day', '')).strip()
            if lot_day == str(curr_day or '').strip():
                continue
            can_take = min(int(lot.get('qty', 0) or 0), need)
            if can_take <= 0:
                continue
            lot['qty'] = int(lot.get('qty', 0) or 0) - can_take
            consumed.append({'qty': can_take, 'unit_cost': float(lot.get('unit_cost', 0.0) or 0.0)})
            need -= can_take
        if need > 0:
            return None
        pos['lots'] = [x for x in lots if int(x.get('qty', 0) or 0) > 0]
        return consumed
        
    def execute_order(self, strategy_id, signal, kline, account=None):
        """
        Execute an order (buy/sell).
        """
        direction = str(signal.get('direction', '') or '').strip().upper()
        code = signal['code']
        qty = int(float(signal['qty']))
        account_ledger = account if account is not None else self.account_ledger
        lot_size = 100
        if direction not in {'BUY', 'SELL'}:
            self.risk_audit_log.record_rejection(strategy_id, 'EXEC_DIR_INVALID', f"Invalid direction: {direction}", kline['dt'])
            return False
        if qty <= 0:
            self.risk_audit_log.record_rejection(strategy_id, 'EXEC_QTY_INVALID', f"Invalid qty: {qty}", kline['dt'])
            return False
        if direction == 'BUY':
            qty = (qty // lot_size) * lot_size
            if qty < lot_size:
                self.risk_audit_log.record_rejection(strategy_id, 'EXEC_LOT_BLOCK', f"BUY qty must be >= {lot_size} and lot-sized", kline['dt'])
                return False
        
        # Match simulated orders
        match_signal = dict(signal)
        match_signal['direction'] = direction
        success, fill_price = self.order_matcher.match_order(match_signal, kline)
        
        if not success:
            self.risk_audit_log.record_rejection(strategy_id, 'EXEC_FAIL', "Execution failed", kline['dt'])
            return False

        if direction == 'BUY':
            cash_available = float(account_ledger.cash)
            if cash_available <= 0:
                self.risk_audit_log.record_rejection(strategy_id, 'EXEC_NO_CASH', "No available cash", kline['dt'])
                return False
            amount_probe = fill_price * qty
            cost_probe, _, _, _ = account_ledger.calculate_cost(amount_probe, direction, fill_price, qty)
            if amount_probe + cost_probe > cash_available:
                lo, hi = 0, qty // lot_size
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    mid_qty = mid * lot_size
                    mid_amount = fill_price * mid_qty
                    mid_cost, _, _, _ = account_ledger.calculate_cost(mid_amount, direction, fill_price, mid_qty)
                    if mid_amount + mid_cost <= cash_available:
                        lo = mid
                    else:
                        hi = mid - 1
                qty = lo * lot_size
                if qty < lot_size:
                    self.risk_audit_log.record_rejection(strategy_id, 'EXEC_NO_CASH', "Insufficient cash after fee/slippage", kline['dt'])
                    return False
        amount = fill_price * qty
        cost, comm, stamp, transfer = account_ledger.calculate_cost(amount, direction, fill_price, qty)
        
        # Update Position
        if direction == 'BUY':
            if strategy_id not in self.positions:
                self.positions[strategy_id] = {}
            
            if code not in self.positions[strategy_id]:
                self.positions[strategy_id][code] = {
                    'qty': 0,
                    'avg_price': 0.0,
                    'market_value': 0.0,
                    'direction': 'BUY', # Default to Long
                    'stop_loss': None,
                    'take_profit': None,
                    'lots': []
                }
            
            pos = self.positions[strategy_id][code]
            lots = self._ensure_lots(pos)
            lots.append({
                'qty': qty,
                'buy_day': self._trade_day(kline.get('dt')),
                # Keep acquisition fees with the lot so each partial close
                # realizes its share of the full round-trip cash cost.
                'unit_cost': float(fill_price) + float(cost) / qty
            })
            self._rebuild_position_from_lots(pos, fill_price)
            # Protection prices are position-level state.  A filled add-on may
            # replace either threshold explicitly; omitted fields retain the
            # existing value, while 0/None disable that threshold.
            if 'stop_loss' in signal:
                pos['stop_loss'] = self._normalize_protection_price(signal.get('stop_loss'))
            if 'take_profit' in signal:
                pos['take_profit'] = self._normalize_protection_price(signal.get('take_profit'))
            
            account_ledger.record_transaction(
                strategy_id,
                kline['dt'],
                'BUY',
                fill_price,
                qty,
                cost,
                0.0,
                commission=comm,
                stamp_duty=stamp,
                transfer_fee=transfer,
                reason=signal.get('reason'),
                signal_dt=signal.get('signal_dt'),
                execution_mode=signal.get('execution_mode'),
                effective_stop_loss=pos.get('stop_loss'),
                effective_take_profit=pos.get('take_profit')
            )

        elif direction == 'SELL':
            if strategy_id not in self.positions or code not in self.positions[strategy_id]:
                 self.risk_audit_log.record_violation(strategy_id, 'SELL_NO_POS', f"Sell {code} without position", kline['dt'])
                 return False
            
            pos = self.positions[strategy_id][code]
            effective_stop_loss = pos.get('stop_loss')
            effective_take_profit = pos.get('take_profit')
            pos_qty = int(pos.get('qty', 0) or 0)
            if qty > pos_qty:
                 self.risk_audit_log.record_violation(strategy_id, 'SELL_OVER_QTY', f"Sell {qty} > Holding {pos_qty}", kline['dt'])
                 return False
            if qty % lot_size != 0 and qty != pos_qty:
                self.risk_audit_log.record_rejection(strategy_id, 'EXEC_LOT_BLOCK', f"SELL qty must be lot-sized or equal to full position ({pos_qty})", kline['dt'])
                return False
            curr_day = self._trade_day(kline.get('dt'))
            sellable_qty = self._sellable_qty_t1(pos, curr_day)
            if qty > sellable_qty:
                self.risk_audit_log.record_rejection(strategy_id, 'EXEC_T1_BLOCK', f"T+1 block: {code} sellable {sellable_qty} < request {qty}", kline['dt'])
                return False
            consumed = self._consume_lots_fifo(pos, qty, curr_day)
            if consumed is None:
                self.risk_audit_log.record_rejection(strategy_id, 'EXEC_T1_BLOCK', f"T+1 block: {code} insufficient sellable lots", kline['dt'])
                return False
            cost_basis = sum(float(x.get('unit_cost', 0.0) or 0.0) * int(x.get('qty', 0) or 0) for x in consumed)
            
            # Calculate Realized PnL
            pnl = (fill_price * qty) - cost_basis - cost
            self._rebuild_position_from_lots(pos, fill_price)
            if int(pos.get('qty', 0) or 0) == 0:
                del self.positions[strategy_id][code]

            account_ledger.record_transaction(
                strategy_id,
                kline['dt'],
                'SELL',
                fill_price,
                qty,
                cost,
                pnl,
                commission=comm,
                stamp_duty=stamp,
                transfer_fee=transfer,
                reason=signal.get('reason'),
                signal_dt=signal.get('signal_dt'),
                execution_mode=signal.get('execution_mode'),
                effective_stop_loss=effective_stop_loss,
                effective_take_profit=effective_take_profit
            )
            
        return True

    def update_holdings_value(self, current_prices):
        """
        Update market value of all holdings based on current prices.
        """
        total_value = 0.0
        for strategy_id, stocks in self.positions.items():
            for code, pos in stocks.items():
                if code in current_prices:
                    price = current_prices[code]
                    pos['market_value'] = pos['qty'] * price
                    total_value += pos['market_value']
        return total_value

    def update_strategy_holdings_value(self, strategy_id, current_prices):
        total_value = 0.0
        stocks = self.positions.get(strategy_id, {})
        for code, pos in stocks.items():
            if code in current_prices:
                price = current_prices[code]
                pos['market_value'] = pos['qty'] * price
                total_value += pos['market_value']
        return total_value

    def get_strategy_settlement(self, strategy_id, account, current_prices, as_of=None):
        """Return a fee-inclusive reconciliation for one strategy account."""
        prices = current_prices or {}
        stocks = self.positions.get(strategy_id, {})
        as_of_timestamp = pd.to_datetime(as_of, errors='coerce') if as_of is not None else pd.NaT
        as_of_day = "" if pd.isna(as_of_timestamp) else as_of_timestamp.strftime("%Y-%m-%d")
        as_of_value = None if pd.isna(as_of_timestamp) else as_of_timestamp.isoformat()

        holdings_value = 0.0
        unrealized_pnl = 0.0
        open_quantity = 0
        positions = []
        for code in sorted(stocks):
            pos = stocks[code]
            qty = int(pos.get('qty', 0) or 0)
            if qty <= 0:
                continue
            lots = self._ensure_lots(pos)
            cost_basis = sum(
                float(lot.get('unit_cost', 0.0) or 0.0) * int(lot.get('qty', 0) or 0)
                for lot in lots
            )
            try:
                mark_price = float(prices[code])
            except (KeyError, TypeError, ValueError):
                mark_price = float(pos.get('market_value', 0.0) or 0.0) / qty
            if not math.isfinite(mark_price):
                mark_price = 0.0
            market_value = mark_price * qty
            position_unrealized = market_value - cost_basis
            item = {
                'code': code,
                'qty': qty,
                'market_value': market_value,
                'cost_basis': cost_basis,
                'unrealized_pnl': position_unrealized,
            }
            if as_of_day:
                item['sellable_qty'] = self._sellable_qty_t1(pos, as_of_day)
            positions.append(item)
            holdings_value += market_value
            unrealized_pnl += position_unrealized
            open_quantity += qty

        strategy_transactions = [
            transaction for transaction in account.transactions
            if transaction.get('strategy_id') == strategy_id
        ]
        realized_pnl = sum(
            float(transaction.get('pnl', 0.0) or 0.0)
            for transaction in strategy_transactions
            if str(transaction.get('direction', '') or '').strip().upper() == 'SELL'
        )
        fees_paid = sum(float(transaction.get('cost', 0.0) or 0.0) for transaction in strategy_transactions)
        cash = float(account.cash)
        initial_capital = float(account.initial_capital)
        reconciliation_delta = (
            cash + holdings_value - (initial_capital + realized_pnl + unrealized_pnl)
        )
        return {
            'as_of': as_of_value,
            'state': 'open' if positions else 'flat',
            'cash': cash,
            'holdings_value': holdings_value,
            'open_position_count': len(positions),
            'open_quantity': open_quantity,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'fees_paid': fees_paid,
            'reconciliation_delta': reconciliation_delta,
            'positions': positions,
        }

    def check_stops(self, kline):
        """
        Check and trigger stop loss/take profit for all positions.
        """
        triggered_orders = []
        code = kline['code']
        
        for strategy_id, stocks in self.positions.items():
            if code in stocks:
                pos = stocks[code]
                curr_day = self._trade_day(kline.get('dt'))
                sellable_qty = self._sellable_qty_t1(pos, curr_day)
                if sellable_qty <= 0:
                    continue
                triggered, type_, price = self.order_matcher.check_stop_orders(pos, kline)
                
                if triggered:
                    # The timed matcher applies the stop/limit trigger and
                    # handles an opening gap before the trigger price.
                    order = {
                        'strategy_id': strategy_id,
                        'code': code,
                        'dt': kline['dt'], # Triggered at this time
                        'direction': 'SELL',
                        'qty': sellable_qty,
                        'price': price,
                        'trigger_price': pos.get('stop_loss') if type_ == 'STOP_LOSS' else pos.get('take_profit'),
                        'execution_mode': 'sell_stop' if type_ == 'STOP_LOSS' else 'sell_limit',
                        'reason': type_,
                        'type': 'MARKET'
                    }
                    triggered_orders.append(order)
                    self.risk_audit_log.record_circuit_break(strategy_id, f"{type_} triggered at {price}", kline['dt'])
        
        return triggered_orders
