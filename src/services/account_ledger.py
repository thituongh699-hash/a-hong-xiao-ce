# src/services/account_ledger.py
import pandas as pd
from src.utils.constants import *
from src.utils.runtime_params import get_value


_UNSET = object()

class AccountLedger:
    """
    账户核算 (Revenue): 逐笔核算资金、净值、手续费、印花税、总成本
    """
    def __init__(self, initial_capital, nav_frequency='bar'):
        nav_frequency = str(nav_frequency or '').strip().lower()
        if nav_frequency not in {'bar', 'daily'}:
            raise ValueError("nav_frequency must be 'bar' or 'daily'")
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.nav_frequency = nav_frequency
        self.transactions = []
        self.daily_nav = [] # Date, NAV, Cash, Holdings Value
        self.total_commission = 0.0
        self.total_stamp_duty = 0.0
        self.total_transfer_fee = 0.0

    def calculate_cost(self, amount, direction, price, quantity):
        """
        Calculate transaction cost.
        Returns: (total_cost, commission, stamp_duty, transfer_fee)
        """
        # Commission: Max(5, amount * 0.00025)
        min_commission = float(get_value("trading_cost.min_commission", MIN_COMMISSION))
        commission_rate = float(get_value("trading_cost.commission_rate", COMMISSION_RATE))
        stamp_duty_rate = float(get_value("trading_cost.stamp_duty", STAMP_DUTY))
        transfer_fee_rate = float(get_value("trading_cost.transfer_fee", TRANSFER_FEE))
        commission = max(min_commission, amount * commission_rate)
        
        # Stamp Duty: 0.1% on SELL only
        stamp_duty = 0.0
        if direction == 'SELL':
            stamp_duty = amount * stamp_duty_rate
            
        # Transfer Fee: amount * 0.00001
        transfer_fee = amount * transfer_fee_rate
        
        total_cost = commission + stamp_duty + transfer_fee
        return total_cost, commission, stamp_duty, transfer_fee

    def record_transaction(
        self,
        strategy_id,
        dt,
        direction,
        price,
        quantity,
        cost,
        pnl=0.0,
        commission=0.0,
        stamp_duty=0.0,
        transfer_fee=0.0,
        reason=_UNSET,
        signal_dt=_UNSET,
        execution_mode=_UNSET,
        effective_stop_loss=_UNSET,
        effective_take_profit=_UNSET,
    ):
        """
        Record a transaction.
        """
        amount = price * quantity
        
        transaction = {
            'strategy_id': strategy_id,
            'dt': dt,
            'direction': direction,
            'price': price,
            'quantity': quantity,
            'amount': amount,
            'cost': cost,
            'pnl': pnl,
            'commission': float(commission or 0.0),
            'stamp_duty': float(stamp_duty or 0.0),
            'transfer_fee': float(transfer_fee or 0.0)
        }
        optional_fields = {
            'reason': reason,
            'signal_dt': signal_dt,
            'execution_mode': execution_mode,
            'effective_stop_loss': effective_stop_loss,
            'effective_take_profit': effective_take_profit,
        }
        transaction.update({key: value for key, value in optional_fields.items() if value is not _UNSET})
        self.transactions.append(transaction)
        
        if direction == 'BUY':
            self.cash -= (amount + cost)
        elif direction == 'SELL':
            self.cash += (amount - cost) # Proceeds - Cost
            
        self.total_commission += float(commission or 0.0)
        self.total_stamp_duty += float(stamp_duty or 0.0)
        self.total_transfer_fee += float(transfer_fee or 0.0)
        
    def update_daily_nav(self, dt, holdings_value):
        try:
            timestamp = pd.Timestamp(dt)
        except (TypeError, ValueError, OverflowError):
            timestamp = pd.NaT
        if pd.isna(timestamp):
            raise ValueError(f"Invalid NAV timestamp: {dt!r}")
        if self.daily_nav:
            previous = pd.Timestamp(self.daily_nav[-1]['dt'])
            if pd.isna(previous):
                raise ValueError("NAV timestamps must be monotonic")
            try:
                timestamps_reversed = timestamp < previous
            except TypeError as exc:
                raise ValueError("NAV timestamps must use compatible timezones") from exc
            if timestamps_reversed:
                raise ValueError("NAV timestamps must be monotonic")
        nav = self.cash + holdings_value
        record = {
            'dt': dt,
            'nav': nav,
            'cash': self.cash,
            'holdings': holdings_value
        }
        if self.nav_frequency == 'daily' and self.daily_nav and timestamp.date() == previous.date():
            self.daily_nav[-1] = record
        else:
            self.daily_nav.append(record)
        return nav

    def get_nav_history(self):
        return pd.DataFrame(self.daily_nav)
