# src/services/order_matcher.py

import math

from src.utils.constants import *
from src.utils.runtime_params import get_value

class OrderMatcher:
    """
    交易信号 (War): 模拟下单、撮合、止盈止损触发
    """
    def __init__(self):
        pass

    def _to_float(self, value, default=0.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    def _is_true(self, value):
        if isinstance(value, bool):
            return value
        txt = str(value or "").strip().lower()
        return txt in {"1", "true", "yes", "y", "t"}

    def _is_limit_up(self, kline):
        if self._is_true(kline.get("is_limit_up", False)):
            return True
        status = str(kline.get("limit_status", "")).strip().lower()
        if status in {"up", "u", "limit_up", "涨停"}:
            return True
        close = self._to_float(kline.get("close", 0.0))
        up_limit = kline.get("up_limit", None)
        if up_limit is not None:
            upl = self._to_float(up_limit, 0.0)
            if upl > 0 and close >= upl * 0.9999:
                return True
        high = self._to_float(kline.get("high", close))
        low = self._to_float(kline.get("low", close))
        open_p = self._to_float(kline.get("open", close))
        if close > 0 and high > 0 and abs(high - low) <= max(1e-6, close * 1e-6) and abs(close - high) <= max(1e-6, close * 1e-6) and close >= open_p:
            return True
        return False

    def _is_limit_down(self, kline):
        if self._is_true(kline.get("is_limit_down", False)):
            return True
        status = str(kline.get("limit_status", "")).strip().lower()
        if status in {"down", "d", "limit_down", "跌停"}:
            return True
        close = self._to_float(kline.get("close", 0.0))
        down_limit = kline.get("down_limit", None)
        if down_limit is not None:
            downl = self._to_float(down_limit, 0.0)
            if downl > 0 and close <= downl * 1.0001:
                return True
        high = self._to_float(kline.get("high", close))
        low = self._to_float(kline.get("low", close))
        open_p = self._to_float(kline.get("open", close))
        if close > 0 and high > 0 and abs(high - low) <= max(1e-6, close * 1e-6) and abs(close - low) <= max(1e-6, close * 1e-6) and close <= open_p:
            return True
        return False

    def _is_suspended_or_invalid(self, kline):
        close = self._to_float(kline.get("close", 0.0))
        high = self._to_float(kline.get("high", 0.0))
        low = self._to_float(kline.get("low", 0.0))
        volume = self._to_float(kline.get("volume", kline.get("vol", 0.0)))
        if close <= 0 or high < low or volume <= 0:
            return True
        if self._is_true(kline.get("is_suspended", False)):
            return True
        return False

    def match_order(self, order, kline):
        """
        Simulate order matching against a K-line.
        Returns: (success, fill_price, cost)
        """
        # Explicit timing is opt-in. Legacy callers keep their existing behavior.
        mode = str(order.get('execution_mode', '') or '').strip().lower()
        if mode:
            return self._match_timed_order(order, kline, mode)
        # Assume market order at open of next bar (as per rules: "Entry: Next 1-min open")
        # Or limit order if price within high/low.
        
        # For simplicity, we assume market orders execute at Open price of the current bar (which is the "next" bar relative to signal generation).
        # Slippage is applied.
        
        direction = str(order.get('direction', '')).upper()
        if direction not in {'BUY', 'SELL'}:
            return False, 0.0
        if self._is_suspended_or_invalid(kline):
            return False, 0.0
        if direction == 'BUY' and self._is_limit_up(kline):
            return False, 0.0
        if direction == 'SELL' and self._is_limit_down(kline):
            return False, 0.0
        price = self._to_float(kline.get('open', 0.0))
        if price <= 0:
            return False, 0.0
        slippage = float(get_value("execution.slippage", SLIPPAGE))
        
        # Apply slippage
        if direction == 'BUY':
            fill_price = price * (1 + slippage)
        else:
            fill_price = price * (1 - slippage)
            
        # Check if price is within K-line range (it is, since we use Open)
        # But if it's a stop loss/take profit trigger, we check against High/Low.
        
        return True, fill_price

    def _match_timed_order(self, order, kline, mode):
        """Match a preplanned order without using end-of-day locks at the open.

        The caller must supply verified/declared exchange-price boundaries.
        A close-locked day can still contain an earlier buy below the upper limit.
        This interface does not infer intraday queue priority from a daily bar.
        """
        direction = str(order.get('direction', '')).upper()
        if direction not in {'BUY', 'SELL'} or self._is_suspended_or_invalid(kline):
            return False, 0.0
        o, h, l, c = [self._to_float(kline.get(key)) for key in ('open', 'high', 'low', 'close')]
        if not all(math.isfinite(x) and x > 0 for x in (o, h, l, c)) or not l <= min(o, c) <= max(o, c) <= h:
            return False, 0.0
        upper = self._to_float(kline.get('up_limit'))
        lower = self._to_float(kline.get('down_limit'))
        if mode == 'at_close':
            up_locked = c >= upper - 1e-8 if upper > 0 else self._is_limit_up(kline)
            down_locked = c <= lower + 1e-8 if lower > 0 else self._is_limit_down(kline)
            if direction == 'BUY' and up_locked:
                return False, 0.0
            if direction == 'SELL' and down_locked:
                return False, 0.0
            price = c
        elif mode == 'at_open':
            price = o
        elif mode == 'buy_stop' and direction == 'BUY':
            trigger = self._to_float(order.get('trigger_price'))
            if trigger <= 0 or h < trigger:
                return False, 0.0
            price = max(o, trigger)
        elif mode == 'sell_stop' and direction == 'SELL':
            trigger = self._to_float(order.get('trigger_price'))
            if trigger <= 0 or l > trigger:
                return False, 0.0
            price = min(o, trigger)
        elif mode == 'sell_limit' and direction == 'SELL':
            trigger = self._to_float(order.get('trigger_price'))
            if trigger <= 0 or h < trigger:
                return False, 0.0
            price = max(o, trigger)
        else:
            return False, 0.0
        # Opening at a price limit is conservatively unfillable, even if the
        # completed daily bar later opened its queue. No intraday path is known.
        if mode in {'at_open', 'buy_stop', 'sell_stop', 'sell_limit'}:
            if direction == 'BUY' and upper > 0 and o >= upper - 1e-8:
                return False, 0.0
            if direction == 'SELL' and lower > 0 and o <= lower + 1e-8:
                return False, 0.0
        if direction == 'BUY' and upper > 0 and price >= upper - 1e-8:
            return False, 0.0
        if direction == 'SELL' and lower > 0 and price <= lower + 1e-8:
            return False, 0.0
        # With no boundaries, refuse a truly one-price bar. A non-limit flat bar
        # is allowed when its supplied exchange limits establish it is tradable.
        if h == l and upper <= 0 and lower <= 0:
            return False, 0.0
        slippage = self._to_float(get_value('execution.slippage', SLIPPAGE))
        if slippage < 0 or not math.isfinite(slippage):
            return False, 0.0
        fill = price * (1 + slippage if direction == 'BUY' else 1 - slippage)
        if fill <= 0 or (upper > 0 and fill > upper + 1e-8) or (lower > 0 and fill < lower - 1e-8):
            return False, 0.0
        # Slippage is an adverse execution-cost assumption; daily OHLC cannot
        # prove the exact tick fill or queue priority at the modeled price.
        return True, fill

    def check_stop_orders(self, position, kline):
        """
        Check if stop loss or take profit is triggered.
        Returns: (triggered, type, price)
        """
        if not position:
            return False, None, 0.0

        # Long position logic
        if position['direction'] == 'BUY':
            stop_loss = position.get('stop_loss', None)
            take_profit = position.get('take_profit', None)
            # Stop Loss
            if stop_loss is not None and kline['low'] <= stop_loss:
                # Triggered at stop price or open if gap down, but rule says "Triggered immediately"
                # Conservative: min(open, stop_loss) if gap? 
                # Rule: "Triggered at high/low" -> We use the stop price itself if within range, else Open.
                fill_price = stop_loss
                # If Open is already below stop loss (gap down), use Open.
                if kline['open'] < stop_loss:
                    fill_price = kline['open']
                return True, 'STOP_LOSS', fill_price
            
            # Take Profit
            if take_profit is not None and kline['high'] >= take_profit:
                 fill_price = take_profit
                 if kline['open'] > take_profit:
                     fill_price = kline['open']
                 return True, 'TAKE_PROFIT', fill_price

        return False, None, 0.0
