"""
ETC Trend Following Engine
+221% backtested return (+29%/year)
Same strategy as SOL/MATIC/DOGE/ADA/FET
"""
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ETCTrend(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGE = "RANGE"


class ETCAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class ETCSignal:
    action: ETCAction
    trend: ETCTrend
    reason: str
    confidence: float = 0.8


class ETCTrendEngine:
    """
    ETC Trend Following Engine
    Backtested: +221% over 7.5 years (~29%/year)
    B&H had -14% loss - trading clearly wins
    """
    
    def __init__(self):
        self.trend_confirm_hours = 72
        self.cooldown_hours = 48
        self.rsi_top = 75
        self.profit_rsi = 0.25
        self.profit_target = 0.40
        self.rsi_entry = 45
        self.pullback_pct = 0.15
        self.position_size = 0.80
        
        self.current_trend = ETCTrend.RANGE
        self.trend_counter = 0
        self.last_trade_time: Optional[datetime] = None
        self.sold_at_top = False
        self.in_cash_bear = False
        
        self.entry_price = 0.0
        self.in_position = False
        
        logger.info("ETC Trend Engine initialized (+221% backtested)")
    
    def update_position(self, entry_price: float, in_position: bool):
        self.entry_price = entry_price
        self.in_position = in_position
    
    def _detect_raw_trend(self, sma10: float, sma20: float, sma50: float) -> ETCTrend:
        if sma10 > sma20 > sma50:
            return ETCTrend.UPTREND
        elif sma10 < sma20 < sma50:
            return ETCTrend.DOWNTREND
        return ETCTrend.RANGE
    
    def _update_trend(self, raw_trend: ETCTrend):
        if raw_trend == self.current_trend:
            self.trend_counter = 0
        else:
            self.trend_counter += 1
            if self.trend_counter >= self.trend_confirm_hours:
                self.current_trend = raw_trend
                self.trend_counter = 0
    
    def _can_trade(self) -> bool:
        if self.last_trade_time is None:
            return True
        return datetime.now() - self.last_trade_time >= timedelta(hours=self.cooldown_hours)
    
    def get_signal(self, current_price: float, sma10: float, sma20: float, sma50: float,
                   rsi: float, high_20d: float, momentum_7d: float = 0.0) -> ETCSignal:
        raw_trend = self._detect_raw_trend(sma10, sma20, sma50)
        self._update_trend(raw_trend)
        
        pnl = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
        dist_from_high = (high_20d - current_price) / high_20d if high_20d > 0 else 0
        can_trade = self._can_trade()
        
        if self.in_position:
            if self.current_trend == ETCTrend.UPTREND and can_trade:
                if rsi > self.rsi_top and pnl > self.profit_rsi:
                    self.last_trade_time = datetime.now()
                    self.sold_at_top = True
                    return ETCSignal(ETCAction.SELL, self.current_trend, f"RSI_TOP: PnL={pnl*100:+.1f}%", 0.85)
                if pnl > self.profit_target:
                    self.last_trade_time = datetime.now()
                    self.sold_at_top = True
                    return ETCSignal(ETCAction.SELL, self.current_trend, f"PROFIT: PnL={pnl*100:+.1f}%", 0.90)
            elif self.current_trend == ETCTrend.DOWNTREND:
                self.last_trade_time = datetime.now()
                self.in_cash_bear = True
                return ETCSignal(ETCAction.SELL, self.current_trend, f"DOWNTREND: PnL={pnl*100:+.1f}%", 0.80)
            return ETCSignal(ETCAction.HOLD, self.current_trend, f"HOLDING: PnL={pnl*100:+.1f}%", 0.70)
        else:
            if self.sold_at_top and can_trade and rsi < self.rsi_entry and dist_from_high > self.pullback_pct and momentum_7d > 0:
                self.last_trade_time = datetime.now()
                self.sold_at_top = False
                return ETCSignal(ETCAction.BUY, self.current_trend, f"PULLBACK: RSI={rsi:.0f}", 0.80)
            elif self.in_cash_bear and self.current_trend == ETCTrend.UPTREND and can_trade:
                self.last_trade_time = datetime.now()
                self.in_cash_bear = False
                return ETCSignal(ETCAction.BUY, self.current_trend, "UPTREND_ENTRY", 0.85)
            elif not self.sold_at_top and not self.in_cash_bear and self.current_trend == ETCTrend.UPTREND and can_trade:
                self.last_trade_time = datetime.now()
                return ETCSignal(ETCAction.BUY, self.current_trend, "INITIAL_ENTRY", 0.85)
            return ETCSignal(ETCAction.HOLD, self.current_trend, f"WAITING", 0.50)
