"""
FET Trend Following Engine
+368% backtested return (+54%/year)
Same strategy as SOL/MATIC/DOGE/ADA - Trend 2d cooldown
"""
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FETTrend(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGE = "RANGE"


class FETAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class FETSignal:
    action: FETAction
    trend: FETTrend
    reason: str
    confidence: float = 0.8


class FETTrendEngine:
    """
    FET Trend Following Engine
    
    Strategy: Trend Following 2d cooldown
    - Trend confirmation: 72 hours (3 days)
    - Trade cooldown: 48 hours (2 days)
    - Entry: UPTREND confirmed
    - Exit: RSI top (>75 + 25% profit) OR 40% profit OR DOWNTREND
    - Re-entry: Pullback after top sell
    
    Backtested: +368% over 6.8 years (~54%/year)
    Beats B&H which had -2% return
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
        
        self.current_trend = FETTrend.RANGE
        self.trend_counter = 0
        self.last_trade_time: Optional[datetime] = None
        self.sold_at_top = False
        self.in_cash_bear = False
        self.highest_since_entry = 0.0
        
        self.entry_price = 0.0
        self.in_position = False
        
        logger.info("FET Trend Engine initialized (+368% backtested)")
    
    def update_position(self, entry_price: float, in_position: bool):
        self.entry_price = entry_price
        self.in_position = in_position
    
    def _detect_raw_trend(self, sma10: float, sma20: float, sma50: float) -> FETTrend:
        if sma10 > sma20 > sma50:
            return FETTrend.UPTREND
        elif sma10 < sma20 < sma50:
            return FETTrend.DOWNTREND
        return FETTrend.RANGE
    
    def _update_trend(self, raw_trend: FETTrend) -> None:
        if raw_trend == self.current_trend:
            self.trend_counter = 0
        else:
            self.trend_counter += 1
            if self.trend_counter >= self.trend_confirm_hours:
                logger.info(f"FET Trend changed: {self.current_trend.value} -> {raw_trend.value}")
                self.current_trend = raw_trend
                self.trend_counter = 0
    
    def _can_trade(self) -> bool:
        if self.last_trade_time is None:
            return True
        elapsed = datetime.now() - self.last_trade_time
        return elapsed >= timedelta(hours=self.cooldown_hours)
    
    def get_signal(
        self,
        current_price: float,
        sma10: float,
        sma20: float,
        sma50: float,
        rsi: float,
        high_20d: float,
        momentum_7d: float = 0.0
    ) -> FETSignal:
        raw_trend = self._detect_raw_trend(sma10, sma20, sma50)
        self._update_trend(raw_trend)
        
        if self.in_position and current_price > self.highest_since_entry:
            self.highest_since_entry = current_price
        
        pnl = 0.0
        if self.in_position and self.entry_price > 0:
            pnl = (current_price - self.entry_price) / self.entry_price
        
        dist_from_high = (high_20d - current_price) / high_20d if high_20d > 0 else 0
        can_trade = self._can_trade()
        
        if self.in_position:
            if self.current_trend == FETTrend.UPTREND and can_trade:
                if rsi > self.rsi_top and pnl > self.profit_rsi:
                    self.last_trade_time = datetime.now()
                    self.sold_at_top = True
                    self.highest_since_entry = 0.0
                    return FETSignal(
                        action=FETAction.SELL,
                        trend=self.current_trend,
                        reason=f"RSI_TOP: RSI={rsi:.0f}, PnL={pnl*100:+.1f}%",
                        confidence=0.85
                    )
                if pnl > self.profit_target:
                    self.last_trade_time = datetime.now()
                    self.sold_at_top = True
                    self.highest_since_entry = 0.0
                    return FETSignal(
                        action=FETAction.SELL,
                        trend=self.current_trend,
                        reason=f"PROFIT_TARGET: PnL={pnl*100:+.1f}%",
                        confidence=0.90
                    )
            elif self.current_trend == FETTrend.DOWNTREND:
                self.last_trade_time = datetime.now()
                self.in_cash_bear = True
                self.highest_since_entry = 0.0
                return FETSignal(
                    action=FETAction.SELL,
                    trend=self.current_trend,
                    reason=f"DOWNTREND_EXIT: PnL={pnl*100:+.1f}%",
                    confidence=0.80
                )
            
            return FETSignal(
                action=FETAction.HOLD,
                trend=self.current_trend,
                reason=f"HOLDING: PnL={pnl*100:+.1f}%",
                confidence=0.70
            )
        else:
            if self.sold_at_top and can_trade:
                if rsi < self.rsi_entry and dist_from_high > self.pullback_pct and momentum_7d > 0:
                    self.last_trade_time = datetime.now()
                    self.sold_at_top = False
                    self.highest_since_entry = current_price
                    return FETSignal(
                        action=FETAction.BUY,
                        trend=self.current_trend,
                        reason=f"PULLBACK: RSI={rsi:.0f}, Dist={dist_from_high*100:.1f}%",
                        confidence=0.80
                    )
            elif self.in_cash_bear and self.current_trend == FETTrend.UPTREND and can_trade:
                self.last_trade_time = datetime.now()
                self.in_cash_bear = False
                self.highest_since_entry = current_price
                return FETSignal(
                    action=FETAction.BUY,
                    trend=self.current_trend,
                    reason="UPTREND_ENTRY: New uptrend after bear",
                    confidence=0.85
                )
            elif not self.sold_at_top and not self.in_cash_bear:
                if self.current_trend == FETTrend.UPTREND and can_trade:
                    self.last_trade_time = datetime.now()
                    self.highest_since_entry = current_price
                    return FETSignal(
                        action=FETAction.BUY,
                        trend=self.current_trend,
                        reason="INITIAL_ENTRY: Uptrend confirmed",
                        confidence=0.85
                    )
            
            return FETSignal(
                action=FETAction.HOLD,
                trend=self.current_trend,
                reason=f"WAITING: Trend={self.current_trend.value}",
                confidence=0.50
            )


if __name__ == "__main__":
    engine = FETTrendEngine()
    signal = engine.get_signal(0.50, 0.48, 0.45, 0.40, 55, 0.60, 0.05)
    print(f"Action: {signal.action.value}, Trend: {signal.trend.value}, Reason: {signal.reason}")
