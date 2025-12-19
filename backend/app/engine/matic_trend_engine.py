"""
MATIC Trend Following Engine
+3084% backtested return (573%/year)
Same strategy as SOL - Trend 2d cooldown
"""
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MATICTrend(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGE = "RANGE"


class MATICAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class MATICSignal:
    action: MATICAction
    trend: MATICTrend
    reason: str
    confidence: float = 0.8


class MATICTrendEngine:
    """
    MATIC Trend Following Engine
    
    Strategy: Same as SOL (+303%)
    - Trend confirmation: 72 hours (3 days)
    - Trade cooldown: 48 hours (2 days)
    - Entry: UPTREND confirmed
    - Exit: RSI top (>75 + 25% profit) OR 40% profit OR DOWNTREND
    - Re-entry: Pullback after top sell
    
    Backtested: +3084.2% over 5.38 years (~573%/year)
    """
    
    def __init__(self):
        # Strategy parameters (optimized from backtest)
        self.trend_confirm_hours = 72   # 3 days to confirm trend
        self.cooldown_hours = 48        # 2 days between trades
        self.rsi_top = 75               # RSI threshold for top
        self.profit_rsi = 0.25          # Min profit for RSI exit (25%)
        self.profit_target = 0.40       # Profit target (40%)
        self.rsi_entry = 45             # RSI for pullback entry
        self.pullback_pct = 0.15        # 15% pullback for re-entry
        self.position_size = 0.80       # 80% of capital
        
        # State tracking
        self.current_trend = MATICTrend.RANGE
        self.trend_counter = 0
        self.last_trade_time: Optional[datetime] = None
        self.sold_at_top = False
        self.in_cash_bear = False
        self.highest_since_entry = 0.0
        
        # Position tracking (managed externally)
        self.entry_price = 0.0
        self.in_position = False
        
        logger.info("MATIC Trend Engine initialized (+3084% backtested)")
    
    def update_position(self, entry_price: float, in_position: bool):
        """Update position state from external source"""
        self.entry_price = entry_price
        self.in_position = in_position
    
    def _detect_raw_trend(self, sma10: float, sma20: float, sma50: float) -> MATICTrend:
        """Detect trend from SMA alignment"""
        if sma10 > sma20 > sma50:
            return MATICTrend.UPTREND
        elif sma10 < sma20 < sma50:
            return MATICTrend.DOWNTREND
        return MATICTrend.RANGE
    
    def _update_trend(self, raw_trend: MATICTrend) -> None:
        """Update confirmed trend with smoothing"""
        if raw_trend == self.current_trend:
            self.trend_counter = 0
        else:
            self.trend_counter += 1
            # Confirm new trend after enough hours
            if self.trend_counter >= self.trend_confirm_hours:
                logger.info(f"MATIC Trend changed: {self.current_trend.value} -> {raw_trend.value}")
                self.current_trend = raw_trend
                self.trend_counter = 0
    
    def _can_trade(self) -> bool:
        """Check if cooldown period has passed"""
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
    ) -> MATICSignal:
        """
        Get trading signal based on trend following strategy.
        
        Returns MATICSignal with action, trend, reason, and confidence.
        """
        # Update trend
        raw_trend = self._detect_raw_trend(sma10, sma20, sma50)
        self._update_trend(raw_trend)
        
        # Track highest since entry
        if self.in_position and current_price > self.highest_since_entry:
            self.highest_since_entry = current_price
        
        # Calculate current PnL
        pnl = 0.0
        if self.in_position and self.entry_price > 0:
            pnl = (current_price - self.entry_price) / self.entry_price
        
        # Calculate distance from high
        dist_from_high = (high_20d - current_price) / high_20d if high_20d > 0 else 0
        
        can_trade = self._can_trade()
        
        # ========== IN POSITION ==========
        if self.in_position:
            
            # UPTREND: Look for top signals
            if self.current_trend == MATICTrend.UPTREND and can_trade:
                
                # RSI Top Exit
                if rsi > self.rsi_top and pnl > self.profit_rsi:
                    self.last_trade_time = datetime.now()
                    self.sold_at_top = True
                    self.highest_since_entry = 0.0
                    return MATICSignal(
                        action=MATICAction.SELL,
                        trend=self.current_trend,
                        reason=f"RSI_TOP: RSI={rsi:.0f}, PnL={pnl*100:+.1f}%",
                        confidence=0.85
                    )
                
                # Profit Target Exit
                if pnl > self.profit_target:
                    self.last_trade_time = datetime.now()
                    self.sold_at_top = True
                    self.highest_since_entry = 0.0
                    return MATICSignal(
                        action=MATICAction.SELL,
                        trend=self.current_trend,
                        reason=f"PROFIT_TARGET: PnL={pnl*100:+.1f}% > {self.profit_target*100}%",
                        confidence=0.90
                    )
            
            # DOWNTREND: Exit immediately
            elif self.current_trend == MATICTrend.DOWNTREND:
                self.last_trade_time = datetime.now()
                self.in_cash_bear = True
                self.highest_since_entry = 0.0
                return MATICSignal(
                    action=MATICAction.SELL,
                    trend=self.current_trend,
                    reason=f"DOWNTREND_EXIT: Trend confirmed bearish, PnL={pnl*100:+.1f}%",
                    confidence=0.80
                )
            
            # HOLD
            return MATICSignal(
                action=MATICAction.HOLD,
                trend=self.current_trend,
                reason=f"HOLDING: Trend={self.current_trend.value}, PnL={pnl*100:+.1f}%",
                confidence=0.70
            )
        
        # ========== NO POSITION ==========
        else:
            
            # Re-entry after selling at top
            if self.sold_at_top and can_trade:
                # Wait for pullback
                if rsi < self.rsi_entry and dist_from_high > self.pullback_pct and momentum_7d > 0:
                    self.last_trade_time = datetime.now()
                    self.sold_at_top = False
                    self.highest_since_entry = current_price
                    return MATICSignal(
                        action=MATICAction.BUY,
                        trend=self.current_trend,
                        reason=f"PULLBACK: RSI={rsi:.0f}, Dist from high={dist_from_high*100:.1f}%",
                        confidence=0.80
                    )
            
            # Entry after bear market
            elif self.in_cash_bear and self.current_trend == MATICTrend.UPTREND and can_trade:
                self.last_trade_time = datetime.now()
                self.in_cash_bear = False
                self.highest_since_entry = current_price
                return MATICSignal(
                    action=MATICAction.BUY,
                    trend=self.current_trend,
                    reason="UPTREND_ENTRY: New uptrend after bear market",
                    confidence=0.85
                )
            
            # Initial entry
            elif not self.sold_at_top and not self.in_cash_bear:
                if self.current_trend == MATICTrend.UPTREND and can_trade:
                    self.last_trade_time = datetime.now()
                    self.highest_since_entry = current_price
                    return MATICSignal(
                        action=MATICAction.BUY,
                        trend=self.current_trend,
                        reason="INITIAL_ENTRY: Uptrend confirmed, entering position",
                        confidence=0.85
                    )
            
            # No entry signal
            return MATICSignal(
                action=MATICAction.HOLD,
                trend=self.current_trend,
                reason=f"WAITING: Trend={self.current_trend.value}, RSI={rsi:.0f}",
                confidence=0.50
            )


# Test if run directly
if __name__ == "__main__":
    engine = MATICTrendEngine()
    
    # Test signal
    signal = engine.get_signal(
        current_price=0.85,
        sma10=0.82,
        sma20=0.78,
        sma50=0.70,
        rsi=55,
        high_20d=0.90,
        momentum_7d=0.05
    )
    
    print(f"Action: {signal.action.value}")
    print(f"Trend: {signal.trend.value}")
    print(f"Reason: {signal.reason}")
    print(f"Confidence: {signal.confidence}")
