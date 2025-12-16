"""
Sentinel Quant - ATR-Based Position Sizer
Dynamic position sizing based on market volatility
"""
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    quantity: float
    position_value_usdt: float
    risk_usdt: float
    stop_loss_distance: float
    atr: float
    volatility_adjustment: float
    reason: str


class PositionSizer:
    """
    ATR-based dynamic position sizing.
    
    Logic:
    - High volatility (high ATR) = Smaller position
    - Low volatility (low ATR) = Larger position
    - Never exceeds max position size
    - Always respects risk per trade limit
    """
    
    def __init__(
        self,
        max_position_usdt: float = 100.0,
        risk_per_trade_percent: float = 1.0,
        atr_multiplier: float = 2.0
    ):
        self.max_position_usdt = max_position_usdt
        self.risk_per_trade_percent = risk_per_trade_percent
        self.atr_multiplier = atr_multiplier
    
    def calculate_atr(self, high_prices: List[float], low_prices: List[float], close_prices: List[float], period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(high_prices) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(high_prices)):
            tr1 = high_prices[i] - low_prices[i]
            tr2 = abs(high_prices[i] - close_prices[i-1])
            tr3 = abs(low_prices[i] - close_prices[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        # Simple moving average of True Range
        if len(true_ranges) >= period:
            atr = sum(true_ranges[-period:]) / period
            return atr
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    
    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float,
        atr: float,
        direction: str = "LONG"
    ) -> PositionSizeResult:
        """
        Calculate optimal position size based on ATR.
        
        Args:
            account_balance: Total account balance in USDT
            current_price: Current asset price
            atr: Current ATR value
            direction: LONG or SHORT
        """
        # Calculate risk amount
        risk_usdt = account_balance * (self.risk_per_trade_percent / 100)
        
        # Stop loss distance based on ATR
        stop_loss_distance = atr * self.atr_multiplier
        
        if stop_loss_distance == 0:
            logger.warning("ATR is zero, using default 2% stop loss")
            stop_loss_distance = current_price * 0.02
        
        # Calculate quantity based on risk
        # Risk = Quantity * Stop Loss Distance
        # Quantity = Risk / Stop Loss Distance
        quantity = risk_usdt / stop_loss_distance
        
        # Position value
        position_value = quantity * current_price
        
        # Apply volatility adjustment
        # Higher ATR = reduce position size further
        volatility_ratio = atr / current_price if current_price > 0 else 0
        volatility_adjustment = 1.0
        
        if volatility_ratio > 0.05:  # High volatility (>5% ATR/price ratio)
            volatility_adjustment = 0.5
            reason = "High volatility - position reduced by 50%"
        elif volatility_ratio > 0.03:  # Medium volatility
            volatility_adjustment = 0.75
            reason = "Medium volatility - position reduced by 25%"
        else:
            reason = "Normal volatility - full position"
        
        # Apply adjustment
        quantity *= volatility_adjustment
        position_value = quantity * current_price
        
        # Ensure we don't exceed max position
        if position_value > self.max_position_usdt:
            quantity = self.max_position_usdt / current_price
            position_value = self.max_position_usdt
            reason += " (capped at max position)"
        
        return PositionSizeResult(
            quantity=quantity,
            position_value_usdt=position_value,
            risk_usdt=risk_usdt * volatility_adjustment,
            stop_loss_distance=stop_loss_distance,
            atr=atr,
            volatility_adjustment=volatility_adjustment,
            reason=reason
        )
    
    def calculate_stop_loss(self, entry_price: float, atr: float, direction: str) -> float:
        """Calculate stop loss price based on ATR"""
        stop_distance = atr * self.atr_multiplier
        
        if direction == "LONG":
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price: float, atr: float, direction: str, rr_ratio: float = 2.0) -> float:
        """Calculate take profit price based on ATR and risk-reward ratio"""
        profit_distance = atr * self.atr_multiplier * rr_ratio
        
        if direction == "LONG":
            return entry_price + profit_distance
        else:  # SHORT
            return entry_price - profit_distance
