"""
Sentinel Quant - Monthly Rebalance Engine
==========================================
Portfolio-level equal weight rebalancing strategy.

Strategy:
- Initial: Divide capital equally across all 10 coins (10% each)
- Monthly: Rebalance if any coin deviates >5% from target weight
- Result: +3311% backtested over 4 years

Configuration:
- REBALANCE_DAY: 1 (first of month)
- THRESHOLD: 0.05 (5% deviation triggers rebalance)
- MIN_TRADE_SIZE: 10.0 (minimum $10 per trade)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# All 10 trading coins with equal weight
PORTFOLIO_COINS = [
    "SOL/USDT",
    "MATIC/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "FET/USDT",
    "VET/USDT",
    "ETC/USDT",
    "HBAR/USDT",
    "AAVE/USDT",
    "BTC/USDT",
]


@dataclass
class Allocation:
    """Current allocation for a coin"""
    symbol: str
    quantity: float
    current_price: float
    current_value: float
    target_value: float
    target_quantity: float
    deviation_percent: float
    action: str  # BUY, SELL, HOLD
    trade_value: float  # Amount to trade (positive)


@dataclass
class RebalanceResult:
    """Result of rebalance check"""
    should_rebalance: bool
    reason: str
    allocations: Dict[str, Allocation]
    total_portfolio_value: float
    cash_available: float


class MonthlyRebalanceEngine:
    """
    Portfolio-level monthly rebalancing engine.
    
    Manages 10 coins with equal weight allocation.
    Rebalances once per month when any coin deviates >5% from target.
    """
    
    def __init__(self):
        # Configuration
        self.rebalance_day = 1  # First of month
        self.deviation_threshold = 0.05  # 5%
        self.min_trade_size = 10.0  # $10 minimum
        self.fee_rate = 0.001  # 0.1% Binance fee
        
        # State
        self.holdings: Dict[str, float] = {}  # symbol -> quantity
        self.last_rebalance_date: Optional[datetime] = None
        self.is_initialized = False
        
        logger.info("Monthly Rebalance Engine initialized")
    
    def initialize_portfolio(
        self, 
        total_capital: float, 
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Initialize portfolio with equal weight allocation.
        
        Args:
            total_capital: Total USD to invest
            prices: Current prices for all coins
            
        Returns:
            Dict of symbol -> quantity bought
        """
        if len(prices) < len(PORTFOLIO_COINS):
            logger.warning(f"Only {len(prices)} prices available, need {len(PORTFOLIO_COINS)}")
        
        per_coin = total_capital / len(PORTFOLIO_COINS)
        
        for symbol in PORTFOLIO_COINS:
            if symbol in prices and prices[symbol] > 0:
                # Apply fee
                buy_value = per_coin * (1 - self.fee_rate)
                quantity = buy_value / prices[symbol]
                self.holdings[symbol] = quantity
                logger.info(f"Initial BUY {symbol}: ${per_coin:.2f} = {quantity:.6f}")
        
        self.is_initialized = True
        self.last_rebalance_date = datetime.now()
        
        return self.holdings.copy()
    
    def check_rebalance_needed(
        self,
        prices: Dict[str, float],
        cash_available: float = 0.0
    ) -> RebalanceResult:
        """
        Check if monthly rebalance is needed.
        
        Returns RebalanceResult with allocations and whether to rebalance.
        """
        now = datetime.now()
        
        # Check if it's time for monthly rebalance
        should_check = False
        reason = ""
        
        if self.last_rebalance_date is None:
            should_check = True
            reason = "First rebalance check"
        elif now.month != self.last_rebalance_date.month:
            should_check = True
            reason = f"New month: {now.strftime('%Y-%m')}"
        elif now.day >= self.rebalance_day and self.last_rebalance_date.day < self.rebalance_day:
            should_check = True
            reason = f"Rebalance day {self.rebalance_day} reached"
        
        if not should_check:
            return RebalanceResult(
                should_rebalance=False,
                reason=f"Not time yet (next: {self.rebalance_day} of next month)",
                allocations={},
                total_portfolio_value=0,
                cash_available=cash_available
            )
        
        # Calculate current allocations
        total_value = cash_available
        allocations: Dict[str, Allocation] = {}
        
        for symbol in PORTFOLIO_COINS:
            quantity = self.holdings.get(symbol, 0.0)
            price = prices.get(symbol, 0.0)
            current_value = quantity * price
            total_value += current_value
        
        # Calculate targets
        target_per_coin = total_value / len(PORTFOLIO_COINS)
        needs_rebalance = False
        
        for symbol in PORTFOLIO_COINS:
            quantity = self.holdings.get(symbol, 0.0)
            price = prices.get(symbol, 0.0)
            current_value = quantity * price if price > 0 else 0
            
            target_value = target_per_coin
            target_quantity = target_value / price if price > 0 else 0
            
            deviation = (current_value - target_value) / target_value if target_value > 0 else 0
            
            # Determine action
            if abs(deviation) > self.deviation_threshold:
                needs_rebalance = True
                if deviation > 0:
                    action = "SELL"
                    trade_value = current_value - target_value
                else:
                    action = "BUY"
                    trade_value = target_value - current_value
            else:
                action = "HOLD"
                trade_value = 0.0
            
            allocations[symbol] = Allocation(
                symbol=symbol,
                quantity=quantity,
                current_price=price,
                current_value=current_value,
                target_value=target_value,
                target_quantity=target_quantity,
                deviation_percent=deviation * 100,
                action=action,
                trade_value=trade_value
            )
        
        return RebalanceResult(
            should_rebalance=needs_rebalance,
            reason=reason if needs_rebalance else "All coins within threshold",
            allocations=allocations,
            total_portfolio_value=total_value,
            cash_available=cash_available
        )
    
    def execute_rebalance(
        self,
        result: RebalanceResult,
        execute_trade_func=None
    ) -> List[Dict]:
        """
        Execute the rebalance trades.
        
        Args:
            result: RebalanceResult from check_rebalance_needed
            execute_trade_func: Optional callback to execute real trades
            
        Returns:
            List of executed trades
        """
        if not result.should_rebalance:
            return []
        
        trades = []
        
        # First: SELL overweight coins to free up cash
        for symbol, alloc in result.allocations.items():
            if alloc.action == "SELL" and alloc.trade_value >= self.min_trade_size:
                sell_qty = alloc.trade_value / alloc.current_price
                
                if execute_trade_func:
                    execute_trade_func("SELL", symbol, sell_qty, alloc.current_price)
                
                self.holdings[symbol] -= sell_qty
                
                trades.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "quantity": sell_qty,
                    "price": alloc.current_price,
                    "value": alloc.trade_value,
                    "reason": f"Rebalance: {alloc.deviation_percent:+.1f}% deviation"
                })
                
                logger.info(f"REBALANCE SELL {symbol}: {sell_qty:.6f} @ ${alloc.current_price:.4f}")
        
        # Second: BUY underweight coins
        for symbol, alloc in result.allocations.items():
            if alloc.action == "BUY" and alloc.trade_value >= self.min_trade_size:
                buy_qty = alloc.trade_value / alloc.current_price
                
                if execute_trade_func:
                    execute_trade_func("BUY", symbol, buy_qty, alloc.current_price)
                
                self.holdings[symbol] = self.holdings.get(symbol, 0) + buy_qty
                
                trades.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": buy_qty,
                    "price": alloc.current_price,
                    "value": alloc.trade_value,
                    "reason": f"Rebalance: {alloc.deviation_percent:+.1f}% deviation"
                })
                
                logger.info(f"REBALANCE BUY {symbol}: {buy_qty:.6f} @ ${alloc.current_price:.4f}")
        
        self.last_rebalance_date = datetime.now()
        
        return trades
    
    def get_portfolio_status(self, prices: Dict[str, float]) -> Dict:
        """Get current portfolio status for API/UI"""
        total_value = 0
        coin_values = {}
        
        for symbol in PORTFOLIO_COINS:
            qty = self.holdings.get(symbol, 0)
            price = prices.get(symbol, 0)
            value = qty * price
            total_value += value
            coin_values[symbol] = {
                "quantity": qty,
                "price": price,
                "value": value,
                "weight": 0  # Will calculate after total known
            }
        
        # Calculate weights
        for symbol in coin_values:
            if total_value > 0:
                coin_values[symbol]["weight"] = coin_values[symbol]["value"] / total_value * 100
        
        return {
            "total_value": total_value,
            "coins": coin_values,
            "last_rebalance": self.last_rebalance_date.isoformat() if self.last_rebalance_date else None,
            "is_initialized": self.is_initialized,
            "next_rebalance_day": self.rebalance_day
        }


# Global instance
monthly_rebalance_engine = MonthlyRebalanceEngine()
