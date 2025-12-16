"""
Sentinel Quant - Base Exchange Interface
Abstract base class for exchange implementations
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class OrderResult:
    """Result of an order execution"""
    success: bool
    order_id: Optional[str]
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float]
    filled_quantity: float
    average_price: Optional[float]
    status: str
    error: Optional[str] = None


@dataclass
class Balance:
    """Account balance"""
    asset: str
    free: float
    locked: float
    total: float


@dataclass
class Ticker:
    """Price ticker"""
    symbol: str
    last_price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h: float
    change_24h_percent: float


class BaseExchange(ABC):
    """
    Abstract base class for exchange implementations.
    All exchange integrations must implement these methods.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self._exchange = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange name"""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Initialize connection to exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection"""
        pass
    
    @abstractmethod
    async def get_balance(self, asset: str = "USDT") -> Balance:
        """Get balance for an asset"""
        pass
    
    @abstractmethod
    async def get_all_balances(self) -> List[Balance]:
        """Get all non-zero balances"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price ticker"""
        pass
    
    @abstractmethod
    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float
    ) -> OrderResult:
        """Place a market order"""
        pass
    
    @abstractmethod
    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float
    ) -> OrderResult:
        """Place a limit order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders"""
        pass
    
    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100
    ) -> List[List[float]]:
        """Get OHLCV candlestick data"""
        pass
    
    async def close_position(self, symbol: str, quantity: float, side: str) -> OrderResult:
        """Close a position by placing opposite order"""
        close_side = OrderSide.SELL if side == "LONG" else OrderSide.BUY
        return await self.place_market_order(symbol, close_side, quantity)
