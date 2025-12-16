"""
Sentinel Quant - Binance Exchange Implementation
"""
from typing import Optional, List, Dict, Any
import ccxt.async_support as ccxt

from .base import BaseExchange, OrderSide, OrderType, OrderResult, Balance, Ticker
from core.config import settings


class BinanceExchange(BaseExchange):
    """Binance exchange implementation using CCXT"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None
    ):
        super().__init__(
            api_key or settings.BINANCE_API_KEY,
            api_secret or settings.BINANCE_API_SECRET
        )
        # Use testnet from settings if not explicitly provided
        self.testnet = testnet if testnet is not None else settings.BINANCE_TESTNET
        self._exchange: Optional[ccxt.binance] = None
    
    @property
    def name(self) -> str:
        return "binance"
    
    async def connect(self) -> bool:
        """Initialize Binance connection"""
        options = {
            "defaultType": "spot",  # or 'future' for futures
            "adjustForTimeDifference": True
        }
        
        if self.testnet:
            options["urls"] = {
                "api": {
                    "public": "https://testnet.binance.vision/api",
                    "private": "https://testnet.binance.vision/api"
                }
            }
        
        self._exchange = ccxt.binance({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "options": options,
            "enableRateLimit": True
        })
        
        # Test connection
        try:
            await self._exchange.load_markets()
            return True
        except Exception:
            return False
    
    async def disconnect(self) -> None:
        """Close Binance connection"""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
    
    async def get_balance(self, asset: str = "USDT") -> Balance:
        """Get balance for specific asset"""
        if not self._exchange:
            await self.connect()
        
        balance = await self._exchange.fetch_balance()
        asset_balance = balance.get(asset, {"free": 0, "used": 0, "total": 0})
        
        return Balance(
            asset=asset,
            free=float(asset_balance.get("free", 0)),
            locked=float(asset_balance.get("used", 0)),
            total=float(asset_balance.get("total", 0))
        )
    
    async def get_all_balances(self) -> List[Balance]:
        """Get all non-zero balances"""
        if not self._exchange:
            await self.connect()
        
        balance = await self._exchange.fetch_balance()
        balances = []
        
        for asset, amounts in balance.get("total", {}).items():
            if float(amounts) > 0:
                balances.append(Balance(
                    asset=asset,
                    free=float(balance.get(asset, {}).get("free", 0)),
                    locked=float(balance.get(asset, {}).get("used", 0)),
                    total=float(amounts)
                ))
        
        return balances
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price ticker"""
        if not self._exchange:
            await self.connect()
        
        ticker = await self._exchange.fetch_ticker(symbol)
        
        return Ticker(
            symbol=symbol,
            last_price=float(ticker["last"]),
            bid=float(ticker["bid"] or ticker["last"]),
            ask=float(ticker["ask"] or ticker["last"]),
            volume_24h=float(ticker["quoteVolume"] or 0),
            change_24h=float(ticker["change"] or 0),
            change_24h_percent=float(ticker["percentage"] or 0)
        )
    
    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float
    ) -> OrderResult:
        """Place a market order"""
        if not self._exchange:
            await self.connect()
        
        try:
            order = await self._exchange.create_order(
                symbol=symbol,
                type="market",
                side=side.value,
                amount=quantity
            )
            
            return OrderResult(
                success=True,
                order_id=order["id"],
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=None,
                filled_quantity=float(order.get("filled", 0)),
                average_price=float(order.get("average", 0)) if order.get("average") else None,
                status=order["status"]
            )
        except Exception as e:
            return OrderResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=None,
                filled_quantity=0,
                average_price=None,
                status="failed",
                error=str(e)
            )
    
    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float
    ) -> OrderResult:
        """Place a limit order"""
        if not self._exchange:
            await self.connect()
        
        try:
            order = await self._exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side.value,
                amount=quantity,
                price=price
            )
            
            return OrderResult(
                success=True,
                order_id=order["id"],
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                filled_quantity=float(order.get("filled", 0)),
                average_price=float(order.get("average", 0)) if order.get("average") else None,
                status=order["status"]
            )
        except Exception as e:
            return OrderResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                filled_quantity=0,
                average_price=None,
                status="failed",
                error=str(e)
            )
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        if not self._exchange:
            await self.connect()
        
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception:
            return False
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders"""
        if not self._exchange:
            await self.connect()
        
        orders = await self._exchange.fetch_open_orders(symbol)
        return orders
    
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100
    ) -> List[List[float]]:
        """Get OHLCV candlestick data"""
        if not self._exchange:
            await self.connect()
        
        ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return ohlcv
