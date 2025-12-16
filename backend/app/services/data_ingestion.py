"""
Sentinel Quant - Data Ingestion Service
Real-time price data collection from exchanges
"""
from typing import Optional, List, Dict, Any
import asyncio
import logging
from datetime import datetime, timezone

from services.exchange.binance import BinanceExchange
from models.price import PriceOHLCV
from api.v1.websocket.stream import manager as ws_manager

logger = logging.getLogger(__name__)


class DataIngestionService:
    """
    Service for ingesting real-time price data from exchanges.
    Stores in database and broadcasts via WebSocket.
    """
    
    def __init__(self):
        self.exchange = BinanceExchange()
        self.running = False
        self.symbols = ["BTC/USDT", "ETH/USDT"]  # Default symbols
        self.update_interval = 5  # seconds
    
    async def start(self, symbols: Optional[List[str]] = None):
        """Start data ingestion loop"""
        if symbols:
            self.symbols = symbols
        
        self.running = True
        logger.info(f"Starting data ingestion for {self.symbols}")
        
        await self.exchange.connect()
        
        while self.running:
            try:
                await self._fetch_and_broadcast()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Data ingestion error: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def stop(self):
        """Stop data ingestion"""
        self.running = False
        await self.exchange.disconnect()
        logger.info("Data ingestion stopped")
    
    async def _fetch_and_broadcast(self):
        """Fetch prices and broadcast to WebSocket clients"""
        for symbol in self.symbols:
            try:
                ticker = await self.exchange.get_ticker(symbol)
                
                # Broadcast to WebSocket clients
                await ws_manager.broadcast_price_update(
                    symbol=symbol,
                    price=ticker.last_price,
                    change_24h=ticker.change_24h_percent
                )
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
    
    async def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 1000
    ) -> List[PriceOHLCV]:
        """Fetch historical OHLCV data"""
        await self.exchange.connect()
        
        ohlcv_data = await self.exchange.get_ohlcv(symbol, timeframe, limit)
        
        candles = [
            PriceOHLCV.from_ccxt(symbol, timeframe, candle)
            for candle in ohlcv_data
        ]
        
        return candles
    
    async def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all tracked symbols"""
        await self.exchange.connect()
        
        prices = {}
        for symbol in self.symbols:
            try:
                ticker = await self.exchange.get_ticker(symbol)
                prices[symbol] = ticker.last_price
            except:
                pass
        
        return prices


# Global instance
data_ingestion_service = DataIngestionService()
