"""
Sentinel Quant - Price Repository
Optimized for TimescaleDB time-series queries
"""
from typing import Optional, List
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseRepository
from models.price import PriceOHLCV


class PriceRepository(BaseRepository[PriceOHLCV]):
    """Repository for PriceOHLCV model operations"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(PriceOHLCV, session)
    
    async def get_latest_price(self, symbol: str) -> Optional[PriceOHLCV]:
        """Get the most recent price for a symbol"""
        result = await self.session.execute(
            select(PriceOHLCV)
            .where(PriceOHLCV.symbol == symbol)
            .order_by(PriceOHLCV.timestamp.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[PriceOHLCV]:
        """Get OHLCV candles for a symbol and timeframe"""
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        
        result = await self.session.execute(
            select(PriceOHLCV)
            .where(PriceOHLCV.symbol == symbol)
            .where(PriceOHLCV.timeframe == timeframe)
            .where(PriceOHLCV.timestamp >= start_time)
            .where(PriceOHLCV.timestamp <= end_time)
            .order_by(PriceOHLCV.timestamp.asc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_recent_candles(
        self,
        symbol: str,
        timeframe: str = "1h",
        count: int = 100
    ) -> List[PriceOHLCV]:
        """Get most recent N candles"""
        result = await self.session.execute(
            select(PriceOHLCV)
            .where(PriceOHLCV.symbol == symbol)
            .where(PriceOHLCV.timeframe == timeframe)
            .order_by(PriceOHLCV.timestamp.desc())
            .limit(count)
        )
        # Return in chronological order
        candles = list(result.scalars().all())
        return candles[::-1]
    
    async def bulk_insert(self, candles: List[PriceOHLCV]) -> int:
        """Bulk insert candles (for data ingestion)"""
        self.session.add_all(candles)
        await self.session.flush()
        return len(candles)
    
    async def get_price_change_24h(self, symbol: str) -> dict:
        """Calculate 24h price change"""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(hours=24)
        
        # Current price
        current = await self.get_latest_price(symbol)
        
        # Price 24h ago
        old_result = await self.session.execute(
            select(PriceOHLCV)
            .where(PriceOHLCV.symbol == symbol)
            .where(PriceOHLCV.timestamp <= yesterday)
            .order_by(PriceOHLCV.timestamp.desc())
            .limit(1)
        )
        old = old_result.scalar_one_or_none()
        
        if not current or not old:
            return {"price": 0, "change": 0, "change_percent": 0}
        
        change = current.close - old.close
        change_percent = (change / old.close) * 100 if old.close else 0
        
        return {
            "price": current.close,
            "change": change,
            "change_percent": change_percent
        }
    
    async def initialize_hypertable(self) -> None:
        """
        Convert price_ohlcv to TimescaleDB hypertable.
        Call this once during database initialization.
        """
        await self.session.execute(
            text("""
                SELECT create_hypertable(
                    'price_ohlcv', 
                    'timestamp',
                    if_not_exists => TRUE
                );
            """)
        )
