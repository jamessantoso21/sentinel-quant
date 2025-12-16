"""
Sentinel Quant - Price OHLCV Model
Time-series price data optimized for TimescaleDB
"""
from sqlalchemy import Column, String, Float, DateTime, Index, BigInteger
from datetime import datetime, timezone

from db.base import Base


class PriceOHLCV(Base):
    """
    OHLCV candlestick data.
    
    This table is designed to be converted to a TimescaleDB hypertable
    for optimal time-series query performance.
    
    Hypertable creation SQL:
    SELECT create_hypertable('price_ohlcv', 'timestamp');
    """
    
    __tablename__ = "price_ohlcv"
    
    # Override id to use BigInteger for high volume
    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    
    # Composite key for unique candle
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 15m, 1h, 4h, 1d
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Technical indicators (pre-calculated for faster queries)
    sma_20 = Column(Float, nullable=True)
    sma_50 = Column(Float, nullable=True)
    rsi_14 = Column(Float, nullable=True)
    atr_14 = Column(Float, nullable=True)
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('ix_price_symbol_timeframe_ts', 'symbol', 'timeframe', 'timestamp'),
    )
    
    def __repr__(self) -> str:
        return f"<PriceOHLCV({self.symbol} {self.timeframe} {self.timestamp})>"
    
    @classmethod
    def from_ccxt(cls, symbol: str, timeframe: str, ohlcv: list) -> "PriceOHLCV":
        """
        Create from CCXT OHLCV format: [timestamp, open, high, low, close, volume]
        """
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.fromtimestamp(ohlcv[0] / 1000, tz=timezone.utc),
            open=ohlcv[1],
            high=ohlcv[2],
            low=ohlcv[3],
            close=ohlcv[4],
            volume=ohlcv[5]
        )
