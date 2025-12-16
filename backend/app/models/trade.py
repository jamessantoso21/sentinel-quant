"""
Sentinel Quant - Trade Model
Records of executed trades
"""
from sqlalchemy import Column, String, Float, Integer, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from db.base import Base


class TradeDirection(str, enum.Enum):
    """Trade direction"""
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, enum.Enum):
    """Trade execution status"""
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class Trade(Base):
    """
    Individual trade execution record
    """
    
    # User reference
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False, index=True)
    
    # Trade details
    symbol = Column(String(20), nullable=False, index=True)  # e.g., "BTC/USDT"
    direction = Column(SQLEnum(TradeDirection), nullable=False)
    status = Column(SQLEnum(TradeStatus), default=TradeStatus.PENDING, nullable=False)
    
    # Pricing
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=False)  # Amount in base currency
    
    # Risk management
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    risk_percent = Column(Float, nullable=True)  # Risk as % of portfolio
    
    # AI Decision factors
    ai_confidence = Column(Float, nullable=True)  # 0.0 - 1.0
    sentiment_score = Column(Float, nullable=True)  # 0 - 100
    technical_signal = Column(String(50), nullable=True)  # e.g., "PPO_BUY"
    
    # Results
    pnl_usdt = Column(Float, nullable=True)  # Profit/Loss in USDT
    pnl_percent = Column(Float, nullable=True)  # Profit/Loss as percentage
    
    # Exchange reference
    exchange = Column(String(20), nullable=False)  # binance, bybit
    exchange_order_id = Column(String(100), nullable=True)
    
    # Notes
    notes = Column(String(500), nullable=True)
    
    def __repr__(self) -> str:
        return f"<Trade(id={self.id}, symbol={self.symbol}, direction={self.direction}, status={self.status})>"
    
    @property
    def is_profitable(self) -> bool:
        """Check if trade was profitable"""
        return self.pnl_usdt is not None and self.pnl_usdt > 0
