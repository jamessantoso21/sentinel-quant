"""
Sentinel Quant - Position Model
Active trading positions
"""
from sqlalchemy import Column, String, Float, Integer, ForeignKey, Boolean, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from db.base import Base


class PositionSide(str, enum.Enum):
    """Position side"""
    LONG = "LONG"
    SHORT = "SHORT"


class Position(Base):
    """
    Currently open position
    """
    
    # User reference
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False, index=True)
    
    # Position details
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(SQLEnum(PositionSide), nullable=False)
    
    # Amounts
    quantity = Column(Float, nullable=False)  # Position size in base currency
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    
    # Risk management
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    leverage = Column(Integer, default=1, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Exchange reference
    exchange = Column(String(20), nullable=False)
    
    def __repr__(self) -> str:
        return f"<Position(id={self.id}, symbol={self.symbol}, side={self.side}, qty={self.quantity})>"
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized PnL"""
        if not self.current_price:
            return 0.0
        
        price_diff = self.current_price - self.entry_price
        if self.side == PositionSide.SHORT:
            price_diff = -price_diff
        
        return price_diff * self.quantity
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized PnL as percentage"""
        if not self.entry_price or self.entry_price == 0:
            return 0.0
        
        price_diff = self.current_price - self.entry_price if self.current_price else 0
        if self.side == PositionSide.SHORT:
            price_diff = -price_diff
        
        return (price_diff / self.entry_price) * 100
    
    @property
    def position_value(self) -> float:
        """Current position value in USDT"""
        price = self.current_price or self.entry_price
        return self.quantity * price
