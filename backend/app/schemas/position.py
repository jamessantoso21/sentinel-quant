"""
Sentinel Quant - Position Schemas
Pydantic models for position API requests/responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionCreate(BaseModel):
    """Properties for creating a position"""
    symbol: str = Field(..., example="BTC/USDT")
    side: PositionSide
    quantity: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: int = Field(default=1, ge=1, le=125)


class PositionResponse(BaseModel):
    """Position response model"""
    id: int
    user_id: int
    symbol: str
    side: PositionSide
    
    quantity: float
    entry_price: float
    current_price: Optional[float]
    
    stop_loss: Optional[float]
    take_profit: Optional[float]
    leverage: int
    
    is_active: bool
    exchange: str
    
    # Calculated fields
    unrealized_pnl: float
    unrealized_pnl_percent: float
    position_value: float
    
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}


class PositionListResponse(BaseModel):
    """List of positions"""
    positions: List[PositionResponse]
    total_value: float
    total_unrealized_pnl: float


class ClosePositionRequest(BaseModel):
    """Request to close a position"""
    position_id: int
    close_percent: float = Field(default=100.0, ge=0, le=100)  # Partial close support
