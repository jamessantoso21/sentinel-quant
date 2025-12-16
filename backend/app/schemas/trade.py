"""
Sentinel Quant - Trade Schemas
Pydantic models for trade API requests/responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class TradeBase(BaseModel):
    """Base trade properties"""
    symbol: str = Field(..., example="BTC/USDT")
    direction: TradeDirection
    quantity: float = Field(..., gt=0)


class TradeCreate(TradeBase):
    """Properties for creating a trade"""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_percent: Optional[float] = Field(None, ge=0, le=100)
    notes: Optional[str] = None


class TradeUpdate(BaseModel):
    """Properties for updating a trade"""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: Optional[str] = None


class TradeResponse(TradeBase):
    """Trade response model"""
    id: int
    user_id: int
    status: TradeStatus
    
    entry_price: Optional[float]
    exit_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    
    ai_confidence: Optional[float]
    sentiment_score: Optional[float]
    technical_signal: Optional[str]
    
    pnl_usdt: Optional[float]
    pnl_percent: Optional[float]
    
    exchange: str
    exchange_order_id: Optional[str]
    
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}


class TradeListResponse(BaseModel):
    """Paginated trade list"""
    trades: List[TradeResponse]
    total: int
    page: int
    page_size: int


class TradeSummary(BaseModel):
    """Trade summary statistics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl_usdt: float
    average_pnl_percent: float
    best_trade_pnl: float
    worst_trade_pnl: float
