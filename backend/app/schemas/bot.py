"""
Sentinel Quant - Bot Control Schemas
Pydantic models for bot status and control
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class BotState(str, Enum):
    """Bot operational state"""
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class BotAction(str, Enum):
    """Bot control actions"""
    START = "START"
    PAUSE = "PAUSE"
    STOP = "STOP"
    KILL_SWITCH = "KILL_SWITCH"  # Emergency: close all + stop


class BotStatus(BaseModel):
    """Current bot status"""
    state: BotState
    trading_enabled: bool
    
    # Connection status
    exchange_connected: bool
    database_connected: bool
    redis_connected: bool
    
    # Current activity
    active_positions: int
    pending_orders: int
    
    # Today's stats
    trades_today: int
    pnl_today: float
    pnl_today_percent: float
    
    # AI status
    last_signal_time: Optional[datetime]
    last_signal: Optional[str]
    current_confidence: Optional[float]
    current_sentiment: Optional[float]
    
    # Risk status
    daily_loss_limit_used_percent: float
    
    updated_at: datetime


class BotCommand(BaseModel):
    """Bot control command"""
    action: BotAction
    confirm: bool = Field(
        default=False,
        description="Must be True for KILL_SWITCH action"
    )


class TradeSignal(BaseModel):
    """AI trade signal"""
    symbol: str
    direction: str  # LONG, SHORT, or HOLD
    confidence: float = Field(..., ge=0, le=1)
    
    # AI analysis
    technical_score: float
    sentiment_score: float
    combined_score: float
    
    # Risk parameters
    suggested_stop_loss: Optional[float]
    suggested_take_profit: Optional[float]
    suggested_position_size: Optional[float]
    
    # Reasoning
    reasons: List[str]
    warnings: List[str]
    
    # Veto status
    is_vetoed: bool = False
    veto_reason: Optional[str] = None
    
    generated_at: datetime


class KillSwitchResponse(BaseModel):
    """Response from kill switch activation"""
    success: bool
    positions_closed: int
    total_value_liquidated: float
    final_balance_usdt: float
    message: str
