"""
Sentinel Quant - Bot Control Endpoints
"""
from fastapi import APIRouter
from datetime import datetime, timezone

from api.deps import DbSession, CurrentUser, TradingUser
from db.repositories.position import PositionRepository
from db.repositories.trade import TradeRepository
from schemas.bot import (
    BotStatus, BotCommand, BotAction, BotState, 
    KillSwitchResponse
)
from core.config import settings
from core.exceptions import raise_bad_request
import redis.asyncio as redis

router = APIRouter()

# In-memory bot state (would be stored in Redis in production)
bot_state = {
    "state": BotState.STOPPED,
    "last_signal_time": None,
    "last_signal": None,
    "current_confidence": None,
    "current_sentiment": None
}


@router.get("/status", response_model=BotStatus)
async def get_bot_status(current_user: CurrentUser, db: DbSession):
    """Get current bot status"""
    position_repo = PositionRepository(db)
    trade_repo = TradeRepository(db)
    
    # Get active positions count
    positions = await position_repo.get_active_positions(current_user.id)
    
    # Get today's stats
    today_trades = await trade_repo.get_today_trades(current_user.id)
    today_pnl = await trade_repo.get_today_pnl(current_user.id)
    
    # Calculate daily loss usage
    # Assuming starting balance of max_position_size for simplicity
    starting_balance = current_user.max_position_size_usdt * 10  # Rough estimate
    max_daily_loss = starting_balance * (current_user.max_daily_loss_percent / 100)
    daily_loss_used = abs(min(today_pnl, 0)) / max_daily_loss * 100 if max_daily_loss > 0 else 0
    
    # Check connections
    redis_connected = True
    try:
        r = redis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.close()
    except:
        redis_connected = False
    
    return BotStatus(
        state=bot_state["state"],
        trading_enabled=current_user.trading_enabled and settings.TRADING_ENABLED,
        exchange_connected=True,  # Placeholder
        database_connected=True,
        redis_connected=redis_connected,
        active_positions=len(positions),
        pending_orders=0,  # TODO: Get from exchange
        trades_today=len(today_trades),
        pnl_today=today_pnl,
        pnl_today_percent=(today_pnl / starting_balance * 100) if starting_balance else 0,
        last_signal_time=bot_state.get("last_signal_time"),
        last_signal=bot_state.get("last_signal"),
        current_confidence=bot_state.get("current_confidence"),
        current_sentiment=bot_state.get("current_sentiment"),
        daily_loss_limit_used_percent=daily_loss_used,
        updated_at=datetime.now(timezone.utc)
    )


@router.post("/command")
async def send_bot_command(
    command: BotCommand,
    current_user: TradingUser,
    db: DbSession
):
    """Send command to bot (start, stop, pause, kill switch)"""
    global bot_state
    
    if command.action == BotAction.START:
        bot_state["state"] = BotState.RUNNING
        return {"message": "Bot started", "state": BotState.RUNNING}
    
    elif command.action == BotAction.PAUSE:
        bot_state["state"] = BotState.PAUSED
        return {"message": "Bot paused", "state": BotState.PAUSED}
    
    elif command.action == BotAction.STOP:
        bot_state["state"] = BotState.STOPPED
        return {"message": "Bot stopped", "state": BotState.STOPPED}
    
    elif command.action == BotAction.KILL_SWITCH:
        if not command.confirm:
            raise_bad_request("Kill switch requires confirmation. Set 'confirm' to true.")
        
        # Execute kill switch
        return await execute_kill_switch(current_user, db)


@router.post("/kill-switch", response_model=KillSwitchResponse)
async def emergency_kill_switch(
    confirm: bool,
    current_user: TradingUser,
    db: DbSession
):
    """Emergency kill switch - close all positions and stop bot"""
    if not confirm:
        raise_bad_request("Kill switch requires confirmation")
    
    return await execute_kill_switch(current_user, db)


async def execute_kill_switch(current_user, db: DbSession) -> KillSwitchResponse:
    """Execute the kill switch logic"""
    global bot_state
    
    position_repo = PositionRepository(db)
    
    # Get all active positions
    positions = await position_repo.get_active_positions(current_user.id)
    total_value = sum(p.position_value for p in positions)
    
    # Close all positions
    closed_count = await position_repo.close_all_positions(current_user.id)
    
    # Stop the bot
    bot_state["state"] = BotState.STOPPED
    
    # TODO: Send actual close orders to exchange
    # TODO: Calculate actual final balance
    
    return KillSwitchResponse(
        success=True,
        positions_closed=closed_count,
        total_value_liquidated=total_value,
        final_balance_usdt=total_value,  # Simplified
        message=f"Kill switch activated. Closed {closed_count} positions. Bot stopped."
    )


@router.get("/signal")
async def get_current_signal(current_user: CurrentUser):
    """Get current AI trading signal"""
    # This would come from the AI engine in production
    from schemas.bot import TradeSignal
    
    # Placeholder - in production this comes from the decision engine
    return {
        "message": "No active signal",
        "last_signal_time": bot_state.get("last_signal_time"),
        "last_signal": bot_state.get("last_signal")
    }
