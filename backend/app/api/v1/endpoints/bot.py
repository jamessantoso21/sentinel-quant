"""
Sentinel Quant - Bot Control Endpoints
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

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
    "current_sentiment": None,
    "voting_result": None,  # Voting details for UI
    "activity_log": []  # Store recent activity
}


def add_activity(action: str, details: str, traded: bool = False):
    """Add activity to log"""
    from datetime import datetime, timezone
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "details": details,
        "traded": traded
    }
    bot_state["activity_log"].insert(0, log_entry)
    # Keep only last 50 entries
    bot_state["activity_log"] = bot_state["activity_log"][:50]


@router.get("/activity")
async def get_bot_activity(current_user: CurrentUser, limit: int = 20):
    """Get recent bot activity/decisions"""
    return {
        "state": bot_state["state"],
        "last_signal_time": bot_state.get("last_signal_time"),
        "last_signal": bot_state.get("last_signal"),
        "current_confidence": bot_state.get("current_confidence"),
        "current_sentiment": bot_state.get("current_sentiment"),
        "voting_result": bot_state.get("voting_result"),
        "activity_log": bot_state["activity_log"][:limit]
    }


@router.get("/voting")
async def get_voting_details(current_user: CurrentUser):
    """Get current voting details from all voters"""
    voting = bot_state.get("voting_result")
    if not voting:
        return {
            "status": "no_data",
            "message": "No voting data yet. Wait for next trading cycle."
        }
    return voting



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


@router.post("/trigger-cycle")
async def trigger_trading_cycle(current_user: TradingUser):
    """Manually trigger one trading cycle for testing"""
    try:
        from engine.trading_engine import trading_engine
        
        # Run one cycle
        await trading_engine._trading_cycle()
        
        return {
            "message": "Trading cycle completed",
            "activity_log": bot_state["activity_log"][:5]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trading cycle failed: {str(e)}")


# Background task for trading loop
_background_task = None

async def run_trading_loop():
    """Background trading loop"""
    import asyncio
    from engine.trading_engine import trading_engine
    
    while bot_state["state"].value == "RUNNING":
        try:
            await trading_engine._trading_cycle()
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
        
        await asyncio.sleep(300)  # 5 minutes


def start_background_trading():
    """Start background trading task"""
    global _background_task
    import asyncio
    
    if _background_task is None or _background_task.done():
        _background_task = asyncio.create_task(run_trading_loop())
        return True
    return False
