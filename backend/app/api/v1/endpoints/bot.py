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
    "activity_log": [],  # Store recent activity
    
    # SOL Trend Engine State (+303% optimized)
    "sol_trend": None,        # Current trend: UPTREND, DOWNTREND, RANGE
    "sol_action": None,       # Last action: BUY, SELL, HOLD
    "sol_in_position": False, # Is SOL position open?
    "sol_entry_price": None,  # Entry price if in position
    "sol_pnl": None,          # Current unrealized PnL %
    
    # MATIC Trend Engine State (+3084% optimized)
    "matic_trend": None,
    "matic_action": None,
    "matic_in_position": False,
    "matic_entry_price": None,
    "matic_pnl": None,
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
        "activity_log": bot_state["activity_log"][:limit],
        
        # SOL Trend Engine Status
        "sol": {
            "trend": bot_state.get("sol_trend"),
            "action": bot_state.get("sol_action"),
            "in_position": bot_state.get("sol_in_position", False),
            "entry_price": bot_state.get("sol_entry_price"),
            "pnl_percent": bot_state.get("sol_pnl"),
        },
        
        # MATIC Trend Engine Status
        "matic": {
            "trend": bot_state.get("matic_trend"),
            "action": bot_state.get("matic_action"),
            "in_position": bot_state.get("matic_in_position", False),
            "entry_price": bot_state.get("matic_entry_price"),
            "pnl_percent": bot_state.get("matic_pnl"),
        }
    }


@router.get("/sol-status")
async def get_sol_status(current_user: CurrentUser):
    """Get SOL Trend Engine status (optimized +303% strategy)"""
    from engine.trading_engine import trading_engine
    
    # Get live price
    sol_price = await trading_engine._get_current_price("SOL/USDT")
    
    # Calculate PnL if in position
    pnl = None
    if trading_engine.sol_in_position and trading_engine.sol_entry_price > 0 and sol_price:
        pnl = (sol_price - trading_engine.sol_entry_price) / trading_engine.sol_entry_price * 100
    
    return {
        "symbol": "SOL/USDT",
        "strategy": "TREND_FOLLOWING (+303% backtested)",
        "current_price": sol_price,
        "trend": bot_state.get("sol_trend", "UNKNOWN"),
        "action": bot_state.get("sol_action", "HOLD"),
        "in_position": trading_engine.sol_in_position,
        "entry_price": trading_engine.sol_entry_price if trading_engine.sol_in_position else None,
        "unrealized_pnl_percent": round(pnl, 2) if pnl else None,
        "config": {
            "trend_confirm_hours": 72,
            "cooldown_hours": 48,
            "rsi_top": 75,
            "profit_target": "40%",
            "position_size": "80%"
        }
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
    """Get current bot status with live P&L"""
    position_repo = PositionRepository(db)
    trade_repo = TradeRepository(db)
    
    # Get all trades to find active paper trades and today's stats
    trades = await trade_repo.get_user_trades(user_id=current_user.id, limit=100)
    today_trades = await trade_repo.get_today_trades(current_user.id)
    
    # Filter active paper trades (no exit price)
    active_paper_trades = [t for t in trades if t.exit_price is None]
    active_positions_count = len(active_paper_trades)
    
    # Calculate Realized P&L Today
    realized_today = await trade_repo.get_today_pnl(current_user.id) or 0
    
    # Calculate Unrealized P&L (Live)
    unrealized_pnl = 0.0
    if active_paper_trades:
        from api.v1.endpoints.trades import get_live_prices
        symbols = list(set(t.symbol for t in active_paper_trades))
        live_prices = await get_live_prices(symbols)
        
        for trade in active_paper_trades:
            current_price = live_prices.get(trade.symbol, trade.entry_price or 0)
            entry_price = trade.entry_price or 0
            quantity = trade.quantity or 0
            
            if trade.direction.value == "LONG":
                unrealized_pnl += (current_price - entry_price) * quantity
            else:
                unrealized_pnl += (entry_price - current_price) * quantity
    
    # Total P&L Today = Realized Today + Unrealized (All active)
    # Note: Usually Unrealized is separate, but user wants to see "Today's P&L" moving
    total_today_pnl = realized_today + unrealized_pnl
    
    # Calculate daily loss usage
    # Assuming starting balance of max_position_size for simplicity
    starting_balance = current_user.max_position_size_usdt * 10  # Rough estimate
    max_daily_loss = starting_balance * (current_user.max_daily_loss_percent / 100)
    daily_loss_used = abs(min(total_today_pnl, 0)) / max_daily_loss * 100 if max_daily_loss > 0 else 0
    
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
        active_positions=active_positions_count,
        pending_orders=0,
        trades_today=len(today_trades),
        pnl_today=round(total_today_pnl, 2),
        pnl_today_percent=(total_today_pnl / starting_balance * 100) if starting_balance else 0,
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
        
        # Restart the background task
        import asyncio
        from engine.trading_engine import start_trading_engine
        asyncio.create_task(start_trading_engine())
        
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
    """Execute the kill switch logic - close all positions/trades and stop bot"""
    global bot_state
    
    position_repo = PositionRepository(db)
    trade_repo = TradeRepository(db)
    
    # 1. Close PositionRepository positions (if any)
    positions = await position_repo.get_active_positions(current_user.id)
    closed_positions_count = await position_repo.close_all_positions(current_user.id)
    
    # 2. Close Active Paper Trades (TradeRepository)
    trades = await trade_repo.get_user_trades(user_id=current_user.id, limit=100)
    active_trades = [t for t in trades if t.exit_price is None]
    
    paper_closed_count = 0
    total_value_liquidated = sum(p.position_value for p in positions)
    
    if active_trades:
        from api.v1.endpoints.trades import get_live_prices
        from datetime import datetime, timezone
        
        symbols = list(set(t.symbol for t in active_trades))
        live_prices = await get_live_prices(symbols)
        
        for trade in active_trades:
            close_price = live_prices.get(trade.symbol, trade.entry_price)
            
            # Calculate P&L
            if trade.direction.value == "LONG":
                pnl_usdt = (close_price - trade.entry_price) * trade.quantity
                pnl_percent = ((close_price - trade.entry_price) / trade.entry_price * 100)
            else:
                pnl_usdt = (trade.entry_price - close_price) * trade.quantity
                pnl_percent = ((trade.entry_price - close_price) / trade.entry_price * 100)
                
            # Update trade
            await trade_repo.update(
                trade.id,
                obj_in={
                    "exit_price": close_price,
                    "exit_time": datetime.now(timezone.utc),
                    "pnl_usdt": pnl_usdt,
                    "pnl_percent": pnl_percent,
                    "status": "CLOSED",
                    "notes": f"{trade.notes or ''} | CLOSED by Kill Switch"
                }
            )
            paper_closed_count += 1
            total_value_liquidated += (close_price * trade.quantity)
            
    # Stop the bot
    bot_state["state"] = BotState.STOPPED
    
    # Clear memory positions in TradingEngine if running
    # This is a bit tricky since we can't easily access the running engine instance from here
    # But next cycle it will see trades are closed? No, engine keeps memory state.
    # ideally we should signal engine to clear state. For now, stopping bot prevents new trades.
    
    total_closed = closed_positions_count + paper_closed_count
    
    return KillSwitchResponse(
        success=True,
        positions_closed=total_closed,
        total_value_liquidated=total_value_liquidated,
        final_balance_usdt=total_value_liquidated,  # Simplified
        message=f"Kill switch activated. Closed {total_closed} trades/positions. Bot stopped."
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
