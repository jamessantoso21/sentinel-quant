"""
Sentinel Quant - Trades Endpoints
"""
from fastapi import APIRouter, Query
from typing import Optional
import httpx

from api.deps import DbSession, CurrentUser
from db.repositories.trade import TradeRepository
from schemas.trade import (
    TradeResponse, TradeListResponse, TradeSummary,
    TradeStatus
)

router = APIRouter()


async def get_live_prices(symbols: list[str]) -> dict:
    """Fetch live prices from CoinGecko"""
    coin_map = {
        "BTC/USDT": "bitcoin",
        "PAXG/USDT": "pax-gold",
        "ETH/USDT": "ethereum",
    }
    
    coin_ids = [coin_map.get(s, "bitcoin") for s in symbols]
    unique_ids = list(set(coin_ids))
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": ",".join(unique_ids), "vs_currencies": "usd"}
            )
            if response.status_code == 200:
                data = response.json()
                result = {}
                for symbol in symbols:
                    coin_id = coin_map.get(symbol, "bitcoin")
                    result[symbol] = data.get(coin_id, {}).get("usd", 0)
                return result
    except Exception:
        pass
    return {}


@router.get("/active")
async def get_active_trades(current_user: CurrentUser, db: DbSession):
    """Get open trades with live P&L calculation"""
    trade_repo = TradeRepository(db)
    
    # Get trades without exit_price (still open)
    trades = await trade_repo.get_user_trades(
        user_id=current_user.id,
        limit=50,
        status=None
    )
    
    # Filter to only open trades (no exit_price)
    open_trades = [t for t in trades if t.exit_price is None]
    
    if not open_trades:
        return {"positions": [], "total_unrealized_pnl": 0, "total_value": 0}
    
    # Get live prices
    symbols = list(set(t.symbol for t in open_trades))
    live_prices = await get_live_prices(symbols)
    
    # Calculate P&L for each position
    positions = []
    total_pnl = 0
    total_value = 0
    
    for trade in open_trades:
        current_price = live_prices.get(trade.symbol, trade.entry_price)
        entry_price = trade.entry_price or 0
        quantity = trade.quantity or 0
        
        # Calculate unrealized P&L
        if trade.direction.value == "LONG":
            pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            pnl_usdt = (current_price - entry_price) * quantity
        else:  # SHORT
            pnl_percent = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
            pnl_usdt = (entry_price - current_price) * quantity
        
        position_value = current_price * quantity
        total_pnl += pnl_usdt
        total_value += position_value
        
        positions.append({
            "id": trade.id,
            "symbol": trade.symbol,
            "side": trade.direction.value,  # LONG or SHORT
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": current_price,
            "unrealized_pnl": round(pnl_usdt, 2),
            "unrealized_pnl_percent": round(pnl_percent, 2),
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
            "created_at": trade.created_at.isoformat() if trade.created_at else None,
        })
    
    return {
        "positions": positions,
        "total_unrealized_pnl": round(total_pnl, 2),
        "total_value": round(total_value, 2)
    }


@router.get("/", response_model=TradeListResponse)
async def get_trades(
    current_user: CurrentUser,
    db: DbSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[TradeStatus] = None
):
    """Get user's trade history with pagination"""
    trade_repo = TradeRepository(db)
    skip = (page - 1) * page_size
    
    trades = await trade_repo.get_user_trades(
        user_id=current_user.id,
        skip=skip,
        limit=page_size,
        status=status
    )
    
    # Get total count for pagination
    total = await trade_repo.count()
    
    return TradeListResponse(
        trades=[TradeResponse.model_validate(t) for t in trades],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/today")
async def get_today_trades(current_user: CurrentUser, db: DbSession):
    """Get today's trades"""
    trade_repo = TradeRepository(db)
    trades = await trade_repo.get_today_trades(current_user.id)
    
    return {
        "trades": [TradeResponse.model_validate(t) for t in trades],
        "pnl_today": await trade_repo.get_today_pnl(current_user.id)
    }


@router.get("/summary", response_model=TradeSummary)
async def get_trade_summary(current_user: CurrentUser, db: DbSession):
    """Get trade statistics summary"""
    trade_repo = TradeRepository(db)
    stats = await trade_repo.get_trade_statistics(current_user.id)
    return TradeSummary(**stats)


@router.get("/{trade_id}", response_model=TradeResponse)
async def get_trade(trade_id: int, current_user: CurrentUser, db: DbSession):
    """Get specific trade details"""
    trade_repo = TradeRepository(db)
    trade = await trade_repo.get_by_id(trade_id)
    
    if not trade or trade.user_id != current_user.id:
        from core.exceptions import raise_not_found
        raise_not_found("Trade")
    
    return trade
