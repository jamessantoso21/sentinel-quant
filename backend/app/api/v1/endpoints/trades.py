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

# Cache for live prices (30 second TTL)
_price_cache: dict = {}
_price_cache_time: float = 0
PRICE_CACHE_TTL = 30  # seconds


async def get_live_prices(symbols: list[str]) -> dict:
    """Fetch live prices from CoinGecko with caching"""
    import time
    global _price_cache, _price_cache_time
    
    # Check cache first
    if time.time() - _price_cache_time < PRICE_CACHE_TTL and _price_cache:
        return {s: _price_cache.get(s, 0) for s in symbols}
    
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
                
                # Update cache
                _price_cache = result
                _price_cache_time = time.time()
                return result
    except Exception:
        pass
    
    # Return cached prices if available, even if stale
    if _price_cache:
        return {s: _price_cache.get(s, 0) for s in symbols}
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
    """Get user's trade history with pagination and live P&L for open trades"""
    trade_repo = TradeRepository(db)
    skip = (page - 1) * page_size
    
    trades = await trade_repo.get_user_trades(
        user_id=current_user.id,
        skip=skip,
        limit=page_size,
        status=status
    )
    
    # Enrich open trades with live P&L
    open_trades_indices = [i for i, t in enumerate(trades) if t.exit_price is None]
    
    response_trades = [TradeResponse.model_validate(t) for t in trades]
    
    if open_trades_indices:
        open_trades = [trades[i] for i in open_trades_indices]
        symbols = list(set(t.symbol for t in open_trades))
        live_prices = await get_live_prices(symbols)
        
        for i in open_trades_indices:
            trade = trades[i]
            current_price = live_prices.get(trade.symbol, trade.entry_price or 0)
            entry_price = trade.entry_price or 0
            quantity = trade.quantity or 0
            
            pnl_usdt = 0
            pnl_percent = 0
            
            if trade.direction.value == "LONG":
                pnl_usdt = (current_price - entry_price) * quantity
                pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            else:
                pnl_usdt = (entry_price - current_price) * quantity
                pnl_percent = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
            
            # Update the response model
            response_trades[i].pnl_usdt = round(pnl_usdt, 2)
            response_trades[i].pnl_percent = round(pnl_percent, 2)
    
    # Get total count for pagination
    total = await trade_repo.count()
    
    return TradeListResponse(
        trades=response_trades,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/today")
async def get_today_trades(current_user: CurrentUser, db: DbSession):
    """Get today's trades with live P&L for open positions"""
    trade_repo = TradeRepository(db)
    trades = await trade_repo.get_today_trades(current_user.id)
    
    # Calculate live P&L for open trades
    open_trades = [t for t in trades if t.exit_price is None]
    realized_pnl = await trade_repo.get_today_pnl(current_user.id)
    
    unrealized_pnl = 0.0
    if open_trades:
        symbols = list(set(t.symbol for t in open_trades))
        live_prices = await get_live_prices(symbols)
        
        for trade in open_trades:
            current_price = live_prices.get(trade.symbol, trade.entry_price or 0)
            entry_price = trade.entry_price or 0
            quantity = trade.quantity or 0
            
            if trade.direction.value == "LONG":
                unrealized_pnl += (current_price - entry_price) * quantity
            else:
                unrealized_pnl += (entry_price - current_price) * quantity
    
    return {
        "trades": [TradeResponse.model_validate(t) for t in trades],
        "realized_pnl": round(realized_pnl or 0, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "pnl_today": round((realized_pnl or 0) + unrealized_pnl, 2)  # Total = realized + unrealized
    }


@router.get("/summary")
async def get_trade_summary(current_user: CurrentUser, db: DbSession):
    """Get trade statistics summary with live unrealized P&L"""
    trade_repo = TradeRepository(db)
    stats = await trade_repo.get_trade_statistics(current_user.id)
    
    # Get all trades to calculate unrealized P&L for open positions
    trades = await trade_repo.get_user_trades(user_id=current_user.id, limit=100)
    open_trades = [t for t in trades if t.exit_price is None]
    
    unrealized_pnl = 0.0
    if open_trades:
        symbols = list(set(t.symbol for t in open_trades))
        live_prices = await get_live_prices(symbols)
        
        for trade in open_trades:
            current_price = live_prices.get(trade.symbol, trade.entry_price or 0)
            entry_price = trade.entry_price or 0
            quantity = trade.quantity or 0
            
            if trade.direction.value == "LONG":
                unrealized_pnl += (current_price - entry_price) * quantity
            else:
                unrealized_pnl += (entry_price - current_price) * quantity
    
    # Add unrealized P&L to total
    realized_pnl = stats.get('total_pnl_usdt', 0) or 0
    total_pnl = realized_pnl + unrealized_pnl
    
    return {
        **stats,
        "realized_pnl_usdt": round(realized_pnl, 2),
        "unrealized_pnl_usdt": round(unrealized_pnl, 2),
        "total_pnl_usdt": round(total_pnl, 2),  # Combined
        "active_positions": len(open_trades)
    }


@router.get("/{trade_id}", response_model=TradeResponse)
async def get_trade(trade_id: int, current_user: CurrentUser, db: DbSession):
    """Get specific trade details"""
    trade_repo = TradeRepository(db)
    trade = await trade_repo.get_by_id(trade_id)
    
    if not trade or trade.user_id != current_user.id:
        from core.exceptions import raise_not_found
        raise_not_found("Trade")
    
    return trade
