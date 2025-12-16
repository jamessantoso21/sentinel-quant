"""
Sentinel Quant - Trades Endpoints
"""
from fastapi import APIRouter, Query
from typing import Optional

from api.deps import DbSession, CurrentUser
from db.repositories.trade import TradeRepository
from schemas.trade import (
    TradeResponse, TradeListResponse, TradeSummary,
    TradeStatus
)

router = APIRouter()


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
