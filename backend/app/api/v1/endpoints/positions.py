"""
Sentinel Quant - Positions Endpoints
"""
from fastapi import APIRouter

from api.deps import DbSession, CurrentUser, TradingUser
from db.repositories.position import PositionRepository
from schemas.position import (
    PositionResponse, PositionListResponse, ClosePositionRequest
)
from core.exceptions import raise_not_found, raise_bad_request

router = APIRouter()


@router.get("/", response_model=PositionListResponse)
async def get_active_positions(current_user: CurrentUser, db: DbSession):
    """Get all active positions"""
    position_repo = PositionRepository(db)
    positions = await position_repo.get_active_positions(current_user.id)
    
    total_value = sum(p.position_value for p in positions)
    total_pnl = sum(p.unrealized_pnl for p in positions)
    
    return PositionListResponse(
        positions=[PositionResponse.model_validate(p) for p in positions],
        total_value=total_value,
        total_unrealized_pnl=total_pnl
    )


@router.get("/{position_id}", response_model=PositionResponse)
async def get_position(position_id: int, current_user: CurrentUser, db: DbSession):
    """Get specific position details"""
    position_repo = PositionRepository(db)
    position = await position_repo.get_by_id(position_id)
    
    if not position or position.user_id != current_user.id:
        raise_not_found("Position")
    
    return position


@router.post("/{position_id}/close")
async def close_position(
    position_id: int,
    request: ClosePositionRequest,
    current_user: TradingUser,
    db: DbSession
):
    """Close a position (partial or full)"""
    position_repo = PositionRepository(db)
    position = await position_repo.get_by_id(position_id)
    
    if not position or position.user_id != current_user.id:
        raise_not_found("Position")
    
    if not position.is_active:
        raise_bad_request("Position is already closed")
    
    # TODO: Execute close order on exchange
    # For now, just mark as closed
    if request.close_percent == 100:
        await position_repo.close_position(position_id)
        return {"message": "Position closed", "position_id": position_id}
    else:
        # Partial close - would need to update quantity
        return {
            "message": f"Partial close ({request.close_percent}%) - Not implemented yet",
            "position_id": position_id
        }


@router.get("/symbol/{symbol}")
async def get_position_by_symbol(
    symbol: str,
    current_user: CurrentUser,
    db: DbSession
):
    """Get active position for a specific symbol"""
    position_repo = PositionRepository(db)
    position = await position_repo.get_position_by_symbol(current_user.id, symbol)
    
    if not position:
        return {"message": f"No active position for {symbol}"}
    
    return PositionResponse.model_validate(position)
