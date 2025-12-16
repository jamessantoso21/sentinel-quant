"""
Sentinel Quant - Position Repository
"""
from typing import Optional, List
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseRepository
from models.position import Position, PositionSide


class PositionRepository(BaseRepository[Position]):
    """Repository for Position model operations"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(Position, session)
    
    async def get_active_positions(self, user_id: int) -> List[Position]:
        """Get all active positions for a user"""
        result = await self.session.execute(
            select(Position)
            .where(Position.user_id == user_id)
            .where(Position.is_active == True)
            .order_by(Position.created_at.desc())
        )
        return list(result.scalars().all())
    
    async def get_position_by_symbol(
        self, 
        user_id: int, 
        symbol: str
    ) -> Optional[Position]:
        """Get active position for a specific symbol"""
        result = await self.session.execute(
            select(Position)
            .where(Position.user_id == user_id)
            .where(Position.symbol == symbol)
            .where(Position.is_active == True)
        )
        return result.scalar_one_or_none()
    
    async def get_total_exposure(self, user_id: int) -> float:
        """Calculate total position value (exposure)"""
        positions = await self.get_active_positions(user_id)
        return sum(p.position_value for p in positions)
    
    async def get_total_unrealized_pnl(self, user_id: int) -> float:
        """Calculate total unrealized PnL"""
        positions = await self.get_active_positions(user_id)
        return sum(p.unrealized_pnl for p in positions)
    
    async def close_position(self, position_id: int) -> Optional[Position]:
        """Mark position as closed"""
        return await self.update(position_id, is_active=False)
    
    async def close_all_positions(self, user_id: int) -> int:
        """Close all active positions for a user (kill switch)"""
        positions = await self.get_active_positions(user_id)
        for position in positions:
            await self.close_position(position.id)
        return len(positions)
    
    async def update_current_price(
        self, 
        symbol: str, 
        current_price: float
    ) -> int:
        """Update current price for all positions of a symbol"""
        result = await self.session.execute(
            select(Position)
            .where(Position.symbol == symbol)
            .where(Position.is_active == True)
        )
        positions = list(result.scalars().all())
        
        for position in positions:
            position.current_price = current_price
        
        await self.session.flush()
        return len(positions)
