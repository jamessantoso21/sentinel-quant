"""
Sentinel Quant - Trade Repository
"""
from typing import Optional, List
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseRepository
from models.trade import Trade, TradeStatus, TradeDirection


class TradeRepository(BaseRepository[Trade]):
    """Repository for Trade model operations"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(Trade, session)
    
    async def get_user_trades(
        self, 
        user_id: int, 
        skip: int = 0, 
        limit: int = 50,
        status: Optional[TradeStatus] = None
    ) -> List[Trade]:
        """Get trades for a specific user with optional status filter"""
        query = select(Trade).where(Trade.user_id == user_id)
        
        if status:
            query = query.where(Trade.status == status)
        
        query = query.order_by(Trade.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_today_trades(self, user_id: int) -> List[Trade]:
        """Get today's trades for a user"""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        result = await self.session.execute(
            select(Trade)
            .where(Trade.user_id == user_id)
            .where(Trade.created_at >= today_start)
            .order_by(Trade.created_at.desc())
        )
        return list(result.scalars().all())
    
    async def get_today_pnl(self, user_id: int) -> float:
        """Calculate today's total PnL for a user"""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        result = await self.session.execute(
            select(func.coalesce(func.sum(Trade.pnl_usdt), 0))
            .where(Trade.user_id == user_id)
            .where(Trade.created_at >= today_start)
            .where(Trade.status == TradeStatus.EXECUTED)
        )
        return float(result.scalar() or 0)
    
    async def get_trade_statistics(self, user_id: int) -> dict:
        """Get comprehensive trade statistics"""
        # Total trades
        total = await self.session.execute(
            select(func.count())
            .select_from(Trade)
            .where(Trade.user_id == user_id)
            .where(Trade.status == TradeStatus.EXECUTED)
        )
        total_trades = total.scalar() or 0
        
        # Winning trades
        wins = await self.session.execute(
            select(func.count())
            .select_from(Trade)
            .where(Trade.user_id == user_id)
            .where(Trade.status == TradeStatus.EXECUTED)
            .where(Trade.pnl_usdt > 0)
        )
        winning_trades = wins.scalar() or 0
        
        # Total PnL
        pnl = await self.session.execute(
            select(func.coalesce(func.sum(Trade.pnl_usdt), 0))
            .where(Trade.user_id == user_id)
            .where(Trade.status == TradeStatus.EXECUTED)
        )
        total_pnl = float(pnl.scalar() or 0)
        
        # Best and worst trades
        best = await self.session.execute(
            select(func.coalesce(func.max(Trade.pnl_usdt), 0))
            .where(Trade.user_id == user_id)
            .where(Trade.status == TradeStatus.EXECUTED)
        )
        
        worst = await self.session.execute(
            select(func.coalesce(func.min(Trade.pnl_usdt), 0))
            .where(Trade.user_id == user_id)
            .where(Trade.status == TradeStatus.EXECUTED)
        )
        
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = (total_pnl / total_trades) if total_trades > 0 else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl_usdt": total_pnl,
            "average_pnl_percent": avg_pnl,
            "best_trade_pnl": float(best.scalar() or 0),
            "worst_trade_pnl": float(worst.scalar() or 0)
        }
