"""
Sentinel Quant - User Repository
"""
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseRepository
from models.user import User


class UserRepository(BaseRepository[User]):
    """Repository for User model operations"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(User, session)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address"""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def get_active_users(self) -> list[User]:
        """Get all active users"""
        result = await self.session.execute(
            select(User).where(User.is_active == True)
        )
        return list(result.scalars().all())
    
    async def get_trading_enabled_users(self) -> list[User]:
        """Get users with trading enabled"""
        result = await self.session.execute(
            select(User)
            .where(User.is_active == True)
            .where(User.trading_enabled == True)
        )
        return list(result.scalars().all())
    
    async def update_fcm_token(self, user_id: int, fcm_token: str) -> Optional[User]:
        """Update user's FCM token for push notifications"""
        return await self.update(user_id, fcm_token=fcm_token)
    
    async def toggle_trading(self, user_id: int, enabled: bool) -> Optional[User]:
        """Enable or disable trading for user"""
        return await self.update(user_id, trading_enabled=enabled)
