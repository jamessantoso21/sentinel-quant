"""
Sentinel Quant - API Dependencies
Dependency injection for endpoints
"""
from typing import Annotated, Optional
from fastapi import Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db as get_db_session
from db.repositories.user import UserRepository
from models.user import User
from core.security import security_manager
from core.exceptions import raise_unauthorized, TokenExpiredError, InvalidTokenError

# Security scheme
security = HTTPBearer()


async def get_db():
    """Database session dependency"""
    async for session in get_db_session():
        yield session


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> User:
    """
    Validate JWT token and return current user.
    Use as dependency in protected endpoints.
    """
    token = credentials.credentials
    
    # Verify token
    user_id = security_manager.verify_access_token(token)
    if not user_id:
        raise_unauthorized("Invalid or expired token")
    
    # Get user from database
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(int(user_id))
    
    if not user:
        raise_unauthorized("User not found")
    
    if not user.is_active:
        raise_unauthorized("User account is disabled")
    
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """Ensure user is active"""
    if not current_user.is_active:
        raise_unauthorized("Inactive user")
    return current_user


async def get_current_trading_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """Ensure user has trading enabled"""
    if not current_user.trading_enabled:
        raise_unauthorized("Trading is not enabled for this account")
    return current_user


# Type aliases for cleaner endpoint signatures
CurrentUser = Annotated[User, Depends(get_current_user)]
TradingUser = Annotated[User, Depends(get_current_trading_user)]
DbSession = Annotated[AsyncSession, Depends(get_db)]
