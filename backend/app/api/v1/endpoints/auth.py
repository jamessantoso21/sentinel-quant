"""
Sentinel Quant - Authentication Endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import DbSession, CurrentUser
from db.repositories.user import UserRepository
from schemas.user import (
    UserCreate, UserResponse, UserLogin, UserUpdate,
    TokenResponse, RefreshTokenRequest
)
from core.security import security_manager
from core.exceptions import raise_bad_request, raise_unauthorized

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: DbSession):
    """Register a new user"""
    user_repo = UserRepository(db)
    
    # Check if email exists
    existing = await user_repo.get_by_email(user_data.email)
    if existing:
        raise_bad_request("Email already registered")
    
    # Create user
    hashed_password = security_manager.hash_password(user_data.password)
    user = await user_repo.create(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name
    )
    
    return user


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: DbSession):
    """Login and get access token"""
    user_repo = UserRepository(db)
    
    # Find user
    user = await user_repo.get_by_email(credentials.email)
    if not user:
        raise_unauthorized("Invalid email or password")
    
    # Verify password
    if not security_manager.verify_password(credentials.password, user.hashed_password):
        raise_unauthorized("Invalid email or password")
    
    # Check if active
    if not user.is_active:
        raise_unauthorized("Account is disabled")
    
    # Generate tokens
    token_pair = security_manager.create_token_pair(str(user.id))
    
    return TokenResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest, db: DbSession):
    """Refresh access token using refresh token"""
    user_id = security_manager.verify_refresh_token(request.refresh_token)
    if not user_id:
        raise_unauthorized("Invalid refresh token")
    
    # Verify user still exists and is active
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(int(user_id))
    if not user or not user.is_active:
        raise_unauthorized("User not found or inactive")
    
    # Generate new tokens
    token_pair = security_manager.create_token_pair(user_id)
    
    return TokenResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: CurrentUser):
    """Get current user information"""
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: CurrentUser,
    db: DbSession
):
    """Update current user information"""
    user_repo = UserRepository(db)
    
    update_data = user_update.model_dump(exclude_unset=True)
    
    # Hash password if provided
    if "password" in update_data:
        update_data["hashed_password"] = security_manager.hash_password(update_data.pop("password"))
    
    updated_user = await user_repo.update(current_user.id, **update_data)
    return updated_user


@router.post("/fcm-token")
async def update_fcm_token(
    fcm_token: str,
    current_user: CurrentUser,
    db: DbSession
):
    """Update FCM token for push notifications"""
    user_repo = UserRepository(db)
    await user_repo.update_fcm_token(current_user.id, fcm_token)
    return {"message": "FCM token updated"}


@router.post("/enable-trading")
async def enable_trading(
    current_user: CurrentUser,
    db: DbSession
):
    """Enable trading for current user (for paper trading setup)"""
    user_repo = UserRepository(db)
    await user_repo.update(current_user.id, trading_enabled=True)
    return {"message": "Trading enabled", "trading_enabled": True}
