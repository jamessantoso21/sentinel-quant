"""
Sentinel Quant - User Schemas
Pydantic models for user API requests/responses
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class UserBase(BaseModel):
    """Base user properties"""
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """Properties for user registration"""
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    """Properties for user update"""
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)
    trading_enabled: Optional[bool] = None
    max_position_size_usdt: Optional[float] = Field(None, gt=0)
    max_daily_loss_percent: Optional[float] = Field(None, gt=0, le=100)
    fcm_token: Optional[str] = None


class UserResponse(UserBase):
    """User response model"""
    id: int
    is_active: bool
    trading_enabled: bool
    max_position_size_usdt: float
    max_daily_loss_percent: float
    created_at: datetime
    
    model_config = {"from_attributes": True}


class UserLogin(BaseModel):
    """Login request"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str
