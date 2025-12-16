"""
Sentinel Quant - Security Module
JWT token management and password hashing
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Any
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import settings


class TokenPayload(BaseModel):
    """JWT Token payload structure"""
    sub: str  # user_id
    exp: datetime
    type: str  # access or refresh


class TokenPair(BaseModel):
    """Access and refresh token pair"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class SecurityManager:
    """
    Handles all security operations:
    - Password hashing/verification
    - JWT token creation/validation
    """
    
    def __init__(self):
        # Use argon2id - modern, secure, no version issues
        self.pwd_context = CryptContext(
            schemes=["argon2"],
            deprecated="auto"
        )
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        self.refresh_token_expire = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    # Password Operations
    def hash_password(self, password: str) -> str:
        """Hash a password using argon2id"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    # Token Operations
    def create_access_token(self, user_id: str, extra_data: Optional[dict] = None) -> str:
        """Create a new access token"""
        expire = datetime.now(timezone.utc) + self.access_token_expire
        payload = {
            "sub": user_id,
            "exp": expire,
            "type": "access",
            **(extra_data or {})
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create a new refresh token"""
        expire = datetime.now(timezone.utc) + self.refresh_token_expire
        payload = {
            "sub": user_id,
            "exp": expire,
            "type": "refresh"
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_token_pair(self, user_id: str, extra_data: Optional[dict] = None) -> TokenPair:
        """Create both access and refresh tokens"""
        return TokenPair(
            access_token=self.create_access_token(user_id, extra_data),
            refresh_token=self.create_refresh_token(user_id)
        )
    
    def decode_token(self, token: str) -> Optional[TokenPayload]:
        """Decode and validate a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return TokenPayload(**payload)
        except JWTError:
            return None
    
    def verify_access_token(self, token: str) -> Optional[str]:
        """Verify access token and return user_id if valid"""
        payload = self.decode_token(token)
        if payload and payload.type == "access":
            return payload.sub
        return None
    
    def verify_refresh_token(self, token: str) -> Optional[str]:
        """Verify refresh token and return user_id if valid"""
        payload = self.decode_token(token)
        if payload and payload.type == "refresh":
            return payload.sub
        return None


# Global security manager instance
security_manager = SecurityManager()
