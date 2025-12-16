"""
Sentinel Quant - User Model
"""
from sqlalchemy import Column, String, Boolean, Float
from db.base import Base


class User(Base):
    """
    User account for app access
    """
    
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Trading settings
    trading_enabled = Column(Boolean, default=False, nullable=False)
    max_position_size_usdt = Column(Float, default=100.0, nullable=False)
    max_daily_loss_percent = Column(Float, default=5.0, nullable=False)
    
    # Firebase token for push notifications
    fcm_token = Column(String(500), nullable=True)
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"
