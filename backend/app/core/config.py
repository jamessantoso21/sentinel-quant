"""
Sentinel Quant - Configuration Management
Pydantic-based settings with environment variable support
"""
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All sensitive data should be set via .env file or environment.
    """
    
    # Application
    APP_NAME: str = "Sentinel Quant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False)
    ENVIRONMENT: str = Field(default="development")
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://sentinel:sentinel_pass@timescaledb:5432/sentinel_quant"
    )
    DATABASE_POOL_SIZE: int = Field(default=10)
    DATABASE_MAX_OVERFLOW: int = Field(default=20)
    
    # Redis
    REDIS_URL: str = Field(default="redis://redis:6379/0")
    REDIS_CACHE_TTL: int = Field(default=300)  # 5 minutes
    
    # Security
    SECRET_KEY: str = Field(default="your-super-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7)
    ALGORITHM: str = Field(default="HS256")
    
    # Exchange APIs
    BINANCE_API_KEY: Optional[str] = Field(default=None)
    BINANCE_API_SECRET: Optional[str] = Field(default=None)
    BINANCE_TESTNET: bool = Field(default=False)  # Use testnet for paper trading
    BYBIT_API_KEY: Optional[str] = Field(default=None)
    BYBIT_API_SECRET: Optional[str] = Field(default=None)
    PRIMARY_EXCHANGE: str = Field(default="binance")  # binance or bybit
    
    # Trading Configuration
    TRADING_ENABLED: bool = Field(default=False)  # Safety: disabled by default
    MAX_POSITION_SIZE_USDT: float = Field(default=100.0)
    MAX_DAILY_LOSS_PERCENT: float = Field(default=5.0)
    CONFIDENCE_THRESHOLD: float = Field(default=0.85)  # 85% minimum
    
    # Validators to handle whitespace in boolean values
    @field_validator('BINANCE_TESTNET', 'TRADING_ENABLED', 'DEBUG', mode='before')
    @classmethod
    def parse_boolean(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            return v in ('true', '1', 'yes', 'on')
        return bool(v)
    
    # Dify (Sentiment Analysis)
    DIFY_API_URL: Optional[str] = Field(default=None)
    DIFY_API_KEY: Optional[str] = Field(default=None)
    SENTIMENT_UPDATE_INTERVAL_MINUTES: int = Field(default=15)
    
    # Firebase (Push Notifications)
    FIREBASE_CREDENTIALS_PATH: Optional[str] = Field(default=None)
    
    # Celery
    CELERY_BROKER_URL: str = Field(default="redis://redis:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://redis:6379/2")
    
    @field_validator("PRIMARY_EXCHANGE")
    @classmethod
    def validate_exchange(cls, v: str) -> str:
        if v.lower() not in ("binance", "bybit"):
            raise ValueError("PRIMARY_EXCHANGE must be 'binance' or 'bybit'")
        return v.lower()
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def async_database_url(self) -> str:
        """Convert sync DB URL to async (asyncpg)"""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
