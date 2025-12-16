"""
Sentinel Quant - Custom Exceptions
Hierarchy of exceptions for better error handling
"""
from typing import Optional, Any
from fastapi import HTTPException, status


class SentinelException(Exception):
    """Base exception for all Sentinel Quant errors"""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Any] = None):
        self.message = message
        self.code = code or "SENTINEL_ERROR"
        self.details = details
        super().__init__(self.message)


class TradingException(SentinelException):
    """Trading-related errors"""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Any] = None):
        super().__init__(message, code or "TRADING_ERROR", details)


class ExchangeConnectionError(TradingException):
    """Failed to connect to exchange"""
    
    def __init__(self, exchange: str, message: str):
        super().__init__(
            f"Failed to connect to {exchange}: {message}",
            "EXCHANGE_CONNECTION_ERROR",
            {"exchange": exchange}
        )


class InsufficientBalanceError(TradingException):
    """Not enough balance for trade"""
    
    def __init__(self, required: float, available: float, asset: str = "USDT"):
        super().__init__(
            f"Insufficient {asset} balance. Required: {required}, Available: {available}",
            "INSUFFICIENT_BALANCE",
            {"required": required, "available": available, "asset": asset}
        )


class RiskLimitExceededError(TradingException):
    """Trade exceeds risk parameters"""
    
    def __init__(self, reason: str, details: Optional[Any] = None):
        super().__init__(
            f"Risk limit exceeded: {reason}",
            "RISK_LIMIT_EXCEEDED",
            details
        )


class ConfidenceTooLowError(TradingException):
    """AI confidence below threshold"""
    
    def __init__(self, confidence: float, threshold: float):
        super().__init__(
            f"Trade confidence ({confidence:.2%}) below threshold ({threshold:.2%})",
            "CONFIDENCE_TOO_LOW",
            {"confidence": confidence, "threshold": threshold}
        )


class SentimentVetoError(TradingException):
    """Trade vetoed by sentiment analysis"""
    
    def __init__(self, sentiment_score: float, reason: str):
        super().__init__(
            f"Trade vetoed by sentiment analysis: {reason}",
            "SENTIMENT_VETO",
            {"sentiment_score": sentiment_score, "reason": reason}
        )


class AuthException(SentinelException):
    """Authentication/Authorization errors"""
    
    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message, code or "AUTH_ERROR")


class InvalidCredentialsError(AuthException):
    """Invalid username or password"""
    
    def __init__(self):
        super().__init__("Invalid email or password", "INVALID_CREDENTIALS")


class TokenExpiredError(AuthException):
    """JWT token has expired"""
    
    def __init__(self):
        super().__init__("Token has expired", "TOKEN_EXPIRED")


class InvalidTokenError(AuthException):
    """JWT token is invalid"""
    
    def __init__(self):
        super().__init__("Invalid token", "INVALID_TOKEN")


# HTTP Exception Helpers
def raise_unauthorized(detail: str = "Could not validate credentials") -> None:
    """Raise 401 Unauthorized"""
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"}
    )


def raise_forbidden(detail: str = "Access denied") -> None:
    """Raise 403 Forbidden"""
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=detail
    )


def raise_not_found(resource: str = "Resource") -> None:
    """Raise 404 Not Found"""
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"{resource} not found"
    )


def raise_bad_request(detail: str) -> None:
    """Raise 400 Bad Request"""
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=detail
    )
