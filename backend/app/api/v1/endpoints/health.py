"""
Sentinel Quant - Health Check Endpoints
"""
from fastapi import APIRouter, Depends
from datetime import datetime, timezone
import redis.asyncio as redis
import ccxt.async_support as ccxt

from core.config import settings

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with all service statuses"""
    results = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {}
    }
    
    # Check Redis
    try:
        r = redis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.close()
        results["services"]["redis"] = {"status": "healthy"}
    except Exception as e:
        results["services"]["redis"] = {"status": "unhealthy", "error": str(e)}
        results["status"] = "degraded"
    
    # Check Exchange (Binance)
    try:
        exchange = ccxt.binance()
        ticker = await exchange.fetch_ticker('BTC/USDT')
        await exchange.close()
        results["services"]["exchange"] = {
            "status": "healthy",
            "btc_price": ticker['last']
        }
    except Exception as e:
        results["services"]["exchange"] = {"status": "unhealthy", "error": str(e)}
        results["status"] = "degraded"
    
    # Trading status
    results["trading"] = {
        "enabled": settings.TRADING_ENABLED,
        "primary_exchange": settings.PRIMARY_EXCHANGE,
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD
    }
    
    return results
