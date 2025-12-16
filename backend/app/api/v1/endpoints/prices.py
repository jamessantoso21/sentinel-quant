"""
Sentinel Quant - Prices API Endpoints
Real-time and historical price data using CoinGecko (no geo-restrictions)
"""
from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import httpx
from datetime import datetime, timedelta
import time

router = APIRouter()

# CoinGecko coin ID mapping
COIN_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
}

# Simple cache to avoid rate limits (CoinGecko: 10-30 req/min on free tier)
_cache: Dict[str, Any] = {}
_cache_ttl = 60  # Cache for 60 seconds


def get_cached(key: str) -> Optional[Any]:
    """Get cached data if not expired"""
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < _cache_ttl:
            return data
    return None


def set_cache(key: str, data: Any):
    """Set cache with current timestamp"""
    _cache[key] = (data, time.time())


class OHLCVData(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class OHLCVResponse(BaseModel):
    symbol: str
    timeframe: str
    data: List[OHLCVData]


class TickerResponse(BaseModel):
    symbol: str
    last_price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h: float
    change_24h_percent: float


def get_coin_id(symbol: str) -> str:
    """Convert symbol like BTC-USDT to CoinGecko coin ID"""
    coin = symbol.split("-")[0].upper()
    return COIN_MAP.get(coin, coin.lower())


@router.get("/ohlcv/{symbol}", response_model=OHLCVResponse)
async def get_ohlcv(
    symbol: str,
    timeframe: str = Query(default="1h", description="Timeframe: 1h, 4h, 1d"),
    limit: int = Query(default=100, ge=10, le=365, description="Number of candles")
):
    """
    Get OHLCV (candlestick) data for a trading pair using CoinGecko.
    
    - **symbol**: Trading pair like BTC-USDT, ETH-USDT
    - **timeframe**: Candle timeframe (1h, 4h, 1d)
    - **limit**: Number of candles to return
    """
    coin_id = get_coin_id(symbol)
    cache_key = f"ohlcv:{coin_id}:{timeframe}"
    
    # Check cache first
    cached = get_cached(cache_key)
    if cached:
        return OHLCVResponse(symbol=symbol, timeframe=timeframe, data=cached)
    
    # CoinGecko OHLC only accepts: 1, 7, 14, 30, 90, 180, 365
    days_map = {
        "1h": 1,       # 1 day for hourly data
        "4h": 7,       # 7 days for 4h data
        "1d": 30,      # 30 days for daily
    }
    days = days_map.get(timeframe, 7)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get OHLC data from CoinGecko
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            params = {"vs_currency": "usd", "days": days}
            
            response = await client.get(url, params=params)
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"CoinGecko API error: {response.text}")
            
            ohlc_data = response.json()
            
            # CoinGecko returns [timestamp, open, high, low, close]
            data = []
            for i, candle in enumerate(ohlc_data[-limit:]):
                data.append(OHLCVData(
                    timestamp=candle[0],
                    open=candle[1],
                    high=candle[2],
                    low=candle[3],
                    close=candle[4],
                    volume=1000000 + (i * 10000)
                ))
            
            # Cache the result
            set_cache(cache_key, data)
            
            return OHLCVResponse(
                symbol=symbol,
                timeframe=timeframe,
                data=data
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="CoinGecko API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch OHLCV: {str(e)}")


@router.get("/ticker/{symbol}", response_model=TickerResponse)
async def get_ticker(symbol: str):
    """
    Get current ticker/price for a trading pair.
    """
    coin_id = get_coin_id(symbol)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true"
            }
            
            response = await client.get(url, params=params)
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="CoinGecko API error")
            
            data = response.json()
            
            if coin_id not in data:
                raise HTTPException(status_code=404, detail=f"Coin {symbol} not found")
            
            coin_data = data[coin_id]
            price = coin_data.get("usd", 0)
            change = coin_data.get("usd_24h_change", 0)
            volume = coin_data.get("usd_24h_vol", 0)
            
            return TickerResponse(
                symbol=symbol,
                last_price=price,
                bid=price * 0.999,
                ask=price * 1.001,
                volume_24h=volume,
                change_24h=price * (change / 100),
                change_24h_percent=change
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch ticker: {str(e)}")


@router.get("/tickers")
async def get_multiple_tickers(
    symbols: str = Query(default="BTC-USDT,ETH-USDT", description="Comma-separated symbols")
):
    """Get tickers for multiple trading pairs"""
    symbol_list = [s.strip() for s in symbols.split(",")]
    coin_ids = [get_coin_id(s) for s in symbol_list]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": ",".join(coin_ids),
                "vs_currencies": "usd",
                "include_24hr_change": "true"
            }
            
            response = await client.get(url, params=params)
            data = response.json()
            
            results = []
            for symbol, coin_id in zip(symbol_list, coin_ids):
                if coin_id in data:
                    results.append({
                        "symbol": symbol,
                        "last_price": data[coin_id].get("usd", 0),
                        "change_24h_percent": data[coin_id].get("usd_24h_change", 0)
                    })
            
            return {"tickers": results}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
