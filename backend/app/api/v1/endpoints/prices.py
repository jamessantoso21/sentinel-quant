"""
Sentinel Quant - Prices API Endpoints
Real-time and historical price data from exchanges
"""
from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import ccxt.async_support as ccxt

from core.config import settings

router = APIRouter()


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


@router.get("/ohlcv/{symbol}", response_model=OHLCVResponse)
async def get_ohlcv(
    symbol: str,
    timeframe: str = Query(default="1h", description="Timeframe: 1m, 5m, 15m, 1h, 4h, 1d"),
    limit: int = Query(default=100, ge=10, le=1000, description="Number of candles")
):
    """
    Get OHLCV (candlestick) data for a trading pair.
    
    - **symbol**: Trading pair like BTC-USDT, ETH-USDT (use dash, not slash)
    - **timeframe**: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
    - **limit**: Number of candles to return (10-1000)
    """
    # Convert dash to slash for ccxt
    ccxt_symbol = symbol.replace("-", "/").upper()
    
    try:
        # Create exchange instance
        exchange = ccxt.binance({
            "apiKey": settings.BINANCE_API_KEY,
            "secret": settings.BINANCE_API_SECRET,
            "enableRateLimit": True
        })
        
        # Use testnet if configured
        if settings.BINANCE_TESTNET:
            exchange.set_sandbox_mode(True)
        
        # Fetch OHLCV data
        ohlcv = await exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)
        
        await exchange.close()
        
        # Convert to response format
        data = [
            OHLCVData(
                timestamp=candle[0],
                open=candle[1],
                high=candle[2],
                low=candle[3],
                close=candle[4],
                volume=candle[5]
            )
            for candle in ohlcv
        ]
        
        return OHLCVResponse(
            symbol=symbol,
            timeframe=timeframe,
            data=data
        )
        
    except Exception as e:
        # Fallback to public API if testnet fails
        try:
            exchange = ccxt.binance({"enableRateLimit": True})
            ohlcv = await exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)
            await exchange.close()
            
            data = [
                OHLCVData(
                    timestamp=candle[0],
                    open=candle[1],
                    high=candle[2],
                    low=candle[3],
                    close=candle[4],
                    volume=candle[5]
                )
                for candle in ohlcv
            ]
            
            return OHLCVResponse(
                symbol=symbol,
                timeframe=timeframe,
                data=data
            )
        except Exception as fallback_error:
            raise HTTPException(status_code=500, detail=f"Failed to fetch OHLCV: {str(fallback_error)}")


@router.get("/ticker/{symbol}", response_model=TickerResponse)
async def get_ticker(symbol: str):
    """
    Get current ticker/price for a trading pair.
    
    - **symbol**: Trading pair like BTC-USDT, ETH-USDT
    """
    ccxt_symbol = symbol.replace("-", "/").upper()
    
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        ticker = await exchange.fetch_ticker(ccxt_symbol)
        await exchange.close()
        
        return TickerResponse(
            symbol=symbol,
            last_price=ticker["last"],
            bid=ticker["bid"] or ticker["last"],
            ask=ticker["ask"] or ticker["last"],
            volume_24h=ticker["quoteVolume"] or 0,
            change_24h=ticker["change"] or 0,
            change_24h_percent=ticker["percentage"] or 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch ticker: {str(e)}")


@router.get("/tickers")
async def get_multiple_tickers(
    symbols: str = Query(default="BTC-USDT,ETH-USDT", description="Comma-separated symbols")
):
    """Get tickers for multiple trading pairs"""
    symbol_list = [s.strip() for s in symbols.split(",")]
    
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        results = []
        
        for symbol in symbol_list:
            ccxt_symbol = symbol.replace("-", "/").upper()
            try:
                ticker = await exchange.fetch_ticker(ccxt_symbol)
                results.append({
                    "symbol": symbol,
                    "last_price": ticker["last"],
                    "change_24h_percent": ticker["percentage"] or 0
                })
            except:
                pass
        
        await exchange.close()
        return {"tickers": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
