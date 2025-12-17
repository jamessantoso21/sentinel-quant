"""
Sentinel Quant - Technical Analyzer
Real-time technical analysis for trading decisions
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, List
import logging
import httpx

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignal:
    """Technical analysis result"""
    rsi: float                    # RSI value (0-100)
    rsi_signal: str               # OVERSOLD, NEUTRAL, OVERBOUGHT
    bb_position: float            # 0=lower band, 0.5=middle, 1=upper
    bb_signal: str                # OVERSOLD, NEUTRAL, OVERBOUGHT
    trend: str                    # BULLISH, BEARISH, SIDEWAYS
    support_distance: float       # % distance to support
    resistance_distance: float    # % distance to resistance
    overall_signal: str           # BUY, SELL, NEUTRAL
    confluence_score: float       # 0-1 how many signals agree


class TechnicalAnalyzer:
    """
    Real-time technical analysis.
    Computes RSI, Bollinger Bands, and trend detection.
    """
    
    # Class-level cache to persist across instances
    _price_cache = {}
    _cache_times = {}
    _cache_duration = 300  # 5 minutes cache
    
    def __init__(self):
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.bb_oversold = 0.2
        self.bb_overbought = 0.8
    
    async def analyze(self, symbol: str = "BTC/USDT") -> Optional[TechnicalSignal]:
        """
        Analyze current market conditions for a symbol.
        Fetches recent price data and computes indicators.
        """
        try:
            # Fetch recent price data
            logger.info(f"Fetching price data for {symbol}")
            df = await self._fetch_price_data(symbol)
            if df is None or len(df) < 40:  # CoinGecko provides ~42 candles
                logger.warning(f"Insufficient price data for analysis: {len(df) if df is not None else 0} rows")
                return None
            
            logger.info(f"Got {len(df)} price rows, computing indicators...")
            
            # Compute RSI
            rsi = self._compute_rsi(df['close'], 14)
            rsi_value = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            logger.info(f"RSI computed: {rsi_value:.1f}")
            
            if rsi_value < self.rsi_oversold:
                rsi_signal = "OVERSOLD"
            elif rsi_value > self.rsi_overbought:
                rsi_signal = "OVERBOUGHT"
            else:
                rsi_signal = "NEUTRAL"
            
            # Compute Bollinger Bands
            sma = df['close'].rolling(20).mean()
            std = df['close'].rolling(20).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            
            current_price = df['close'].iloc[-1]
            bb_position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            bb_position = max(0, min(1, bb_position))  # Clamp to 0-1
            
            if bb_position < self.bb_oversold:
                bb_signal = "OVERSOLD"
            elif bb_position > self.bb_overbought:
                bb_signal = "OVERBOUGHT"
            else:
                bb_signal = "NEUTRAL"
            
            # Trend detection (SMA crossover)
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            
            if sma_20 > sma_50 * 1.01:
                trend = "BULLISH"
            elif sma_20 < sma_50 * 0.99:
                trend = "BEARISH"
            else:
                trend = "SIDEWAYS"
            
            # Support/Resistance (simplified: recent lows/highs)
            recent_low = df['low'].iloc[-20:].min()
            recent_high = df['high'].iloc[-20:].max()
            
            support_distance = (current_price - recent_low) / current_price * 100
            resistance_distance = (recent_high - current_price) / current_price * 100
            
            # Determine overall signal and confluence
            signals = []
            
            # RSI signal
            if rsi_signal == "OVERSOLD":
                signals.append("BUY")
            elif rsi_signal == "OVERBOUGHT":
                signals.append("SELL")
            else:
                signals.append("NEUTRAL")
            
            # BB signal
            if bb_signal == "OVERSOLD":
                signals.append("BUY")
            elif bb_signal == "OVERBOUGHT":
                signals.append("SELL")
            else:
                signals.append("NEUTRAL")
            
            # Trend signal
            if trend == "BULLISH":
                signals.append("BUY")
            elif trend == "BEARISH":
                signals.append("SELL")
            else:
                signals.append("NEUTRAL")
            
            # Count confluence
            buy_count = signals.count("BUY")
            sell_count = signals.count("SELL")
            
            if buy_count >= 2:
                overall_signal = "BUY"
                confluence_score = buy_count / 3
            elif sell_count >= 2:
                overall_signal = "SELL"
                confluence_score = sell_count / 3
            else:
                overall_signal = "NEUTRAL"
                confluence_score = 0.5
            
            return TechnicalSignal(
                rsi=rsi_value,
                rsi_signal=rsi_signal,
                bb_position=bb_position,
                bb_signal=bb_signal,
                trend=trend,
                support_distance=support_distance,
                resistance_distance=resistance_distance,
                overall_signal=overall_signal,
                confluence_score=confluence_score
            )
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return None
    
    async def _fetch_price_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data - try CoinGecko first (no geo-block), fallback to Binance"""
        import time
        
        # Check cache first
        cache_key = f"{symbol}_{limit}"
        if cache_key in self._price_cache:
            cache_age = time.time() - self._cache_times.get(cache_key, 0)
            if cache_age < self._cache_duration:
                logger.info(f"Using cached price data for {symbol} (age: {cache_age:.0f}s)")
                return self._price_cache[cache_key].copy()
        
        # Try CoinGecko first (no geo-restrictions)
        df = await self._fetch_from_coingecko(symbol, limit)
        if df is not None and len(df) >= 40:  # CoinGecko provides ~42 candles
            # Cache the result
            self._price_cache[cache_key] = df.copy()
            self._cache_times[cache_key] = time.time()
            logger.info(f"Cached price data for {symbol}")
            return df
        
        # Fallback to Binance
        df = await self._fetch_from_binance(symbol, limit)
        if df is not None and len(df) >= 40:
            self._price_cache[cache_key] = df.copy()
            self._cache_times[cache_key] = time.time()
        return df
    
    async def _fetch_from_coingecko(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch from CoinGecko API (no geo-restrictions)"""
        try:
            # Map symbol to CoinGecko ID
            coin_map = {
                "BTC/USDT": "bitcoin",
                "ETH/USDT": "ethereum",
                "BNB/USDT": "binancecoin",
                "SOL/USDT": "solana",
                "PAXG/USDT": "pax-gold"  # Gold-backed token
            }
            coin_id = coin_map.get(symbol, "bitcoin")
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Get OHLC data (1 day, 4h candles)
                response = await client.get(
                    f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
                    params={"vs_currency": "usd", "days": "7"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # CoinGecko format: [timestamp, open, high, low, close]
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                    df['volume'] = 0  # CoinGecko OHLC doesn't include volume
                    
                    logger.info(f"CoinGecko: Got {len(df)} candles for {coin_id}")
                    return df
                else:
                    logger.warning(f"CoinGecko API error: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed: {e}")
            return None
    
    async def _fetch_from_binance(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Binance (may be geo-blocked)"""
        try:
            binance_symbol = symbol.replace("/", "")
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://api.binance.com/api/v3/klines",
                    params={
                        "symbol": binance_symbol,
                        "interval": "15m",
                        "limit": limit
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    logger.info(f"Binance: Got {len(df)} candles")
                    return df
                else:
                    logger.warning(f"Binance API error: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Binance fetch failed: {e}")
            return None
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


# Singleton instance
technical_analyzer = TechnicalAnalyzer()
