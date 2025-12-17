"""
Sentinel Quant - Trading Engine
Background trading loop with Dify sentiment analysis
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import httpx
import json

from core.config import settings

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine that runs the trading loop"""
    
    def __init__(self):
        self.is_running = False
        self.current_symbol = "BTC/USDT"
        self.last_signal_time: Optional[datetime] = None
        self.last_sentiment: Optional[Dict] = None
        self.last_ai_prediction: Optional[float] = None
        
    async def start(self):
        """Start the trading loop"""
        if self.is_running:
            logger.warning("Trading engine already running")
            return
            
        self.is_running = True
        logger.info("Trading engine started")
        
        while self.is_running:
            try:
                await self._trading_cycle()
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
            
            # Wait 5 minutes between cycles
            await asyncio.sleep(300)
    
    def stop(self):
        """Stop the trading loop"""
        self.is_running = False
        logger.info("Trading engine stopped")
    
    async def _trading_cycle(self):
        """Single trading cycle"""
        logger.info(f"Running trading cycle for {self.current_symbol}")
        
        # 1. Get market sentiment from Dify
        sentiment = await self._get_sentiment()
        if sentiment:
            self.last_sentiment = sentiment
            logger.info(f"Sentiment: {sentiment}")
        
        # 2. Get AI prediction (from trained models)
        ai_confidence = await self._get_ai_prediction()
        self.last_ai_prediction = ai_confidence
        
        # 3. Combine signals
        combined_confidence = self._combine_signals(sentiment, ai_confidence)
        
        # 4. Check if should trade
        if combined_confidence >= settings.CONFIDENCE_THRESHOLD:
            signal = "BUY" if (sentiment and sentiment.get("score", 0) > 0) else "SELL"
            logger.info(f"Trade signal: {signal} with confidence {combined_confidence:.2f}")
            
            # 5. Execute trade (if trading enabled)
            if settings.TRADING_ENABLED:
                await self._execute_trade(signal, combined_confidence)
        else:
            logger.info(f"No trade - confidence {combined_confidence:.2f} < threshold {settings.CONFIDENCE_THRESHOLD}")
        
        self.last_signal_time = datetime.now(timezone.utc)
    
    async def _get_sentiment(self) -> Optional[Dict]:
        """Get market sentiment from Dify"""
        if not settings.DIFY_API_KEY or not settings.DIFY_API_URL:
            logger.warning("Dify not configured, skipping sentiment")
            return None
        
        try:
            # Get recent crypto news (simplified - in production use news API)
            news_text = f"Analyze current market sentiment for {self.current_symbol}. Consider recent price action, market trends, and general crypto market conditions."
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{settings.DIFY_API_URL}/chat-messages",
                    headers={
                        "Authorization": f"Bearer {settings.DIFY_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": {},
                        "query": news_text,
                        "response_mode": "blocking",
                        "user": "trading-bot"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    
                    # Parse JSON from response
                    try:
                        # Try to extract JSON from the answer
                        if "{" in answer and "}" in answer:
                            json_start = answer.index("{")
                            json_end = answer.rindex("}") + 1
                            json_str = answer[json_start:json_end]
                            return json.loads(json_str)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Failed to parse sentiment JSON: {e}")
                        return {"sentiment": "neutral", "score": 0, "confidence": 0.5}
                else:
                    logger.error(f"Dify API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
        
        return None
    
    async def _get_ai_prediction(self) -> float:
        """Get prediction from trained AI models"""
        # In production, this would load and run the trained PPO/LSTM models
        # For now, return a placeholder
        import random
        return random.uniform(0.5, 0.9)
    
    def _combine_signals(self, sentiment: Optional[Dict], ai_confidence: float) -> float:
        """Combine sentiment and AI signals"""
        if sentiment is None:
            return ai_confidence * 0.8  # Reduce confidence if no sentiment
        
        sentiment_conf = sentiment.get("confidence", 0.5)
        sentiment_score = abs(sentiment.get("score", 0))
        
        # Weighted average: 60% AI, 40% sentiment
        combined = (ai_confidence * 0.6) + (sentiment_conf * sentiment_score * 0.4)
        return min(combined, 1.0)
    
    async def _execute_trade(self, signal: str, confidence: float):
        """Execute trade on exchange"""
        logger.info(f"Executing {signal} trade with confidence {confidence:.2f}")
        
        try:
            from services.exchange.binance import BinanceExchange
            from db.session import async_session
            from db.repositories.trade import TradeRepository
            
            exchange = BinanceExchange()
            
            # Calculate position size based on confidence
            position_size = settings.MAX_POSITION_SIZE_USDT * min(confidence, 1.0)
            
            if signal == "BUY":
                # Get current price
                ticker = await exchange.fetch_ticker(self.current_symbol)
                price = ticker["last"]
                quantity = position_size / price
                
                # Place order
                order = await exchange.create_order(
                    symbol=self.current_symbol,
                    order_type="market",
                    side="buy",
                    quantity=quantity
                )
                
                logger.info(f"BUY order placed: {order}")
                
                # Record trade in database
                async with async_session() as db:
                    trade_repo = TradeRepository(db)
                    await trade_repo.create(
                        user_id=1,  # Default user for bot trades
                        symbol=self.current_symbol,
                        side="buy",
                        quantity=quantity,
                        price=price,
                        order_id=order.get("id"),
                        ai_confidence=confidence,
                        sentiment_score=self.last_sentiment.get("score") if self.last_sentiment else None
                    )
                    await db.commit()
                    
            elif signal == "SELL":
                # Similar logic for sell
                pass
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")


# Global trading engine instance
trading_engine = TradingEngine()


async def start_trading_engine():
    """Start the trading engine in background"""
    await trading_engine.start()


def stop_trading_engine():
    """Stop the trading engine"""
    trading_engine.stop()
