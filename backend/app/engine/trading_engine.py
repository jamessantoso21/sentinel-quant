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
        """Single trading cycle with Sentiment + Technical confluence"""
        logger.info(f"Running trading cycle for {self.current_symbol}")
        
        # Import activity logger
        from api.v1.endpoints.bot import add_activity, bot_state
        
        # 1. Get market sentiment from Dify
        sentiment = await self._get_sentiment()
        if sentiment:
            self.last_sentiment = sentiment
            bot_state["current_sentiment"] = sentiment.get("score", 0.0)
            bot_state["current_confidence"] = sentiment.get("confidence", 0.5)
            add_activity(
                "SENTIMENT_ANALYSIS",
                f"Dify: {sentiment.get('sentiment')} (score: {sentiment.get('score', 0):.2f}, confidence: {sentiment.get('confidence', 0):.2f})",
                traded=False
            )
            logger.info(f"Sentiment: {sentiment}")
        else:
            add_activity("SENTIMENT_ANALYSIS", "Failed to get sentiment from Dify", traded=False)
        
        # 2. Get Technical Analysis (RSI, BB, Trend)
        from engine.technical_analyzer import technical_analyzer
        technical = await technical_analyzer.analyze(self.current_symbol)
        
        if technical:
            add_activity(
                "TECHNICAL_ANALYSIS",
                f"RSI: {technical.rsi:.1f} ({technical.rsi_signal}), BB: {technical.bb_position:.2f} ({technical.bb_signal}), Trend: {technical.trend}",
                traded=False
            )
            logger.info(f"Technical: RSI={technical.rsi:.1f}, BB={technical.bb_position:.2f}, Signal={technical.overall_signal}")
            bot_state["technical_signal"] = technical.overall_signal
            bot_state["rsi"] = technical.rsi
        else:
            add_activity("TECHNICAL_ANALYSIS", "Failed to get technical data", traded=False)
        
        # 3. Enhanced Decision Logic (Confluence)
        decision = self._make_enhanced_decision(sentiment, technical)
        
        action = decision["action"]
        confidence = decision["confidence"]
        reasoning = decision["reasoning"]
        market_condition = decision["market_condition"]
        
        bot_state["last_signal"] = action
        bot_state["market_condition"] = market_condition
        
        # 4. Execute based on decision
        if action in ["BUY", "SELL"] and confidence >= settings.CONFIDENCE_THRESHOLD:
            logger.info(f"Trade signal: {action} with confidence {confidence:.2f}")
            
            if settings.TRADING_ENABLED:
                add_activity(
                    f"TRADE_{action}",
                    f"{market_condition}: {reasoning} (confidence: {confidence:.2f})",
                    traded=True
                )
                await self._execute_trade(action, confidence)
            else:
                add_activity(
                    "TRADE_SKIPPED",
                    f"Trading disabled. {market_condition}: {reasoning}",
                    traded=False
                )
        else:
            add_activity(
                "NO_TRADE",
                f"{market_condition}: {reasoning} (confidence: {confidence:.2f} vs threshold: {settings.CONFIDENCE_THRESHOLD})",
                traded=False
            )
            logger.info(f"No trade - {reasoning}")
        
        self.last_signal_time = datetime.now(timezone.utc)
        bot_state["last_signal_time"] = self.last_signal_time.isoformat()
    
    def _make_enhanced_decision(self, sentiment: Optional[Dict], technical) -> Dict:
        """
        Confluence-based decision making.
        
        IDEAL_BUY: Bullish sentiment + Oversold (RSI<30)
        IDEAL_SELL: Bearish sentiment + Overbought (RSI>70)
        RISKY: Sentiment and technical disagree
        """
        # Default values
        if not sentiment and not technical:
            return {
                "action": "HOLD",
                "confidence": 0.3,
                "reasoning": "No data available",
                "market_condition": "NO_DATA"
            }
        
        # Extract sentiment
        sent_label = sentiment.get("sentiment", "neutral") if sentiment else "neutral"
        sent_score = sentiment.get("score", 0) if sentiment else 0
        sent_conf = sentiment.get("confidence", 0.5) if sentiment else 0.5
        
        is_bullish = sent_label.lower() == "bullish" or sent_score > 0.3
        is_bearish = sent_label.lower() == "bearish" or sent_score < -0.3
        
        # Extract technical
        if technical:
            rsi = technical.rsi
            is_oversold = rsi < 35 or technical.rsi_signal == "OVERSOLD"
            is_overbought = rsi > 65 or technical.rsi_signal == "OVERBOUGHT"
            tech_signal = technical.overall_signal
            tech_conf = technical.confluence_score
        else:
            rsi = 50
            is_oversold = False
            is_overbought = False
            tech_signal = "NEUTRAL"
            tech_conf = 0.5
        
        # ========== CONFLUENCE DECISION MATRIX ==========
        
        # IDEAL BUY: Bullish + Oversold = Buy the dip!
        if is_bullish and is_oversold:
            return {
                "action": "BUY",
                "confidence": min((sent_conf + tech_conf) / 2 * 1.3, 1.0),
                "reasoning": f"CONFLUENCE: Bullish sentiment + Oversold (RSI:{rsi:.0f}). Buy the dip!",
                "market_condition": "IDEAL_BUY"
            }
        
        # IDEAL SELL: Bearish + Overbought = Sell the top!
        if is_bearish and is_overbought:
            return {
                "action": "SELL",
                "confidence": min((sent_conf + tech_conf) / 2 * 1.3, 1.0),
                "reasoning": f"CONFLUENCE: Bearish sentiment + Overbought (RSI:{rsi:.0f}). Sell the top!",
                "market_condition": "IDEAL_SELL"
            }
        
        # RISKY: Bullish + Overbought = Don't chase!
        if is_bullish and is_overbought:
            return {
                "action": "HOLD",
                "confidence": 0.4,
                "reasoning": f"RISKY: Bullish news but OVERBOUGHT (RSI:{rsi:.0f}). Don't chase the pump!",
                "market_condition": "RISKY_OVERBOUGHT"
            }
        
        # RISKY: Bearish + Oversold = Could bounce!
        if is_bearish and is_oversold:
            return {
                "action": "HOLD",
                "confidence": 0.4,
                "reasoning": f"RISKY: Bearish news but OVERSOLD (RSI:{rsi:.0f}). Could bounce or capitulate.",
                "market_condition": "RISKY_OVERSOLD"
            }
        
        # MODERATE: Sentiment aligns with technical
        if is_bullish and tech_signal == "BUY":
            return {
                "action": "BUY",
                "confidence": (sent_conf + tech_conf) / 2,
                "reasoning": f"MODERATE: Bullish sentiment + Technical BUY (RSI:{rsi:.0f})",
                "market_condition": "MODERATE_BUY"
            }
        
        if is_bearish and tech_signal == "SELL":
            return {
                "action": "SELL",
                "confidence": (sent_conf + tech_conf) / 2,
                "reasoning": f"MODERATE: Bearish sentiment + Technical SELL (RSI:{rsi:.0f})",
                "market_condition": "MODERATE_SELL"
            }
        
        # NEUTRAL: No clear confluence
        return {
            "action": "HOLD",
            "confidence": 0.5,
            "reasoning": f"NO CONFLUENCE: Sentiment={sent_label}, Technical={tech_signal}, RSI={rsi:.0f}",
            "market_condition": "NEUTRAL"
        }
    
    async def _get_sentiment(self) -> Optional[Dict]:
        """Get market sentiment from Dify"""
        if not settings.DIFY_API_KEY or not settings.DIFY_API_URL:
            logger.warning(f"Dify not configured. API_KEY: {bool(settings.DIFY_API_KEY)}, API_URL: {bool(settings.DIFY_API_URL)}")
            return None
        
        try:
            # Market analysis text
            news_text = f"Current market conditions for {self.current_symbol}: The crypto market is showing mixed signals with Bitcoin consolidating near recent highs. Institutional interest remains strong with continued ETF inflows. Market sentiment appears cautiously optimistic with traders watching for the next major catalyst."
            
            url = f"{settings.DIFY_API_URL}/chat-messages".strip()
            logger.info(f"Calling Dify API: {url}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {settings.DIFY_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": {"text": news_text},
                        "query": "Analyze this market news",
                        "response_mode": "blocking",
                        "user": "trading-bot"
                    }
                )
                
                logger.info(f"Dify response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    logger.info(f"Dify answer: {answer[:200]}...")
                    
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
                    logger.error(f"Dify API error: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {type(e).__name__}: {e}")
        
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
