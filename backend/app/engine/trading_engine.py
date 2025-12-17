"""
Sentinel Quant - Trading Engine
Background trading loop with Dify sentiment analysis
Multi-symbol support: BTC, PAXG (Gold)
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import httpx
import json

from core.config import settings

logger = logging.getLogger(__name__)

# Supported trading pairs
TRADING_SYMBOLS = [
    "BTC/USDT",   # Bitcoin
    "PAXG/USDT",  # Pax Gold (Gold-backed token)
]

# TP/SL Configuration (from backtest optimization)
STOP_LOSS_PERCENT = 0.02    # 2% stop loss
TAKE_PROFIT_PERCENT = 0.04  # 4% take profit
POSITION_SIZE_PERCENT = 0.10  # 10% of capital per trade


class TradingEngine:
    """Main trading engine that runs the trading loop"""
    
    def __init__(self):
        self.is_running = False
        self.symbols = TRADING_SYMBOLS
        self.current_symbol_index = 0
        self.current_symbol = self.symbols[0]
        self.last_signal_time: Optional[datetime] = None
        self.last_sentiment: Optional[Dict] = None
        self.last_ai_prediction: Optional[float] = None
        
        # Position tracking for TP/SL
        self.positions: Dict[str, Dict] = {}  # {symbol: {side, entry_price, entry_time}}
        
    async def start(self):
        """Start the trading loop"""
        if self.is_running:
            logger.warning("Trading engine already running")
            return
            
        self.is_running = True
        logger.info(f"Trading engine started - Symbols: {self.symbols}")
        
        while self.is_running:
            try:
                # Cycle through all symbols
                for symbol in self.symbols:
                    self.current_symbol = symbol
                    await self._trading_cycle()
                    await asyncio.sleep(10)  # Small delay between symbols
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
            
            # Wait 5 minutes between full cycles
            await asyncio.sleep(290)  # 290 + 10*2 = 310 seconds total
    
    def stop(self):
        """Stop the trading loop"""
        self.is_running = False
        logger.info("Trading engine stopped")
    
    async def _trading_cycle(self):
        """Single trading cycle with Sentiment + Technical confluence + SMA validation"""
        logger.info(f"Running trading cycle for {self.current_symbol}")
        
        # Import activity logger
        from api.v1.endpoints.bot import add_activity, bot_state
        from engine.sentiment_tracker import sentiment_tracker
        
        # 0. Check existing positions for TP/SL
        if self.current_symbol in self.positions:
            current_price = await self._get_current_price(self.current_symbol)
            if current_price:
                result = await self._check_tp_sl(self.current_symbol, current_price)
                if result:
                    logger.info(f"{self.current_symbol} position closed via {result}")
        
        # 1. Get market sentiment from Dify
        sentiment = await self._get_sentiment()
        if sentiment:
            self.last_sentiment = sentiment
            bot_state["current_sentiment"] = sentiment.get("score", 0.0)
            bot_state["current_confidence"] = sentiment.get("confidence", 0.5)
            
            # Record to sentiment history for SMA
            symbol = self.current_symbol.split("/")[0]
            sentiment_tracker.record(
                sentiment=sentiment.get("sentiment", "neutral"),
                score=sentiment.get("score", 0),
                confidence=sentiment.get("confidence", 0.5),
                symbol=symbol
            )
            
            # Check SMA - validate against history to detect fake pumps
            trust_check = sentiment_tracker.should_trust_current_sentiment(
                current_sentiment=sentiment.get("sentiment", "neutral"),
                current_score=sentiment.get("score", 0),
                period_hours=6
            )
            
            sma = trust_check.get("sma")
            sma_info = f"SMA-6h: {sma.avg_score:.2f} ({sma.data_points}pts)" if sma else "SMA: N/A"
            trust_info = f"Trust: {trust_check['trust_score']:.0%}"
            
            add_activity(
                "SENTIMENT_ANALYSIS",
                f"Dify: {sentiment.get('sentiment')} (score: {sentiment.get('score', 0):.2f}). {sma_info}. {trust_info}",
                traded=False
            )
            
            # Store trust info in bot_state
            bot_state["sentiment_trust"] = trust_check["trust_score"]
            bot_state["sentiment_sma"] = sma.avg_score if sma else None
            
            # Log warning if fake pump detected
            if not trust_check["trust"]:
                logger.warning(f"FAKE PUMP WARNING: {trust_check['reason']}")
                add_activity("SENTIMENT_WARNING", trust_check["reason"], traded=False)
            
            logger.info(f"Sentiment: {sentiment}. {trust_check['reason']}")
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
        
        # 3. Multi-Model Voting (4 voters)
        from engine.multi_voter import multi_voter
        
        voting_result = multi_voter.vote(
            sentiment_data=sentiment,
            technical_data=technical,
            lstm_prediction=None,  # Not trained yet
            ppo_action=None        # Not trained yet
        )
        
        # Log voting result
        add_activity(
            "VOTING",
            f"Result: {voting_result.final_action} ({voting_result.consensus_level:.0%} consensus, {voting_result.active_voters}/{voting_result.total_voters} active)",
            traded=False
        )
        logger.info(f"Voting: {voting_result.reasoning}")
        
        action = voting_result.final_action
        confidence = voting_result.consensus_level
        reasoning = voting_result.reasoning
        market_condition = f"VOTE_{voting_result.final_action}"
        
        # Store in bot_state with individual voter details
        voters = []
        for vote in voting_result.individual_votes:
            voters.append({
                "name": vote.voter_name,
                "vote": vote.vote.value,
                "confidence": vote.confidence,
                "reasoning": vote.reasoning,
                "active": vote.is_active
            })
        
        bot_state["voting_result"] = {
            "action": action,
            "consensus": voting_result.consensus_level,
            "active_voters": voting_result.active_voters,
            "total_voters": voting_result.total_voters,
            "buy_votes": voting_result.buy_votes,
            "sell_votes": voting_result.sell_votes,
            "hold_votes": voting_result.hold_votes,
            "should_trade": voting_result.should_trade,
            "voters": voters
        }
        
        bot_state["last_signal"] = action
        bot_state["market_condition"] = market_condition
        
        # 4. Execute based on voting result
        if voting_result.should_trade and action in ["BUY", "SELL"]:
            # Skip if already have position in this symbol
            if self.current_symbol in self.positions:
                logger.info(f"Already have {self.positions[self.current_symbol]['side']} position in {self.current_symbol}, skipping new entry")
            else:
                logger.info(f"Trade signal: {action} with consensus {confidence:.0%}")
                
                if settings.TRADING_ENABLED:
                    add_activity(
                        f"TRADE_{action}",
                        f"Voting: {voting_result.buy_votes}B/{voting_result.sell_votes}S/{voting_result.hold_votes}H | {reasoning}",
                        traded=True
                    )
                    
                    # Get entry price and record position with TP/SL
                    entry_price = await self._get_current_price(self.current_symbol)
                    if entry_price:
                        self._open_position(self.current_symbol, action, entry_price)
                    
                    await self._execute_trade(action, confidence)
                else:
                    add_activity(
                        "TRADE_SKIPPED",
                        f"Trading disabled. Voting: {voting_result.buy_votes}B/{voting_result.sell_votes}S/{voting_result.hold_votes}H",
                        traded=False
                    )
        else:
            add_activity(
                "NO_TRADE",
                f"Voting: {voting_result.buy_votes}B/{voting_result.sell_votes}S/{voting_result.hold_votes}H (consensus: {confidence:.0%}) | {reasoning}",
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
        """Get market sentiment from Dify using real news"""
        if not settings.DIFY_API_KEY or not settings.DIFY_API_URL:
            logger.warning(f"Dify not configured. API_KEY: {bool(settings.DIFY_API_KEY)}, API_URL: {bool(settings.DIFY_API_URL)}")
            return None
        
        try:
            # Fetch real news from CryptoCompare
            from engine.news_fetcher import news_fetcher
            
            # Extract symbol (BTC from BTC/USDT)
            symbol = self.current_symbol.split("/")[0]
            
            news_items = await news_fetcher.fetch_news(symbol, limit=5)
            news_text = news_fetcher.format_for_dify(news_items, symbol)
            
            if news_items:
                logger.info(f"Fetched {len(news_items)} news articles for {symbol}")
                # Log first headline
                logger.info(f"Latest headline: {news_items[0].title[:80]}...")
            else:
                logger.warning("No news found, using fallback text")
            
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
    
    async def _check_tp_sl(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if any open position should be closed due to TP or SL.
        Returns: 'TP', 'SL', or None
        """
        from api.v1.endpoints.bot import add_activity
        
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        side = position['side']
        
        # Calculate P&L percentage
        if side == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SELL
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check TP/SL
        if pnl_pct <= -STOP_LOSS_PERCENT:
            logger.info(f"STOP LOSS triggered for {symbol}: {pnl_pct*100:.2f}%")
            add_activity(
                "STOP_LOSS",
                f"{symbol} closed at ${current_price:.2f} | Loss: {pnl_pct*100:.2f}%",
                traded=True
            )
            del self.positions[symbol]
            return 'SL'
        
        elif pnl_pct >= TAKE_PROFIT_PERCENT:
            logger.info(f"TAKE PROFIT triggered for {symbol}: {pnl_pct*100:.2f}%")
            add_activity(
                "TAKE_PROFIT",
                f"{symbol} closed at ${current_price:.2f} | Profit: {pnl_pct*100:.2f}%",
                traded=True
            )
            del self.positions[symbol]
            return 'TP'
        
        else:
            logger.debug(f"Position {symbol}: P&L = {pnl_pct*100:.2f}% (TP: +{TAKE_PROFIT_PERCENT*100}%, SL: -{STOP_LOSS_PERCENT*100}%)")
            return None
    
    def _open_position(self, symbol: str, side: str, entry_price: float):
        """Record a new position"""
        from api.v1.endpoints.bot import add_activity
        
        self.positions[symbol] = {
            'side': side,
            'entry_price': entry_price,
            'entry_time': datetime.now(timezone.utc),
            'tp_price': entry_price * (1 + TAKE_PROFIT_PERCENT) if side == 'BUY' else entry_price * (1 - TAKE_PROFIT_PERCENT),
            'sl_price': entry_price * (1 - STOP_LOSS_PERCENT) if side == 'BUY' else entry_price * (1 + STOP_LOSS_PERCENT),
        }
        
        pos = self.positions[symbol]
        logger.info(f"Opened {side} position for {symbol} at ${entry_price:.2f} | TP: ${pos['tp_price']:.2f} | SL: ${pos['sl_price']:.2f}")
        add_activity(
            f"OPEN_{side}",
            f"{symbol} at ${entry_price:.2f} | TP: ${pos['tp_price']:.2f} | SL: ${pos['sl_price']:.2f}",
            traded=True
        )
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from CoinGecko"""
        try:
            coin_map = {
                "BTC/USDT": "bitcoin",
                "PAXG/USDT": "pax-gold",
                "ETH/USDT": "ethereum",
            }
            coin_id = coin_map.get(symbol, "bitcoin")
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://api.coingecko.com/api/v3/simple/price",
                    params={"ids": coin_id, "vs_currencies": "usd"}
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get(coin_id, {}).get('usd')
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
        return None


# Global trading engine instance
trading_engine = TradingEngine()


async def start_trading_engine():
    """Start the trading engine in background"""
    await trading_engine.start()


def stop_trading_engine():
    """Stop the trading engine"""
    trading_engine.stop()
