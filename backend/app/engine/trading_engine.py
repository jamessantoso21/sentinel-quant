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

# Supported trading pairs - Only trend-following coins that beat B&H
TRADING_SYMBOLS = [
    "SOL/USDT",   # Solana - uses optimized Trend Following (+303% backtested)
    "MATIC/USDT", # Polygon - uses optimized Trend Following (+3084% backtested)
    "DOGE/USDT",  # Dogecoin - uses optimized Trend Following (+33723% backtested)
    "ADA/USDT",   # Cardano - uses optimized Trend Following (+1195% backtested)
    # BTC removed: B&H (+1803%) beats all trend strategies (+61% to +157%)
    # PAXG removed: Gold-backed token, not suitable for trend trading
]

# Per-Asset Configuration - All use trend engines
ASSET_SETTINGS = {
    "SOL/USDT": {
        "stop_loss": None,       # Trend engine handles exits
        "take_profit": None,     # Trend engine handles exits
        "use_trend_engine": True,  # Use optimized trend following (+303%)
    },
    "MATIC/USDT": {
        "stop_loss": None,       # Trend engine handles exits
        "take_profit": None,     # Trend engine handles exits
        "use_trend_engine": True,  # Use optimized trend following (+3084%)
    },
    "DOGE/USDT": {
        "stop_loss": None,       # Trend engine handles exits
        "take_profit": None,     # Trend engine handles exits
        "use_trend_engine": True,  # Use optimized trend following (+33723%)
    },
    "ADA/USDT": {
        "stop_loss": None,       # Trend engine handles exits
        "take_profit": None,     # Trend engine handles exits  
        "use_trend_engine": True,  # Use optimized trend following (+1195%)
    },
}

# Default settings (fallback)
DEFAULT_STOP_LOSS = 0.02
DEFAULT_TAKE_PROFIT = 0.04
POSITION_SIZE_PERCENT = 0.10  # 10% of capital per trade


def get_asset_settings(symbol: str) -> dict:
    """Get optimized settings for a specific asset"""
    return ASSET_SETTINGS.get(symbol, {
        "stop_loss": DEFAULT_STOP_LOSS,
        "take_profit": DEFAULT_TAKE_PROFIT,
        "consensus": 0.25,
    })


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
        
        # SOL Trend Engine (+303% backtested)
        self.sol_trend_engine = None
        self.sol_entry_price = 0.0
        self.sol_in_position = False
        self._init_sol_engine()
        
        # MATIC Trend Engine (+3084% backtested)
        self.matic_trend_engine = None
        self.matic_entry_price = 0.0
        self.matic_in_position = False
        self._init_matic_engine()
        
        # DOGE Trend Engine (+33723% backtested)
        self.doge_trend_engine = None
        self.doge_entry_price = 0.0
        self.doge_in_position = False
        self._init_doge_engine()
        
        # ADA Trend Engine (+1195% backtested)
        self.ada_trend_engine = None
        self.ada_entry_price = 0.0
        self.ada_in_position = False
        self._init_ada_engine()
        
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
        
        # Get asset settings
        asset_settings = get_asset_settings(self.current_symbol)
        
        # ========== Trend Engine Coins (bypass consensus) ==========
        if asset_settings.get("use_trend_engine"):
            if self.current_symbol == "SOL/USDT":
                await self._sol_trend_cycle(add_activity, bot_state)
            elif self.current_symbol == "MATIC/USDT":
                await self._matic_trend_cycle(add_activity, bot_state)
            elif self.current_symbol == "DOGE/USDT":
                await self._doge_trend_cycle(add_activity, bot_state)
            elif self.current_symbol == "ADA/USDT":
                await self._ada_trend_cycle(add_activity, bot_state)
            return  # Trend coins use their own logic, skip consensus voting
        
        # ========== BTC/PAXG: Use Consensus Voting ==========
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
        """Execute trade (paper trading mode - saves to database without real exchange)"""
        logger.info(f"Executing {signal} trade with confidence {confidence:.2f}")
        
        try:
            from db.session import AsyncSessionLocal
            from db.repositories.trade import TradeRepository
            
            # Get current price from our cached method
            price = await self._get_current_price(self.current_symbol)
            if not price:
                logger.error("Could not get current price for trade")
                return
            
            # Calculate position size based on confidence
            position_size = settings.MAX_POSITION_SIZE_USDT * min(confidence, 1.0)
            quantity = position_size / price
            
            # Generate simulated order ID
            import uuid
            order_id = f"PAPER_{uuid.uuid4().hex[:12].upper()}"
            
            logger.info(f"PAPER TRADE: {signal} {quantity:.6f} {self.current_symbol} @ ${price:.2f}")
            
            # Get per-asset settings for SL/TP
            settings_asset = get_asset_settings(self.current_symbol)
            sl_price = price * (1 - settings_asset['stop_loss']) if signal == "BUY" else price * (1 + settings_asset['stop_loss'])
            tp_price = price * (1 + settings_asset['take_profit']) if signal == "BUY" else price * (1 - settings_asset['take_profit'])
            
            # Record trade in database
            async with AsyncSessionLocal() as db:
                trade_repo = TradeRepository(db)
                await trade_repo.create(
                    user_id=1,  # Default user for bot trades
                    symbol=self.current_symbol,
                    direction="LONG" if signal == "BUY" else "SHORT",
                    status="EXECUTED",
                    entry_price=price,
                    quantity=quantity,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    ai_confidence=confidence,
                    sentiment_score=self.last_sentiment.get("score") if self.last_sentiment else None,
                    exchange="paper",
                    exchange_order_id=order_id,
                    notes=f"Paper trade via bot"
                )
                await db.commit()
                logger.info(f"Trade saved to database: {order_id}")
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            import traceback
            traceback.print_exc()
    
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
        
        # Get per-asset settings
        settings = get_asset_settings(symbol)
        stop_loss = settings['stop_loss']
        take_profit = settings['take_profit']
        
        # Calculate P&L percentage
        if side == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SELL
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check TP/SL
        if pnl_pct <= -stop_loss:
            logger.info(f"STOP LOSS triggered for {symbol}: {pnl_pct*100:.2f}%")
            add_activity(
                "STOP_LOSS",
                f"{symbol} closed at ${current_price:.2f} | Loss: {pnl_pct*100:.2f}%",
                traded=True
            )
            del self.positions[symbol]
            return 'SL'
        
        elif pnl_pct >= take_profit:
            logger.info(f"TAKE PROFIT triggered for {symbol}: {pnl_pct*100:.2f}%")
            add_activity(
                "TAKE_PROFIT",
                f"{symbol} closed at ${current_price:.2f} | Profit: {pnl_pct*100:.2f}%",
                traded=True
            )
            del self.positions[symbol]
            return 'TP'
        
        else:
            logger.debug(f"Position {symbol}: P&L = {pnl_pct*100:.2f}% (TP: +{take_profit*100}%, SL: -{stop_loss*100}%)")
            return None
    
    def _open_position(self, symbol: str, side: str, entry_price: float):
        """Record a new position with per-asset TP/SL"""
        from api.v1.endpoints.bot import add_activity
        
        # Get per-asset settings
        settings = get_asset_settings(symbol)
        stop_loss = settings['stop_loss']
        take_profit = settings['take_profit']
        
        self.positions[symbol] = {
            'side': side,
            'entry_price': entry_price,
            'entry_time': datetime.now(timezone.utc),
            'tp_price': entry_price * (1 + take_profit) if side == 'BUY' else entry_price * (1 - take_profit),
            'sl_price': entry_price * (1 - stop_loss) if side == 'BUY' else entry_price * (1 + stop_loss),
        }
        
        pos = self.positions[symbol]
        logger.info(f"Opened {side} position for {symbol} at ${entry_price:.2f} | TP: ${pos['tp_price']:.2f} | SL: ${pos['sl_price']:.2f}")
        add_activity(
            f"OPEN_{side}",
            f"{symbol} at ${entry_price:.2f} | TP: ${pos['tp_price']:.2f} | SL: ${pos['sl_price']:.2f}",
            traded=True
        )
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from CryptoCompare (primary) or CoinGecko (fallback)."""
        crypto_compare_map = {
            "SOL/USDT": "SOL",
            "MATIC/USDT": "MATIC", 
            "DOGE/USDT": "DOGE",
            "ADA/USDT": "ADA",
            "BTC/USDT": "BTC",
            "PAXG/USDT": "PAXG",
            "ETH/USDT": "ETH",
        }
        coingecko_map = {
            "SOL/USDT": "solana",
            "MATIC/USDT": "matic-network",
            "DOGE/USDT": "dogecoin",
            "ADA/USDT": "cardano",
            "BTC/USDT": "bitcoin",
            "PAXG/USDT": "pax-gold",
            "ETH/USDT": "ethereum",
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try CryptoCompare first (more generous rate limits)
                cc_symbol = crypto_compare_map.get(symbol)
                if cc_symbol:
                    try:
                        response = await client.get(
                            "https://min-api.cryptocompare.com/data/price",
                            params={"fsym": cc_symbol, "tsyms": "USD"}
                        )
                        if response.status_code == 200:
                            data = response.json()
                            price = data.get('USD')
                            if price:
                                return float(price)
                    except Exception:
                        pass
                
                # Fallback to CoinGecko
                coin_id = coingecko_map.get(symbol)
                if coin_id:
                    response = await client.get(
                        "https://api.coingecko.com/api/v3/simple/price",
                        params={"ids": coin_id, "vs_currencies": "usd"}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return data.get(coin_id, {}).get('usd')
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
        return None
    
    def _init_sol_engine(self):
        """Initialize SOL Trend Engine"""
        try:
            from engine.sol_trend_engine import SOLTrendEngine
            self.sol_trend_engine = SOLTrendEngine()
            logger.info("SOL Trend Engine initialized (+303% backtested)")
        except ImportError as e:
            logger.warning(f"Could not load SOL Trend Engine: {e}")
            self.sol_trend_engine = None
    
    async def _sol_trend_cycle(self, add_activity, bot_state):
        """
        SOL trading using optimized Trend Following strategy.
        Bypasses consensus voting for +303% backtested performance.
        """
        from engine.technical_analyzer import technical_analyzer
        
        if not self.sol_trend_engine:
            logger.warning("SOL Trend Engine not available, skipping")
            return
        
        # Get current price
        current_price = await self._get_current_price("SOL/USDT")
        if not current_price:
            add_activity("SOL_TREND", "Could not get SOL price", traded=False)
            return
        
        # Get technical data for indicators
        technical = await technical_analyzer.analyze("SOL/USDT")
        
        # Prepare data for trend engine
        sma10 = technical.sma_10 if technical and hasattr(technical, 'sma_10') else current_price
        sma20 = technical.sma_20 if technical and hasattr(technical, 'sma_20') else current_price
        sma50 = technical.sma_50 if technical and hasattr(technical, 'sma_50') else current_price
        rsi = technical.rsi if technical else 50
        high_20d = current_price * 1.1  # Approximate
        
        # Update engine position state
        self.sol_trend_engine.update_position(self.sol_entry_price, self.sol_in_position)
        
        # Get signal from trend engine
        signal = self.sol_trend_engine.get_signal(
            current_price=current_price,
            sma10=sma10,
            sma20=sma20,
            sma50=sma50,
            rsi=rsi,
            high_20d=high_20d,
            momentum_7d=0.0
        )
        
        # Log current state
        trend_info = f"Trend: {signal.trend.value} | RSI: {rsi:.0f}"
        add_activity("SOL_TREND", f"${current_price:.2f} | {trend_info} | Signal: {signal.action.value}", traded=False)
        bot_state["sol_trend"] = signal.trend.value
        bot_state["sol_action"] = signal.action.value
        
        # Execute trade based on signal
        if signal.action.value == "BUY" and not self.sol_in_position:
            if settings.TRADING_ENABLED:
                add_activity("SOL_BUY", f"{signal.reason} @ ${current_price:.2f}", traded=True)
                self.sol_entry_price = current_price
                self.sol_in_position = True
                
                # Update bot_state for monitoring
                bot_state["sol_in_position"] = True
                bot_state["sol_entry_price"] = current_price
                bot_state["sol_pnl"] = 0.0
                
                # Record position
                self.positions["SOL/USDT"] = {
                    'side': 'BUY',
                    'entry_price': current_price,
                    'entry_time': datetime.now(timezone.utc),
                    'tp_price': None,  # Trend engine handles exits
                    'sl_price': None,
                }
                
                await self._execute_trade("BUY", signal.confidence)
                logger.info(f"SOL BUY executed: {signal.reason}")
            else:
                add_activity("SOL_BUY_SKIPPED", f"Trading disabled: {signal.reason}", traded=False)
        
        elif signal.action.value == "SELL" and self.sol_in_position:
            if settings.TRADING_ENABLED:
                pnl = (current_price - self.sol_entry_price) / self.sol_entry_price * 100
                add_activity("SOL_SELL", f"{signal.reason} @ ${current_price:.2f} (PnL: {pnl:+.1f}%)", traded=True)
                self.sol_entry_price = 0.0
                self.sol_in_position = False
                
                # Update bot_state for monitoring
                bot_state["sol_in_position"] = False
                bot_state["sol_entry_price"] = None
                bot_state["sol_pnl"] = None
                
                # Clear position
                if "SOL/USDT" in self.positions:
                    del self.positions["SOL/USDT"]
                
                await self._execute_trade("SELL", signal.confidence)
                logger.info(f"SOL SELL executed: {signal.reason} (PnL: {pnl:+.1f}%)")
            else:
                add_activity("SOL_SELL_SKIPPED", f"Trading disabled: {signal.reason}", traded=False)
        
        else:
            # HOLD - update PnL if in position
            if self.sol_in_position and self.sol_entry_price > 0:
                pnl = (current_price - self.sol_entry_price) / self.sol_entry_price * 100
                bot_state["sol_pnl"] = round(pnl, 2)
            logger.debug(f"SOL HOLD: {signal.reason}")
    
    def _init_matic_engine(self):
        """Initialize MATIC Trend Engine"""
        try:
            from engine.matic_trend_engine import MATICTrendEngine
            self.matic_trend_engine = MATICTrendEngine()
            logger.info("MATIC Trend Engine initialized (+3084% backtested)")
        except ImportError as e:
            logger.warning(f"Could not load MATIC Trend Engine: {e}")
            self.matic_trend_engine = None
    
    async def _matic_trend_cycle(self, add_activity, bot_state):
        """
        MATIC trading using optimized Trend Following strategy.
        Bypasses consensus voting for +3084% backtested performance.
        """
        from engine.technical_analyzer import technical_analyzer
        
        if not self.matic_trend_engine:
            logger.warning("MATIC Trend Engine not available, skipping")
            return
        
        # Get current price
        current_price = await self._get_current_price("MATIC/USDT")
        if not current_price:
            add_activity("MATIC_TREND", "Could not get MATIC price", traded=False)
            return
        
        # Get technical data for indicators
        technical = await technical_analyzer.analyze("MATIC/USDT")
        
        # Prepare data for trend engine
        sma10 = technical.sma_10 if technical and hasattr(technical, 'sma_10') else current_price
        sma20 = technical.sma_20 if technical and hasattr(technical, 'sma_20') else current_price
        sma50 = technical.sma_50 if technical and hasattr(technical, 'sma_50') else current_price
        rsi = technical.rsi if technical else 50
        high_20d = current_price * 1.1  # Approximate
        
        # Update engine position state
        self.matic_trend_engine.update_position(self.matic_entry_price, self.matic_in_position)
        
        # Get signal from trend engine
        signal = self.matic_trend_engine.get_signal(
            current_price=current_price,
            sma10=sma10,
            sma20=sma20,
            sma50=sma50,
            rsi=rsi,
            high_20d=high_20d,
            momentum_7d=0.0
        )
        
        # Log current state
        trend_info = f"Trend: {signal.trend.value} | RSI: {rsi:.0f}"
        add_activity("MATIC_TREND", f"${current_price:.4f} | {trend_info} | Signal: {signal.action.value}", traded=False)
        bot_state["matic_trend"] = signal.trend.value
        bot_state["matic_action"] = signal.action.value
        
        # Execute trade based on signal
        if signal.action.value == "BUY" and not self.matic_in_position:
            if settings.TRADING_ENABLED:
                add_activity("MATIC_BUY", f"{signal.reason} @ ${current_price:.4f}", traded=True)
                self.matic_entry_price = current_price
                self.matic_in_position = True
                
                # Update bot_state for monitoring
                bot_state["matic_in_position"] = True
                bot_state["matic_entry_price"] = current_price
                bot_state["matic_pnl"] = 0.0
                
                # Record position
                self.positions["MATIC/USDT"] = {
                    'side': 'BUY',
                    'entry_price': current_price,
                    'entry_time': datetime.now(timezone.utc),
                    'tp_price': None,
                    'sl_price': None,
                }
                
                await self._execute_trade("BUY", signal.confidence)
                logger.info(f"MATIC BUY executed: {signal.reason}")
            else:
                add_activity("MATIC_BUY_SKIPPED", f"Trading disabled: {signal.reason}", traded=False)
        
        elif signal.action.value == "SELL" and self.matic_in_position:
            if settings.TRADING_ENABLED:
                pnl = (current_price - self.matic_entry_price) / self.matic_entry_price * 100
                add_activity("MATIC_SELL", f"{signal.reason} @ ${current_price:.4f} (PnL: {pnl:+.1f}%)", traded=True)
                self.matic_entry_price = 0.0
                self.matic_in_position = False
                
                # Update bot_state for monitoring
                bot_state["matic_in_position"] = False
                bot_state["matic_entry_price"] = None
                bot_state["matic_pnl"] = None
                
                # Clear position
                if "MATIC/USDT" in self.positions:
                    del self.positions["MATIC/USDT"]
                
                await self._execute_trade("SELL", signal.confidence)
                logger.info(f"MATIC SELL executed: {signal.reason} (PnL: {pnl:+.1f}%)")
            else:
                add_activity("MATIC_SELL_SKIPPED", f"Trading disabled: {signal.reason}", traded=False)
        
        else:
            # HOLD - update PnL if in position
            if self.matic_in_position and self.matic_entry_price > 0:
                pnl = (current_price - self.matic_entry_price) / self.matic_entry_price * 100
                bot_state["matic_pnl"] = round(pnl, 2)
            logger.debug(f"MATIC HOLD: {signal.reason}")
    
    def _init_doge_engine(self):
        """Initialize DOGE Trend Engine"""
        try:
            from engine.doge_trend_engine import DOGETrendEngine
            self.doge_trend_engine = DOGETrendEngine()
            logger.info("DOGE Trend Engine initialized (+33723% backtested)")
        except ImportError as e:
            logger.warning(f"Could not load DOGE Trend Engine: {e}")
            self.doge_trend_engine = None
    
    async def _doge_trend_cycle(self, add_activity, bot_state):
        """DOGE trading using optimized Trend Following strategy."""
        from engine.technical_analyzer import technical_analyzer
        
        if not self.doge_trend_engine:
            logger.warning("DOGE Trend Engine not available, skipping")
            return
        
        current_price = await self._get_current_price("DOGE/USDT")
        if not current_price:
            add_activity("DOGE_TREND", "Could not get DOGE price", traded=False)
            return
        
        technical = await technical_analyzer.analyze("DOGE/USDT")
        
        sma10 = technical.sma_10 if technical and hasattr(technical, 'sma_10') else current_price
        sma20 = technical.sma_20 if technical and hasattr(technical, 'sma_20') else current_price
        sma50 = technical.sma_50 if technical and hasattr(technical, 'sma_50') else current_price
        rsi = technical.rsi if technical else 50
        high_20d = current_price * 1.1
        
        self.doge_trend_engine.update_position(self.doge_entry_price, self.doge_in_position)
        
        signal = self.doge_trend_engine.get_signal(
            current_price=current_price, sma10=sma10, sma20=sma20, sma50=sma50,
            rsi=rsi, high_20d=high_20d, momentum_7d=0.0
        )
        
        trend_info = f"Trend: {signal.trend.value} | RSI: {rsi:.0f}"
        add_activity("DOGE_TREND", f"${current_price:.6f} | {trend_info} | Signal: {signal.action.value}", traded=False)
        bot_state["doge_trend"] = signal.trend.value
        bot_state["doge_action"] = signal.action.value
        
        if signal.action.value == "BUY" and not self.doge_in_position:
            if settings.TRADING_ENABLED:
                add_activity("DOGE_BUY", f"{signal.reason} @ ${current_price:.6f}", traded=True)
                self.doge_entry_price = current_price
                self.doge_in_position = True
                bot_state["doge_in_position"] = True
                bot_state["doge_entry_price"] = current_price
                bot_state["doge_pnl"] = 0.0
                self.positions["DOGE/USDT"] = {
                    'side': 'BUY', 'entry_price': current_price,
                    'entry_time': datetime.now(timezone.utc), 'tp_price': None, 'sl_price': None,
                }
                await self._execute_trade("BUY", signal.confidence)
                logger.info(f"DOGE BUY executed: {signal.reason}")
            else:
                add_activity("DOGE_BUY_SKIPPED", f"Trading disabled: {signal.reason}", traded=False)
        
        elif signal.action.value == "SELL" and self.doge_in_position:
            if settings.TRADING_ENABLED:
                pnl = (current_price - self.doge_entry_price) / self.doge_entry_price * 100
                add_activity("DOGE_SELL", f"{signal.reason} @ ${current_price:.6f} (PnL: {pnl:+.1f}%)", traded=True)
                self.doge_entry_price = 0.0
                self.doge_in_position = False
                bot_state["doge_in_position"] = False
                bot_state["doge_entry_price"] = None
                bot_state["doge_pnl"] = None
                if "DOGE/USDT" in self.positions:
                    del self.positions["DOGE/USDT"]
                await self._execute_trade("SELL", signal.confidence)
                logger.info(f"DOGE SELL executed: {signal.reason} (PnL: {pnl:+.1f}%)")
            else:
                add_activity("DOGE_SELL_SKIPPED", f"Trading disabled: {signal.reason}", traded=False)
        else:
            if self.doge_in_position and self.doge_entry_price > 0:
                pnl = (current_price - self.doge_entry_price) / self.doge_entry_price * 100
                bot_state["doge_pnl"] = round(pnl, 2)
            logger.debug(f"DOGE HOLD: {signal.reason}")
    
    def _init_ada_engine(self):
        """Initialize ADA Trend Engine"""
        try:
            from engine.ada_trend_engine import ADATrendEngine
            self.ada_trend_engine = ADATrendEngine()
            logger.info("ADA Trend Engine initialized (+1195% backtested)")
        except ImportError as e:
            logger.warning(f"Could not load ADA Trend Engine: {e}")
            self.ada_trend_engine = None
    
    async def _ada_trend_cycle(self, add_activity, bot_state):
        """ADA trading using optimized Trend Following strategy."""
        from engine.technical_analyzer import technical_analyzer
        
        if not self.ada_trend_engine:
            logger.warning("ADA Trend Engine not available, skipping")
            return
        
        current_price = await self._get_current_price("ADA/USDT")
        if not current_price:
            add_activity("ADA_TREND", "Could not get ADA price", traded=False)
            return
        
        technical = await technical_analyzer.analyze("ADA/USDT")
        
        sma10 = technical.sma_10 if technical and hasattr(technical, 'sma_10') else current_price
        sma20 = technical.sma_20 if technical and hasattr(technical, 'sma_20') else current_price
        sma50 = technical.sma_50 if technical and hasattr(technical, 'sma_50') else current_price
        rsi = technical.rsi if technical else 50
        high_20d = current_price * 1.1
        
        self.ada_trend_engine.update_position(self.ada_entry_price, self.ada_in_position)
        
        signal = self.ada_trend_engine.get_signal(
            current_price=current_price, sma10=sma10, sma20=sma20, sma50=sma50,
            rsi=rsi, high_20d=high_20d, momentum_7d=0.0
        )
        
        bot_state["ada_trend"] = signal.trend.value
        bot_state["ada_action"] = signal.action.value
        
        pnl = 0
        if signal.action.value == "BUY" and not self.ada_in_position:
            add_activity("ADA_BUY_SIGNAL", f"Trend: {signal.trend.value} | {signal.reason}", traded=False)
            if TRADING_ENABLED:
                self.ada_entry_price = current_price
                self.ada_in_position = True
                bot_state["ada_in_position"] = True
                bot_state["ada_entry_price"] = current_price
                await self._execute_trade("BUY", signal.confidence)
                logger.info(f"ADA BUY executed at ${current_price:.4f}: {signal.reason}")
            else:
                add_activity("ADA_BUY_SKIPPED", f"Trading disabled: {signal.reason}", traded=False)
        elif signal.action.value == "SELL" and self.ada_in_position:
            pnl = (current_price - self.ada_entry_price) / self.ada_entry_price * 100
            add_activity("ADA_SELL_SIGNAL", f"PnL: {pnl:+.1f}% | {signal.reason}", traded=False)
            if TRADING_ENABLED:
                self.ada_entry_price = 0.0
                self.ada_in_position = False
                bot_state["ada_in_position"] = False
                bot_state["ada_entry_price"] = None
                bot_state["ada_pnl"] = None
                if "ADA/USDT" in self.positions:
                    del self.positions["ADA/USDT"]
                await self._execute_trade("SELL", signal.confidence)
                logger.info(f"ADA SELL executed: {signal.reason} (PnL: {pnl:+.1f}%)")
            else:
                add_activity("ADA_SELL_SKIPPED", f"Trading disabled: {signal.reason}", traded=False)
        else:
            if self.ada_in_position and self.ada_entry_price > 0:
                pnl = (current_price - self.ada_entry_price) / self.ada_entry_price * 100
                bot_state["ada_pnl"] = round(pnl, 2)
            logger.debug(f"ADA HOLD: {signal.reason}")


# Global trading engine instance
trading_engine = TradingEngine()


async def start_trading_engine():
    """Start the trading engine in background"""
    await trading_engine.start()


def stop_trading_engine():
    """Stop the trading engine"""
    trading_engine.stop()
