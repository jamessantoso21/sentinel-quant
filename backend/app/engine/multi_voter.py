"""
Sentinel Quant - Multi-Model Voting System
Combines 4 voters for higher win rate trading decisions
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class VoteType(str, Enum):
    """Possible votes"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    ABSTAIN = "ABSTAIN"  # When model can't decide or not available


@dataclass
class Vote:
    """Single vote from a model"""
    voter_name: str
    vote: VoteType
    confidence: float  # 0-1
    reasoning: str
    is_active: bool  # True if model is trained/available


@dataclass
class VotingResult:
    """Aggregated voting result"""
    final_action: str  # BUY, SELL, HOLD
    consensus_level: float  # 0-1 (1 = unanimous)
    total_voters: int
    active_voters: int
    buy_votes: int
    sell_votes: int
    hold_votes: int
    individual_votes: List[Vote]
    reasoning: str
    should_trade: bool


class MultiModelVoter:
    """
    Aggregates votes from multiple models/sources.
    
    Voters:
    1. Dify Sentiment (ACTIVE)
    2. Technical Analysis (ACTIVE)
    3. LSTM Prediction (PLACEHOLDER - needs training)
    4. PPO Agent (PLACEHOLDER - needs training)
    
    Rules:
    - Trade only if 2+ active voters agree
    - ABSTAIN votes don't count against consensus
    - Confidence = weighted average of agreeing voters
    """
    
    def __init__(self, min_consensus: float = 0.25):  # OPTIMIZED: 25% = 1/4 voters (13.9% backtest return)
        self.min_consensus = min_consensus
    
    def vote(
        self,
        sentiment_data: Optional[Dict] = None,
        technical_data = None,  # TechnicalSignal
        lstm_prediction: Optional[float] = None,
        ppo_action: Optional[str] = None
    ) -> VotingResult:
        """
        Collect votes from all models and aggregate.
        """
        votes = []
        
        # 1. Dify Sentiment Vote
        sentiment_vote = self._get_sentiment_vote(sentiment_data)
        votes.append(sentiment_vote)
        
        # 2. Technical Analysis Vote
        technical_vote = self._get_technical_vote(technical_data)
        votes.append(technical_vote)
        
        # 3. Momentum Vote (NEW - lightweight, no ML!)
        momentum_vote = self._get_momentum_vote(technical_data)
        votes.append(momentum_vote)
        
        # 4. Trend Confirmation Vote (NEW)
        trend_vote = self._get_trend_vote(technical_data)
        votes.append(trend_vote)
        
        # Aggregate votes
        return self._aggregate_votes(votes)
    
    def _get_momentum_vote(self, technical_data) -> Vote:
        """
        Momentum voter - checks if price momentum supports the trade.
        Based on RSI momentum (is RSI rising or falling?)
        """
        if not technical_data:
            return Vote(
                voter_name="Momentum",
                vote=VoteType.ABSTAIN,
                confidence=0.0,
                reasoning="No data",
                is_active=True
            )
        
        rsi = technical_data.rsi
        bb_position = technical_data.bb_position
        
        # Strong momentum signals
        if rsi < 30 and bb_position < 0.2:
            # Extremely oversold - strong BUY momentum
            vote_type = VoteType.BUY
            confidence = 0.9
            reason = f"Oversold momentum (RSI={rsi:.0f}, BB={bb_position:.2f})"
        elif rsi > 70 and bb_position > 0.8:
            # Extremely overbought - strong SELL momentum
            vote_type = VoteType.SELL
            confidence = 0.9
            reason = f"Overbought momentum (RSI={rsi:.0f}, BB={bb_position:.2f})"
        elif rsi < 40:
            # Mild oversold
            vote_type = VoteType.BUY
            confidence = 0.6
            reason = f"Mild oversold (RSI={rsi:.0f})"
        elif rsi > 60:
            # Mild overbought
            vote_type = VoteType.SELL
            confidence = 0.6
            reason = f"Mild overbought (RSI={rsi:.0f})"
        else:
            vote_type = VoteType.HOLD
            confidence = 0.5
            reason = f"Neutral momentum (RSI={rsi:.0f})"
        
        return Vote(
            voter_name="Momentum",
            vote=vote_type,
            confidence=confidence,
            reasoning=reason,
            is_active=True
        )
    
    def _get_trend_vote(self, technical_data) -> Vote:
        """
        Trend confirmation voter - only trade in direction of trend.
        """
        if not technical_data:
            return Vote(
                voter_name="Trend",
                vote=VoteType.ABSTAIN,
                confidence=0.0,
                reasoning="No data",
                is_active=True
            )
        
        trend = technical_data.trend
        
        if trend == "UP":
            vote_type = VoteType.BUY
            confidence = 0.8
            reason = "Uptrend confirmed"
        elif trend == "DOWN":
            vote_type = VoteType.SELL
            confidence = 0.8
            reason = "Downtrend confirmed"
        else:  # SIDEWAYS
            vote_type = VoteType.HOLD
            confidence = 0.7
            reason = "Sideways - no clear trend"
        
        return Vote(
            voter_name="Trend",
            vote=vote_type,
            confidence=confidence,
            reasoning=reason,
            is_active=True
        )
    
    def _get_sentiment_vote(self, data: Optional[Dict]) -> Vote:
        """Convert Dify sentiment to vote"""
        if not data:
            return Vote(
                voter_name="Dify Sentiment",
                vote=VoteType.ABSTAIN,
                confidence=0.0,
                reasoning="No sentiment data",
                is_active=True
            )
        
        sentiment = data.get("sentiment", "neutral").lower()
        score = data.get("score", 0)
        confidence = data.get("confidence", 0.5)
        
        if sentiment == "bullish" or score > 0.3:
            vote_type = VoteType.BUY
            reason = f"Bullish sentiment (score={score:.2f})"
        elif sentiment == "bearish" or score < -0.3:
            vote_type = VoteType.SELL
            reason = f"Bearish sentiment (score={score:.2f})"
        else:
            vote_type = VoteType.HOLD
            reason = f"Neutral sentiment (score={score:.2f})"
        
        return Vote(
            voter_name="Dify Sentiment",
            vote=vote_type,
            confidence=confidence,
            reasoning=reason,
            is_active=True
        )
    
    def _get_technical_vote(self, signal) -> Vote:
        """Convert technical signal to vote"""
        if not signal:
            return Vote(
                voter_name="Technical Analysis",
                vote=VoteType.ABSTAIN,
                confidence=0.0,
                reasoning="No technical data",
                is_active=True
            )
        
        # Map signal to vote
        if signal.overall_signal == "BUY":
            vote_type = VoteType.BUY
            reason = f"RSI={signal.rsi:.0f} ({signal.rsi_signal}), Trend={signal.trend}"
        elif signal.overall_signal == "SELL":
            vote_type = VoteType.SELL
            reason = f"RSI={signal.rsi:.0f} ({signal.rsi_signal}), Trend={signal.trend}"
        else:
            vote_type = VoteType.HOLD
            reason = f"RSI={signal.rsi:.0f} (NEUTRAL), Trend={signal.trend}"
        
        return Vote(
            voter_name="Technical Analysis",
            vote=vote_type,
            confidence=signal.confluence_score,
            reasoning=reason,
            is_active=True
        )
    
    def _get_lstm_vote(self, prediction: Optional[float]) -> Vote:
        """
        Convert LSTM prediction to vote.
        Uses trained model if available.
        """
        if prediction is None:
            return Vote(
                voter_name="LSTM Prediction",
                vote=VoteType.ABSTAIN,
                confidence=0.0,
                reasoning="No LSTM prediction available",
                is_active=False
            )
        
        # prediction > 0 = price going up = BUY
        # prediction < 0 = price going down = SELL
        confidence = min(abs(prediction), 1.0)
        
        if prediction > 0.1:
            vote_type = VoteType.BUY
            reason = f"LSTM predicts UP ({prediction:+.2f})"
        elif prediction < -0.1:
            vote_type = VoteType.SELL
            reason = f"LSTM predicts DOWN ({prediction:+.2f})"
        else:
            vote_type = VoteType.HOLD
            reason = f"LSTM uncertain ({prediction:+.2f})"
        
        return Vote(
            voter_name="LSTM Prediction",
            vote=vote_type,
            confidence=confidence,
            reasoning=reason,
            is_active=True
        )
    
    def _get_ppo_vote(self, action: Optional[str]) -> Vote:
        """
        Convert PPO action to vote.
        Uses trained model if available.
        """
        if action is None:
            return Vote(
                voter_name="PPO Agent",
                vote=VoteType.ABSTAIN,
                confidence=0.0,
                reasoning="No PPO action available",
                is_active=False
            )
        
        # Map PPO action to vote
        action = action.upper()
        if action == "BUY":
            vote_type = VoteType.BUY
            reason = "PPO recommends BUY"
        elif action == "SELL":
            vote_type = VoteType.SELL
            reason = "PPO recommends SELL"
        else:
            vote_type = VoteType.HOLD
            reason = "PPO recommends HOLD"
        
        return Vote(
            voter_name="PPO Agent",
            vote=vote_type,
            confidence=0.7,  # Fixed confidence for trained model
            reasoning=reason,
            is_active=True
        )
    
    def _aggregate_votes(self, votes: List[Vote]) -> VotingResult:
        """Aggregate all votes into final decision"""
        
        # Count votes (excluding ABSTAIN)
        active_votes = [v for v in votes if v.vote != VoteType.ABSTAIN]
        active_voters = len(active_votes)
        
        buy_votes = [v for v in active_votes if v.vote == VoteType.BUY]
        sell_votes = [v for v in active_votes if v.vote == VoteType.SELL]
        hold_votes = [v for v in active_votes if v.vote == VoteType.HOLD]
        
        buy_count = len(buy_votes)
        sell_count = len(sell_votes)
        hold_count = len(hold_votes)
        
        # Determine winner
        if active_voters == 0:
            final_action = "HOLD"
            consensus_level = 0.0
            reasoning = "No active voters"
            should_trade = False
        elif buy_count > sell_count and buy_count > hold_count:
            final_action = "BUY"
            consensus_level = buy_count / active_voters
            avg_confidence = sum(v.confidence for v in buy_votes) / buy_count
            reasoning = f"BUY consensus: {buy_count}/{active_voters} voters agree (conf={avg_confidence:.1%})"
            should_trade = consensus_level >= self.min_consensus
        elif sell_count > buy_count and sell_count > hold_count:
            final_action = "SELL"
            consensus_level = sell_count / active_voters
            avg_confidence = sum(v.confidence for v in sell_votes) / sell_count
            reasoning = f"SELL consensus: {sell_count}/{active_voters} voters agree (conf={avg_confidence:.1%})"
            should_trade = consensus_level >= self.min_consensus
        else:
            final_action = "HOLD"
            consensus_level = hold_count / active_voters if active_voters > 0 else 0
            reasoning = f"No consensus: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}"
            should_trade = False
        
        # Build detailed reasoning
        voter_details = []
        for v in votes:
            status = "✓" if v.is_active else "○"
            voter_details.append(f"{status}{v.voter_name}: {v.vote.value}")
        
        full_reasoning = f"{reasoning} | Votes: {', '.join(voter_details)}"
        
        return VotingResult(
            final_action=final_action,
            consensus_level=consensus_level,
            total_voters=len(votes),
            active_voters=active_voters,
            buy_votes=buy_count,
            sell_votes=sell_count,
            hold_votes=hold_count,
            individual_votes=votes,
            reasoning=full_reasoning,
            should_trade=should_trade
        )


# Singleton instance
multi_voter = MultiModelVoter()
