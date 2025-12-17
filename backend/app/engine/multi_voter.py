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
    
    def __init__(self, min_consensus: float = 0.6):
        self.min_consensus = min_consensus  # 60% = 2/3 or 3/4 agree
    
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
        
        # 3. LSTM Vote (placeholder)
        lstm_vote = self._get_lstm_vote(lstm_prediction)
        votes.append(lstm_vote)
        
        # 4. PPO Vote (placeholder)
        ppo_vote = self._get_ppo_vote(ppo_action)
        votes.append(ppo_vote)
        
        # Aggregate votes
        return self._aggregate_votes(votes)
    
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
        PLACEHOLDER: Returns ABSTAIN until model is trained.
        """
        # TODO: When LSTM is trained, this will return actual predictions
        # prediction > 0 = price going up = BUY
        # prediction < 0 = price going down = SELL
        
        return Vote(
            voter_name="LSTM Prediction",
            vote=VoteType.ABSTAIN,
            confidence=0.0,
            reasoning="Model not trained yet",
            is_active=False
        )
    
    def _get_ppo_vote(self, action: Optional[str]) -> Vote:
        """
        Convert PPO action to vote.
        PLACEHOLDER: Returns ABSTAIN until model is trained.
        """
        # TODO: When PPO is trained, this will return actual actions
        
        return Vote(
            voter_name="PPO Agent",
            vote=VoteType.ABSTAIN,
            confidence=0.0,
            reasoning="Model not trained yet",
            is_active=False
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
