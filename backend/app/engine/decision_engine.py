"""
Sentinel Quant - Decision Engine
The Brain: Combines technical analysis, sentiment, and risk management
"""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging

from .risk.position_sizer import PositionSizer, PositionSizeResult
from .risk.confidence_gate import ConfidenceGate, GateResult
from core.config import settings

logger = logging.getLogger(__name__)


class TradeAction(str, Enum):
    """Possible trade actions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TechnicalSignal:
    """Signal from technical analysis"""
    action: TradeAction
    confidence: float
    source: str  # e.g., "PPO", "LSTM"
    reasons: List[str]


@dataclass
class SentimentSignal:
    """Signal from sentiment analysis"""
    score: float  # 0-100
    veto: bool
    reason: Optional[str]


@dataclass
class TradeDecision:
    """Final trade decision from the engine"""
    should_trade: bool
    action: TradeAction
    symbol: str
    
    # From technical analysis
    technical_confidence: float
    technical_signal: str
    
    # From sentiment
    sentiment_score: Optional[float]
    sentiment_veto: bool
    
    # Combined
    combined_confidence: float
    
    # Position sizing
    quantity: Optional[float]
    position_value: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    
    # Risk info
    risk_percent: float
    
    # Reasoning
    reasons: List[str]
    warnings: List[str]
    
    timestamp: datetime


class DecisionEngine:
    """
    The Brain of Sentinel Quant.
    
    Workflow:
    1. Get technical signal (from PPO/LSTM models)
    2. Get sentiment signal (from Dify)
    3. Apply confidence gate
    4. Calculate position size if approved
    5. Return final decision
    """
    
    def __init__(
        self,
        confidence_threshold: float = None,
        max_position_usdt: float = None,
        risk_per_trade_percent: float = 1.0
    ):
        self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
        self.max_position_usdt = max_position_usdt or settings.MAX_POSITION_SIZE_USDT
        
        self.position_sizer = PositionSizer(
            max_position_usdt=self.max_position_usdt,
            risk_per_trade_percent=risk_per_trade_percent
        )
        self.confidence_gate = ConfidenceGate(
            confidence_threshold=self.confidence_threshold
        )
    
    def make_decision(
        self,
        symbol: str,
        current_price: float,
        account_balance: float,
        technical_signal: TechnicalSignal,
        sentiment_signal: Optional[SentimentSignal] = None,
        atr: float = 0.0,
        daily_pnl_percent: float = 0.0
    ) -> TradeDecision:
        """
        Make a trade decision based on all inputs.
        """
        reasons = []
        warnings = []
        
        # Step 1: Check technical signal
        if technical_signal.action == TradeAction.HOLD:
            return self._create_hold_decision(
                symbol=symbol,
                technical_signal=technical_signal,
                sentiment_signal=sentiment_signal,
                reason="Technical signal is HOLD"
            )
        
        reasons.append(f"Technical: {technical_signal.action.value} ({technical_signal.confidence:.1%})")
        reasons.extend(technical_signal.reasons)
        
        # Step 2: Check sentiment
        sentiment_score = sentiment_signal.score if sentiment_signal else None
        sentiment_veto = sentiment_signal.veto if sentiment_signal else False
        
        if sentiment_signal:
            if sentiment_veto:
                reasons.append(f"Sentiment VETO: {sentiment_signal.reason}")
            else:
                reasons.append(f"Sentiment OK: {sentiment_score}/100")
        
        # Step 3: Apply confidence gate
        gate_result = self.confidence_gate.check(
            ai_confidence=technical_signal.confidence,
            sentiment_score=sentiment_score,
            sentiment_veto=sentiment_veto,
            daily_pnl_percent=daily_pnl_percent
        )
        
        reasons.extend(gate_result.reasons)
        warnings.extend(gate_result.warnings)
        
        if not gate_result.passed:
            return self._create_hold_decision(
                symbol=symbol,
                technical_signal=technical_signal,
                sentiment_signal=sentiment_signal,
                reason="Failed confidence gate",
                reasons=reasons,
                warnings=warnings
            )
        
        # Step 4: Calculate position size
        direction = "LONG" if technical_signal.action == TradeAction.BUY else "SHORT"
        
        # Apply daily PnL reduction
        size_multiplier = self.confidence_gate.should_reduce_size(daily_pnl_percent)
        if size_multiplier < 1.0:
            warnings.append(f"Position reduced to {size_multiplier:.0%} due to daily drawdown")
        
        position_result = self.position_sizer.calculate_position_size(
            account_balance=account_balance * size_multiplier,
            current_price=current_price,
            atr=atr,
            direction=direction
        )
        
        reasons.append(f"Position: {position_result.reason}")
        
        # Calculate stop loss and take profit
        stop_loss = self.position_sizer.calculate_stop_loss(current_price, atr, direction)
        take_profit = self.position_sizer.calculate_take_profit(current_price, atr, direction)
        
        # Step 5: Create final decision
        combined_confidence = technical_signal.confidence
        if sentiment_score is not None:
            # Weight: 70% technical, 30% sentiment
            sentiment_normalized = sentiment_score / 100
            combined_confidence = (0.7 * technical_signal.confidence) + (0.3 * sentiment_normalized)
        
        return TradeDecision(
            should_trade=True,
            action=technical_signal.action,
            symbol=symbol,
            technical_confidence=technical_signal.confidence,
            technical_signal=f"{technical_signal.source}_{technical_signal.action.value}",
            sentiment_score=sentiment_score,
            sentiment_veto=False,
            combined_confidence=combined_confidence,
            quantity=position_result.quantity,
            position_value=position_result.position_value_usdt,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percent=position_result.risk_usdt / account_balance * 100,
            reasons=reasons,
            warnings=warnings,
            timestamp=datetime.now(timezone.utc)
        )
    
    def _create_hold_decision(
        self,
        symbol: str,
        technical_signal: TechnicalSignal,
        sentiment_signal: Optional[SentimentSignal],
        reason: str,
        reasons: List[str] = None,
        warnings: List[str] = None
    ) -> TradeDecision:
        """Create a HOLD decision"""
        return TradeDecision(
            should_trade=False,
            action=TradeAction.HOLD,
            symbol=symbol,
            technical_confidence=technical_signal.confidence,
            technical_signal=f"{technical_signal.source}_{technical_signal.action.value}",
            sentiment_score=sentiment_signal.score if sentiment_signal else None,
            sentiment_veto=sentiment_signal.veto if sentiment_signal else False,
            combined_confidence=0.0,
            quantity=None,
            position_value=None,
            stop_loss=None,
            take_profit=None,
            risk_percent=0.0,
            reasons=reasons or [reason],
            warnings=warnings or [],
            timestamp=datetime.now(timezone.utc)
        )
