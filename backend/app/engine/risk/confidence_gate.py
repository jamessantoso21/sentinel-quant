"""
Sentinel Quant - Confidence Gate
Only allows trades above confidence threshold
"""
from dataclasses import dataclass
from typing import Optional, List
import logging

from core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of confidence gate check"""
    passed: bool
    confidence: float
    threshold: float
    reasons: List[str]
    warnings: List[str]


class ConfidenceGate:
    """
    Confidence gating mechanism.
    
    Only allows trades when:
    1. AI confidence >= threshold (default 85%)
    2. No veto from sentiment analysis
    3. Daily loss limit not exceeded
    """
    
    def __init__(
        self,
        confidence_threshold: float = None,
        max_daily_loss_percent: float = None
    ):
        self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
        self.max_daily_loss_percent = max_daily_loss_percent or settings.MAX_DAILY_LOSS_PERCENT
    
    def check(
        self,
        ai_confidence: float,
        sentiment_score: Optional[float] = None,
        sentiment_veto: bool = False,
        daily_pnl_percent: float = 0.0
    ) -> GateResult:
        """
        Check if trade should be allowed.
        
        Args:
            ai_confidence: AI model confidence (0.0 - 1.0)
            sentiment_score: Sentiment score (0-100)
            sentiment_veto: Whether sentiment analysis vetoed the trade
            daily_pnl_percent: Today's PnL as percentage
        """
        reasons = []
        warnings = []
        passed = True
        
        # Check 1: AI Confidence
        if ai_confidence < self.confidence_threshold:
            passed = False
            reasons.append(
                f"AI confidence ({ai_confidence:.1%}) below threshold ({self.confidence_threshold:.1%})"
            )
        else:
            reasons.append(f"AI confidence OK: {ai_confidence:.1%}")
        
        # Check 2: Sentiment Veto
        if sentiment_veto:
            passed = False
            score_str = f" (score: {sentiment_score})" if sentiment_score is not None else ""
            reasons.append(f"Trade vetoed by sentiment analysis{score_str}")
        elif sentiment_score is not None:
            if sentiment_score < 30:
                warnings.append(f"Low sentiment score: {sentiment_score}/100")
            elif sentiment_score > 80:
                warnings.append(f"High greed detected: {sentiment_score}/100")
            else:
                reasons.append(f"Sentiment OK: {sentiment_score}/100")
        
        # Check 3: Daily Loss Limit
        if daily_pnl_percent < -self.max_daily_loss_percent:
            passed = False
            reasons.append(
                f"Daily loss limit exceeded ({daily_pnl_percent:.1f}% < -{self.max_daily_loss_percent:.1f}%)"
            )
        elif daily_pnl_percent < 0:
            warnings.append(f"Currently in daily drawdown: {daily_pnl_percent:.1f}%")
        
        return GateResult(
            passed=passed,
            confidence=ai_confidence,
            threshold=self.confidence_threshold,
            reasons=reasons,
            warnings=warnings
        )
    
    def should_reduce_size(self, daily_pnl_percent: float) -> float:
        """
        Calculate position size reduction based on daily PnL.
        
        Returns multiplier (0.0 - 1.0):
        - 1.0 = full size
        - 0.5 = half size
        - 0.0 = no trading
        """
        if daily_pnl_percent >= 0:
            return 1.0
        
        # Progressive reduction as losses increase
        loss_percent = abs(daily_pnl_percent)
        max_loss = self.max_daily_loss_percent
        
        if loss_percent >= max_loss:
            return 0.0  # Stop trading
        
        # Linear reduction: at 50% of max loss, reduce to 50%
        reduction = 1.0 - (loss_percent / max_loss)
        return max(0.25, reduction)  # Minimum 25% size
