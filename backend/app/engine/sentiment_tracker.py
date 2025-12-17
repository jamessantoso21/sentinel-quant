"""
Sentinel Quant - Sentiment History Tracker
Stores sentiment history and calculates moving averages to avoid fake pumps
"""
import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SentimentRecord:
    """Single sentiment data point"""
    timestamp: datetime
    sentiment: str  # bullish, bearish, neutral
    score: float    # -1 to 1
    confidence: float  # 0 to 1
    symbol: str
    source: str     # headline/source info


@dataclass 
class SentimentSMA:
    """Sentiment Moving Average result"""
    avg_score: float       # Average score over period
    avg_confidence: float  # Average confidence
    data_points: int       # Number of data points used
    period_hours: int      # Period used for calculation
    dominant_sentiment: str  # Most common sentiment
    trend: str             # IMPROVING, WORSENING, STABLE
    is_reliable: bool      # True if enough data points


class SentimentHistoryTracker:
    """
    Tracks sentiment history in memory and calculates SMAs.
    Prevents trading on single fake pump news.
    
    For production: Should use database (SentimentScore model).
    For now: Uses in-memory deque for simplicity.
    """
    
    def __init__(self, max_history_hours: int = 24):
        self.max_history_hours = max_history_hours
        self._history: deque = deque(maxlen=500)  # Keep last 500 records
        self._min_data_points = 3  # Minimum points for reliable SMA
    
    def record(
        self,
        sentiment: str,
        score: float,
        confidence: float,
        symbol: str = "BTC",
        source: str = "dify"
    ) -> SentimentRecord:
        """
        Record a new sentiment data point.
        """
        record = SentimentRecord(
            timestamp=datetime.now(timezone.utc),
            sentiment=sentiment.lower() if sentiment else "neutral",
            score=score,
            confidence=confidence,
            symbol=symbol,
            source=source
        )
        
        self._history.append(record)
        self._cleanup_old_records()
        
        logger.info(f"Recorded sentiment: {sentiment}, score={score:.2f}, total_history={len(self._history)}")
        return record
    
    def _cleanup_old_records(self):
        """Remove records older than max_history_hours"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.max_history_hours)
        
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()
    
    def get_sma(self, period_hours: int = 6, symbol: str = "BTC") -> SentimentSMA:
        """
        Calculate Sentiment Moving Average over the specified period.
        
        Args:
            period_hours: Lookback period in hours (default 6)
            symbol: Filter by symbol
            
        Returns:
            SentimentSMA with average values
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=period_hours)
        
        # Filter relevant records
        relevant = [
            r for r in self._history 
            if r.timestamp >= cutoff and r.symbol == symbol
        ]
        
        if not relevant:
            return SentimentSMA(
                avg_score=0.0,
                avg_confidence=0.5,
                data_points=0,
                period_hours=period_hours,
                dominant_sentiment="unknown",
                trend="STABLE",
                is_reliable=False
            )
        
        # Calculate averages
        avg_score = sum(r.score for r in relevant) / len(relevant)
        avg_confidence = sum(r.confidence for r in relevant) / len(relevant)
        
        # Determine dominant sentiment
        sentiment_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        for r in relevant:
            if r.sentiment in sentiment_counts:
                sentiment_counts[r.sentiment] += 1
        dominant = max(sentiment_counts, key=sentiment_counts.get)
        
        # Determine trend (compare first half vs second half)
        if len(relevant) >= 4:
            mid = len(relevant) // 2
            first_half_avg = sum(r.score for r in relevant[:mid]) / mid
            second_half_avg = sum(r.score for r in relevant[mid:]) / (len(relevant) - mid)
            
            if second_half_avg > first_half_avg + 0.1:
                trend = "IMPROVING"
            elif second_half_avg < first_half_avg - 0.1:
                trend = "WORSENING"
            else:
                trend = "STABLE"
        else:
            trend = "STABLE"
        
        is_reliable = len(relevant) >= self._min_data_points
        
        return SentimentSMA(
            avg_score=avg_score,
            avg_confidence=avg_confidence,
            data_points=len(relevant),
            period_hours=period_hours,
            dominant_sentiment=dominant,
            trend=trend,
            is_reliable=is_reliable
        )
    
    def should_trust_current_sentiment(
        self,
        current_sentiment: str,
        current_score: float,
        period_hours: int = 6
    ) -> Dict:
        """
        Check if current sentiment aligns with historical average.
        Helps avoid trading on fake pump news.
        
        Returns:
            Dict with trust_score and reasoning
        """
        sma = self.get_sma(period_hours)
        
        # Not enough data - be cautious
        if not sma.is_reliable:
            return {
                "trust": True,  # Trust current (no history to compare)
                "trust_score": 0.6,
                "reason": f"Limited history ({sma.data_points} points). Using current sentiment.",
                "sma": sma
            }
        
        # Check alignment
        current_is_bullish = current_sentiment.lower() == "bullish" or current_score > 0.3
        current_is_bearish = current_sentiment.lower() == "bearish" or current_score < -0.3
        
        history_is_bullish = sma.dominant_sentiment == "bullish" or sma.avg_score > 0.3
        history_is_bearish = sma.dominant_sentiment == "bearish" or sma.avg_score < -0.3
        
        # Perfect alignment
        if current_is_bullish and history_is_bullish:
            return {
                "trust": True,
                "trust_score": 0.9,
                "reason": f"Bullish sentiment confirmed by {period_hours}h history (avg={sma.avg_score:.2f})",
                "sma": sma
            }
        
        if current_is_bearish and history_is_bearish:
            return {
                "trust": True,
                "trust_score": 0.9,
                "reason": f"Bearish sentiment confirmed by {period_hours}h history (avg={sma.avg_score:.2f})",
                "sma": sma
            }
        
        # Sudden change - be cautious (possible fake pump)
        if current_is_bullish and history_is_bearish:
            return {
                "trust": False,
                "trust_score": 0.3,
                "reason": f"⚠️ SUDDEN BULLISH - History was bearish (avg={sma.avg_score:.2f}). Possible fake pump!",
                "sma": sma
            }
        
        if current_is_bearish and history_is_bullish:
            return {
                "trust": False,
                "trust_score": 0.3,
                "reason": f"⚠️ SUDDEN BEARISH - History was bullish (avg={sma.avg_score:.2f}). Verify before acting.",
                "sma": sma
            }
        
        # Neutral transitions
        return {
            "trust": True,
            "trust_score": 0.7,
            "reason": f"Sentiment transition acceptable (history avg={sma.avg_score:.2f}, trend={sma.trend})",
            "sma": sma
        }
    
    def get_history_stats(self, symbol: str = "BTC") -> Dict:
        """Get statistics about stored history"""
        symbol_records = [r for r in self._history if r.symbol == symbol]
        
        if not symbol_records:
            return {"total_records": 0, "oldest": None, "newest": None}
        
        return {
            "total_records": len(symbol_records),
            "oldest": symbol_records[0].timestamp.isoformat() if symbol_records else None,
            "newest": symbol_records[-1].timestamp.isoformat() if symbol_records else None,
            "sma_6h": self.get_sma(6, symbol).__dict__,
            "sma_12h": self.get_sma(12, symbol).__dict__
        }


# Singleton instance
sentiment_tracker = SentimentHistoryTracker()
