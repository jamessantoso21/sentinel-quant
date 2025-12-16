"""
Sentinel Quant - Sentiment Score Model
Stores sentiment analysis results from Dify
"""
from sqlalchemy import Column, String, Float, Text, Enum as SQLEnum
import enum

from db.base import Base


class SentimentLevel(str, enum.Enum):
    """Sentiment classification"""
    EXTREME_FEAR = "EXTREME_FEAR"
    FEAR = "FEAR"
    NEUTRAL = "NEUTRAL"
    GREED = "GREED"
    EXTREME_GREED = "EXTREME_GREED"


class SentimentSource(str, enum.Enum):
    """Source of sentiment data"""
    NEWS = "NEWS"
    SOCIAL = "SOCIAL"
    MARKET = "MARKET"  # Fear & Greed Index
    COMBINED = "COMBINED"


class SentimentScore(Base):
    """
    Sentiment analysis result from Dify.
    Updated every 15 minutes by Celery worker.
    """
    
    # Target asset
    symbol = Column(String(20), nullable=False, index=True)  # e.g., "BTC", "ETH", or "MARKET"
    
    # Score (0-100, where 0 = extreme fear, 100 = extreme greed)
    score = Column(Float, nullable=False)
    level = Column(SQLEnum(SentimentLevel), nullable=False)
    source = Column(SQLEnum(SentimentSource), default=SentimentSource.COMBINED, nullable=False)
    
    # Analysis details
    summary = Column(Text, nullable=True)  # Brief summary from Dify
    key_factors = Column(Text, nullable=True)  # JSON string of key factors
    news_headlines = Column(Text, nullable=True)  # JSON string of analyzed headlines
    
    # Veto signal
    should_veto_buys = Column(Float, default=False)  # True if sentiment suggests avoiding buys
    veto_reason = Column(String(500), nullable=True)
    
    def __repr__(self) -> str:
        return f"<SentimentScore({self.symbol}: {self.score} - {self.level.value})>"
    
    @classmethod
    def from_score(cls, symbol: str, score: float, source: SentimentSource = SentimentSource.COMBINED) -> "SentimentScore":
        """Create sentiment from numeric score with auto-classification"""
        if score < 20:
            level = SentimentLevel.EXTREME_FEAR
            veto = True
            reason = "Extreme fear detected - high risk of continued downside"
        elif score < 40:
            level = SentimentLevel.FEAR
            veto = True
            reason = "Fear in market - consider waiting for stabilization"
        elif score < 60:
            level = SentimentLevel.NEUTRAL
            veto = False
            reason = None
        elif score < 80:
            level = SentimentLevel.GREED
            veto = False
            reason = None
        else:
            level = SentimentLevel.EXTREME_GREED
            veto = True  # Also veto on extreme greed (potential top)
            reason = "Extreme greed detected - potential market top"
        
        return cls(
            symbol=symbol,
            score=score,
            level=level,
            source=source,
            should_veto_buys=veto,
            veto_reason=reason
        )
