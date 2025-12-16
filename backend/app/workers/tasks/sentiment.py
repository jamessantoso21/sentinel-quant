"""
Sentinel Quant - Sentiment Analysis Tasks
Celery tasks for updating market sentiment via Dify
"""
from workers.celery_app import celery_app
from core.config import settings
import logging
import httpx

logger = logging.getLogger(__name__)


@celery_app.task(name="workers.tasks.sentiment.update_market_sentiment")
def update_market_sentiment():
    """
    Update market sentiment score via Dify API.
    Runs every 15 minutes.
    """
    logger.info("Starting sentiment analysis...")
    
    if not settings.DIFY_API_URL or not settings.DIFY_API_KEY:
        logger.warning("Dify not configured, skipping sentiment update")
        return {"status": "skipped", "reason": "Dify not configured"}
    
    try:
        # Call Dify API for sentiment analysis
        prompt = """
        Analyze the current cryptocurrency market sentiment based on recent news.
        Consider:
        1. Major news headlines from the last 24 hours
        2. Social media sentiment (Twitter, Reddit)
        3. Fear & Greed Index
        4. Any FUD or major announcements
        
        Return a JSON response with:
        - score: 0-100 (0=extreme fear, 100=extreme greed)
        - level: "EXTREME_FEAR", "FEAR", "NEUTRAL", "GREED", "EXTREME_GREED"
        - summary: Brief summary of market sentiment
        - should_veto: true if conditions suggest avoiding trades
        - veto_reason: reason for veto if applicable
        """
        
        # Make API call to Dify
        # This is a placeholder - actual implementation depends on Dify API format
        response = httpx.post(
            f"{settings.DIFY_API_URL}/chat-messages",
            headers={
                "Authorization": f"Bearer {settings.DIFY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": {},
                "query": prompt,
                "response_mode": "blocking",
                "user": "sentinel_quant"
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Sentiment updated: {result}")
            
            # TODO: Store sentiment in database
            # sentiment_repo.create(...)
            
            return {"status": "success", "result": result}
        else:
            logger.error(f"Dify API error: {response.status_code}")
            return {"status": "error", "code": response.status_code}
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {"status": "error", "error": str(e)}


@celery_app.task(name="workers.tasks.sentiment.analyze_specific_news")
def analyze_specific_news(news_text: str, symbol: str = "BTC"):
    """
    Analyze specific news article for sentiment.
    Called on-demand when major news is detected.
    """
    logger.info(f"Analyzing news for {symbol}...")
    
    if not settings.DIFY_API_URL:
        return {"status": "skipped", "reason": "Dify not configured"}
    
    try:
        prompt = f"""
        Analyze this cryptocurrency news for {symbol}:
        
        "{news_text}"
        
        Is this:
        1. Bullish or bearish?
        2. High impact or low impact?
        3. Should we avoid trading based on this?
        
        Return JSON with: sentiment (-1 to 1), impact (0-10), avoid_trading (bool)
        """
        
        # TODO: Call Dify API
        return {"status": "pending", "message": "Not implemented yet"}
        
    except Exception as e:
        logger.error(f"News analysis failed: {e}")
        return {"status": "error", "error": str(e)}
