"""
Sentinel Quant - Crypto News Fetcher
Fetches real-time crypto news for sentiment analysis
"""
import httpx
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Single news article"""
    title: str
    body: str
    source: str
    published_at: datetime
    url: str
    categories: List[str]


class CryptoNewsFetcher:
    """
    Fetches crypto news from multiple sources.
    Uses CryptoCompare as primary (free, no API key needed).
    """
    
    def __init__(self):
        self.cryptocompare_url = "https://min-api.cryptocompare.com/data/v2/news/"
        self.cache_duration_minutes = 5
        self._cache: Dict[str, List[NewsItem]] = {}
        self._cache_time: Dict[str, datetime] = {}
    
    async def fetch_news(self, symbol: str = "BTC", limit: int = 5) -> List[NewsItem]:
        """
        Fetch latest news for a crypto symbol.
        
        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
            limit: Number of articles to fetch
            
        Returns:
            List of NewsItem objects
        """
        # Check cache
        cache_key = f"{symbol}_{limit}"
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached news for {symbol}")
            return self._cache[cache_key]
        
        # Fetch fresh news
        news = await self._fetch_from_cryptocompare(symbol, limit)
        
        # Cache results
        if news:
            self._cache[cache_key] = news
            self._cache_time[cache_key] = datetime.now()
        
        return news
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid"""
        if cache_key not in self._cache_time:
            return False
        
        age = datetime.now() - self._cache_time[cache_key]
        return age < timedelta(minutes=self.cache_duration_minutes)
    
    async def _fetch_from_cryptocompare(self, symbol: str, limit: int) -> List[NewsItem]:
        """Fetch news from CryptoCompare API"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # CryptoCompare uses category names like BTC, ETH
                response = await client.get(
                    self.cryptocompare_url,
                    params={
                        "categories": symbol,
                        "extraParams": "SentinelQuant"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("Data", [])[:limit]
                    
                    news_items = []
                    for article in articles:
                        try:
                            # Safely get body text
                            body = article.get("body", "") or ""
                            if isinstance(body, str):
                                body = body[:500]
                            else:
                                body = str(body)[:500]
                            
                            # Safely get categories
                            cats = article.get("categories", "")
                            if isinstance(cats, str):
                                categories = cats.split("|")
                            elif isinstance(cats, list):
                                categories = cats
                            else:
                                categories = []
                            
                            news_items.append(NewsItem(
                                title=str(article.get("title", "")),
                                body=body,
                                source=str(article.get("source", "unknown")),
                                published_at=datetime.fromtimestamp(article.get("published_on", 0)),
                                url=str(article.get("url", "")),
                                categories=categories
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to parse article: {e}")
                            continue
                    
                    logger.info(f"CryptoCompare: Fetched {len(news_items)} news for {symbol}")
                    return news_items
                else:
                    logger.warning(f"CryptoCompare API error: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            return []
    
    def format_for_dify(self, news_items: List[NewsItem], symbol: str) -> str:
        """
        Format news items into a prompt for Dify sentiment analysis.
        
        Args:
            news_items: List of news articles
            symbol: The crypto symbol being analyzed
            
        Returns:
            Formatted text for Dify
        """
        # Special context for PAXG (gold-backed token)
        asset_context = ""
        if symbol == "PAXG":
            asset_context = "Note: PAXG (Pax Gold) is a gold-backed cryptocurrency. 1 PAXG = 1 oz of physical gold. Consider gold market factors and safe-haven demand.\n\n"
        
        if not news_items:
            # Fallback to generic market context
            return f"""{asset_context}Current market conditions for {symbol}/USDT: 
The market is showing mixed signals with {symbol} consolidating near recent price levels. 
No specific news available at this time. Analyze general market sentiment."""
        
        # Build news summary
        news_text = f"{asset_context}Latest {symbol} News Headlines:\n\n"
        
        for i, item in enumerate(news_items, 1):
            age = datetime.now() - item.published_at
            age_str = f"{int(age.total_seconds() // 3600)}h ago" if age.total_seconds() > 3600 else f"{int(age.total_seconds() // 60)}m ago"
            
            news_text += f"{i}. [{item.source}] ({age_str})\n"
            news_text += f"   {item.title}\n"
            if item.body:
                # Truncate body to first sentence or 100 chars
                body_preview = item.body[:100].split('.')[0] + "..."
                news_text += f"   {body_preview}\n"
            news_text += "\n"
        
        news_text += f"\nAnalyze the sentiment of these {symbol} news articles and their potential market impact."
        
        return news_text


# Singleton instance
news_fetcher = CryptoNewsFetcher()
