"""
Sentinel Quant - AI Chat Endpoints
Integration with Dify for natural language queries
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone

from api.deps import CurrentUser
from core.config import settings

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message request"""
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response"""
    response: str
    conversation_id: str
    timestamp: datetime


@router.post("/message", response_model=ChatResponse)
async def send_chat_message(chat: ChatMessage, current_user: CurrentUser):
    """
    Send message to AI analyst (Dify integration).
    
    Example questions:
    - "Why didn't the bot trade last night?"
    - "What's the current market sentiment?"
    - "Show me today's trading summary"
    """
    # TODO: Integrate with Dify API
    # For now, return a placeholder response
    
    if not settings.DIFY_API_URL or not settings.DIFY_API_KEY:
        return ChatResponse(
            response="AI Chat is not configured. Please set DIFY_API_URL and DIFY_API_KEY.",
            conversation_id=chat.conversation_id or "new",
            timestamp=datetime.now(timezone.utc)
        )
    
    # Placeholder responses based on keywords
    message_lower = chat.message.lower()
    
    if "sentiment" in message_lower:
        response = "Current market sentiment analysis:\n\n📊 **Overall: Neutral (Score: 52/100)**\n\n• Bitcoin showing sideways movement\n• No major FUD detected in recent news\n• Fear & Greed Index: Neutral\n\n*Bot is clear to take trades if technical signals are strong.*"
    elif "why" in message_lower and "trade" in message_lower:
        response = "The bot didn't execute trades for the following reasons:\n\n1. **Confidence below threshold**: AI confidence was 78% (threshold: 85%)\n2. **Low volatility**: ATR was below optimal range\n\n*The bot is designed to be conservative and only trades when conditions are optimal.*"
    elif "summary" in message_lower or "today" in message_lower:
        response = "📈 **Today's Trading Summary**\n\n• Trades Executed: 3\n• Win Rate: 66.7%\n• Total PnL: +$45.20 (+1.2%)\n\n**Best Trade**: BTC Long +$32.50\n**Worst Trade**: ETH Short -$12.30"
    else:
        response = f"I received your message: \"{chat.message}\"\n\nI'm the Sentinel Quant AI analyst. I can help you understand:\n• Market sentiment\n• Why trades were or weren't executed\n• Trading performance summaries\n• Risk analysis\n\n*Full AI integration coming soon with Dify.*"
    
    return ChatResponse(
        response=response,
        conversation_id=chat.conversation_id or f"conv_{current_user.id}_{int(datetime.now().timestamp())}",
        timestamp=datetime.now(timezone.utc)
    )


@router.get("/history")
async def get_chat_history(
    current_user: CurrentUser,
    conversation_id: Optional[str] = None
):
    """Get chat history for a conversation"""
    # TODO: Store and retrieve chat history from database
    return {
        "messages": [],
        "message": "Chat history not implemented yet"
    }
