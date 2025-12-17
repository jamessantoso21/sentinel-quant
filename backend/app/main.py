"""
Sentinel Quant - Main FastAPI Application
Enterprise-grade hybrid AI trading system
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio

from core.config import settings
from core.security import security_manager
from api.router import api_router
from api.v1.websocket.stream import manager as ws_manager
from db.session import init_db, close_db

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Initialize database tables
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Auto-start bot if trading is enabled
    if settings.TRADING_ENABLED:
        try:
            from api.v1.endpoints.bot import bot_state
            from schemas.bot import BotState
            bot_state["state"] = BotState.RUNNING
            logger.info("Trading bot auto-started (TRADING_ENABLED=true)")
            
            # Start trading engine in background
            from engine.trading_engine import start_trading_engine
            asyncio.create_task(start_trading_engine())
            logger.info("Trading engine background task started")
        except Exception as e:
            logger.error(f"Failed to auto-start bot: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await close_db()


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    Sentinel Quant - Institutional-Grade Hybrid AI Trading Ecosystem
    
    ## Features
    - 🤖 AI-powered trading with PPO & LSTM models
    - 📰 Sentiment analysis via Dify LLM
    - 📊 Real-time WebSocket streaming
    - 🔒 JWT authentication
    - 🚨 Emergency kill switch
    - 📱 Mobile app ready API
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs",
        "trading_enabled": settings.TRADING_ENABLED
    }


# WebSocket endpoint for real-time streaming
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    """
    WebSocket endpoint for real-time updates.
    
    Channels:
    - prices: Real-time price updates
    - positions: Position PnL updates
    - signals: Trading signals
    - notifications: Push notifications
    
    Send JSON to subscribe/unsubscribe:
    {"action": "subscribe", "channel": "prices"}
    {"action": "unsubscribe", "channel": "positions"}
    """
    await ws_manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            action = data.get("action")
            channel = data.get("channel")
            
            if action == "subscribe" and channel:
                ws_manager.subscribe(user_id, channel)
                await ws_manager.send_personal(user_id, {
                    "type": "subscribed",
                    "channel": channel
                })
            
            elif action == "unsubscribe" and channel:
                ws_manager.unsubscribe(user_id, channel)
                await ws_manager.send_personal(user_id, {
                    "type": "unsubscribed",
                    "channel": channel
                })
            
            elif action == "ping":
                await ws_manager.send_personal(user_id, {"type": "pong"})
    
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, user_id)


# Health check (simple, no auth)
@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )