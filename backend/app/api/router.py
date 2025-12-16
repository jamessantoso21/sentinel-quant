"""
Sentinel Quant - Main API Router
Combines all endpoint routers
"""
from fastapi import APIRouter

from .v1.endpoints import auth, health, trades, positions, bot, chat, ai

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

api_router.include_router(
    trades.router,
    prefix="/trades",
    tags=["Trades"]
)

api_router.include_router(
    positions.router,
    prefix="/positions",
    tags=["Positions"]
)

api_router.include_router(
    bot.router,
    prefix="/bot",
    tags=["Bot Control"]
)

api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["AI Chat"]
)

api_router.include_router(
    ai.router,
    prefix="/ai",
    tags=["AI Models"]
)

