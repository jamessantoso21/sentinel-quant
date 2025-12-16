"""
Sentinel Quant - WebSocket Stream Manager
Real-time price and PnL streaming to mobile app
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, Optional
import json
import asyncio
from datetime import datetime, timezone


class ConnectionManager:
    """
    Manages WebSocket connections for real-time streaming.
    Supports multiple channels: prices, positions, signals
    """
    
    def __init__(self):
        # Active connections by user_id
        self.active_connections: Dict[int, Set[WebSocket]] = {}
        # Subscriptions by channel
        self.subscriptions: Dict[str, Set[int]] = {
            "prices": set(),
            "positions": set(),
            "signals": set(),
            "notifications": set()
        }
    
    async def connect(self, websocket: WebSocket, user_id: int):
        """Accept connection and register user"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        
        self.active_connections[user_id].add(websocket)
        
        # Auto-subscribe to notifications
        self.subscriptions["notifications"].add(user_id)
    
    def disconnect(self, websocket: WebSocket, user_id: int):
        """Remove connection on disconnect"""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                # Remove from all subscriptions
                for channel in self.subscriptions:
                    self.subscriptions[channel].discard(user_id)
    
    def subscribe(self, user_id: int, channel: str):
        """Subscribe user to a channel"""
        if channel in self.subscriptions:
            self.subscriptions[channel].add(user_id)
    
    def unsubscribe(self, user_id: int, channel: str):
        """Unsubscribe user from a channel"""
        if channel in self.subscriptions:
            self.subscriptions[channel].discard(user_id)
    
    async def send_personal(self, user_id: int, message: dict):
        """Send message to specific user"""
        if user_id in self.active_connections:
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass  # Connection might be closed
    
    async def broadcast_to_channel(self, channel: str, message: dict):
        """Broadcast message to all users subscribed to channel"""
        if channel not in self.subscriptions:
            return
        
        message["channel"] = channel
        message["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        for user_id in self.subscriptions[channel]:
            await self.send_personal(user_id, message)
    
    async def broadcast_price_update(self, symbol: str, price: float, change_24h: float):
        """Broadcast price update to price subscribers"""
        await self.broadcast_to_channel("prices", {
            "type": "price_update",
            "symbol": symbol,
            "price": price,
            "change_24h": change_24h
        })
    
    async def broadcast_position_update(self, user_id: int, position_data: dict):
        """Send position update to specific user"""
        await self.send_personal(user_id, {
            "type": "position_update",
            "channel": "positions",
            **position_data
        })
    
    async def broadcast_trade_signal(self, user_id: int, signal_data: dict):
        """Send trade signal to specific user"""
        await self.send_personal(user_id, {
            "type": "trade_signal",
            "channel": "signals",
            **signal_data
        })
    
    async def send_notification(self, user_id: int, title: str, message: str, data: Optional[dict] = None):
        """Send notification to user"""
        await self.send_personal(user_id, {
            "type": "notification",
            "channel": "notifications",
            "title": title,
            "message": message,
            "data": data or {}
        })
    
    @property
    def connected_users(self) -> int:
        """Get count of connected users"""
        return len(self.active_connections)


# Global connection manager
manager = ConnectionManager()
