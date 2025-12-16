"""
Sentinel Quant - Firebase Push Notification Service
"""
from typing import Optional, Dict, Any
import json
import logging

from core.config import settings

logger = logging.getLogger(__name__)


class FirebaseService:
    """
    Firebase Cloud Messaging service for push notifications.
    Requires firebase-admin SDK and credentials file.
    """
    
    def __init__(self):
        self._initialized = False
        self._app = None
    
    async def initialize(self) -> bool:
        """Initialize Firebase Admin SDK"""
        if not settings.FIREBASE_CREDENTIALS_PATH:
            logger.warning("Firebase credentials not configured")
            return False
        
        try:
            import firebase_admin
            from firebase_admin import credentials
            
            cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
            self._app = firebase_admin.initialize_app(cred)
            self._initialized = True
            logger.info("Firebase initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            return False
    
    async def send_notification(
        self,
        token: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        deep_link: Optional[str] = None
    ) -> bool:
        """
        Send push notification to a device.
        
        Args:
            token: FCM device token
            title: Notification title
            body: Notification body
            data: Optional data payload for deep linking
            deep_link: Optional deep link URL
        """
        if not self._initialized:
            logger.warning("Firebase not initialized, skipping notification")
            return False
        
        try:
            from firebase_admin import messaging
            
            # Build notification
            notification = messaging.Notification(
                title=title,
                body=body
            )
            
            # Build data payload
            payload = data or {}
            if deep_link:
                payload["deep_link"] = deep_link
            
            # Convert all values to strings (FCM requirement)
            payload = {k: str(v) for k, v in payload.items()}
            
            # Build message
            message = messaging.Message(
                notification=notification,
                data=payload,
                token=token,
                android=messaging.AndroidConfig(
                    priority="high",
                    notification=messaging.AndroidNotification(
                        click_action="FLUTTER_NOTIFICATION_CLICK"
                    )
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(sound="default")
                    )
                )
            )
            
            # Send
            response = messaging.send(message)
            logger.info(f"Notification sent: {response}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
    
    async def send_trade_notification(
        self,
        token: str,
        symbol: str,
        direction: str,
        entry_price: float,
        risk_percent: float,
        trade_id: int
    ) -> bool:
        """Send trade execution notification"""
        title = f"🚀 {symbol} {direction} Executed"
        body = f"Entry: ${entry_price:,.2f} | Risk: {risk_percent:.1f}%"
        
        return await self.send_notification(
            token=token,
            title=title,
            body=body,
            data={
                "type": "trade_executed",
                "trade_id": trade_id,
                "symbol": symbol,
                "direction": direction
            },
            deep_link=f"sentinel://trade/{trade_id}"
        )
    
    async def send_alert_notification(
        self,
        token: str,
        alert_type: str,
        message: str
    ) -> bool:
        """Send alert notification (risk warning, sentiment alert, etc.)"""
        title_map = {
            "risk_warning": "⚠️ Risk Warning",
            "sentiment_alert": "📰 Sentiment Alert",
            "position_closed": "📊 Position Closed",
            "kill_switch": "🚨 Kill Switch Activated"
        }
        
        return await self.send_notification(
            token=token,
            title=title_map.get(alert_type, "📢 Alert"),
            body=message,
            data={"type": alert_type}
        )


# Global instance
firebase_service = FirebaseService()
