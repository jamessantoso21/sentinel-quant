"""
AI Model API Endpoints
Endpoints for model training and status
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any

from api.deps import CurrentUser
from core.config import settings

router = APIRouter()


class TrainRequest(BaseModel):
    """Model training request"""
    symbol: str = "BTC/USDT"
    lstm_epochs: int = 50
    ppo_timesteps: int = 100000
    lookback_days: int = 365


class TrainResponse(BaseModel):
    """Training response"""
    status: str
    message: str
    task_id: Optional[str] = None


class ModelStatus(BaseModel):
    """Model status response"""
    is_ready: bool
    lstm_trained: bool
    ppo_trained: bool
    confidence_threshold: float
    last_training: Optional[str] = None


# Global model instance (initialized on demand)
_hybrid_intelligence = None


def get_hybrid_intelligence():
    """Get or create HybridIntelligence instance"""
    global _hybrid_intelligence
    
    if _hybrid_intelligence is None:
        try:
            from engine.models import HybridIntelligence
            _hybrid_intelligence = HybridIntelligence(
                sequence_length=60,
                model_dir="models",
                confidence_threshold=settings.CONFIDENCE_THRESHOLD
            )
            # Try to load existing models
            _hybrid_intelligence.load()
        except ImportError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"AI models not available: {e}"
            )
    
    return _hybrid_intelligence


@router.get("/status", response_model=ModelStatus)
async def get_model_status(current_user: CurrentUser):
    """Get AI model training status"""
    try:
        hybrid = get_hybrid_intelligence()
        status_info = hybrid.get_status()
        return ModelStatus(**status_info)
    except HTTPException:
        return ModelStatus(
            is_ready=False,
            lstm_trained=False,
            ppo_trained=False,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD
        )


@router.post("/train", response_model=TrainResponse)
async def train_models(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser
):
    """
    Start model training (runs in background).
    
    Note: Training can take 10-30 minutes depending on data size.
    """
    # Check admin permission (only admins can train)
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only active users can train models"
        )
    
    # Add training task to background
    background_tasks.add_task(
        _run_training,
        request.symbol,
        request.lstm_epochs,
        request.ppo_timesteps,
        request.lookback_days
    )
    
    return TrainResponse(
        status="started",
        message=f"Training started for {request.symbol}. This may take 10-30 minutes."
    )


async def _run_training(
    symbol: str,
    lstm_epochs: int,
    ppo_timesteps: int,
    lookback_days: int
):
    """Background training task"""
    import logging
    import pandas as pd
    import numpy as np
    
    logger = logging.getLogger(__name__)
    
    try:
        hybrid = get_hybrid_intelligence()
        
        # TODO: Fetch real historical data from exchange
        # For now, generate mock data for testing
        logger.info(f"Starting training for {symbol}...")
        
        # Generate mock OHLCV data
        n_samples = lookback_days * 24  # Hourly data
        np.random.seed(42)
        
        base_price = 40000 if "BTC" in symbol else 2000
        returns = np.random.normal(0, 0.02, n_samples)
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq='H'),
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_samples)),
            'high': prices * (1 + np.random.uniform(0, 0.02, n_samples)),
            'low': prices * (1 - np.random.uniform(0, 0.02, n_samples)),
            'close': prices,
            'volume': np.random.uniform(100, 10000, n_samples)
        })
        
        # Train models
        result = hybrid.train(
            df=df,
            lstm_epochs=lstm_epochs,
            ppo_timesteps=ppo_timesteps
        )
        
        logger.info(f"Training complete: {result}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")


@router.get("/predict/{symbol}")
async def get_prediction(symbol: str, current_user: CurrentUser) -> Dict[str, Any]:
    """Get AI prediction for a symbol"""
    import pandas as pd
    import numpy as np
    
    hybrid = get_hybrid_intelligence()
    
    if not hybrid.is_ready:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "reasoning": "Models not trained yet",
            "is_ready": False
        }
    
    # TODO: Get real market data
    # Mock data for testing
    n_samples = 100
    base_price = 40000 if "BTC" in symbol else 2000
    prices = base_price * (1 + np.cumsum(np.random.normal(0, 0.01, n_samples)))
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq='H'),
        'open': prices * 0.999,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.uniform(100, 10000, n_samples)
    })
    
    signal = hybrid.predict(df)
    
    return {
        "action": signal.action,
        "confidence": signal.confidence,
        "position_size": signal.position_size,
        "reasoning": signal.reasoning,
        "lstm_direction": signal.lstm_prediction.predicted_direction,
        "ppo_action": signal.ppo_action.action,
        "is_ready": True
    }
