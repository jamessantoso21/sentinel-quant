"""
Hybrid Intelligence
Combines PPO (Reinforcement Learning) and LSTM (Deep Learning) for trading decisions
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import os

from .feature_engineer import FeatureEngineer, MarketFeatures
from .lstm_predictor import LSTMPredictor, PricePrediction
from .ppo_agent import PPOTradingAgent, TradingAction

logger = logging.getLogger(__name__)


@dataclass
class HybridSignal:
    """Combined signal from PPO and LSTM"""
    action: str                    # BUY, SELL, HOLD
    confidence: float              # Combined confidence 0-1
    ppo_action: TradingAction      # Raw PPO output
    lstm_prediction: PricePrediction  # Raw LSTM output
    position_size: float           # Recommended position size
    reasoning: str                 # Human-readable explanation


class HybridIntelligence:
    """
    Combines multiple AI models for robust trading decisions:
    
    1. PPO Agent: Learns trading policy through reinforcement learning
       - Trained on historical data with reward based on PnL
       - Outputs: BUY/SELL/HOLD with confidence
       
    2. LSTM Predictor: Predicts price movements
       - Trained on price sequences with attention mechanism
       - Outputs: Price direction and magnitude prediction
       
    The hybrid approach:
    - Both models must agree for high-confidence signals
    - Disagreement results in HOLD or reduced position size
    - LSTM provides direction hint, PPO provides action confirmation
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        model_dir: str = "models",
        confidence_threshold: float = 0.7
    ):
        self.sequence_length = sequence_length
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(
            lookback_periods=sequence_length,
            normalize=True
        )
        
        self.lstm_predictor = LSTMPredictor(
            sequence_length=sequence_length,
            hidden_size=128,
            num_layers=2
        )
        
        self.ppo_agent = PPOTradingAgent(
            learning_rate=3e-4,
            n_steps=2048
        )
        
        self.is_ready = False
        
        logger.info("HybridIntelligence initialized")
    
    def train(
        self,
        df: pd.DataFrame,
        lstm_epochs: int = 50,
        ppo_timesteps: int = 100000
    ) -> Dict[str, Any]:
        """
        Train both models on historical data.
        
        Args:
            df: DataFrame with OHLCV columns
            lstm_epochs: Training epochs for LSTM
            ppo_timesteps: Training timesteps for PPO
            
        Returns:
            Training results dict
        """
        logger.info("Starting Hybrid Intelligence training...")
        
        # Compute features
        market_features = self.feature_engineer.compute_features(df)
        
        # Create sequences for LSTM
        sequences = self.feature_engineer.create_sequences(
            market_features.features,
            self.sequence_length
        )
        
        # Targets: next-step returns
        returns = np.diff(market_features.prices) / market_features.prices[:-1]
        targets = returns[self.sequence_length - 1:]
        
        # Align sequences and targets
        min_len = min(len(sequences), len(targets))
        sequences = sequences[:min_len]
        targets = targets[:min_len]
        
        # Train LSTM
        logger.info("Training LSTM Predictor...")
        lstm_history = self.lstm_predictor.train(
            features=sequences,
            targets=targets,
            epochs=lstm_epochs,
            batch_size=32
        )
        
        # Train PPO
        logger.info("Training PPO Agent...")
        # Use non-sequential features for PPO
        ppo_features = market_features.features[self.sequence_length:]
        ppo_prices = market_features.prices[self.sequence_length:]
        
        ppo_result = self.ppo_agent.train(
            features=ppo_features,
            prices=ppo_prices,
            total_timesteps=ppo_timesteps
        )
        
        self.is_ready = True
        
        # Save models
        self.save()
        
        return {
            'lstm_history': lstm_history,
            'ppo_result': ppo_result,
            'is_ready': True
        }
    
    def predict(self, df: pd.DataFrame) -> HybridSignal:
        """
        Generate trading signal from current market data.
        
        Args:
            df: Recent OHLCV DataFrame (at least sequence_length rows)
            
        Returns:
            HybridSignal with combined recommendation
        """
        if not self.is_ready:
            return HybridSignal(
                action="HOLD",
                confidence=0.0,
                ppo_action=TradingAction("HOLD", 0.0, 0.0, 0),
                lstm_prediction=PricePrediction("NEUTRAL", 0.0, 0.0, np.array([0])),
                position_size=0.0,
                reasoning="Models not trained yet"
            )
        
        # Compute features
        market_features = self.feature_engineer.compute_features(df)
        
        # LSTM prediction
        sequence = market_features.features[-self.sequence_length:]
        lstm_pred = self.lstm_predictor.predict(sequence)
        
        # PPO action
        current_features = market_features.features[-1]
        ppo_action = self.ppo_agent.predict(current_features)
        
        # Combine signals
        signal = self._combine_signals(ppo_action, lstm_pred)
        
        return signal
    
    def _combine_signals(
        self,
        ppo: TradingAction,
        lstm: PricePrediction
    ) -> HybridSignal:
        """Combine PPO and LSTM signals into final decision"""
        
        # Map LSTM direction to action
        lstm_action = {
            "UP": "BUY",
            "DOWN": "SELL",
            "NEUTRAL": "HOLD"
        }.get(lstm.predicted_direction, "HOLD")
        
        # Agreement logic
        ppo_confidence = ppo.confidence
        lstm_confidence = lstm.confidence
        
        # Both agree
        if ppo.action == lstm_action and ppo.action != "HOLD":
            combined_confidence = (ppo_confidence + lstm_confidence) / 2
            action = ppo.action
            reasoning = f"Strong signal: PPO and LSTM agree on {action}"
            
        # PPO says act, LSTM neutral
        elif ppo.action != "HOLD" and lstm_action == "HOLD":
            combined_confidence = ppo_confidence * 0.7
            action = ppo.action
            reasoning = f"Moderate signal: PPO suggests {action}, LSTM neutral"
            
        # LSTM has direction, PPO neutral
        elif lstm_action != "HOLD" and ppo.action == "HOLD":
            combined_confidence = lstm_confidence * 0.6
            action = lstm_action
            reasoning = f"Weak signal: LSTM predicts {lstm.predicted_direction}, PPO holding"
            
        # Disagreement
        elif ppo.action != lstm_action and ppo.action != "HOLD" and lstm_action != "HOLD":
            combined_confidence = 0.3
            action = "HOLD"
            reasoning = f"Conflict: PPO says {ppo.action}, LSTM says {lstm.predicted_direction}"
            
        # Both neutral
        else:
            combined_confidence = 0.5
            action = "HOLD"
            reasoning = "No clear signal from either model"
        
        # Determine position size based on confidence
        if combined_confidence >= self.confidence_threshold:
            position_size = min(combined_confidence, 1.0)
        else:
            position_size = 0.0
            if action != "HOLD":
                action = "HOLD"
                reasoning += f" (Confidence {combined_confidence:.2%} below threshold)"
        
        return HybridSignal(
            action=action,
            confidence=combined_confidence,
            ppo_action=ppo,
            lstm_prediction=lstm,
            position_size=position_size,
            reasoning=reasoning
        )
    
    def save(self):
        """Save all models"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        if self.lstm_predictor.is_trained:
            self.lstm_predictor.save(os.path.join(self.model_dir, "lstm_predictor.pt"))
        
        if self.ppo_agent.is_trained:
            self.ppo_agent.save(os.path.join(self.model_dir, "ppo_agent"))
        
        logger.info(f"Models saved to {self.model_dir}")
    
    def load(self):
        """Load all models"""
        lstm_path = os.path.join(self.model_dir, "lstm_predictor.pt")
        ppo_path = os.path.join(self.model_dir, "ppo_agent")
        
        if os.path.exists(lstm_path):
            self.lstm_predictor.load(lstm_path)
        
        if os.path.exists(ppo_path + ".zip"):
            self.ppo_agent.load(ppo_path)
        
        self.is_ready = self.lstm_predictor.is_trained and self.ppo_agent.is_trained
        
        if self.is_ready:
            logger.info("Models loaded successfully")
        else:
            logger.warning("Some models not found or not trained")
        
        return self.is_ready
    
    def get_status(self) -> Dict[str, Any]:
        """Get model status"""
        return {
            'is_ready': self.is_ready,
            'lstm_trained': self.lstm_predictor.is_trained,
            'ppo_trained': self.ppo_agent.is_trained,
            'confidence_threshold': self.confidence_threshold,
            'sequence_length': self.sequence_length
        }
