"""
Sentinel Quant - Model Loader
Loads trained LSTM and PPO models for inference
"""
import os
import logging
import numpy as np
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
LSTM_PATH = os.path.join(MODELS_DIR, "lstm_model.pt")
PPO_PATH = os.path.join(MODELS_DIR, "ppo_model.zip")


class LSTMInference:
    """LSTM model for price prediction inference"""
    
    def __init__(self):
        self.model = None
        self.mean = None
        self.std = None
        self.seq_len = 60
        self.loaded = False
        
    def load(self) -> bool:
        """Load trained LSTM model"""
        if not os.path.exists(LSTM_PATH):
            logger.warning(f"LSTM model not found at {LSTM_PATH}")
            return False
        
        try:
            import torch
            
            checkpoint = torch.load(LSTM_PATH, map_location='cpu', weights_only=False)
            
            # Recreate model
            from training.train_lstm import LSTMPredictor
            self.model = LSTMPredictor(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.mean = checkpoint['mean']
            self.std = checkpoint['std']
            self.seq_len = checkpoint['seq_len']
            self.loaded = True
            
            logger.info("LSTM model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Optional[float]:
        """
        Predict price direction.
        
        Args:
            features: 2D array of shape (seq_len, n_features)
            
        Returns:
            float between -1 (bearish) and 1 (bullish), or None if failed
        """
        if not self.loaded:
            return None
        
        try:
            import torch
            
            # Normalize features
            features = (features - self.mean) / self.std
            
            # Ensure correct shape
            if len(features) < self.seq_len:
                return None
            features = features[-self.seq_len:]
            
            # Convert to tensor
            x = torch.FloatTensor(features).unsqueeze(0)  # (1, seq_len, features)
            
            # Predict
            with torch.no_grad():
                prediction = self.model(x).item()
            
            return prediction
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return None


class PPOInference:
    """PPO model for trading action inference"""
    
    def __init__(self):
        self.model = None
        self.loaded = False
        
    def load(self) -> bool:
        """Load trained PPO model"""
        if not os.path.exists(PPO_PATH):
            logger.warning(f"PPO model not found at {PPO_PATH}")
            return False
        
        try:
            from stable_baselines3 import PPO
            
            self.model = PPO.load(PPO_PATH)
            self.loaded = True
            
            logger.info("PPO model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PPO model: {e}")
            return False
    
    def predict(self, observation: np.ndarray) -> Optional[str]:
        """
        Predict trading action.
        
        Args:
            observation: Environment observation array
            
        Returns:
            str: "HOLD", "BUY", or "SELL", or None if failed
        """
        if not self.loaded:
            return None
        
        try:
            action, _ = self.model.predict(observation, deterministic=True)
            
            # Map action to string
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            return action_map.get(int(action), "HOLD")
            
        except Exception as e:
            logger.error(f"PPO prediction failed: {e}")
            return None


# Singleton instances
lstm_inference = LSTMInference()
ppo_inference = PPOInference()


def load_models():
    """Load all models"""
    results = {
        "lstm": lstm_inference.load(),
        "ppo": ppo_inference.load()
    }
    logger.info(f"Model loading results: {results}")
    return results
