"""
PPO Trading Agent
Proximal Policy Optimization for trading decisions
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
import os

try:
    import gymnasium as gym
    from gymnasium import spaces
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

logger = logging.getLogger(__name__)


@dataclass
class TradingAction:
    """PPO trading action result"""
    action: str           # BUY, SELL, HOLD
    position_size: float  # 0.0 to 1.0 (fraction of capital)
    confidence: float     # 0.0 to 1.0
    raw_action: int       # Raw action from PPO


class TradingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for trading.
    Used to train PPO agent.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0
    ):
        super().__init__()
        
        self.features = features
        self.prices = prices
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        # Observation space: features + position info
        n_features = features.shape[1] if len(features.shape) > 1 else features.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + 3,),  # features + [position, balance_ratio, unrealized_pnl]
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # Units held
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        features = self.features[self.current_step]
        if isinstance(features, np.ndarray):
            features = features.flatten()
        
        current_price = self.prices[self.current_step]
        
        # Position info
        position_value = self.position * current_price
        unrealized_pnl = 0.0
        if self.position != 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * np.sign(self.position)
        
        extra_features = np.array([
            self.position / self.max_position,  # Normalized position
            self.balance / self.initial_balance,  # Balance ratio
            unrealized_pnl  # Unrealized PnL
        ], dtype=np.float32)
        
        return np.concatenate([features, extra_features]).astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute trading action"""
        current_price = self.prices[self.current_step]
        reward = 0.0
        
        # Execute action
        if action == 1:  # BUY
            if self.position <= 0:  # Close short or open long
                # Close short if exists
                if self.position < 0:
                    pnl = (self.entry_price - current_price) * abs(self.position)
                    pnl -= abs(self.position) * current_price * self.transaction_cost
                    self.balance += pnl
                    reward += pnl / self.initial_balance
                    if pnl > 0:
                        self.winning_trades += 1
                    self.total_trades += 1
                
                # Open long
                position_value = self.balance * self.max_position * 0.5
                self.position = position_value / current_price
                self.entry_price = current_price
                self.balance -= position_value * self.transaction_cost
                
        elif action == 2:  # SELL
            if self.position >= 0:  # Close long or open short
                # Close long if exists
                if self.position > 0:
                    pnl = (current_price - self.entry_price) * self.position
                    pnl -= self.position * current_price * self.transaction_cost
                    self.balance += pnl
                    reward += pnl / self.initial_balance
                    if pnl > 0:
                        self.winning_trades += 1
                    self.total_trades += 1
                
                # Open short
                position_value = self.balance * self.max_position * 0.5
                self.position = -position_value / current_price
                self.entry_price = current_price
                self.balance -= position_value * self.transaction_cost
        
        # Small penalty for holding to encourage action
        if action == 0 and self.position != 0:
            reward -= 0.0001
        
        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False
        
        # Check for bankruptcy
        if self.balance <= 0:
            terminated = True
            reward = -1.0
        
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape)
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades
        }
        
        return obs, reward, terminated, truncated, info


class PPOTradingAgent:
    """
    PPO-based trading agent using Stable Baselines 3.
    Uses reinforcement learning for trading decisions.
    """
    
    def __init__(
        self,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        device: str = "auto"
    ):
        if not HAS_SB3:
            raise ImportError("Stable Baselines 3 required. Install with: pip install stable-baselines3")
        
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.clip_range = clip_range
        self.device = device
        
        self.model: Optional[PPO] = None
        self.is_trained = False
        
        logger.info("PPOTradingAgent initialized")
    
    def train(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        total_timesteps: int = 100000,
        initial_balance: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Train the PPO agent.
        
        Args:
            features: 2D array (timesteps, features)
            prices: 1D array of prices
            total_timesteps: Total training timesteps
            initial_balance: Starting balance for simulation
            
        Returns:
            Training info dict
        """
        # Create environment
        env = TradingEnvironment(
            features=features,
            prices=prices,
            initial_balance=initial_balance
        )
        
        # Wrap in DummyVecEnv for SB3
        vec_env = DummyVecEnv([lambda: env])
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            clip_range=self.clip_range,
            verbose=1,
            device=self.device
        )
        
        # Train
        logger.info(f"Starting PPO training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        
        self.is_trained = True
        
        # Get final performance
        final_info = self._evaluate(vec_env)
        
        logger.info(f"Training complete. Final balance: ${final_info.get('balance', 0):.2f}")
        
        return final_info
    
    def _evaluate(self, env: DummyVecEnv, n_episodes: int = 1) -> Dict[str, Any]:
        """Evaluate agent performance"""
        obs = env.reset()
        total_reward = 0
        info = {}
        
        for _ in range(10000):  # Max steps
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            
            if done[0]:
                break
        
        return {
            'total_reward': total_reward,
            'balance': info[0].get('balance', 0) if info else 0,
            'total_trades': info[0].get('total_trades', 0) if info else 0,
            'winning_trades': info[0].get('winning_trades', 0) if info else 0
        }
    
    def predict(self, features: np.ndarray) -> TradingAction:
        """
        Get trading action for current state.
        
        Args:
            features: Feature array for current state
            
        Returns:
            TradingAction with action, position size, and confidence
        """
        if not self.is_trained:
            logger.warning("Agent not trained, returning HOLD")
            return TradingAction(
                action="HOLD",
                position_size=0.0,
                confidence=0.0,
                raw_action=0
            )
        
        # Add position info (assume neutral state for prediction)
        extra_features = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        obs = np.concatenate([features.flatten(), extra_features])
        
        # Get action
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        
        # Get action probabilities for confidence
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs[0].numpy()
        
        confidence = float(probs[action])
        
        # Map action
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_str = action_map.get(action, "HOLD")
        
        # Position size based on confidence
        if action_str == "HOLD":
            position_size = 0.0
        else:
            position_size = min(confidence, 1.0)
        
        return TradingAction(
            action=action_str,
            position_size=position_size,
            confidence=confidence,
            raw_action=action
        )
    
    def save(self, path: str):
        """Save agent to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent from disk"""
        if not os.path.exists(path + ".zip"):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = PPO.load(path)
        self.is_trained = True
        logger.info(f"Agent loaded from {path}")
