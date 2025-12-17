"""
Sentinel Quant - PPO Training Script
Trains PPO agent for trading decisions using Stable-Baselines3
"""
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Custom trading environment for PPO training.
    
    Actions: 0=HOLD, 1=BUY, 2=SELL
    State: Technical features + position info
    Reward: Based on PnL
    """
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, 
                 commission: float = 0.001, window_size: int = 60):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        
        # Compute features
        self.features = self._compute_features()
        self.n_features = self.features.shape[1] + 2  # +2 for position and pnl
        
        # Action space: HOLD, BUY, SELL
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.n_features), 
            dtype=np.float32
        )
        
        self.reset()
    
    def _compute_features(self) -> np.ndarray:
        """Compute technical features"""
        features = {}
        
        features['returns'] = self.df['close'].pct_change().fillna(0)
        features['hl_ratio'] = (self.df['high'] - self.df['low']) / self.df['close']
        
        for period in [5, 10, 20]:
            sma = self.df['close'].rolling(period).mean()
            features[f'sma_{period}_ratio'] = (self.df['close'] / sma).fillna(1)
        
        # RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = ((100 - (100 / (1 + rs))) / 100).fillna(0.5)
        
        # Normalize volume
        features['volume_ratio'] = (self.df['volume'] / self.df['volume'].rolling(20).mean()).fillna(1)
        
        feature_df = pd.DataFrame(features)
        feature_df = feature_df.replace([np.inf, -np.inf], 0).fillna(0)
        
        return feature_df.values.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0  # BTC held
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Get current observation"""
        start_idx = self.current_step - self.window_size
        feature_window = self.features[start_idx:self.current_step]
        
        # Add position info to each timestep
        position_info = np.full((self.window_size, 2), 
                                [1 if self.position > 0 else 0, self.total_pnl / self.initial_balance])
        
        obs = np.concatenate([feature_window, position_info], axis=1)
        return obs.astype(np.float32)
    
    def step(self, action):
        """Execute trading action"""
        price = self.df.loc[self.current_step, 'close']
        reward = 0.0
        
        # Execute action
        if action == 1:  # BUY
            if self.position == 0:  # Only buy if no position
                amount_to_buy = self.balance * 0.95  # Use 95% of balance
                self.position = amount_to_buy / price * (1 - self.commission)
                self.balance -= amount_to_buy
                self.entry_price = price
                self.trades += 1
                reward = -0.001  # Small penalty for trading (to avoid overtrading)
        
        elif action == 2:  # SELL
            if self.position > 0:  # Only sell if has position
                sell_value = self.position * price * (1 - self.commission)
                pnl = sell_value - (self.position * self.entry_price)
                self.total_pnl += pnl
                self.balance += sell_value
                self.position = 0
                self.trades += 1
                
                # Reward based on PnL
                reward = pnl / self.initial_balance * 100  # Percentage gain
        
        else:  # HOLD
            if self.position > 0:
                # Unrealized PnL
                unrealized_pnl = (price - self.entry_price) * self.position
                reward = unrealized_pnl / self.initial_balance * 0.1  # Small reward for holding profitable position
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        if done:
            # Close any open position at end
            if self.position > 0:
                sell_value = self.position * price * (1 - self.commission)
                pnl = sell_value - (self.position * self.entry_price)
                self.total_pnl += pnl
                self.balance += sell_value
        
        obs = self._get_obs() if not done else np.zeros_like(self._get_obs())
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_pnl': self.total_pnl,
            'trades': self.trades
        }
        
        return obs, reward, done, truncated, info


class TradingCallback(BaseCallback):
    """Callback to log training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if self.model.ep_info_buffer:
            mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            logger.info(f"Episode reward: {mean_reward:.2f}")


def train_ppo(data_path: str = "btc_historical.csv",
              total_timesteps: int = 200000,
              save_path: str = "../models/ppo_model"):
    """Train PPO agent"""
    
    # Load data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Split data (80% train, 20% eval)
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx].copy()
    eval_df = df[split_idx:].copy()
    
    logger.info(f"Train: {len(train_df)} rows, Eval: {len(eval_df)} rows")
    
    # Create environments
    train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
    eval_env = DummyVecEnv([lambda: TradingEnv(eval_df)])
    
    # Create model
    logger.info("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./ppo_tensorboard/"
    )
    
    # Callbacks
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(save_path),
        log_path="./ppo_logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, TradingCallback()]
    )
    
    # Save final model
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")
    
    # Evaluate
    logger.info("Evaluating model...")
    eval_env_single = TradingEnv(eval_df)
    obs, _ = eval_env_single.reset()
    
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env_single.step(action)
        total_reward += reward
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  Total PnL: ${info['total_pnl']:.2f}")
    logger.info(f"  Total Trades: {info['trades']}")
    logger.info(f"  Final Balance: ${info['balance']:.2f}")
    
    return model


if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), "btc_historical.csv")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "ppo_model")
    
    train_ppo(data_path=data_file, total_timesteps=200000, save_path=model_path)
