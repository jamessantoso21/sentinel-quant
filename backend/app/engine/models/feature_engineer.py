"""
Feature Engineering for Trading Models
Transforms raw OHLCV data into features for ML models
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False
    logging.warning("ta library not installed, using basic indicators")

logger = logging.getLogger(__name__)


@dataclass
class MarketFeatures:
    """Computed market features"""
    prices: np.ndarray
    features: np.ndarray
    feature_names: List[str]
    timestamp: pd.Timestamp


class FeatureEngineer:
    """
    Transforms raw market data into ML-ready features.
    Computes technical indicators and normalizes data.
    """
    
    def __init__(
        self,
        lookback_periods: int = 60,
        include_volume: bool = True,
        normalize: bool = True
    ):
        self.lookback_periods = lookback_periods
        self.include_volume = include_volume
        self.normalize = normalize
        self.feature_names: List[str] = []
        
        # Normalization params
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
    
    def compute_features(self, df: pd.DataFrame) -> MarketFeatures:
        """
        Compute features from OHLCV dataframe.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume, timestamp]
            
        Returns:
            MarketFeatures object with computed features
        """
        if len(df) < self.lookback_periods:
            raise ValueError(f"Need at least {self.lookback_periods} rows, got {len(df)}")
        
        features = {}
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios
        features['hl_ratio'] = (df['high'] - df['low']) / df['close']
        features['oc_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}'] = sma
            features[f'sma_{period}_ratio'] = df['close'] / sma
            
            ema = df['close'].ewm(span=period).mean()
            features[f'ema_{period}'] = ema
            features[f'ema_{period}_ratio'] = df['close'] / ema
        
        # Volatility (ATR-like)
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        features['atr_14'] = tr.rolling(14).mean()
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Momentum
        features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        features['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # RSI
        features['rsi_14'] = self._compute_rsi(df['close'], 14)
        features['rsi_7'] = self._compute_rsi(df['close'], 7)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_upper'] = bb_sma + 2 * bb_std
        features['bb_lower'] = bb_sma - 2 * bb_std
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_sma
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volume features
        if self.include_volume and 'volume' in df.columns:
            features['volume_sma_20'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma_20']
            features['volume_momentum'] = df['volume'].pct_change()
        
        # High resolution TA indicators if available
        if HAS_TA:
            try:
                features['stoch_k'] = ta.momentum.stochrsi_k(df['close'])
                features['stoch_d'] = ta.momentum.stochrsi_d(df['close'])
                features['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
                features['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
            except Exception as e:
                logger.warning(f"Error computing TA indicators: {e}")
        
        # Convert to DataFrame and clean
        feature_df = pd.DataFrame(features)
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(method='ffill').fillna(0)
        
        # Store feature names
        self.feature_names = list(feature_df.columns)
        
        # Get numpy array
        feature_array = feature_df.values
        
        # Normalize if needed
        if self.normalize:
            feature_array = self._normalize(feature_array)
        
        return MarketFeatures(
            prices=df['close'].values,
            features=feature_array,
            feature_names=self.feature_names,
            timestamp=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else pd.Timestamp.now()
        )
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        if self._means is None:
            self._means = np.nanmean(features, axis=0)
            self._stds = np.nanstd(features, axis=0)
            self._stds[self._stds == 0] = 1  # Avoid division by zero
        
        return (features - self._means) / self._stds
    
    def fit_normalization(self, features: np.ndarray):
        """Fit normalization parameters from training data"""
        self._means = np.nanmean(features, axis=0)
        self._stds = np.nanstd(features, axis=0)
        self._stds[self._stds == 0] = 1
    
    def get_feature_count(self) -> int:
        """Get number of features"""
        return len(self.feature_names) if self.feature_names else 0
    
    def create_sequences(
        self,
        features: np.ndarray,
        sequence_length: int = 60
    ) -> np.ndarray:
        """
        Create sequences for LSTM input.
        
        Args:
            features: 2D array (timesteps, features)
            sequence_length: Number of timesteps per sequence
            
        Returns:
            3D array (samples, sequence_length, features)
        """
        sequences = []
        for i in range(len(features) - sequence_length + 1):
            sequences.append(features[i:i + sequence_length])
        return np.array(sequences)
