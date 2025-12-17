"""
Sentinel Quant - LSTM Training Script
Trains LSTM model for price prediction on GPU
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


class LSTMPredictor(nn.Module):
    """LSTM model for price direction prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


def compute_features(df: pd.DataFrame) -> np.ndarray:
    """Compute technical features from OHLCV data"""
    features = {}
    
    # Returns
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Price ratios
    features['hl_ratio'] = (df['high'] - df['low']) / df['close']
    features['oc_ratio'] = (df['close'] - df['open']) / df['open']
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        sma = df['close'].rolling(period).mean()
        features[f'sma_{period}_ratio'] = df['close'] / sma
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = (100 - (100 / (1 + rs))) / 100  # Normalize to 0-1
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    features['macd'] = (ema_12 - ema_26) / df['close']
    
    # Volatility
    features['volatility'] = features['returns'].rolling(20).std()
    
    # Volume
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Create DataFrame and clean
    feature_df = pd.DataFrame(features)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(method='ffill').fillna(0)
    
    return feature_df.values


def create_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int = 60):
    """Create sequences for LSTM training"""
    X, y = [], []
    
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(targets[i + seq_len])
    
    return np.array(X), np.array(y)


def train_lstm(data_path: str = "btc_historical.csv", 
               epochs: int = 50,
               batch_size: int = 64,
               seq_len: int = 60,
               save_path: str = "../models/lstm_model.pt"):
    """Train LSTM model"""
    
    # Load data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Compute features
    logger.info("Computing features...")
    features = compute_features(df)
    
    # Target: next-step return direction (1 = up, -1 = down)
    returns = df['close'].pct_change().shift(-1).values
    targets = np.sign(returns)
    
    # Remove NaN rows
    valid_idx = ~(np.isnan(features).any(axis=1) | np.isnan(targets))
    features = features[valid_idx]
    targets = targets[valid_idx]
    
    logger.info(f"Features shape: {features.shape}")
    
    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1
    features = (features - mean) / std
    
    # Create sequences
    logger.info(f"Creating sequences (seq_len={seq_len})...")
    X, y = create_sequences(features, targets, seq_len)
    logger.info(f"Sequences: {X.shape}, Targets: {y.shape}")
    
    # Train/val split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    input_size = X.shape[2]
    model = LSTMPredictor(input_size=input_size).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            
            # Calculate accuracy (direction prediction)
            predictions = torch.sign(val_outputs)
            accuracy = (predictions == y_val).float().mean().item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'mean': mean,
                'std': std,
                'seq_len': seq_len
            }, save_path)
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Accuracy: {accuracy:.2%}")
    
    logger.info(f"Training complete! Best val loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to {save_path}")
    
    return model


if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), "btc_historical.csv")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "lstm_model.pt")
    
    train_lstm(data_path=data_file, epochs=50, save_path=model_path)
