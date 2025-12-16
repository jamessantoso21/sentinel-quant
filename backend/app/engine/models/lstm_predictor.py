"""
LSTM Price Predictor
Deep learning model for price movement prediction
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging
import os

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class PricePrediction:
    """LSTM prediction result"""
    predicted_direction: str  # UP, DOWN, NEUTRAL
    predicted_change: float   # Percentage change
    confidence: float         # 0-1 prediction confidence
    predictions: np.ndarray   # Raw prediction values


class LSTMModel(nn.Module):
    """PyTorch LSTM Model for price prediction"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence, features)
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output
        output = self.fc(context)
        return output


class LSTMPredictor:
    """
    LSTM-based price predictor for trading signals.
    Uses deep learning to predict price movements.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model: Optional[LSTMModel] = None
        self.optimizer: Optional[torch.optim.Adam] = None
        self.input_size: Optional[int] = None
        self.is_trained = False
        
        logger.info(f"LSTMPredictor initialized on {self.device}")
    
    def _build_model(self, input_size: int):
        """Build LSTM model"""
        self.input_size = input_size
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        logger.info(f"Built LSTM model with {input_size} input features")
    
    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10
    ) -> dict:
        """
        Train the LSTM model.
        
        Args:
            features: 3D array (samples, sequence_length, features)
            targets: 1D array of next-step returns/prices
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation data ratio
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Training history dict
        """
        if self.model is None:
            self._build_model(features.shape[2])
        
        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = targets[:split_idx], targets[split_idx:]
        
        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val).unsqueeze(1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self._best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f} - "
                    f"Val Loss: {val_loss:.6f}"
                )
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if hasattr(self, '_best_state'):
            self.model.load_state_dict(self._best_state)
        
        self.is_trained = True
        logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
        
        return history
    
    def predict(self, features: np.ndarray) -> PricePrediction:
        """
        Make price prediction.
        
        Args:
            features: 2D or 3D array of features
            
        Returns:
            PricePrediction with direction, change, and confidence
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning neutral prediction")
            return PricePrediction(
                predicted_direction="NEUTRAL",
                predicted_change=0.0,
                confidence=0.0,
                predictions=np.array([0.0])
            )
        
        self.model.eval()
        
        # Reshape if needed
        if features.ndim == 2:
            features = features.reshape(1, *features.shape)
        
        with torch.no_grad():
            X = torch.FloatTensor(features).to(self.device)
            predictions = self.model(X).cpu().numpy()
        
        # Get prediction
        pred_value = predictions[-1, 0]
        
        # Determine direction and confidence
        abs_change = abs(pred_value)
        
        if pred_value > 0.001:  # Threshold for UP
            direction = "UP"
            confidence = min(abs_change * 100, 0.95)  # Scale confidence
        elif pred_value < -0.001:  # Threshold for DOWN
            direction = "DOWN"
            confidence = min(abs_change * 100, 0.95)
        else:
            direction = "NEUTRAL"
            confidence = 0.5
        
        return PricePrediction(
            predicted_direction=direction,
            predicted_change=float(pred_value * 100),  # Convert to percentage
            confidence=float(confidence),
            predictions=predictions.flatten()
        )
    
    def save(self, path: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'is_trained': self.is_trained
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self._build_model(checkpoint['input_size'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.is_trained = checkpoint['is_trained']
        
        logger.info(f"Model loaded from {path}")
