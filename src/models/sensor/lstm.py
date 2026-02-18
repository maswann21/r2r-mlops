"""
LSTM Models for Time-series Sensor Data Prediction
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset for time-series sensor data"""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 10
    ):
        """
        Args:
            X: Features of shape (n_samples, n_features)
            y: Labels of shape (n_samples,)
            sequence_length: Length of time series sequences
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y = self.y[idx + self.sequence_length - 1]
        return X_seq, y


class LSTMModel(pl.LightningModule):
    """LSTM-based Time Series Classification Model"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,  # Number of defect types
        dropout: float = 0.2,
        bidirectional: bool = True,
        learning_rate: float = 0.001
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden size of LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Fully connected layers
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last output
        last_output = lstm_out[:, -1, :]

        # Fully connected
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        """Training step"""
        X, y = batch
        logits = self(X)
        loss = self.criterion(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        X, y = batch
        logits = self(X)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
