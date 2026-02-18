"""
GRU Models for Time-series Sensor Data Prediction
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GRUModel(pl.LightningModule):
    """GRU-based Time Series Classification Model (lighter than LSTM)"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.2,
        bidirectional: bool = True,
        learning_rate: float = 0.001
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden size of GRU
            num_layers: Number of GRU layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional GRU
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Fully connected layers
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(gru_output_size, 128)
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
            Output logits
        """
        # GRU
        gru_out, h_n = self.gru(x)

        # Use last output
        last_output = gru_out[:, -1, :]

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

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50
        )
        return [optimizer], [scheduler]
