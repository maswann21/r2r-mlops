"""
1D CNN Models for Time-series Sensor Data Prediction
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)


class CNN1DModel(pl.LightningModule):
    """1D CNN-based Time Series Classification Model"""

    def __init__(
        self,
        input_size: int,
        num_classes: int = 5,
        num_filters: list = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Args:
            input_size: Number of input features (sequence length)
            num_classes: Number of output classes
            num_filters: List of filter sizes for each conv layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        if num_filters is None:
            num_filters = [32, 64, 128]

        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # 1D Convolutional layers
        layers = []
        in_channels = 1

        for out_channels in num_filters:
            layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate size after conv layers
        # After each pooling, size is halved
        conv_output_size = (input_size // (2 ** len(num_filters))) * num_filters[-1]

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 1, sequence_length)

        Returns:
            Output logits
        """
        # Convolutional layers
        x = self.conv_layers(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.fc1(x)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
