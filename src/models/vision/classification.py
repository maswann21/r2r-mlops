"""
ResNet-based Classification Model for Multi-label Defect Detection
Based on: R2Rmachine/CoatingVision/ResNet.ipynb
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import pytorch_lightning as pl
from sklearn.metrics import f1_score, recall_score, precision_score
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class CSVImageDataset(Dataset):
    """Dataset for loading images with CSV labels (Multi-label Classification)"""

    def __init__(
        self, 
        img_dir: str, 
        csv_path: str, 
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            img_dir: Directory containing images
            csv_path: Path to CSV file with labels
            transform: Optional image transformations
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        self.label_cols = [
            "Surface_Crack",
            "Delamination",
            "Pinhole",
            "unclassified"
        ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["file_name"])
        
        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load labels
        label = torch.tensor(
            row[self.label_cols].astype(float).values,
            dtype=torch.float32
        )

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label


class ResNetClassifier(pl.LightningModule):
    """ResNet18-based Multi-label Classification Model"""

    def __init__(
        self,
        num_classes: int = 4,
        learning_rate: float = 1e-4,
        pretrained: bool = True
    ):
        """
        Args:
            num_classes: Number of output classes (4: Surface_Crack, Delamination, Pinhole, unclassified)
            learning_rate: Learning rate for optimizer
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Load pretrained ResNet18
        if pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.model = models.resnet18(weights=None)

        # Modify final layer for multi-label classification
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Loss function (Binary Cross-Entropy for multi-label)
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.val_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step"""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        """Validation step"""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        self.val_outputs.append({
            "outputs": outputs.detach(),
            "labels": labels.detach()
        })

        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self) -> None:
        """Calculate metrics at end of validation epoch"""
        if not self.val_outputs:
            return

        # Concatenate all outputs and labels
        all_outputs = torch.cat([x["outputs"] for x in self.val_outputs])
        all_labels = torch.cat([x["labels"] for x in self.val_outputs])

        # Convert to predictions
        all_preds = (torch.sigmoid(all_outputs) > 0.5).cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        # Calculate multi-label metrics
        f1_macro = f1_score(all_labels_np, all_preds, average="macro", zero_division=0)
        recall_macro = recall_score(all_labels_np, all_preds, average="macro", zero_division=0)
        precision_macro = precision_score(all_labels_np, all_preds, average="macro", zero_division=0)

        self.log("val_f1_macro", f1_macro, prog_bar=True)
        self.log("val_recall_macro", recall_macro)
        self.log("val_precision_macro", precision_macro)

        self.val_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=3,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1_macro",
            }
        }


def get_transforms():
    """Get data transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def create_dataloaders(
    img_dir: str,
    csv_path: str,
    batch_size: int = 16,
    train_size: float = 0.8,
    num_workers: int = 4
) -> tuple:
    """Create train and validation dataloaders"""
    
    train_transform, val_transform = get_transforms()

    # Create dataset
    dataset = CSVImageDataset(
        img_dir=img_dir,
        csv_path=csv_path,
        transform=train_transform
    )

    # Split dataset
    train_size_int = int(train_size * len(dataset))
    val_size_int = len(dataset) - train_size_int
    train_ds, val_ds = random_split(dataset, [train_size_int, val_size_int])

    # Update transform for val dataset
    val_ds.dataset.transform = val_transform

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
