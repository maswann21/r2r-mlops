"""
Semantic Segmentation Models for Defect Region Segmentation
Support: UNet, DeepLab
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    fcn_resnet50
)
import pytorch_lightning as pl
from typing import Dict, Optional


class UNetSegmentor(pl.LightningModule):
    """U-Net based Semantic Segmentation Model"""

    def __init__(
        self,
        num_classes: int = 4,  # Background + 3 defect types
        learning_rate: float = 0.001,
        in_channels: int = 3
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            learning_rate: Learning rate
            in_channels: Input image channels (3 for RGB)
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.model = self._build_unet(in_channels, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def _build_unet(self, in_channels: int, num_classes: int) -> nn.Module:
        """Build U-Net architecture"""
        # Simplified U-Net structure
        class SimpleUNet(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                # Encoder
                self.maxpool = nn.MaxPool2d(2, 2)
                self.conv1 = self._conv_block(in_ch, 64)
                self.conv2 = self._conv_block(64, 128)
                self.conv3 = self._conv_block(128, 256)

                # Bottleneck
                self.conv4 = self._conv_block(256, 512)

                # Decoder
                self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                self.conv5 = self._conv_block(512, 256)
                self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.conv6 = self._conv_block(256, 128)
                self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.conv7 = self._conv_block(128, 64)

                # Final layer
                self.final = nn.Conv2d(64, out_ch, kernel_size=1)

            def _conv_block(self, in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                # Encoder with skip connections
                conv1 = self.conv1(x)
                x = self.maxpool(conv1)
                conv2 = self.conv2(x)
                x = self.maxpool(conv2)
                conv3 = self.conv3(x)
                x = self.maxpool(conv3)

                x = self.conv4(x)

                # Decoder with concatenation
                x = self.upconv3(x)
                x = torch.cat([x, conv3], 1)
                x = self.conv5(x)

                x = self.upconv2(x)
                x = torch.cat([x, conv2], 1)
                x = self.conv6(x)

                x = self.upconv1(x)
                x = torch.cat([x, conv1], 1)
                x = self.conv7(x)

                x = self.final(x)
                return x

        return SimpleUNet(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks.long())
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks.long())
        
        # Calculate IoU
        preds = logits.argmax(dim=1)
        iou = self._calculate_iou(preds, masks)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

    def _calculate_iou(self, pred, target):
        """Calculate Intersection over Union"""
        intersection = (pred == target).sum().item()
        union = (pred != target).sum().item() + (target != pred).sum().item()
        iou = intersection / (union + 1e-6) if union > 0 else 0
        return torch.tensor(iou)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class DeepLabSegmentor(pl.LightningModule):
    """DeepLabv3 based Semantic Segmentation Model"""

    def __init__(
        self,
        num_classes: int = 4,
        learning_rate: float = 0.001,
        pretrained: bool = True
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            learning_rate: Learning rate
            pretrained: Use pretrained weights
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Load pretrained DeepLabv3
        self.model = deeplabv3_resnet50(
            pretrained=pretrained,
            num_classes=num_classes,
            aux_loss=True
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> Dict:
        """Forward pass"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        images, masks = batch
        output = self(images)
        
        loss = self.criterion(output["out"], masks.long())
        if output.get("aux") is not None:
            aux_loss = self.criterion(output["aux"], masks.long())
            loss = loss + 0.4 * aux_loss
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images, masks = batch
        output = self(images)
        loss = self.criterion(output["out"], masks.long())
        
        # Calculate mIoU
        preds = output["out"].argmax(dim=1)
        miou = self._calculate_miou(preds, masks)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_miou", miou, prog_bar=True)

    def _calculate_miou(self, pred, target):
        """Calculate Mean Intersection over Union"""
        ious = []
        for cls in range(self.num_classes):
            pred_mask = (pred == cls).float()
            target_mask = (target == cls).float()
            
            intersection = (pred_mask * target_mask).sum().item()
            union = (pred_mask + target_mask).clamp(max=1).sum().item()
            
            iou = intersection / (union + 1e-6) if union > 0 else 0
            ious.append(iou)
        
        miou = sum(ious) / len(ious) if ious else 0
        return torch.tensor(miou)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=50,
            power=1.0
        )
        return [optimizer], [scheduler]
