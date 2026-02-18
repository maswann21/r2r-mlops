"""Vision Model Definitions for R2R Coating Defect Detection"""

from .classification import ResNetClassifier
from .detection import YOLODetector, FasterRCNNDetector
from .segmentation import UNetSegmentor, DeepLabSegmentor

__all__ = [
    "ResNetClassifier",
    "YOLODetector",
    "FasterRCNNDetector",
    "UNetSegmentor",
    "DeepLabSegmentor",
]
