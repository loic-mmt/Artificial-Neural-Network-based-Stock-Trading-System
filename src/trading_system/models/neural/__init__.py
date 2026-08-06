"""Optional PyTorch sequence-model scaffolding.

This package must remain importable without importing PyTorch eagerly.
"""

from .base import TorchSequenceClassifier
from .config import (
    CommonTrainingConfig,
    GRUConfig,
    LSTMConfig,
    RNNConfig,
    TransformerConfig,
)

__all__ = [
    "CommonTrainingConfig",
    "GRUConfig",
    "LSTMConfig",
    "RNNConfig",
    "TorchSequenceClassifier",
    "TransformerConfig",
]
