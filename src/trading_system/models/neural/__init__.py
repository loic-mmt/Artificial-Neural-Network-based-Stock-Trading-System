"""Optional PyTorch sequence models.

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
from .gru import GRUClassifier
from .lstm import LSTMClassifier
from .rnn import RNNClassifier
from .transformer import TransformerClassifier

__all__ = [
    "CommonTrainingConfig",
    "GRUConfig",
    "GRUClassifier",
    "LSTMConfig",
    "LSTMClassifier",
    "RNNConfig",
    "RNNClassifier",
    "TorchSequenceClassifier",
    "TransformerConfig",
    "TransformerClassifier",
]
