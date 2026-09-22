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
from .gru_base import GRUVariantClassifier, create_gru_variant_classifier
from .lstm import LSTMClassifier
from .rnn import RNNClassifier
from .transformer import TransformerClassifier

__all__ = [
    "CommonTrainingConfig",
    "GRUConfig",
    "GRUClassifier",
    "GRUVariantClassifier",
    "LSTMConfig",
    "LSTMClassifier",
    "RNNConfig",
    "RNNClassifier",
    "TorchSequenceClassifier",
    "TransformerConfig",
    "TransformerClassifier",
    "create_gru_variant_classifier",
]
