from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DeviceMode = Literal["auto", "cpu", "cuda", "mps"]
PoolingMode = Literal["last", "mean", "cls"]
PositionEncodingMode = Literal["sinusoidal", "learned"]


@dataclass(frozen=True)
class CommonTrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    gradient_clip_norm: float | None = 1.0
    seed: int = 1
    device: DeviceMode = "auto"
    deterministic: bool = True
    num_workers: int = 0

    def __post_init__(self) -> None:
        # TODO(neural-config-common-1): Require positive epochs, batch size,
        # learning rate and patience; non-negative weight decay, min delta, seed
        # and worker count; and a positive gradient clip when it is enabled.
        #
        # TODO(neural-config-common-2): Validate the requested device against the
        # declared `DeviceMode` values. Availability belongs to `resolve_device`.
        raise NotImplementedError


@dataclass(frozen=True)
class RNNConfig(CommonTrainingConfig):
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False
    nonlinearity: Literal["tanh", "relu"] = "tanh"

    def __post_init__(self) -> None:
        # TODO(rnn-config-1): Call common validation, then require positive hidden
        # size/layer count, dropout in `[0, 1)`, and a supported nonlinearity.
        # Warn or normalize dropout to zero when `num_layers == 1`, because
        # PyTorch applies recurrent dropout only between stacked layers.
        raise NotImplementedError


@dataclass(frozen=True)
class LSTMConfig(CommonTrainingConfig):
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False

    def __post_init__(self) -> None:
        # TODO(lstm-config-1): Call common validation, then validate hidden size,
        # layer count and dropout. Apply the same one-layer dropout policy as RNN.
        raise NotImplementedError


@dataclass(frozen=True)
class GRUConfig(CommonTrainingConfig):
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False

    def __post_init__(self) -> None:
        # TODO(gru-config-1): Call common validation, then validate hidden size,
        # layer count and dropout. Apply the same one-layer dropout policy as RNN.
        raise NotImplementedError


@dataclass(frozen=True)
class TransformerConfig(CommonTrainingConfig):
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    pooling: PoolingMode = "last"
    positional_encoding: PositionEncodingMode = "sinusoidal"
    causal_attention: bool = False
    activation: Literal["relu", "gelu"] = "gelu"

    def __post_init__(self) -> None:
        # TODO(transformer-config-1): Call common validation and require positive
        # dimensions/layer count, dropout in `[0, 1)`, supported pooling,
        # positional encoding and activation values.
        #
        # TODO(transformer-config-2): Require `d_model % n_heads == 0`. Reject the
        # config immediately rather than waiting for a PyTorch construction error.
        raise NotImplementedError


__all__ = [
    "CommonTrainingConfig",
    "DeviceMode",
    "GRUConfig",
    "LSTMConfig",
    "PoolingMode",
    "PositionEncodingMode",
    "RNNConfig",
    "TransformerConfig",
]
