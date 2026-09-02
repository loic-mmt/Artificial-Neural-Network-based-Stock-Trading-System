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
        for name in (
            "epochs", "batch_size", "early_stopping_patience", "seed", "num_workers"
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0 or self.early_stopping_min_delta < 0:
            raise ValueError("weight_decay and early_stopping_min_delta cannot be negative.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when enabled.")
        if self.seed < 0 or self.num_workers < 0:
            raise ValueError("seed and num_workers cannot be negative.")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'.")
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be boolean.")


@dataclass(frozen=True)
class RNNConfig(CommonTrainingConfig):
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False
    nonlinearity: Literal["tanh", "relu"] = "tanh"

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_recurrent(self.hidden_size, self.num_layers, self.dropout)
        if not isinstance(self.bidirectional, bool):
            raise TypeError("bidirectional must be boolean.")
        if self.nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity must be 'tanh' or 'relu'.")
        if self.num_layers == 1 and self.dropout:
            object.__setattr__(self, "dropout", 0.0)


@dataclass(frozen=True)
class LSTMConfig(CommonTrainingConfig):
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_recurrent(self.hidden_size, self.num_layers, self.dropout)
        if not isinstance(self.bidirectional, bool):
            raise TypeError("bidirectional must be boolean.")
        if self.num_layers == 1 and self.dropout:
            object.__setattr__(self, "dropout", 0.0)


@dataclass(frozen=True)
class GRUConfig(CommonTrainingConfig):
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_recurrent(self.hidden_size, self.num_layers, self.dropout)
        if not isinstance(self.bidirectional, bool):
            raise TypeError("bidirectional must be boolean.")
        if self.num_layers == 1 and self.dropout:
            object.__setattr__(self, "dropout", 0.0)


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
        super().__post_init__()
        for name in ("d_model", "n_heads", "num_layers", "dim_feedforward"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.pooling not in ("last", "mean", "cls"):
            raise ValueError("pooling must be 'last', 'mean', or 'cls'.")
        if self.positional_encoding not in ("sinusoidal", "learned"):
            raise ValueError("Unsupported positional_encoding.")
        if self.activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")


def _validate_recurrent(hidden_size: int, num_layers: int, dropout: float) -> None:
    for name, value in (("hidden_size", hidden_size), ("num_layers", num_layers)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1).")


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
