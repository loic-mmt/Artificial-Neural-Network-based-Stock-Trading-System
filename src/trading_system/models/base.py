from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


@dataclass
class FitResult:
    best_epoch: int
    stop_reason: str
    history: TrainingHistory


@runtime_checkable
class ProbabilisticClassifier(Protocol):
    classes_: np.ndarray

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class ProbabilisticSequenceClassifier(Protocol):
    """Future model contract for canonical 3D sequence inputs."""

    classes_: np.ndarray
    model_name: str

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        # TODO(sequence-contract-fit): Implementations must validate
        # `(N, T, F)` inputs, preserve the fixed Sell/Hold/Buy schema, train only
        # from train/validation, and return the framework-neutral `FitResult`.
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO(sequence-contract-predict): Implementations must accept 3D input
        # and return finite normalized probabilities shaped `(N, 3)` in fixed
        # class order. Inference must not update weights or preprocessing state.
        ...

    def state_dict(self) -> dict[str, object]:
        # TODO(sequence-contract-state): Return framework-independent metadata
        # plus copy-safe model state suitable for the artifact serializer.
        ...


__all__ = [
    "FitResult",
    "ProbabilisticClassifier",
    "ProbabilisticSequenceClassifier",
    "TrainingHistory",
]
