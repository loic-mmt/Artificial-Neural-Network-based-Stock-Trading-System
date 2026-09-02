from __future__ import annotations

from collections.abc import Mapping
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

import numpy as np

from trading_system.models.base import FitResult
from trading_system.models.specs import ModelBuildContext

from .config import CommonTrainingConfig
from .trainer import (
    clone_torch_state_dict,
    fit_torch_model,
    predict_torch_probabilities,
    require_torch,
    resolve_device,
    restore_torch_state_dict,
    seed_torch_run,
)


class TorchSequenceClassifier(ABC):
    """Shared framework adapter for 3D PyTorch sequence classifiers."""

    model_name = "torch_sequence_base"

    def __init__(
        self,
        context: ModelBuildContext,
        config: CommonTrainingConfig,
    ) -> None:
        if not isinstance(context, ModelBuildContext):
            raise TypeError("context must be a ModelBuildContext.")
        if not isinstance(config, CommonTrainingConfig):
            raise TypeError("config must be a CommonTrainingConfig.")
        if context.num_classes != 3:
            raise ValueError("Trading sequence classifiers require three classes.")
        if config.seed != context.seed or config.device != context.device:
            raise ValueError("Model config seed/device must match ModelBuildContext.")
        self.context = context
        self.config = config
        self.classes_ = np.arange(context.num_classes, dtype=np.int64)
        self.torch, self.nn = require_torch()
        self.device = resolve_device(config.device, self.torch)
        seed_torch_run(config.seed, config.deterministic, self.torch)
        module = self._build_module(self.torch, self.nn)
        if not isinstance(module, self.nn.Module):
            raise TypeError("_build_module must return torch.nn.Module.")
        self.module = module.to(self.device)
        self.fit_result_: FitResult | None = None
        self.fitted_ = False

    @abstractmethod
    def _build_module(self, torch_module: Any, nn_module: Any) -> Any:
        """Build a module mapping (batch, time, features) to class logits."""

    def _validate_sequences(
        self, X: np.ndarray, *, allow_empty: bool = False
    ) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError("X must have shape (N, T, F).")
        if not allow_empty and len(values) == 0:
            raise ValueError("X cannot be empty.")
        if values.shape[1:] != (self.context.context_len, self.context.input_size):
            raise ValueError(
                "X sequence dimensions do not match ModelBuildContext."
            )
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite values.")
        return np.ascontiguousarray(values)

    def _validate_labels(self, y: np.ndarray, rows: int) -> np.ndarray:
        labels = np.asarray(y)
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("y must contain integer labels.")
        labels = labels.astype(np.int64, copy=False)
        if labels.ndim != 1 or len(labels) != rows:
            raise ValueError("y must be a 1D array aligned with X.")
        if (labels < 0).any() or (labels >= self.context.num_classes).any():
            raise ValueError("y contains labels outside configured class range.")
        return np.ascontiguousarray(labels)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together.")
        train = self._validate_sequences(X_train)
        train_labels = self._validate_labels(y_train, len(train))
        validation = None
        validation_labels = None
        if X_val is not None and y_val is not None:
            validation = self._validate_sequences(X_val)
            validation_labels = self._validate_labels(y_val, len(validation))
        result = fit_torch_model(
            self.module,
            train,
            train_labels,
            validation,
            validation_labels,
            num_classes=self.context.num_classes,
            config=self.config,
        )
        self.fit_result_ = result
        self.fitted_ = True
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError(f"{type(self).__name__} is not fitted.")
        values = self._validate_sequences(X)
        probabilities = predict_torch_probabilities(
            self.module,
            values,
            batch_size=self.config.batch_size,
            device=self.device,
            torch_module=self.torch,
        )
        expected = (len(values), self.context.num_classes)
        if probabilities.shape != expected:
            raise RuntimeError(
                f"Model returned probabilities shaped {probabilities.shape}; "
                f"expected {expected}."
            )
        return probabilities

    def parameter_count(self) -> int:
        return int(
            sum(parameter.numel() for parameter in self.module.parameters() if parameter.requires_grad)
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "model_name": self.model_name,
            "context": asdict(self.context),
            "config": asdict(self.config),
            "classes": self.classes_.copy(),
            "fitted": self.fitted_,
            "module_state": clone_torch_state_dict(self.module, self.torch),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping.")
        required = {
            "format_version",
            "model_name",
            "context",
            "config",
            "classes",
            "fitted",
            "module_state",
        }
        missing = sorted(required - set(state))
        unexpected = sorted(set(state) - required)
        if missing or unexpected:
            raise ValueError(
                f"Invalid classifier state keys; missing={missing}, "
                f"unexpected={unexpected}."
            )
        if state["format_version"] != 1:
            raise ValueError("Unsupported classifier state format_version.")
        if state["model_name"] != self.model_name:
            raise ValueError("Classifier state model_name differs.")
        if state["context"] != asdict(self.context):
            raise ValueError("Classifier state context differs.")
        if state["config"] != asdict(self.config):
            raise ValueError("Classifier state config differs.")
        classes = np.asarray(state["classes"], dtype=np.int64)
        if not np.array_equal(classes, self.classes_):
            raise ValueError("Classifier state classes differ.")
        if not isinstance(state["fitted"], bool):
            raise TypeError("Classifier fitted state must be boolean.")
        module_state = state["module_state"]
        if not isinstance(module_state, Mapping):
            raise TypeError("module_state must be a mapping.")
        restore_torch_state_dict(
            self.module, module_state, torch_module=self.torch
        )
        self.fitted_ = state["fitted"]
        self.fit_result_ = None


__all__ = ["TorchSequenceClassifier"]
