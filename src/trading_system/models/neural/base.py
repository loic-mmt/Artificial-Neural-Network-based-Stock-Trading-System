from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from trading_system.models.base import FitResult
from trading_system.models.specs import ModelBuildContext

from .config import CommonTrainingConfig


class TorchSequenceClassifier:
    """Framework adapter shared by all future PyTorch sequence classifiers."""

    model_name = "torch_sequence_base"

    def __init__(
        self,
        context: ModelBuildContext,
        config: CommonTrainingConfig,
    ) -> None:
        # TODO(torch-classifier-init-1): Store validated context/config, expose
        # `classes_ = np.arange(num_classes)`, and resolve no data-dependent state.
        #
        # TODO(torch-classifier-init-2): Require torch lazily, build the network
        # through `_build_module`, resolve the device once, and initialize fitted
        # state/history. Do not train or inspect global data in the constructor.
        raise NotImplementedError

    def _build_module(self, torch_module: Any, nn_module: Any) -> Any:
        # TODO(torch-classifier-build): Override in each architecture and return
        # an `nn.Module` mapping `(batch, time, features)` to `(batch, classes)`
        # logits. Base implementation must remain abstract.
        raise NotImplementedError

    def _validate_sequences(
        self, X: np.ndarray, *, allow_empty: bool = False
    ) -> np.ndarray:
        # TODO(torch-classifier-validate): Convert to float32, require 3D finite
        # input, compare `T` and `F` with `ModelBuildContext`, and control whether
        # an empty batch is accepted. Return a contiguous array.
        raise NotImplementedError

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        # TODO(torch-classifier-fit): Require validation arrays together, validate
        # all dimensions/labels, call `fit_torch_model`, store the result and mark
        # the instance fitted. Keep the architecture out of the training loop.
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO(torch-classifier-predict): Require fitted state, validate sequences
        # and call `predict_torch_probabilities`. Verify output class count and
        # fixed class ordering before returning.
        raise NotImplementedError

    def parameter_count(self) -> int:
        # TODO(torch-classifier-parameters): Sum trainable parameter counts only
        # and return a plain integer for experiment metadata.
        raise NotImplementedError

    def state_dict(self) -> dict[str, object]:
        # TODO(torch-classifier-state): Export model name, build context, typed
        # config, fitted flag and cloned CPU tensor state. Do not include optimizer
        # state unless resume-training support is added explicitly.
        raise NotImplementedError

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        # TODO(torch-classifier-load): Verify architecture, context dimensions,
        # classes and config before strict tensor restoration. Mark fitted only
        # after every validation succeeds.
        raise NotImplementedError


__all__ = ["TorchSequenceClassifier"]
