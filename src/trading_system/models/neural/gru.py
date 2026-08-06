from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trading_system.models.specs import ModelBuildContext

from .base import TorchSequenceClassifier
from .config import GRUConfig


def build_gru_module(
    context: ModelBuildContext,
    config: GRUConfig,
    torch_module: Any,
    nn_module: Any,
) -> Any:
    # TODO(gru-module-1): Build `nn.GRU(batch_first=True)` and a linear class head
    # using the configured hidden size, layers, dropout and directions.
    #
    # TODO(gru-module-2): In `forward`, select the final layer's hidden state,
    # concatenate directions only when enabled, and return unnormalized logits.
    raise NotImplementedError


class GRUClassifier(TorchSequenceClassifier):
    model_name = "gru"

    def __init__(self, context: ModelBuildContext, config: GRUConfig):
        # TODO(gru-classifier-init): Store typed config and delegate all common
        # setup/training behavior to `TorchSequenceClassifier`.
        raise NotImplementedError

    def _build_module(self, torch_module: Any, nn_module: Any) -> Any:
        # TODO(gru-classifier-build): Delegate to `build_gru_module` only.
        raise NotImplementedError


def create_gru_classifier(
    context: ModelBuildContext,
    parameters: Mapping[str, Any],
) -> GRUClassifier:
    # TODO(gru-factory): Validate parameter keys, build `GRUConfig`, reconcile
    # seed/device with context, and return the classifier.
    raise NotImplementedError


__all__ = ["GRUClassifier", "build_gru_module", "create_gru_classifier"]
