from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trading_system.models.specs import ModelBuildContext

from .base import TorchSequenceClassifier
from .config import LSTMConfig


def build_lstm_module(
    context: ModelBuildContext,
    config: LSTMConfig,
    torch_module: Any,
    nn_module: Any,
) -> Any:
    # TODO(lstm-module-1): Build `nn.LSTM(batch_first=True)` plus a classification
    # head. Respect layers, dropout and direction count; output exactly three (or
    # `context.num_classes`) logits.
    #
    # TODO(lstm-module-2): In `forward`, use the final hidden state `h_n`, not the
    # cell state `c_n`. Select the final layer and combine directions correctly,
    # then return logits without softmax.
    raise NotImplementedError


class LSTMClassifier(TorchSequenceClassifier):
    model_name = "lstm"

    def __init__(self, context: ModelBuildContext, config: LSTMConfig):
        # TODO(lstm-classifier-init): Store typed config and delegate all shared
        # setup/training behavior to `TorchSequenceClassifier`.
        raise NotImplementedError

    def _build_module(self, torch_module: Any, nn_module: Any) -> Any:
        # TODO(lstm-classifier-build): Delegate to `build_lstm_module` only.
        raise NotImplementedError


def create_lstm_classifier(
    context: ModelBuildContext,
    parameters: Mapping[str, Any],
) -> LSTMClassifier:
    # TODO(lstm-factory): Validate parameter keys, build `LSTMConfig`, reconcile
    # seed/device with context, and return the classifier.
    raise NotImplementedError


__all__ = ["LSTMClassifier", "build_lstm_module", "create_lstm_classifier"]
