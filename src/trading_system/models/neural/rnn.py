from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trading_system.models.specs import ModelBuildContext

from .base import TorchSequenceClassifier
from .config import RNNConfig


def build_rnn_module(
    context: ModelBuildContext,
    config: RNNConfig,
    torch_module: Any,
    nn_module: Any,
) -> Any:
    # TODO(rnn-module-1): Create an `nn.Module` containing `nn.RNN` with
    # `batch_first=True`, the configured nonlinearity/layers/dropout/directions,
    # and a linear classification head producing `context.num_classes` logits.
    #
    # TODO(rnn-module-2): In `forward`, consume `(B, T, F)`, obtain the final
    # hidden state, select the last layer correctly, concatenate forward/backward
    # directions only when configured, and return logits without softmax.
    raise NotImplementedError


class RNNClassifier(TorchSequenceClassifier):
    model_name = "rnn"

    def __init__(self, context: ModelBuildContext, config: RNNConfig):
        # TODO(rnn-classifier-init): Store typed config and delegate common setup
        # to `TorchSequenceClassifier`; do not duplicate training code.
        raise NotImplementedError

    def _build_module(self, torch_module: Any, nn_module: Any) -> Any:
        # TODO(rnn-classifier-build): Delegate only architecture construction to
        # `build_rnn_module` with this instance's context and typed config.
        raise NotImplementedError


def create_rnn_classifier(
    context: ModelBuildContext,
    parameters: Mapping[str, Any],
) -> RNNClassifier:
    # TODO(rnn-factory): Reject unknown keys, construct `RNNConfig` from a copy of
    # parameters, reconcile context seed/device with explicit config policy, and
    # return `RNNClassifier`.
    raise NotImplementedError


__all__ = ["RNNClassifier", "build_rnn_module", "create_rnn_classifier"]
