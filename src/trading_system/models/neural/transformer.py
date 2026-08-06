from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trading_system.models.specs import ModelBuildContext

from .base import TorchSequenceClassifier
from .config import TransformerConfig


def sinusoidal_positional_encoding(
    context_len: int,
    d_model: int,
    *,
    torch_module: Any,
) -> Any:
    # TODO(transformer-position-sinusoidal): Validate dimensions and build the
    # standard sine/cosine matrix with shape `(1, context_len, d_model)`. Handle an
    # odd `d_model`, use stable float32 calculations, and return a non-trainable
    # tensor suitable for module buffer registration.
    raise NotImplementedError


def build_causal_attention_mask(context_len: int, *, torch_module: Any) -> Any:
    # TODO(transformer-causal-mask): Build a square mask that prevents a timestep
    # from attending to later timesteps. Match the mask dtype/API expected by the
    # supported PyTorch encoder and test the exact allowed triangle.
    raise NotImplementedError


def build_transformer_module(
    context: ModelBuildContext,
    config: TransformerConfig,
    torch_module: Any,
    nn_module: Any,
) -> Any:
    # TODO(transformer-module-1): Build a feature projection `F -> d_model`,
    # positional encoding, `TransformerEncoder` with `batch_first=True`, pooling,
    # final LayerNorm and a linear class head.
    #
    # TODO(transformer-module-2): Support sinusoidal and learned positions. For a
    # CLS token, increase the effective sequence length and keep its position/mask
    # handling explicit. Reject sequences longer than the configured context.
    #
    # TODO(transformer-module-3): In `forward`, apply the optional causal mask
    # consistently, implement `last`, `mean` and `cls` pooling, then return logits
    # without applying softmax.
    raise NotImplementedError


class TransformerClassifier(TorchSequenceClassifier):
    model_name = "transformer"

    def __init__(self, context: ModelBuildContext, config: TransformerConfig):
        # TODO(transformer-classifier-init): Store typed config and delegate common
        # setup/training behavior to `TorchSequenceClassifier`.
        raise NotImplementedError

    def _build_module(self, torch_module: Any, nn_module: Any) -> Any:
        # TODO(transformer-classifier-build): Delegate only to
        # `build_transformer_module`; keep optimization in the shared trainer.
        raise NotImplementedError


def create_transformer_classifier(
    context: ModelBuildContext,
    parameters: Mapping[str, Any],
) -> TransformerClassifier:
    # TODO(transformer-factory): Validate parameter keys, build typed config,
    # reconcile seed/device with context, and return the classifier.
    raise NotImplementedError


__all__ = [
    "TransformerClassifier",
    "build_causal_attention_mask",
    "build_transformer_module",
    "create_transformer_classifier",
    "sinusoidal_positional_encoding",
]
