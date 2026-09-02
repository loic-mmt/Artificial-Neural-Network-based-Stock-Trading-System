from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields
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
    if context_len <= 0 or d_model <= 0:
        raise ValueError("context_len and d_model must be positive.")
    position = torch_module.arange(context_len, dtype=torch_module.float32).unsqueeze(1)
    frequency = torch_module.exp(
        torch_module.arange(0, d_model, 2, dtype=torch_module.float32)
        * (-math.log(10_000.0) / d_model)
    )
    encoding = torch_module.zeros((context_len, d_model), dtype=torch_module.float32)
    encoding[:, 0::2] = torch_module.sin(position * frequency)
    odd_width = encoding[:, 1::2].shape[1]
    encoding[:, 1::2] = torch_module.cos(position * frequency[:odd_width])
    return encoding.unsqueeze(0)


def build_causal_attention_mask(context_len: int, *, torch_module: Any) -> Any:
    if context_len <= 0:
        raise ValueError("context_len must be positive.")
    return torch_module.triu(
        torch_module.ones((context_len, context_len), dtype=torch_module.bool),
        diagonal=1,
    )


def build_transformer_module(
    context: ModelBuildContext,
    config: TransformerConfig,
    torch_module: Any,
    nn_module: Any,
) -> Any:
    class TransformerModule(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn_module.Linear(context.input_size, config.d_model)
            self.use_cls = config.pooling == "cls"
            length = context.context_len + int(self.use_cls)
            if self.use_cls:
                self.cls_token = nn_module.Parameter(
                    torch_module.zeros((1, 1, config.d_model))
                )
            else:
                self.register_parameter("cls_token", None)
            if config.positional_encoding == "learned":
                self.position = nn_module.Parameter(
                    torch_module.zeros((1, length, config.d_model))
                )
                nn_module.init.normal_(self.position, mean=0.0, std=0.02)
            else:
                self.register_buffer(
                    "position",
                    sinusoidal_positional_encoding(
                        length, config.d_model, torch_module=torch_module
                    ),
                    persistent=True,
                )
            layer = nn_module.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=True,
            )
            self.encoder = nn_module.TransformerEncoder(layer, config.num_layers)
            self.norm = nn_module.LayerNorm(config.d_model)
            self.head = nn_module.Linear(config.d_model, context.num_classes)
            mask = (
                build_causal_attention_mask(length, torch_module=torch_module)
                if config.causal_attention
                else None
            )
            self.register_buffer("attention_mask", mask, persistent=False)

        def forward(self, sequences: Any) -> Any:
            if sequences.ndim != 3 or sequences.shape[1:] != (
                context.context_len,
                context.input_size,
            ):
                raise ValueError("Transformer input dimensions differ from context.")
            encoded = self.projection(sequences)
            if self.use_cls:
                token = self.cls_token.expand(len(sequences), -1, -1)
                encoded = torch_module.cat((encoded, token), dim=1)
            encoded = self.encoder(encoded + self.position, mask=self.attention_mask)
            if config.pooling in ("last", "cls"):
                pooled = encoded[:, -1]
            else:
                pooled = encoded.mean(dim=1)
            return self.head(self.norm(pooled))

    return TransformerModule()


class TransformerClassifier(TorchSequenceClassifier):
    model_name = "transformer"

    def __init__(self, context: ModelBuildContext, config: TransformerConfig):
        if not isinstance(config, TransformerConfig):
            raise TypeError("config must be TransformerConfig.")
        self.transformer_config = config
        super().__init__(context, config)

    def _build_module(self, torch_module: Any, nn_module: Any) -> Any:
        return build_transformer_module(
            self.context, self.transformer_config, torch_module, nn_module
        )


def create_transformer_classifier(
    context: ModelBuildContext,
    parameters: Mapping[str, Any],
) -> TransformerClassifier:
    allowed = {item.name for item in fields(TransformerConfig)}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"Unknown transformer parameters: {unknown}")
    values = dict(parameters)
    for name, required in (("seed", context.seed), ("device", context.device)):
        if name in values and values[name] != required:
            raise ValueError(f"{name} is controlled by ModelBuildContext.")
        values[name] = required
    return TransformerClassifier(context, TransformerConfig(**values))


__all__ = [
    "TransformerClassifier",
    "build_causal_attention_mask",
    "build_transformer_module",
    "create_transformer_classifier",
    "sinusoidal_positional_encoding",
]
