from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
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
    class GRUModule(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.directions = 2 if config.bidirectional else 1
            self.recurrent = nn_module.GRU(
                context.input_size,
                config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout,
                bidirectional=config.bidirectional,
            )
            self.head = nn_module.Linear(
                config.hidden_size * self.directions, context.num_classes
            )

        def forward(self, sequences: Any) -> Any:
            _, hidden = self.recurrent(sequences)
            hidden = hidden.reshape(
                config.num_layers,
                self.directions,
                len(sequences),
                config.hidden_size,
            )[-1]
            features = hidden.transpose(0, 1).reshape(len(sequences), -1)
            return self.head(features)

    return GRUModule()


class GRUClassifier(TorchSequenceClassifier):
    model_name = "gru"

    def __init__(self, context: ModelBuildContext, config: GRUConfig):
        if not isinstance(config, GRUConfig):
            raise TypeError("config must be GRUConfig.")
        self.gru_config = config
        super().__init__(context, config)

    def _build_module(self, torch_module: Any, nn_module: Any) -> Any:
        return build_gru_module(
            self.context, self.gru_config, torch_module, nn_module
        )


def create_gru_classifier(
    context: ModelBuildContext,
    parameters: Mapping[str, Any],
) -> GRUClassifier:
    allowed = {item.name for item in fields(GRUConfig)}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"Unknown gru parameters: {unknown}")
    values = dict(parameters)
    for name, required in (("seed", context.seed), ("device", context.device)):
        if name in values and values[name] != required:
            raise ValueError(f"{name} is controlled by ModelBuildContext.")
        values[name] = required
    return GRUClassifier(context, GRUConfig(**values))


__all__ = ["GRUClassifier", "build_gru_module", "create_gru_classifier"]
