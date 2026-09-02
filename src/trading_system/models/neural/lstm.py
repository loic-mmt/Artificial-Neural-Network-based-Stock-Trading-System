from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
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
    class LSTMModule(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.directions = 2 if config.bidirectional else 1
            self.recurrent = nn_module.LSTM(
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
            _, (hidden, _) = self.recurrent(sequences)
            hidden = hidden.reshape(
                config.num_layers,
                self.directions,
                len(sequences),
                config.hidden_size,
            )[-1]
            features = hidden.transpose(0, 1).reshape(len(sequences), -1)
            return self.head(features)

    return LSTMModule()


class LSTMClassifier(TorchSequenceClassifier):
    model_name = "lstm"

    def __init__(self, context: ModelBuildContext, config: LSTMConfig):
        if not isinstance(config, LSTMConfig):
            raise TypeError("config must be LSTMConfig.")
        self.lstm_config = config
        super().__init__(context, config)

    def _build_module(self, torch_module: Any, nn_module: Any) -> Any:
        return build_lstm_module(
            self.context, self.lstm_config, torch_module, nn_module
        )


def create_lstm_classifier(
    context: ModelBuildContext,
    parameters: Mapping[str, Any],
) -> LSTMClassifier:
    allowed = {item.name for item in fields(LSTMConfig)}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"Unknown lstm parameters: {unknown}")
    values = dict(parameters)
    for name, required in (("seed", context.seed), ("device", context.device)):
        if name in values and values[name] != required:
            raise ValueError(f"{name} is controlled by ModelBuildContext.")
        values[name] = required
    return LSTMClassifier(context, LSTMConfig(**values))


__all__ = ["LSTMClassifier", "build_lstm_module", "create_lstm_classifier"]
