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
            self.output_size = config.hidden_size * self.directions
            self.recurrent = nn_module.GRU(
                context.input_size,
                config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout,
                bidirectional=config.bidirectional,
            )
            if config.temporal_pooling in ("attention", "last_attention"):
                attention_size = config.attention_hidden_size or self.output_size
                self.attention_projection = nn_module.Linear(
                    self.output_size, attention_size
                )
                self.attention_score = nn_module.Linear(
                    attention_size, 1, bias=False
                )

            if config.temporal_pooling == "flatten":
                head_input_size = context.context_len * self.output_size
            elif config.temporal_pooling == "last_attention":
                head_input_size = 2 * self.output_size
            else:
                head_input_size = self.output_size
            self.head = nn_module.Linear(head_input_size, context.num_classes)

        def _final_hidden(self, hidden: Any, batch_size: int) -> Any:
            hidden = hidden.reshape(
                config.num_layers,
                self.directions,
                batch_size,
                config.hidden_size,
            )[-1]
            return hidden.transpose(0, 1).reshape(batch_size, -1)

        def _attention_pool(self, outputs: Any) -> tuple[Any, Any]:
            scores = self.attention_score(
                torch_module.tanh(self.attention_projection(outputs))
            ).squeeze(-1)
            weights = torch_module.softmax(scores, dim=1)
            pooled = torch_module.sum(outputs * weights.unsqueeze(-1), dim=1)
            return pooled, weights

        def _pool_features(self, outputs: Any, hidden: Any) -> tuple[Any, Any]:
            batch_size = len(outputs)
            if config.temporal_pooling == "last":
                return self._final_hidden(hidden, batch_size), None
            if config.temporal_pooling == "mean":
                return outputs.mean(dim=1), None
            if config.temporal_pooling == "flatten":
                return outputs.reshape(batch_size, -1), None

            attended, weights = self._attention_pool(outputs)
            if config.temporal_pooling == "attention":
                return attended, weights
            final = self._final_hidden(hidden, batch_size)
            return torch_module.cat((final, attended), dim=1), weights

        def forward(self, sequences: Any) -> Any:
            outputs, hidden = self.recurrent(sequences)
            features, _ = self._pool_features(outputs, hidden)
            return self.head(features)

        def forward_with_attention(self, sequences: Any) -> tuple[Any, Any]:
            if config.temporal_pooling not in ("attention", "last_attention"):
                raise RuntimeError(
                    "forward_with_attention requires attention-based pooling."
                )
            outputs, hidden = self.recurrent(sequences)
            features, weights = self._pool_features(outputs, hidden)
            return self.head(features), weights

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

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Load current states and earlier GRU states with compatible defaults."""
        migrated = dict(state)
        raw_config = migrated.get("config")
        if isinstance(raw_config, Mapping):
            migrated_config = dict(raw_config)
            migrated_config.setdefault("temporal_pooling", "last")
            migrated_config.setdefault("attention_hidden_size", None)
            migrated["config"] = migrated_config
        super().load_state_dict(migrated)


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
