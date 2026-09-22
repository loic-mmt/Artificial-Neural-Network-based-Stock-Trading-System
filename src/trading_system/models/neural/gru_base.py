"""Shared GRU encoder, pooling, and extension points for model variants.

The ordinary GRU keeps its historical ``recurrent`` and ``head`` state keys.
Variants can adapt inputs or replace the classification head without copying the
encoder. This is a GRU-only interface; independent GNN branches use their own
node features and a separate date-aligned multimodal data contract.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import Any

from trading_system.models.specs import ModelBuildContext

from .base import TorchSequenceClassifier
from .config import GRUConfig
from .causal_normalization import build_window_normalizer


def build_gru_head_factory(
    config: GRUConfig, nn_module: Any
) -> Callable[[int, int], Any] | None:
    """Use the same configurable classification head across GRU adapters."""

    if config.head_type == "linear":
        return None

    def create_head(width: int, classes: int) -> Any:
        layers: OrderedDict[str, Any] = OrderedDict()
        if config.head_type in ("layernorm_linear", "layernorm_mlp"):
            layers["norm"] = nn_module.LayerNorm(width)
        if config.head_type in ("mlp", "layernorm_mlp"):
            hidden_size = config.head_hidden_size or max(1, width // 2)
            layers["hidden"] = nn_module.Linear(width, hidden_size)
            layers["activation"] = nn_module.GELU()
            layers["dropout"] = nn_module.Dropout(config.head_dropout)
            layers["output"] = nn_module.Linear(hidden_size, classes)
        else:
            layers["output"] = nn_module.Linear(width, classes)
        return nn_module.Sequential(layers)

    return create_head


def build_gru_module(
    context: ModelBuildContext,
    config: GRUConfig,
    torch_module: Any,
    nn_module: Any,
    *,
    input_adapter_factory: Callable[[], Any] | None = None,
    head_factory: Callable[[int, int], Any] | None = None,
) -> Any:
    """Build a GRU with optional shape-preserving input and logit-head hooks.

    ``encode`` returns one embedding per input sequence. It does not group
    assets or construct graphs. An independent GNN branch must not use these
    embeddings as its node features.
    """

    class GRUModule(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            if config.input_normalization != "none":
                self.window_normalizer = build_window_normalizer(
                    config, context.input_size, torch_module, nn_module
                )
            self.directions = 2 if config.bidirectional else 1
            self.output_size = config.hidden_size * self.directions
            if input_adapter_factory is not None:
                self.input_adapter = input_adapter_factory()
                if not isinstance(self.input_adapter, nn_module.Module):
                    raise TypeError("input_adapter_factory must return an nn.Module.")
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
                self.embedding_size = context.context_len * self.output_size
            elif config.temporal_pooling == "last_attention":
                self.embedding_size = 2 * self.output_size
            else:
                self.embedding_size = self.output_size
            if config.input_normalization == "revin_side":
                self.embedding_size += 2 * context.input_size
            self.head = (
                nn_module.Linear(self.embedding_size, context.num_classes)
                if head_factory is None
                else head_factory(self.embedding_size, context.num_classes)
            )
            if not isinstance(self.head, nn_module.Module):
                raise TypeError("head_factory must return an nn.Module.")

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

        def _encode_with_attention(self, sequences: Any) -> tuple[Any, Any]:
            side = None
            if hasattr(self, "window_normalizer"):
                sequences, side = self.window_normalizer(sequences)
            if hasattr(self, "input_adapter"):
                adapted = self.input_adapter(sequences)
                if adapted.shape != sequences.shape:
                    raise ValueError(
                        "GRU input adapter must preserve (batch, time, features)."
                    )
                sequences = adapted
            outputs, hidden = self.recurrent(sequences)
            features, weights = self._pool_features(outputs, hidden)
            if side is not None:
                features = torch_module.cat((features, side), dim=1)
            return features, weights

        def encode(self, sequences: Any) -> Any:
            """Return embeddings shaped (batch, embedding_size), before logits."""
            features, _ = self._encode_with_attention(sequences)
            return features

        def forward(self, sequences: Any) -> Any:
            return self.head(self.encode(sequences))

        def forward_with_attention(self, sequences: Any) -> tuple[Any, Any]:
            if config.temporal_pooling not in ("attention", "last_attention"):
                raise RuntimeError(
                    "forward_with_attention requires attention-based pooling."
                )
            features, weights = self._encode_with_attention(sequences)
            return self.head(features), weights

    return GRUModule()


class GRUVariantClassifier(TorchSequenceClassifier):
    """Reusable sequence-classifier adapter for registered GRU variants.

    Subclasses must provide a unique ``model_name`` and may override the two
    hooks. The default hooks preserve the historical GRU architecture exactly.
    """

    model_name = "gru_variant_base"

    def __init__(self, context: ModelBuildContext, config: GRUConfig):
        if not isinstance(config, GRUConfig):
            raise TypeError("config must be GRUConfig.")
        self.gru_config = config
        super().__init__(context, config)

    def _input_adapter_factory(self, nn_module: Any) -> Callable[[], Any] | None:
        return None

    def _head_factory(self, nn_module: Any) -> Callable[[int, int], Any] | None:
        return build_gru_head_factory(self.gru_config, nn_module)

    def _build_module(self, torch_module: Any, nn_module: Any) -> Any:
        return build_gru_module(
            self.context,
            self.gru_config,
            torch_module,
            nn_module,
            input_adapter_factory=self._input_adapter_factory(nn_module),
            head_factory=self._head_factory(nn_module),
        )

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Load current states and earlier GRU states with compatible defaults."""
        migrated = dict(state)
        raw_config = migrated.get("config")
        if isinstance(raw_config, Mapping):
            migrated_config = dict(raw_config)
            migrated_config.setdefault("temporal_pooling", "last")
            migrated_config.setdefault("attention_hidden_size", None)
            migrated_config.setdefault("head_type", "linear")
            migrated_config.setdefault("head_hidden_size", None)
            migrated_config.setdefault("head_dropout", 0.0)
            migrated_config.setdefault("input_normalization", "none")
            migrated_config.setdefault("normalization_window", 20)
            migrated_config.setdefault("normalization_rate", 0.1)
            migrated_config.setdefault("normalization_epsilon", 1e-5)
            migrated_config.setdefault("normalization_feature_indices", None)
            migrated["config"] = migrated_config
        super().load_state_dict(migrated)


def create_gru_variant_classifier(
    context: ModelBuildContext,
    parameters: Mapping[str, Any],
    classifier_type: type[GRUVariantClassifier],
    *,
    config_type: type[GRUConfig] = GRUConfig,
) -> GRUVariantClassifier:
    """Validate context-owned settings and build a named GRU variant."""

    if not issubclass(classifier_type, GRUVariantClassifier):
        raise TypeError("classifier_type must extend GRUVariantClassifier.")
    allowed = {item.name for item in fields(config_type)}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"Unknown {classifier_type.model_name} parameters: {unknown}")
    values = dict(parameters)
    for name, required in (("seed", context.seed), ("device", context.device)):
        if name in values and values[name] != required:
            raise ValueError(f"{name} is controlled by ModelBuildContext.")
        values[name] = required
    return classifier_type(context, config_type(**values))


__all__ = [
    "GRUVariantClassifier",
    "build_gru_head_factory",
    "build_gru_module",
    "create_gru_variant_classifier",
]
