from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import ProbabilisticSequenceClassifier
from .specs import ModelBuildContext, normalize_model_name

ModelFactory = Callable[
    [ModelBuildContext, Mapping[str, Any]], ProbabilisticSequenceClassifier
]


class ModelRegistry:
    """Registry that keeps architecture selection out of experiment runners."""

    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}

    def register(
        self,
        name: str,
        factory: ModelFactory,
        *,
        replace: bool = False,
    ) -> None:
        normalized = normalize_model_name(name)
        if not callable(factory):
            raise TypeError("factory must be callable.")
        if normalized in self._factories and not replace:
            raise ValueError(f"Model already registered: {normalized}")
        self._factories[normalized] = factory

    def build(
        self,
        name: str,
        context: ModelBuildContext,
        parameters: Mapping[str, Any] | None = None,
    ) -> ProbabilisticSequenceClassifier:
        normalized = normalize_model_name(name)
        if not isinstance(context, ModelBuildContext):
            raise TypeError("context must be a ModelBuildContext.")
        if normalized not in self._factories:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"Unknown model {normalized!r}; available: {available}")
        copied = dict(parameters or {})
        model = self._factories[normalized](context, copied)
        if not isinstance(model, ProbabilisticSequenceClassifier):
            raise TypeError(
                f"Factory {normalized!r} did not return a sequence classifier."
            )
        return model

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def _lazy_factory(module_name: str, factory_name: str) -> ModelFactory:
    def build(
        context: ModelBuildContext, parameters: Mapping[str, Any]
    ) -> ProbabilisticSequenceClassifier:
        from importlib import import_module

        factory = getattr(import_module(module_name), factory_name)
        return factory(context, dict(parameters))

    return build


def create_default_model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    factories = {
        "manual_ann": (
            "trading_system.models.manual_ann.sequence_adapter",
            "create_manual_ann_sequence_classifier",
        ),
        "rnn": ("trading_system.models.neural.rnn", "create_rnn_classifier"),
        "lstm": ("trading_system.models.neural.lstm", "create_lstm_classifier"),
        "gru": ("trading_system.models.neural.gru", "create_gru_classifier"),
        "transformer": (
            "trading_system.models.neural.transformer",
            "create_transformer_classifier",
        ),
    }
    for name, (module_name, factory_name) in factories.items():
        registry.register(name, _lazy_factory(module_name, factory_name))
    return registry


__all__ = ["ModelFactory", "ModelRegistry", "create_default_model_registry"]
