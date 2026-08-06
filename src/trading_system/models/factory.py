from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import ProbabilisticSequenceClassifier
from .specs import ModelBuildContext

ModelFactory = Callable[
    [ModelBuildContext, Mapping[str, Any]], ProbabilisticSequenceClassifier
]


class ModelRegistry:
    """Registry that keeps architecture selection out of experiment runners."""

    def __init__(self) -> None:
        # TODO(model-registry-init): Create a private mapping from normalized
        # model names to factories. Do not register defaults implicitly here;
        # tests need to construct an empty isolated registry.
        raise NotImplementedError

    def register(
        self,
        name: str,
        factory: ModelFactory,
        *,
        replace: bool = False,
    ) -> None:
        # TODO(model-registry-register-1): Normalize and validate `name`, ensure
        # `factory` is callable, and reject duplicate names unless `replace` is
        # explicitly true.
        #
        # TODO(model-registry-register-2): Store only the factory. Do not create
        # a model during registration because input dimensions are not known yet.
        raise NotImplementedError

    def build(
        self,
        name: str,
        context: ModelBuildContext,
        parameters: Mapping[str, Any] | None = None,
    ) -> ProbabilisticSequenceClassifier:
        # TODO(model-registry-build-1): Resolve a registered factory and report
        # an unknown name together with the sorted available names.
        #
        # TODO(model-registry-build-2): Pass a defensive parameter dictionary and
        # the complete build context to the factory. Validate the returned object
        # against `ProbabilisticSequenceClassifier` before returning it.
        raise NotImplementedError

    def names(self) -> tuple[str, ...]:
        # TODO(model-registry-names): Return registered names in deterministic
        # sorted order so CLI choices and artifacts remain stable.
        raise NotImplementedError


def create_default_model_registry() -> ModelRegistry:
    # TODO(default-registry-1): Create a fresh registry and register
    # `manual_ann`, `rnn`, `lstm`, `gru`, and `transformer` with lazy imports.
    # Lazy imports are required so importing the project does not require torch.
    #
    # TODO(default-registry-2): Each factory must parse only its typed config,
    # reject unknown parameter keys, and receive dimensions from
    # `ModelBuildContext` rather than inferring them from global state.
    raise NotImplementedError


__all__ = ["ModelFactory", "ModelRegistry", "create_default_model_registry"]
