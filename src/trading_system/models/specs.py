from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelBuildContext:
    """Dimensions and runtime choices known after data preparation."""

    input_size: int
    context_len: int
    num_classes: int = 3
    seed: int = 1
    device: str = "auto"

    def __post_init__(self) -> None:
        # TODO(model-context-1): Require positive `input_size`, `context_len` and
        # `num_classes > 1`. Require a non-negative seed.
        #
        # TODO(model-context-2): Accept only `auto`, `cpu`, `cuda` and `mps` as
        # generic device requests. Device availability is checked later by the
        # PyTorch layer, not in this framework-independent dataclass.
        raise NotImplementedError


@dataclass(frozen=True)
class ModelSelection:
    """Serializable request used by the experiment runner and model registry."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # TODO(model-selection-1): Normalize and validate a non-empty stable model
        # name without mutating the frozen instance accidentally.
        #
        # TODO(model-selection-2): Copy parameters into an ordinary dictionary
        # and reject values that cannot be represented in run metadata JSON.
        raise NotImplementedError


__all__ = ["ModelBuildContext", "ModelSelection"]
