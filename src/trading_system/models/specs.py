from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from trading_system.artifacts.serialization import to_jsonable

_MODEL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def normalize_model_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Model name must be a string.")
    name = value.strip().lower().replace("-", "_")
    if not _MODEL_NAME.fullmatch(name):
        raise ValueError(
            "Model name must start with a letter and contain only letters, "
            "numbers, or underscores."
        )
    return name


@dataclass(frozen=True)
class ModelBuildContext:
    """Dimensions and runtime choices known after data preparation."""

    input_size: int
    context_len: int
    num_classes: int = 3
    seed: int = 1
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in ("input_size", "context_len", "num_classes", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.input_size <= 0 or self.context_len <= 0:
            raise ValueError("input_size and context_len must be positive.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must exceed one.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'.")


@dataclass(frozen=True)
class ModelSelection:
    """Serializable request used by the experiment runner and model registry."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, dict):
            raise TypeError("parameters must be a dictionary.")
        try:
            parameters = to_jsonable(self.parameters)
        except (TypeError, ValueError) as error:
            raise TypeError("parameters must contain JSON metadata values.") from error
        if not isinstance(parameters, dict):
            raise TypeError("parameters must encode as a JSON object.")
        object.__setattr__(self, "name", normalize_model_name(self.name))
        object.__setattr__(self, "parameters", parameters)


__all__ = ["ModelBuildContext", "ModelSelection", "normalize_model_name"]
