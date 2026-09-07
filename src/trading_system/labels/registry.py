from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import LabelConfig, LabelSemantics, normalize_label_method


@dataclass(frozen=True)
class LabelContext:
    price_col: str = "adj_close"
    date_col: str = "date"
    group_col: str | None = None
    split_col: str | None = "_experiment_split"

    def __post_init__(self) -> None:
        for name in ("price_col", "date_col"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a non-empty string.")
        if self.group_col is not None and (
            not isinstance(self.group_col, str) or not self.group_col
        ):
            raise TypeError("group_col must be a non-empty string or None.")
        if self.split_col is not None and (
            not isinstance(self.split_col, str) or not self.split_col
        ):
            raise TypeError("split_col must be a non-empty string or None.")


@dataclass
class LabelResult:
    frame: pd.DataFrame
    known_mask: np.ndarray
    class_names: tuple[str, ...]
    semantics: LabelSemantics
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pd.DataFrame) or self.frame.empty:
            raise ValueError("Label result frame must be a non-empty DataFrame.")
        missing = {"Label", "Label_id"} - set(self.frame.columns)
        if missing:
            raise ValueError(f"Label result is missing columns: {sorted(missing)}")
        mask = np.asarray(self.known_mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != len(self.frame):
            raise ValueError("known_mask must align one-to-one with label rows.")
        classes = tuple(self.class_names)
        if not classes or any(not isinstance(name, str) or not name for name in classes):
            raise ValueError("class_names must contain non-empty strings.")
        if len(classes) != len(set(classes)):
            raise ValueError("class_names must be unique.")
        label_ids = pd.to_numeric(self.frame["Label_id"], errors="coerce")
        known_ids = label_ids[mask]
        if known_ids.isna().any() or not known_ids.between(0, len(classes) - 1).all():
            raise ValueError("Known Label_id values must match class_names.")
        if self.semantics not in ("action", "target_position", "meta"):
            raise ValueError(f"Unknown label semantics: {self.semantics}")
        if not isinstance(self.metadata, dict):
            raise TypeError("Label metadata must be a dictionary.")
        self.known_mask = mask.copy()
        self.class_names = classes
        self.metadata = dict(self.metadata)


LabelBuilder = Callable[[pd.DataFrame, LabelConfig, LabelContext], LabelResult]


class LabelRegistry:
    """Registry keeping label selection out of experiment runners."""

    def __init__(self) -> None:
        self._builders: dict[str, LabelBuilder] = {}

    def register(
        self,
        name: str,
        builder: LabelBuilder,
        *,
        replace: bool = False,
    ) -> None:
        method = normalize_label_method(name)
        if not callable(builder):
            raise TypeError("Label builder must be callable.")
        if method in self._builders and not replace:
            raise ValueError(f"Label method already registered: {method}")
        self._builders[method] = builder

    def generate(
        self,
        frame: pd.DataFrame,
        config: LabelConfig,
        context: LabelContext | None = None,
    ) -> LabelResult:
        if not isinstance(config, LabelConfig):
            raise TypeError("config must be a LabelConfig.")
        if config.method not in self._builders:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(
                f"Unknown label method {config.method!r}; available: {available}"
            )
        result = self._builders[config.method](
            frame,
            config,
            context or LabelContext(),
        )
        if not isinstance(result, LabelResult):
            raise TypeError(f"Label builder {config.method!r} returned an invalid result.")
        return result

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))


def create_default_label_registry() -> LabelRegistry:
    from .breakout import build_breakout_label_result
    from .forward_return import build_forward_return_label_result
    from .triple_barrier import build_triple_barrier_label_result
    from .volatility_position import build_volatility_position_label_result

    registry = LabelRegistry()
    registry.register("breakout", build_breakout_label_result)
    registry.register("forward_return", build_forward_return_label_result)
    registry.register("triple_barrier", build_triple_barrier_label_result)
    registry.register("volatility_position", build_volatility_position_label_result)
    return registry


__all__ = [
    "LabelBuilder",
    "LabelContext",
    "LabelRegistry",
    "LabelResult",
    "create_default_label_registry",
]
