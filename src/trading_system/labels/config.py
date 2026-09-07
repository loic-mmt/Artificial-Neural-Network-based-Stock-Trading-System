from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

LabelObjective = Literal["absolute_return", "excess_buy_hold", "cross_sectional_alpha"]
LabelSemantics = Literal["action", "target_position", "meta"]

_METHOD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_OBJECTIVES = {"absolute_return", "excess_buy_hold", "cross_sectional_alpha"}
_SEMANTICS = {"action", "target_position", "meta"}


def normalize_label_method(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Label method must be a string.")
    method = value.strip().lower().replace("-", "_")
    if not _METHOD_NAME.fullmatch(method):
        raise ValueError(
            "Label method must start with a letter and contain only letters, "
            "numbers, or underscores."
        )
    return method


@dataclass(frozen=True)
class LabelConfig:
    """Serializable, method-neutral label configuration."""

    method: str
    objective: LabelObjective = "absolute_return"
    semantics: LabelSemantics = "action"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        method = normalize_label_method(self.method)
        if self.objective not in _OBJECTIVES:
            raise ValueError(f"Unknown label objective: {self.objective}")
        if self.semantics not in _SEMANTICS:
            raise ValueError(f"Unknown label semantics: {self.semantics}")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("Label parameters must be a mapping.")
        parameters = copy.deepcopy(dict(self.parameters))
        if any(not isinstance(key, str) or not key for key in parameters):
            raise TypeError("Label parameter names must be non-empty strings.")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "parameters", parameters)

    @property
    def class_names(self) -> tuple[str, ...]:
        if self.semantics == "action":
            return ("Sell", "Hold", "Buy")
        if self.semantics == "target_position":
            return ("Short", "Flat", "Long")
        return ("Skip", "Take")

    @classmethod
    def breakout(
        cls,
        *,
        window: int = 20,
        buy_buffer: float = 0.0,
        sell_buffer: float = 0.0,
        alternating: bool = True,
        objective: LabelObjective = "absolute_return",
    ) -> "LabelConfig":
        return cls(
            method="breakout",
            objective=objective,
            semantics="action",
            parameters={
                "window": window,
                "buy_buffer": buy_buffer,
                "sell_buffer": sell_buffer,
                "alternating": alternating,
            },
        )

    @classmethod
    def forward_return(
        cls,
        *,
        horizon: int = 1,
        buy_threshold: float = 0.002,
        sell_threshold: float = 0.002,
        objective: LabelObjective = "absolute_return",
    ) -> "LabelConfig":
        return cls(
            method="forward_return",
            objective=objective,
            semantics="action",
            parameters={
                "horizon": horizon,
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold,
            },
        )

    @classmethod
    def volatility_position(
        cls,
        *,
        horizon: int = 10,
        volatility_window: int = 20,
        long_threshold: float = 1.0,
        short_threshold: float = 1.5,
        exit_threshold: float = 0.25,
        min_holding_period: int = 5,
        cooldown: int = 0,
        cost_bps: float = 5.0,
        position_mode: Literal["long_flat", "long_short"] = "long_flat",
        objective: LabelObjective = "excess_buy_hold",
    ) -> "LabelConfig":
        return cls(
            method="volatility_position",
            objective=objective,
            semantics="target_position",
            parameters={
                "horizon": horizon,
                "volatility_window": volatility_window,
                "long_threshold": long_threshold,
                "short_threshold": short_threshold,
                "exit_threshold": exit_threshold,
                "min_holding_period": min_holding_period,
                "cooldown": cooldown,
                "cost_bps": cost_bps,
                "position_mode": position_mode,
            },
        )

    @classmethod
    def triple_barrier(
        cls,
        *,
        max_holding: int = 10,
        volatility_window: int = 20,
        volatility_estimator: Literal["rolling_std", "atr", "bollinger"] = "rolling_std",
        profit_barrier: float = 1.0,
        stop_barrier: float = 1.0,
        event_filter: Literal["all", "cusum"] = "all",
        cusum_threshold: float = 0.5,
        cost_bps: float = 5.0,
        between_event_policy: Literal["hold", "flat", "carry"] = "hold",
        objective: LabelObjective = "absolute_return",
    ) -> "LabelConfig":
        semantics: LabelSemantics = (
            "action" if between_event_policy == "hold" else "target_position"
        )
        return cls(
            method="triple_barrier",
            objective=objective,
            semantics=semantics,
            parameters={
                "max_holding": max_holding,
                "volatility_window": volatility_window,
                "volatility_estimator": volatility_estimator,
                "profit_barrier": profit_barrier,
                "stop_barrier": stop_barrier,
                "event_filter": event_filter,
                "cusum_threshold": cusum_threshold,
                "cost_bps": cost_bps,
                "between_event_policy": between_event_policy,
            },
        )


__all__ = [
    "LabelConfig",
    "LabelObjective",
    "LabelSemantics",
    "normalize_label_method",
]
