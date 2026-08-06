from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from trading_system.models.manual_ann.manual_nn import ManualANNConfig

UniverseMode = Literal["single", "multi"]
FeatureSet = Literal["technical", "market"]
LabelMode = Literal["breakout", "forward_return", "oracle_train_only", "oracle_all"]
EvaluationMode = Literal["static", "walk_forward"]


@dataclass(frozen=True)
class ExperimentConfig:
    universe: UniverseMode = "single"
    feature_set: FeatureSet = "technical"
    label_mode: LabelMode = "breakout"
    evaluation_mode: EvaluationMode = "static"
    position_mode: Literal["long_only", "long_short"] = "long_only"
    ticker: str | None = None
    group_col: str = "ticker"
    date_col: str = "date"
    price_col: str = "adj_close"
    label_window: int = 20
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    context_len: int = 20
    initial_capital: float = 10_000.0
    fee_per_trade: float = 0.0
    execution_delay: int = 1
    decision_mode: Literal["argmax", "thresholds"] = "thresholds"
    min_action_rate: float = 0.0
    forward_horizon: int = 1
    forward_buy_threshold: float = 0.002
    forward_sell_threshold: float = 0.002
    oracle_fee_per_trade: float = 0.0
    # TODO(model-neutral-experiment-config): Move this ANN-specific field into a
    # separate `ModelSelection`/model config after the 3D runner is ready. Keep it
    # temporarily so current pipelines remain compatible during migration.
    manual_ann: ManualANNConfig = field(default_factory=ManualANNConfig)

    def __post_init__(self) -> None:
        if self.universe not in ("single", "multi"):
            raise ValueError(f"Unknown universe: {self.universe}")
        if self.feature_set not in ("technical", "market"):
            raise ValueError(f"Unknown feature_set: {self.feature_set}")
        if self.label_mode not in (
            "breakout",
            "forward_return",
            "oracle_train_only",
            "oracle_all",
        ):
            raise ValueError(f"Unknown label_mode: {self.label_mode}")
        if self.position_mode not in ("long_only", "long_short"):
            raise ValueError(f"Unknown position_mode: {self.position_mode}")
        if not 0 < self.train_ratio < 1 or not 0 < self.val_ratio < 1:
            raise ValueError("train_ratio and val_ratio must be between 0 and 1.")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1.")
        if self.label_window <= 0 or self.context_len <= 0:
            raise ValueError("label_window and context_len must be positive.")
        if self.initial_capital <= 0 or self.fee_per_trade < 0:
            raise ValueError("Invalid capital or fee configuration.")
        if self.execution_delay < 0:
            raise ValueError("execution_delay must be non-negative.")
        if not 0 <= self.min_action_rate <= 1:
            raise ValueError("min_action_rate must be between 0 and 1.")


__all__ = [
    "EvaluationMode",
    "ExperimentConfig",
    "FeatureSet",
    "LabelMode",
    "UniverseMode",
]
