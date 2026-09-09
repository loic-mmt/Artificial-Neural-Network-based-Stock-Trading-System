from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

from trading_system.labels.config import LabelConfig
from trading_system.features.fracdiff import FracDiffConfig
from trading_system.features.expanded import DEFAULT_GROUPS, feature_columns
from trading_system.training.sample_weighting import SampleWeightConfig
from trading_system.models.manual_ann.manual_nn import ManualANNConfig
from trading_system.models.specs import ModelSelection
from trading_system.training.overfitting import OverfittingControlConfig
from trading_system.data.purged_cv import PurgedSplit

UniverseMode = Literal["single", "multi"]
FeatureSet = Literal["technical", "market", "expanded"]
LabelMode = Literal[
    "breakout",
    "forward_return",
    "triple_barrier",
    "volatility_position",
    "oracle_train_only",
    "oracle_all",
]
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
    breakout_buy_buffer: float = 0.0
    breakout_sell_buffer: float = 0.0
    breakout_alternating: bool = True
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
    volatility_horizon: int = 10
    volatility_window: int = 20
    volatility_long_threshold: float = 1.0
    volatility_short_threshold: float = 1.5
    volatility_exit_threshold: float = 0.25
    volatility_min_holding_period: int = 5
    volatility_cooldown: int = 0
    volatility_cost_bps: float = 5.0
    volatility_position_mode: Literal["long_flat", "long_short"] = "long_flat"
    triple_barrier_max_holding: int = 10
    triple_barrier_volatility_window: int = 20
    triple_barrier_volatility_estimator: Literal[
        "rolling_std", "atr", "bollinger"
    ] = "rolling_std"
    triple_barrier_profit_barrier: float = 1.0
    triple_barrier_stop_barrier: float = 1.0
    triple_barrier_event_filter: Literal["all", "cusum"] = "all"
    triple_barrier_cusum_threshold: float = 0.5
    triple_barrier_cost_bps: float = 5.0
    triple_barrier_between_event_policy: Literal["hold", "flat", "carry"] = "hold"
    oracle_fee_per_trade: float = 0.0
    model: ModelSelection = field(
        default_factory=lambda: ModelSelection("manual_ann")
    )
    seed: int = 1
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    # Input-only compatibility field. __post_init__ migrates it into `model`.
    manual_ann: ManualANNConfig | None = field(default=None, repr=False, compare=False)
    fracdiff: FracDiffConfig | None = None
    sample_weighting: SampleWeightConfig | None = None
    expanded_feature_groups: tuple[str, ...] = DEFAULT_GROUPS
    expanded_min_coverage: float = 0.5
    overfitting_control: OverfittingControlConfig | None = None
    purged_split: PurgedSplit | None = None

    def __post_init__(self) -> None:
        if isinstance(self.purged_split, dict):
            object.__setattr__(self, "purged_split", PurgedSplit(**self.purged_split))
        if self.purged_split is not None and not isinstance(self.purged_split, PurgedSplit):
            raise TypeError("purged_split must be PurgedSplit or None.")
        if self.purged_split is not None and self.label_mode.startswith("oracle"):
            raise ValueError("Oracle labels cannot be used for purged CV.")
        if isinstance(self.overfitting_control, dict):
            object.__setattr__(
                self,
                "overfitting_control",
                OverfittingControlConfig(**self.overfitting_control),
            )
        if self.overfitting_control is not None and not isinstance(
            self.overfitting_control, OverfittingControlConfig
        ):
            raise TypeError(
                "overfitting_control must be OverfittingControlConfig or None."
            )
        feature_columns(self.expanded_feature_groups)
        object.__setattr__(self, "expanded_feature_groups", tuple(self.expanded_feature_groups))
        if not 0 < self.expanded_min_coverage <= 1:
            raise ValueError("expanded_min_coverage must be in (0, 1].")
        if isinstance(self.sample_weighting, dict):
            object.__setattr__(self, "sample_weighting", SampleWeightConfig(**self.sample_weighting))
        if self.sample_weighting is not None:
            if not isinstance(self.sample_weighting, SampleWeightConfig):
                raise TypeError("sample_weighting must be SampleWeightConfig or None.")
            if self.label_mode != "triple_barrier":
                raise ValueError("Event sample weighting requires label_mode='triple_barrier'.")
        if isinstance(self.fracdiff, dict):
            object.__setattr__(self, "fracdiff", FracDiffConfig(**self.fracdiff))
        if self.fracdiff is not None and not isinstance(self.fracdiff, FracDiffConfig):
            raise TypeError("fracdiff must be FracDiffConfig or None.")
        if self.manual_ann is not None:
            if not isinstance(self.manual_ann, ManualANNConfig):
                raise TypeError("manual_ann must be ManualANNConfig or None.")
            if self.model.name != "manual_ann":
                raise ValueError("manual_ann compatibility config requires manual_ann model.")
            parameters = asdict(self.manual_ann)
            seed = int(parameters.pop("seed"))
            parameters.pop("num_classes")
            object.__setattr__(self, "model", ModelSelection("manual_ann", parameters))
            object.__setattr__(self, "seed", seed)
            object.__setattr__(self, "manual_ann", None)
        if not isinstance(self.model, ModelSelection):
            raise TypeError("model must be a ModelSelection.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'.")
        if self.universe not in ("single", "multi"):
            raise ValueError(f"Unknown universe: {self.universe}")
        if self.feature_set not in ("technical", "market", "expanded"):
            raise ValueError(f"Unknown feature_set: {self.feature_set}")
        if self.label_mode not in (
            "breakout",
            "forward_return",
            "triple_barrier",
            "volatility_position",
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
        for name in ("breakout_buy_buffer", "breakout_sell_buffer"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1).")
        if not isinstance(self.breakout_alternating, bool):
            raise TypeError("breakout_alternating must be a boolean.")
        if isinstance(self.forward_horizon, bool) or not isinstance(
            self.forward_horizon, int
        ):
            raise TypeError("forward_horizon must be an integer.")
        if self.forward_horizon <= 0:
            raise ValueError("forward_horizon must be positive.")
        for name in ("forward_buy_threshold", "forward_sell_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in ("volatility_horizon", "volatility_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("volatility_min_holding_period", "volatility_cooldown"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name in (
            "volatility_long_threshold",
            "volatility_short_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("volatility_exit_threshold", "volatility_cost_bps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.volatility_exit_threshold >= min(
            self.volatility_long_threshold,
            self.volatility_short_threshold,
        ):
            raise ValueError(
                "volatility_exit_threshold must be below entry thresholds."
            )
        if self.volatility_position_mode not in ("long_flat", "long_short"):
            raise ValueError(
                "volatility_position_mode must be 'long_flat' or 'long_short'."
            )
        for name in (
            "triple_barrier_max_holding",
            "triple_barrier_volatility_window",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "triple_barrier_profit_barrier",
            "triple_barrier_stop_barrier",
            "triple_barrier_cost_bps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        value = self.triple_barrier_cusum_threshold
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("triple_barrier_cusum_threshold must be numeric.")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(
                "triple_barrier_cusum_threshold must be finite and positive."
            )
        if self.triple_barrier_profit_barrier == 0.0 and (
            self.triple_barrier_stop_barrier == 0.0
        ):
            raise ValueError("At least one triple-barrier horizontal barrier is required.")
        if self.triple_barrier_event_filter not in ("all", "cusum"):
            raise ValueError("triple_barrier_event_filter must be 'all' or 'cusum'.")
        if self.triple_barrier_volatility_estimator not in (
            "rolling_std",
            "atr",
            "bollinger",
        ):
            raise ValueError(
                "triple_barrier_volatility_estimator must be "
                "rolling_std, atr, or bollinger."
            )
        if self.triple_barrier_between_event_policy not in (
            "hold",
            "flat",
            "carry",
        ):
            raise ValueError(
                "triple_barrier_between_event_policy must be hold, flat, or carry."
            )
        if self.initial_capital <= 0 or self.fee_per_trade < 0:
            raise ValueError("Invalid capital or fee configuration.")
        if self.execution_delay < 0:
            raise ValueError("execution_delay must be non-negative.")
        if not 0 <= self.min_action_rate <= 1:
            raise ValueError("min_action_rate must be between 0 and 1.")

    def resolved_label_config(self) -> LabelConfig | None:
        """Translate legacy experiment fields into the canonical label contract."""

        if self.label_mode == "breakout":
            return LabelConfig.breakout(
                window=self.label_window,
                buy_buffer=self.breakout_buy_buffer,
                sell_buffer=self.breakout_sell_buffer,
                alternating=self.breakout_alternating,
            )
        if self.label_mode == "forward_return":
            return LabelConfig.forward_return(
                horizon=self.forward_horizon,
                buy_threshold=self.forward_buy_threshold,
                sell_threshold=self.forward_sell_threshold,
            )
        if self.label_mode == "volatility_position":
            return LabelConfig.volatility_position(
                horizon=self.volatility_horizon,
                volatility_window=self.volatility_window,
                long_threshold=self.volatility_long_threshold,
                short_threshold=self.volatility_short_threshold,
                exit_threshold=self.volatility_exit_threshold,
                min_holding_period=self.volatility_min_holding_period,
                cooldown=self.volatility_cooldown,
                cost_bps=self.volatility_cost_bps,
                position_mode=self.volatility_position_mode,
            )
        if self.label_mode == "triple_barrier":
            return LabelConfig.triple_barrier(
                max_holding=self.triple_barrier_max_holding,
                volatility_window=self.triple_barrier_volatility_window,
                volatility_estimator=self.triple_barrier_volatility_estimator,
                profit_barrier=self.triple_barrier_profit_barrier,
                stop_barrier=self.triple_barrier_stop_barrier,
                event_filter=self.triple_barrier_event_filter,
                cusum_threshold=self.triple_barrier_cusum_threshold,
                cost_bps=self.triple_barrier_cost_bps,
                between_event_policy=self.triple_barrier_between_event_policy,
            )
        return None

    def resolved_label_semantics(self) -> str:
        label = self.resolved_label_config()
        return label.semantics if label is not None else "action"

    def resolved_class_names(self) -> tuple[str, ...]:
        label = self.resolved_label_config()
        return label.class_names if label is not None else ("Sell", "Hold", "Buy")

    def resolved_backtest_position_mode(self) -> str:
        label = self.resolved_label_config()
        if (
            label is not None
            and label.method == "volatility_position"
            and label.parameters["position_mode"] == "long_flat"
        ):
            return "long_only"
        return self.position_mode


__all__ = [
    "EvaluationMode",
    "ExperimentConfig",
    "FeatureSet",
    "LabelMode",
    "UniverseMode",
]
