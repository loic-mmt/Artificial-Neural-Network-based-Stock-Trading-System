from __future__ import annotations

import argparse
from dataclasses import replace

from trading_system.experiments.config import ExperimentConfig
from trading_system.labels.config import LabelConfig
from trading_system.labels.triple_barrier import VOLATILITY_ESTIMATORS

_CLI_TO_INTERNAL = {
    "breakout": "breakout",
    "forward-return": "forward_return",
    "triple-barrier": "triple_barrier",
    "volatility-position": "volatility_position",
    "oracle-train-only": "oracle_train_only",
    "oracle-all": "oracle_all",
}
_INTERNAL_TO_CLI = {value: key for key, value in _CLI_TO_INTERNAL.items()}
_LABEL_CHOICES = tuple(sorted(set(_CLI_TO_INTERNAL) | set(_CLI_TO_INTERNAL.values())))
_BREAKOUT_DESTINATIONS = {
    "label_window",
    "label_buy_buffer",
    "label_sell_buffer",
    "label_alternating",
}
_HORIZON_DESTINATIONS = {"label_horizon"}
_FORWARD_RETURN_DESTINATIONS = {
    "label_buy_threshold",
    "label_sell_threshold",
}
_VOLATILITY_POSITION_DESTINATIONS = {
    "label_long_threshold",
    "label_short_threshold",
    "label_exit_threshold",
    "label_min_hold",
    "label_cooldown",
    "label_position_mode",
}
_VOLATILITY_SHARED_DESTINATIONS = {"label_vol_window", "label_cost_bps"}
_TRIPLE_BARRIER_DESTINATIONS = {
    "label_volatility_estimator",
    "label_profit_barrier",
    "label_stop_barrier",
    "label_event_filter",
    "label_cusum_threshold",
    "label_between_events",
}


def add_barrier_estimator_argument(
    parser: argparse.ArgumentParser, *, default: str = "rolling_std"
) -> None:
    parser.add_argument(
        "--label-volatility-estimator",
        choices=VOLATILITY_ESTIMATORS,
        default=default,
        help="Triple-barrier scale: return std, ATR/price, or Bollinger half-width/mean.",
    )


def add_label_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared label arguments without overriding preset defaults."""

    parser.add_argument(
        "--label-method",
        "--label-mode",
        dest="label_method",
        choices=_LABEL_CHOICES,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--label-window",
        type=int,
        default=argparse.SUPPRESS,
        help="Breakout rolling window.",
    )
    parser.add_argument(
        "--label-buy-buffer",
        type=float,
        default=argparse.SUPPRESS,
        help="Required fractional move below the previous rolling minimum.",
    )
    parser.add_argument(
        "--label-sell-buffer",
        type=float,
        default=argparse.SUPPRESS,
        help="Required fractional move above the previous rolling maximum.",
    )
    parser.add_argument(
        "--label-alternating",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Suppress repeated consecutive Buy or Sell breakout actions.",
    )
    parser.add_argument(
        "--label-horizon",
        "--forward-horizon",
        "--label-max-holding",
        dest="label_horizon",
        type=int,
        default=argparse.SUPPRESS,
        help="Forward horizon or maximum holding period in rows.",
    )
    parser.add_argument(
        "--label-buy-threshold",
        "--forward-buy-threshold",
        dest="label_buy_threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="Forward return strictly above this value becomes Buy.",
    )
    parser.add_argument(
        "--label-sell-threshold",
        "--forward-sell-threshold",
        dest="label_sell_threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="Forward return strictly below the negative value becomes Sell.",
    )
    parser.add_argument(
        "--label-vol-window",
        type=int,
        default=argparse.SUPPRESS,
        help="Historical volatility window in rows.",
    )
    parser.add_argument(
        "--label-long-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="Volatility-adjusted score required to enter Long.",
    )
    parser.add_argument(
        "--label-short-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="Negative volatility-adjusted score required to enter Short.",
    )
    parser.add_argument(
        "--label-exit-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="Absolute score below which the target position becomes Flat.",
    )
    parser.add_argument(
        "--label-min-hold",
        type=int,
        default=argparse.SUPPRESS,
        help="Minimum number of rows before leaving an open position.",
    )
    parser.add_argument(
        "--label-cooldown",
        type=int,
        default=argparse.SUPPRESS,
        help="Minimum number of rows without another position transition.",
    )
    parser.add_argument(
        "--label-cost-bps",
        type=float,
        default=argparse.SUPPRESS,
        help="Estimated round-trip cost used by volatility-scaled labels.",
    )
    parser.add_argument(
        "--label-position-mode",
        choices=("long-flat", "long-short", "long_flat", "long_short"),
        default=argparse.SUPPRESS,
        help="Allowed volatility-position target states.",
    )
    parser.add_argument(
        "--label-profit-barrier",
        type=float,
        default=argparse.SUPPRESS,
        help="Triple-barrier profit multiple of historical volatility.",
    )
    add_barrier_estimator_argument(parser, default=argparse.SUPPRESS)
    parser.add_argument(
        "--label-stop-barrier",
        type=float,
        default=argparse.SUPPRESS,
        help="Triple-barrier stop multiple of historical volatility.",
    )
    parser.add_argument(
        "--label-event-filter",
        choices=("all", "cusum"),
        default=argparse.SUPPRESS,
        help="Rows that start triple-barrier events.",
    )
    parser.add_argument(
        "--label-cusum-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="CUSUM threshold as a multiple of historical volatility.",
    )
    parser.add_argument(
        "--label-between-events",
        choices=("hold", "flat", "carry"),
        default=argparse.SUPPRESS,
        help="Explicit label policy on rows without a triple-barrier event.",
    )


def apply_label_arguments(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> ExperimentConfig:
    """Resolve explicit CLI overrides over an experiment preset."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig.")
    values = vars(args)
    raw_method = values.get(
        "label_method", _INTERNAL_TO_CLI.get(config.label_mode, config.label_mode)
    )
    label_mode = _CLI_TO_INTERNAL.get(raw_method, raw_method)
    explicit_breakout = sorted(_BREAKOUT_DESTINATIONS.intersection(values))
    if explicit_breakout and label_mode != "breakout":
        options = ["--" + name.replace("_", "-") for name in explicit_breakout]
        raise ValueError(
            f"Breakout-only arguments require --label-method breakout: {options}"
        )
    explicit_forward = sorted(_FORWARD_RETURN_DESTINATIONS.intersection(values))
    if explicit_forward and label_mode != "forward_return":
        options = ["--" + name.replace("_", "-") for name in explicit_forward]
        raise ValueError(
            "Forward-return-only arguments require --label-method "
            f"forward-return: {options}"
        )
    explicit_horizon = sorted(_HORIZON_DESTINATIONS.intersection(values))
    if explicit_horizon and label_mode not in (
        "forward_return",
        "triple_barrier",
        "volatility_position",
    ):
        options = ["--" + name.replace("_", "-") for name in explicit_horizon]
        raise ValueError(
            "Horizon arguments require --label-method forward-return, "
            f"volatility-position, or triple-barrier: {options}"
        )
    explicit_volatility_shared = sorted(
        _VOLATILITY_SHARED_DESTINATIONS.intersection(values)
    )
    if explicit_volatility_shared and label_mode not in (
        "volatility_position",
        "triple_barrier",
    ):
        options = [
            "--" + name.replace("_", "-") for name in explicit_volatility_shared
        ]
        raise ValueError(
            "Volatility arguments require --label-method volatility-position or "
            f"triple-barrier: {options}"
        )
    explicit_triple = sorted(_TRIPLE_BARRIER_DESTINATIONS.intersection(values))
    if explicit_triple and label_mode != "triple_barrier":
        options = ["--" + name.replace("_", "-") for name in explicit_triple]
        raise ValueError(
            "Triple-barrier-only arguments require --label-method "
            f"triple-barrier: {options}"
        )
    explicit_volatility = sorted(
        _VOLATILITY_POSITION_DESTINATIONS.intersection(values)
    )
    if explicit_volatility and label_mode != "volatility_position":
        options = ["--" + name.replace("_", "-") for name in explicit_volatility]
        raise ValueError(
            "Volatility-position-only arguments require --label-method "
            f"volatility-position: {options}"
        )
    updates = {"label_mode": label_mode}
    if "label_window" in values:
        updates["label_window"] = values["label_window"]
    if "label_buy_buffer" in values:
        updates["breakout_buy_buffer"] = values["label_buy_buffer"]
    if "label_sell_buffer" in values:
        updates["breakout_sell_buffer"] = values["label_sell_buffer"]
    if "label_alternating" in values:
        updates["breakout_alternating"] = values["label_alternating"]
    if "label_horizon" in values:
        if label_mode == "volatility_position":
            destination = "volatility_horizon"
        elif label_mode == "triple_barrier":
            destination = "triple_barrier_max_holding"
        else:
            destination = "forward_horizon"
        updates[destination] = values["label_horizon"]
    if "label_buy_threshold" in values:
        updates["forward_buy_threshold"] = values["label_buy_threshold"]
    if "label_sell_threshold" in values:
        updates["forward_sell_threshold"] = values["label_sell_threshold"]
    volatility_updates = {
        "label_long_threshold": "volatility_long_threshold",
        "label_short_threshold": "volatility_short_threshold",
        "label_exit_threshold": "volatility_exit_threshold",
        "label_min_hold": "volatility_min_holding_period",
        "label_cooldown": "volatility_cooldown",
    }
    for source, destination in volatility_updates.items():
        if source in values:
            updates[destination] = values[source]
    if "label_position_mode" in values:
        updates["volatility_position_mode"] = values[
            "label_position_mode"
        ].replace("-", "_")
    if "label_vol_window" in values:
        destination = (
            "triple_barrier_volatility_window"
            if label_mode == "triple_barrier"
            else "volatility_window"
        )
        updates[destination] = values["label_vol_window"]
    if "label_cost_bps" in values:
        destination = (
            "triple_barrier_cost_bps"
            if label_mode == "triple_barrier"
            else "volatility_cost_bps"
        )
        updates[destination] = values["label_cost_bps"]
    triple_updates = {
        "label_volatility_estimator": "triple_barrier_volatility_estimator",
        "label_profit_barrier": "triple_barrier_profit_barrier",
        "label_stop_barrier": "triple_barrier_stop_barrier",
        "label_event_filter": "triple_barrier_event_filter",
        "label_cusum_threshold": "triple_barrier_cusum_threshold",
        "label_between_events": "triple_barrier_between_event_policy",
    }
    for source, destination in triple_updates.items():
        if source in values:
            updates[destination] = values[source]
    return replace(config, **updates)


def resolved_label_config(config: ExperimentConfig) -> LabelConfig | None:
    """Expose the registry configuration represented by an experiment preset."""

    return config.resolved_label_config()


def label_config_from_args(
    args: argparse.Namespace,
    base_config: ExperimentConfig | None = None,
) -> LabelConfig:
    """Build the normalized label configuration represented by CLI arguments."""

    config = apply_label_arguments(base_config or ExperimentConfig(), args)
    resolved = resolved_label_config(config)
    return resolved or LabelConfig(method=config.label_mode)


__all__ = [
    "add_barrier_estimator_argument",
    "add_label_arguments",
    "apply_label_arguments",
    "label_config_from_args",
    "resolved_label_config",
]
