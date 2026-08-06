from __future__ import annotations

from typing import Literal

import numpy as np

from trading_system.labels.schema import TradeLabel

PositionMode = Literal["long_only", "long_short"]


def labels_to_positions(
    predicted_labels: np.ndarray,
    *,
    position_mode: PositionMode = "long_short",
) -> np.ndarray:
    """Convert action labels into persistent target positions."""

    if position_mode not in ("long_only", "long_short"):
        raise ValueError(f"Unknown position_mode: {position_mode}")
    labels = np.asarray(predicted_labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("predicted_labels must be a 1D array.")
    invalid = ~np.isin(labels, [label.value for label in TradeLabel])
    if invalid.any():
        raise ValueError(f"Unknown label IDs: {np.unique(labels[invalid]).tolist()}")
    positions = np.empty(len(labels), dtype=np.float64)
    current = 0.0
    for index, label in enumerate(labels):
        if label == TradeLabel.BUY.value:
            current = 1.0
        elif label == TradeLabel.SELL.value:
            current = -1.0 if position_mode == "long_short" else 0.0
        positions[index] = current
    return positions


def apply_execution_delay(target_positions: np.ndarray, delay: int = 1) -> np.ndarray:
    if not isinstance(delay, (int, np.integer)) or delay < 0:
        raise ValueError("execution delay must be a non-negative integer.")
    target = np.asarray(target_positions, dtype=np.float64)
    if target.ndim != 1:
        raise ValueError("target_positions must be a 1D array.")
    executed = np.zeros_like(target)
    if delay == 0:
        executed[:] = target
    elif delay < len(target):
        executed[delay:] = target[:-delay]
    return executed


def position_turnover(executed_positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(executed_positions, dtype=np.float64)
    if positions.ndim != 1:
        raise ValueError("executed_positions must be a 1D array.")
    previous = np.zeros_like(positions)
    if len(positions) > 1:
        previous[1:] = positions[:-1]
    return np.abs(positions - previous)


def target_positions_to_actions(target_positions: np.ndarray) -> np.ndarray:
    target = np.clip(np.asarray(target_positions, dtype=np.int64), -1, 1)
    if target.ndim != 1:
        raise ValueError("target_positions must be a 1D array.")
    previous = np.zeros_like(target)
    if len(target) > 1:
        previous[1:] = target[:-1]
    delta = target - previous
    return np.where(delta > 0, "buy", np.where(delta < 0, "sell", "hold"))


# Compatibility names.
signals_to_positions = labels_to_positions
pred_labels_to_target_positions = labels_to_positions


__all__ = [
    "PositionMode",
    "apply_execution_delay",
    "labels_to_positions",
    "position_turnover",
    "pred_labels_to_target_positions",
    "signals_to_positions",
    "target_positions_to_actions",
]
