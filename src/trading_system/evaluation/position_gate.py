"""Post-training signal-strength gate for continuous trading positions.

The absolute position is a signal magnitude, not a calibrated probability.
Thresholds are selected on an inner validation panel and then frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PositionGate:
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Position gate threshold must be finite and in [0, 1].")

    def apply(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or (np.abs(values) > 1 + 1e-6).any():
            raise ValueError("Positions must be finite, bounded, and one-dimensional.")
        return np.where(np.abs(values) < self.threshold, 0.0, values)


@dataclass(frozen=True)
class GateSearchConfig:
    quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    min_coverage: float = 0.2
    score_metric: str = "regularized_sharpe"

    def __post_init__(self) -> None:
        if not self.quantiles or any(
            not math.isfinite(q) or not 0 < q < 1 for q in self.quantiles
        ) or len(set(self.quantiles)) != len(self.quantiles):
            raise ValueError("Gate quantiles must be unique finite values in (0, 1).")
        if not math.isfinite(self.min_coverage) or not 0 <= self.min_coverage <= 1:
            raise ValueError("Gate min_coverage must be finite and in [0, 1].")
        if self.score_metric not in ("regularized_sharpe", "net_return", "net_pnl"):
            raise ValueError("Unsupported gate score_metric.")


@dataclass(frozen=True)
class GateSelection:
    gate: PositionGate
    candidates: tuple[dict[str, float | bool | None], ...]


def signal_coverage(positions: np.ndarray) -> float:
    values = np.asarray(positions)
    if values.ndim != 1 or not len(values):
        raise ValueError("Signal coverage requires non-empty 1D positions.")
    return float(np.count_nonzero(values) / len(values))


def fit_position_gate(positions, panel, loss_config, initial_capital, search: GateSearchConfig) -> GateSelection:
    """Select threshold using inner validation returns only; include raw baseline."""
    if not isinstance(search, GateSearchConfig):
        raise TypeError("search must be GateSearchConfig.")
    raw = PositionGate().apply(positions)
    if len(raw) != panel.rows:
        raise ValueError("Positions and validation panel must be aligned.")
    thresholds = [0.0, *(float(x) for x in np.quantile(np.abs(raw), search.quantiles))]
    candidates = []
    best_gate = PositionGate()
    best_score = -np.inf
    for threshold in sorted(set(thresholds)):
        gate = PositionGate(threshold)
        filtered = gate.apply(raw)
        coverage = signal_coverage(filtered)
        eligible = threshold == 0 or coverage >= search.min_coverage
        score = float(panel.metrics(filtered, loss_config, initial_capital)[search.score_metric]) if eligible else None
        candidates.append({"threshold": threshold, "coverage": coverage,
                           "eligible": eligible, "score": score})
        # Sorted thresholds plus strict improvement prefer the least filtering on ties.
        if eligible and score is not None and np.isfinite(score) and score > best_score:
            best_gate, best_score = gate, score
    if not np.isfinite(best_score):
        raise ValueError("No finite validation score for position gate.")
    return GateSelection(best_gate, tuple(candidates))


__all__ = ["GateSearchConfig", "GateSelection", "PositionGate", "fit_position_gate", "signal_coverage"]
