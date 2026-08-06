from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from trading_system.labels.schema import N_CLASSES, TradeLabel

from .classification import evaluate_predictions

DecisionMode = Literal["argmax", "thresholds"]


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_CLASSES:
        raise ValueError(f"probabilities must have shape (n, {N_CLASSES}).")
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values.")
    return values


def predict_with_thresholds(
    probabilities: np.ndarray,
    buy_threshold: float = 0.75,
    sell_threshold: float = 0.75,
) -> np.ndarray:
    values = _validate_probabilities(probabilities)
    if not 0.0 <= buy_threshold <= 1.0 or not 0.0 <= sell_threshold <= 1.0:
        raise ValueError("thresholds must be between 0 and 1.")
    predictions = np.full(len(values), TradeLabel.HOLD.value, dtype=np.int64)
    best_class = values.argmax(axis=1)
    predictions[
        (best_class == TradeLabel.BUY.value)
        & (values[:, TradeLabel.BUY.value] >= buy_threshold)
    ] = TradeLabel.BUY.value
    predictions[
        (best_class == TradeLabel.SELL.value)
        & (values[:, TradeLabel.SELL.value] >= sell_threshold)
    ] = TradeLabel.SELL.value
    return predictions


def predict_from_probs(
    probabilities: np.ndarray,
    *,
    decision_mode: DecisionMode = "thresholds",
    buy_threshold: float = 0.75,
    sell_threshold: float = 0.75,
) -> np.ndarray:
    values = _validate_probabilities(probabilities)
    if decision_mode == "argmax":
        return values.argmax(axis=1).astype(np.int64)
    if decision_mode == "thresholds":
        return predict_with_thresholds(values, buy_threshold, sell_threshold)
    raise ValueError(f"Unknown decision_mode: {decision_mode}")


def threshold_gridsearch(
    probabilities: np.ndarray,
    y_val: np.ndarray,
    *,
    min_action_rate: float = 0.0,
    buy_thresholds: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85),
    sell_thresholds: tuple[float, ...] = (
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ),
) -> tuple[float, float]:
    values = _validate_probabilities(probabilities)
    targets = np.asarray(y_val, dtype=np.int64)
    if len(values) != len(targets):
        raise ValueError("Validation probabilities and labels have different lengths.")
    if not 0.0 <= min_action_rate <= 1.0:
        raise ValueError("min_action_rate must be between 0 and 1.")
    best_score = -np.inf
    best_thresholds = (0.75, 0.75)
    for buy_threshold in buy_thresholds:
        for sell_threshold in sell_thresholds:
            predictions = predict_with_thresholds(values, buy_threshold, sell_threshold)
            if float((predictions != TradeLabel.HOLD.value).mean()) < min_action_rate:
                continue
            score = evaluate_predictions(targets, predictions)["macro_f1"]
            if score > best_score:
                best_score = score
                best_thresholds = (buy_threshold, sell_threshold)
    return best_thresholds if np.isfinite(best_score) else (0.55, 0.35)


@dataclass(frozen=True)
class DecisionPolicy:
    mode: DecisionMode = "thresholds"
    buy_threshold: float = 0.75
    sell_threshold: float = 0.75

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return predict_from_probs(
            probabilities,
            decision_mode=self.mode,
            buy_threshold=self.buy_threshold,
            sell_threshold=self.sell_threshold,
        )

    @classmethod
    def calibrate(
        cls,
        probabilities: np.ndarray,
        y_val: np.ndarray,
        *,
        mode: DecisionMode = "thresholds",
        min_action_rate: float = 0.0,
    ) -> "DecisionPolicy":
        if mode == "argmax":
            return cls(mode="argmax")
        buy, sell = threshold_gridsearch(
            probabilities,
            y_val,
            min_action_rate=min_action_rate,
        )
        return cls(mode="thresholds", buy_threshold=buy, sell_threshold=sell)


__all__ = [
    "DecisionMode",
    "DecisionPolicy",
    "predict_from_probs",
    "predict_with_thresholds",
    "threshold_gridsearch",
]
