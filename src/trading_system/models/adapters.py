from __future__ import annotations

from typing import Any

import numpy as np

from .base import FitResult, TrainingHistory


class SklearnClassifierAdapter:
    """Adapt any fitted-style classifier exposing ``fit`` and ``predict_proba``."""

    def __init__(self, estimator: Any):
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict_proba"):
            raise TypeError("Estimator must expose fit and predict_proba.")
        self.estimator = estimator
        self.classes_ = np.asarray([], dtype=np.int64)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        del X_val, y_val
        self.estimator.fit(X_train, y_train)
        if not hasattr(self.estimator, "classes_"):
            raise TypeError("Fitted estimator does not expose classes_.")
        self.classes_ = np.asarray(self.estimator.classes_, dtype=np.int64)
        return FitResult(
            best_epoch=1, stop_reason="estimator_fit", history=TrainingHistory()
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(self.estimator.predict_proba(X), dtype=np.float64)
        if probabilities.ndim != 2:
            raise ValueError("Estimator predict_proba must return a 2D array.")
        return probabilities


__all__ = ["SklearnClassifierAdapter"]
