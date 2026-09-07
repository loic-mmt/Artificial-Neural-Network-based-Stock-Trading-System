"""Opt-in controls for model capacity, feature noise, and run stability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from trading_system.models.base import FitResult
from trading_system.models.specs import ModelSelection


@dataclass(frozen=True)
class OverfittingControlConfig:
    """Conservative defaults; explicit model parameters always take precedence."""

    profile: Literal["compact_regularized"] = "compact_regularized"
    max_features: int | None = 32
    min_feature_variance: float = 1e-8
    max_feature_correlation: float = 0.98
    max_validation_metric_std: float = 0.05
    max_train_validation_gap: float = 0.15
    require_all_seeds: bool = True

    def __post_init__(self) -> None:
        if self.profile != "compact_regularized":
            raise ValueError("Unknown overfitting-control profile.")
        if self.max_features is not None and (
            isinstance(self.max_features, bool) or self.max_features <= 0
        ):
            raise ValueError("max_features must be positive or None.")
        for name in (
            "min_feature_variance",
            "max_validation_metric_std",
            "max_train_validation_gap",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not 0 < self.max_feature_correlation <= 1:
            raise ValueError("max_feature_correlation must be in (0, 1].")
        if not isinstance(self.require_all_seeds, bool):
            raise TypeError("require_all_seeds must be boolean.")


_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "manual_ann": {
        "hidden_size": 32,
        "dropout_probability": 0.30,
        "weight_decay": 1e-4,
        "early_stopping_patience": 15,
    },
    "rnn": {
        "hidden_size": 32,
        "weight_decay": 1e-4,
        "early_stopping_patience": 15,
    },
    "lstm": {
        "hidden_size": 32,
        "weight_decay": 1e-4,
        "early_stopping_patience": 15,
    },
    "gru": {
        "hidden_size": 32,
        "weight_decay": 1e-4,
        "early_stopping_patience": 15,
    },
    "transformer": {
        "d_model": 32,
        "n_heads": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.30,
        "weight_decay": 1e-4,
        "early_stopping_patience": 15,
    },
}


def apply_overfitting_profile(
    selection: ModelSelection,
    config: OverfittingControlConfig | None,
) -> ModelSelection:
    """Apply conservative defaults without overriding an explicit experiment."""

    if config is None:
        return selection
    defaults = _PROFILE_DEFAULTS.get(selection.name)
    if defaults is None:
        return selection
    parameters = dict(defaults)
    parameters.update(selection.parameters)
    return ModelSelection(selection.name, parameters)


class TrainOnlyFeatureSelector:
    """Deterministic supervised filter fitted exclusively on training rows."""

    def __init__(self, config: OverfittingControlConfig):
        if not isinstance(config, OverfittingControlConfig):
            raise TypeError("config must be OverfittingControlConfig.")
        self.config = config
        self.state: dict[str, Any] | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        if self.state is None:
            raise RuntimeError("Feature selector is not fitted.")
        return tuple(self.state["selected_columns"])

    @staticmethod
    def _relevance(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Between-class variance / total variance, robust to absent classes."""

        overall = values.mean(axis=0)
        total = ((values - overall) ** 2).sum(axis=0)
        between = np.zeros(values.shape[1], dtype=np.float64)
        for label in np.unique(labels):
            group = values[labels == label]
            between += len(group) * (group.mean(axis=0) - overall) ** 2
        return np.divide(between, total, out=np.zeros_like(between), where=total > 0)

    def fit(
        self,
        train: pd.DataFrame,
        columns: tuple[str, ...],
        *,
        label_col: str = "Label_id",
        supervised: bool = True,
    ) -> "TrainOnlyFeatureSelector":
        if not columns:
            raise ValueError("Feature selection requires at least one column.")
        missing = [column for column in columns if column not in train]
        if missing:
            raise ValueError(f"Missing feature-selection columns: {missing}")
        numeric = train.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        medians = numeric.median().fillna(0.0)
        values = numeric.fillna(medians).to_numpy(dtype=np.float64)
        variance = values.var(axis=0)
        eligible = np.isfinite(variance) & (variance >= self.config.min_feature_variance)
        if not eligible.any():
            raise ValueError("No feature survives minimum variance filtering.")
        candidates = np.flatnonzero(eligible)
        if supervised:
            if label_col not in train:
                raise ValueError(f"Training data has no {label_col!r} column.")
            labels = pd.to_numeric(train[label_col], errors="coerce").to_numpy()
            known = np.isfinite(labels)
            if "_label_known" in train:
                known &= train["_label_known"].fillna(False).to_numpy(dtype=bool)
            if not known.any():
                raise ValueError("Feature selection requires observed training labels.")
            scores = self._relevance(values[known], labels[known].astype(np.int64))
        else:
            scale = float(np.max(variance[candidates]))
            scores = variance / scale if scale > 0 else variance
        order = sorted(candidates, key=lambda idx: (-float(scores[idx]), columns[idx]))
        selected: list[int] = []
        for index in order:
            if selected:
                correlations = np.corrcoef(values[:, [index, *selected]], rowvar=False)[0, 1:]
                correlations = np.nan_to_num(np.abs(correlations), nan=0.0)
                if np.any(correlations > self.config.max_feature_correlation):
                    continue
            selected.append(index)
            if self.config.max_features is not None and len(selected) >= self.config.max_features:
                break
        if not selected:
            raise ValueError("No feature survives correlation filtering.")
        self.state = {
            "fit_scope": "train_only",
            "supervised": supervised,
            "input_columns": list(columns),
            "selected_columns": [columns[index] for index in selected],
            "dropped_columns": [column for index, column in enumerate(columns) if index not in selected],
            "scores": {columns[index]: float(scores[index]) for index in candidates},
            "variances": {columns[index]: float(variance[index]) for index in range(len(columns))},
            "config": {
                "max_features": self.config.max_features,
                "min_feature_variance": self.config.min_feature_variance,
                "max_feature_correlation": self.config.max_feature_correlation,
            },
        }
        return self

    def state_dict(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Feature selector is not fitted.")
        return dict(self.state)


def generalization_diagnostics(fit: FitResult) -> dict[str, float | int | str | None]:
    train, val = fit.history.train_loss, fit.history.val_loss
    if not train:
        return {"best_epoch": fit.best_epoch, "stop_reason": fit.stop_reason, "loss_gap": None}
    index = min(max(fit.best_epoch - 1, 0), len(train) - 1)
    train_loss = float(train[index])
    val_loss = float(val[index]) if index < len(val) else None
    return {
        "best_epoch": int(fit.best_epoch),
        "stop_reason": fit.stop_reason,
        "train_loss_at_best": train_loss,
        "val_loss_at_best": val_loss,
        "loss_gap": None if val_loss is None else float(val_loss - train_loss),
        "epochs_ran": len(train),
    }


__all__ = [
    "OverfittingControlConfig",
    "TrainOnlyFeatureSelector",
    "apply_overfitting_profile",
    "generalization_diagnostics",
]
