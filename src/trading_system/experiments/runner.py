from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_system.backtest.engine import evaluate_strategy_vs_buy_hold
from trading_system.data.scaling import SequenceStandardizer, Standardizer
from trading_system.data.splits import chronological_train_val_test_split
from trading_system.data.windows import build_sequence_dataset_with_history
from trading_system.evaluation.classification import evaluate_predictions
from trading_system.evaluation.thresholds import DecisionPolicy
from trading_system.features.market import (
    MARKET_FEATURE_COLUMNS,
    compute_market_features,
)
from trading_system.features.technical import (
    TECHNICAL_FEATURE_COLUMNS,
    compute_technical_features,
)
from trading_system.labels.breakout import (
    generate_breakout_labels,
    generate_breakout_labels_by_ticker,
    label_statistics,
)
from trading_system.labels.forward_return import build_forward_return_labels
from trading_system.labels.oracle_dp import build_oracle_labels_train_only
from trading_system.models.base import (
    FitResult,
    ProbabilisticClassifier,
    ProbabilisticSequenceClassifier,
)
from trading_system.models.manual_ann.manual_nn import ManualANNClassifier
from trading_system.models.manual_ann.sequence_adapter import ManualANNSequenceAdapter

from .config import ExperimentConfig


@dataclass
class TrainedModelBundle:
    estimator: ProbabilisticClassifier | ProbabilisticSequenceClassifier
    # Standardizer remains temporarily supported by legacy walk-forward code.
    # Static experiments use SequenceStandardizer and canonical 3D arrays.
    scaler: Standardizer | SequenceStandardizer
    feature_columns: tuple[str, ...]
    context_len: int
    decision_policy: DecisionPolicy
    feature_fill_values: pd.Series
    fit_result: FitResult

    def predict_proba(self, raw_windows: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_windows, dtype=np.float32)
        if isinstance(self.scaler, SequenceStandardizer):
            if values.ndim != 3 or values.shape[1:] != (
                self.context_len,
                len(self.feature_columns),
            ):
                raise ValueError(
                    "Raw sequence windows must match bundle dimensions (T, F)."
                )
        probabilities = self.estimator.predict_proba(self.scaler.transform(values))
        return align_probability_columns(self.estimator, probabilities)

    def predict(self, raw_windows: np.ndarray) -> np.ndarray:
        return self.decision_policy.predict(self.predict_proba(raw_windows))


@dataclass
class ExperimentResult:
    bundle: TrainedModelBundle
    config: ExperimentConfig
    label_stats: dict[str, int]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    backtest: dict[str, float]
    train_probabilities: np.ndarray
    val_probabilities: np.ndarray
    test_probabilities: np.ndarray
    train_predictions: np.ndarray
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    aligned_test_frame: pd.DataFrame
    split_sizes: dict[str, int]


def align_probability_columns(
    estimator: ProbabilisticClassifier | ProbabilisticSequenceClassifier,
    probabilities: np.ndarray,
    *,
    num_classes: int = 3,
) -> np.ndarray:
    """Align estimator probability columns to fixed Sell/Hold/Buy IDs."""

    if isinstance(num_classes, (bool, np.bool_)) or not isinstance(
        num_classes, (int, np.integer)
    ):
        raise TypeError("num_classes must be an integer.")
    num_classes = int(num_classes)
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or values.shape[1] == 0:
        raise ValueError("Estimator probabilities must be a non-empty 2D matrix.")
    if not np.isfinite(values).all():
        raise ValueError("Estimator probabilities must contain only finite values.")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("Estimator probabilities must be between zero and one.")
    source_row_sums = values.sum(axis=1)
    if not np.allclose(source_row_sums, 1.0, atol=1e-6):
        raise ValueError("Estimator probability rows must sum to one.")

    raw_classes = np.asarray(
        getattr(estimator, "classes_", np.arange(values.shape[1]))
    )
    if raw_classes.ndim != 1 or not np.issubdtype(raw_classes.dtype, np.integer):
        raise ValueError("Estimator classes_ must be a 1D array of integer IDs.")
    classes = raw_classes.astype(np.int64, copy=False)
    if values.shape[1] != len(classes):
        raise ValueError("Estimator probabilities do not match estimator classes_.")
    if (
        (classes < 0).any()
        or (classes >= num_classes).any()
        or len(np.unique(classes)) != len(classes)
    ):
        raise ValueError(
            "Estimator classes_ cannot be aligned to trading label schema."
        )
    aligned = np.zeros((len(values), num_classes), dtype=np.float64)
    for source_column, class_id in enumerate(classes):
        aligned[:, int(class_id)] = values[:, source_column]
    row_sums = aligned.sum(axis=1, keepdims=True)
    # Normalize only tiny accepted floating-point drift after column placement.
    return (aligned / row_sums).astype(np.float32)


def _filter_universe(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    work = frame.copy()
    if work.empty:
        raise ValueError("Input frame must not be empty.")
    if config.universe == "multi":
        if config.group_col not in work.columns:
            raise ValueError(f"Multi-universe experiment requires {config.group_col}.")
        return work
    if config.ticker is not None:
        if config.group_col not in work.columns:
            raise ValueError(f"Ticker filtering requires {config.group_col}.")
        work = work[work[config.group_col] == config.ticker].copy()
    elif (
        config.group_col in work.columns
        and work[config.group_col].nunique(dropna=False) > 1
    ):
        raise ValueError(
            "Single-universe input has multiple tickers; configure ticker explicitly."
        )
    if work.empty:
        raise ValueError("No rows remain after universe filtering.")
    return work


def _group_apply(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    function,
) -> pd.DataFrame:
    if config.universe == "multi" and config.group_col in frame.columns:
        parts = [
            function(group.copy())
            for _, group in frame.groupby(config.group_col, sort=False, dropna=False)
        ]
        return (
            pd.concat(parts, ignore_index=True)
            .sort_values([config.group_col, config.date_col])
            .reset_index(drop=True)
        )
    return function(frame.copy()).sort_values(config.date_col).reset_index(drop=True)


def _apply_labels(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    if config.label_mode == "breakout":
        if config.universe == "multi":
            return generate_breakout_labels_by_ticker(
                frame,
                config.label_window,
                price_col=config.price_col,
                group_col=config.group_col,
                date_col=config.date_col,
            )
        return generate_breakout_labels(
            frame,
            config.label_window,
            price_col=config.price_col,
            date_col=config.date_col,
        )

    if config.label_mode == "forward_return":
        return _group_apply(
            frame,
            config,
            lambda group: build_forward_return_labels(
                group,
                price_col=config.price_col,
                horizon=config.forward_horizon,
                buy_threshold=config.forward_buy_threshold,
                sell_threshold=config.forward_sell_threshold,
                date_col=config.date_col,
            )[0],
        )

    if config.label_mode == "oracle_all":
        return _group_apply(
            frame,
            config,
            lambda group: build_oracle_labels_train_only(
                group,
                price_col=config.price_col,
                initial_capital=config.initial_capital,
                fee_per_trade=config.oracle_fee_per_trade,
            )[0],
        )

    # Oracle train-only uses ordinary breakout labels for validation and test.
    breakout = (
        generate_breakout_labels_by_ticker(
            frame,
            config.label_window,
            price_col=config.price_col,
            group_col=config.group_col,
            date_col=config.date_col,
        )
        if config.universe == "multi"
        else generate_breakout_labels(
            frame,
            config.label_window,
            price_col=config.price_col,
            date_col=config.date_col,
        )
    )
    train, val, test = chronological_train_val_test_split(
        breakout,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        group_col=config.group_col if config.universe == "multi" else None,
        date_col=config.date_col,
    )
    oracle_train = _group_apply(
        train,
        config,
        lambda group: build_oracle_labels_train_only(
            group,
            price_col=config.price_col,
            initial_capital=config.initial_capital,
            fee_per_trade=config.oracle_fee_per_trade,
        )[0],
    )
    sort_columns = (
        [config.group_col, config.date_col]
        if config.universe == "multi"
        else [config.date_col]
    )
    return (
        pd.concat([oracle_train, val, test], ignore_index=True)
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )


def _build_features(
    labeled: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if config.feature_set == "technical":
        featured = compute_technical_features(
            labeled,
            group_col=config.group_col if config.universe == "multi" else None,
            date_col=config.date_col,
        )
        columns = tuple(TECHNICAL_FEATURE_COLUMNS)
    else:
        featured = compute_market_features(
            labeled,
            group_col=config.group_col,
            date_col=config.date_col,
        )
        columns = tuple(MARKET_FEATURE_COLUMNS)
    missing = [
        column for column in [*columns, "Label_id"] if column not in featured.columns
    ]
    if missing:
        raise ValueError(f"Feature builder did not produce required columns: {missing}")
    featured.loc[:, columns] = featured[list(columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    featured.loc[:, columns] = featured[list(columns)].replace(
        [np.inf, -np.inf], np.nan
    )
    featured["Label_id"] = pd.to_numeric(featured["Label_id"], errors="coerce")
    return featured.dropna(subset=["Label_id"]).copy(), columns


def _prepare_splits(
    featured: pd.DataFrame,
    columns: tuple[str, ...],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    train, val, test = chronological_train_val_test_split(
        featured,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        group_col=config.group_col if config.universe == "multi" else None,
        date_col=config.date_col,
    )
    train = train.dropna(subset=[*columns, "Label_id"]).copy()
    if train.empty:
        raise ValueError("No training rows remain after feature NaN removal.")
    fill_values = train[list(columns)].median(numeric_only=True).fillna(0.0)
    for split in (val, test):
        split.loc[:, columns] = split[list(columns)].fillna(fill_values).fillna(0.0)
    if val.empty or test.empty:
        raise ValueError("Validation and test splits must not be empty.")
    return train, val, test, fill_values


def _build_windows(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    columns: tuple[str, ...],
    config: ExperimentConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """Build identical 3D train/validation/test sequence layouts."""

    group_col = config.group_col if config.universe == "multi" else None
    X_train, y_train = build_sequence_dataset_with_history(
        train,
        columns,
        config.context_len,
        group_col=group_col,
        date_col=config.date_col,
    )
    X_val, y_val = build_sequence_dataset_with_history(
        val,
        columns,
        config.context_len,
        history_frame=train,
        group_col=group_col,
        date_col=config.date_col,
    )
    history = pd.concat([train, val], ignore_index=True)
    X_test, y_test, aligned_test = build_sequence_dataset_with_history(
        test,
        columns,
        config.context_len,
        history_frame=history,
        group_col=group_col,
        date_col=config.date_col,
        return_aligned_rows=True,
    )
    if not len(X_train) or not len(X_val) or not len(X_test):
        raise ValueError(
            "Context windowing produced an empty train, validation, or test set."
        )

    # The runner owns the canonical model boundary. Builders retain time and
    # feature axes; no architecture-specific flattening is allowed here.
    expected_tail = (config.context_len, len(columns))
    split_arrays = (
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    )
    for split_name, sequences, labels in split_arrays:
        if sequences.ndim != 3 or sequences.shape[1:] != expected_tail:
            raise RuntimeError(
                f"{split_name} sequences do not match expected dimensions (T, F)."
            )
        if labels.ndim != 1 or len(labels) != len(sequences):
            raise RuntimeError(f"{split_name} sequences and labels are misaligned.")
        if (labels < 0).any() or (labels >= 3).any():
            raise ValueError(
                f"{split_name} labels must use Sell/Hold/Buy IDs 0, 1 and 2."
            )
    if len(aligned_test) != len(X_test):
        raise RuntimeError("Test sequences and aligned backtest rows are misaligned.")
    return X_train, y_train, X_val, y_val, X_test, y_test, aligned_test


def _resolve_sequence_estimator(
    model: ProbabilisticSequenceClassifier | ManualANNClassifier | None,
    config: ExperimentConfig,
) -> ProbabilisticSequenceClassifier:
    """Return a classifier accepting canonical ``(N, T, F)`` inputs."""

    if model is None:
        return ManualANNSequenceAdapter(config.manual_ann)

    # Preserve the old public injection point while keeping flattening inside the
    # one dedicated adapter. This branch can disappear with the legacy 2D API.
    if isinstance(model, ManualANNClassifier):
        adapter = ManualANNSequenceAdapter(model.config)
        adapter.estimator = model
        adapter.classes_ = model.classes_.copy()
        return adapter

    if not isinstance(model, ProbabilisticSequenceClassifier):
        raise TypeError(
            "model must implement ProbabilisticSequenceClassifier and accept "
            "3D arrays shaped (N, T, F)."
        )
    return model


def run_experiment(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    model: ProbabilisticSequenceClassifier | ManualANNClassifier | None = None,
) -> ExperimentResult:
    """Run one leakage-aware static experiment on canonical 3D sequences."""

    work = _filter_universe(frame, config)
    labeled = _apply_labels(work, config)
    labels_summary = label_statistics(labeled)
    featured, feature_columns = _build_features(labeled, config)
    train, val, test, fill_values = _prepare_splits(featured, feature_columns, config)
    windows = _build_windows(train, val, test, feature_columns, config)
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, aligned_test = windows

    # Fit statistics only on training windows. Validation and test receive the
    # exact same per-feature normalization, without data-dependent refitting.
    scaler = SequenceStandardizer()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    estimator = _resolve_sequence_estimator(model, config)
    fit_result = estimator.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    if not isinstance(fit_result, FitResult):
        raise TypeError("model.fit() must return a FitResult.")

    train_probabilities = align_probability_columns(
        estimator, estimator.predict_proba(X_train)
    )
    val_probabilities = align_probability_columns(
        estimator, estimator.predict_proba(X_val)
    )
    test_probabilities = align_probability_columns(
        estimator, estimator.predict_proba(X_test)
    )
    decision_policy = DecisionPolicy.calibrate(
        val_probabilities,
        y_val,
        mode=config.decision_mode,
        min_action_rate=config.min_action_rate,
    )
    train_predictions = decision_policy.predict(train_probabilities)
    val_predictions = decision_policy.predict(val_probabilities)
    test_predictions = decision_policy.predict(test_probabilities)
    train_metrics = evaluate_predictions(y_train, train_predictions)
    val_metrics = evaluate_predictions(y_val, val_predictions)
    test_metrics = evaluate_predictions(y_test, test_predictions)
    backtest = evaluate_strategy_vs_buy_hold(
        aligned_test,
        test_predictions,
        initial_capital=config.initial_capital,
        price_col=config.price_col,
        fee_per_trade=config.fee_per_trade,
        position_mode=config.position_mode,
        execution_delay=config.execution_delay,
        group_col=config.group_col if config.universe == "multi" else None,
        date_col=config.date_col,
    )
    bundle = TrainedModelBundle(
        estimator=estimator,
        scaler=scaler,
        feature_columns=feature_columns,
        context_len=config.context_len,
        decision_policy=decision_policy,
        feature_fill_values=fill_values.copy(),
        fit_result=fit_result,
    )
    return ExperimentResult(
        bundle=bundle,
        config=config,
        label_stats=labels_summary,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        backtest=backtest,
        train_probabilities=train_probabilities,
        val_probabilities=val_probabilities,
        test_probabilities=test_probabilities,
        train_predictions=train_predictions,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        aligned_test_frame=aligned_test,
        split_sizes={"train": len(train), "val": len(val), "test": len(test)},
    )


__all__ = [
    "ExperimentResult",
    "TrainedModelBundle",
    "align_probability_columns",
    "run_experiment",
]
