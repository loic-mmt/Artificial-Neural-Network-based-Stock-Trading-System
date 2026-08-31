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
class ValidationResult:
    """Selection-safe result: no final-test predictions, metrics or prices."""

    bundle: TrainedModelBundle
    config: ExperimentConfig
    label_stats: dict[str, int]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    val_backtest: dict[str, float]
    train_probabilities: np.ndarray
    val_probabilities: np.ndarray
    train_predictions: np.ndarray
    val_predictions: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    # Unobserved forward-return targets stay in inference/backtests, not scores.
    val_label_mask: np.ndarray
    aligned_val_frame: pd.DataFrame
    split_sizes: dict[str, int]


@dataclass
class ExperimentResult(ValidationResult):
    test_metrics: dict[str, float]
    backtest: dict[str, float]
    test_probabilities: np.ndarray
    test_predictions: np.ndarray
    y_test: np.ndarray
    test_label_mask: np.ndarray
    aligned_test_frame: pd.DataFrame


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
    if "_experiment_split" not in breakout:
        raise ValueError("Oracle train-only labels require frozen split boundaries.")
    train_mask = breakout["_experiment_split"] == "train"
    oracle_train = _group_apply(
        breakout.loc[train_mask],
        config,
        lambda group: build_oracle_labels_train_only(
            group,
            price_col=config.price_col,
            initial_capital=config.initial_capital,
            fee_per_trade=config.oracle_fee_per_trade,
        )[0],
    )
    breakout.loc[train_mask, ["Label", "Label_id"]] = oracle_train[
        ["Label", "Label_id"]
    ].to_numpy()
    return breakout


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
    frame: pd.DataFrame,
    config: ExperimentConfig,
    *,
    include_test: bool = False,
    fill_values: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, tuple[str, ...]]:
    """Freeze raw split boundaries before labels/features; withhold test by default."""

    raw_splits = chronological_train_val_test_split(
        frame,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        group_col=config.group_col if config.universe == "multi" else None,
        date_col=config.date_col,
    )
    parts = []
    for name, split in zip(("train", "val", "test"), raw_splits):
        if name == "test" and not include_test:
            continue
        split = split.copy()
        split["_experiment_split"] = name
        split["_label_known"] = True
        if config.label_mode == "forward_return":
            # Exclude targets whose future price belongs to another partition,
            # but keep their feature rows as chronological context.
            groups = (
                split.groupby(config.group_col, sort=False, dropna=False)
                if config.universe == "multi"
                else [(None, split)]
            )
            for _, group in groups:
                unknown = group.tail(config.forward_horizon).index
                split.loc[unknown, "_label_known"] = False
        parts.append(split)
    source = pd.concat(parts, ignore_index=True)
    labeled = _apply_labels(source, config)
    featured, columns = _build_features(labeled, config)
    train, val, test = (
        featured.loc[featured["_experiment_split"] == name].copy()
        for name in ("train", "val", "test")
    )
    train = train.dropna(subset=[*columns, "Label_id"]).copy()
    if train.empty:
        raise ValueError("No training rows remain after feature NaN removal.")
    if fill_values is None:
        fill_values = train[list(columns)].median(numeric_only=True).fillna(0.0)
    for split in (val, test):
        split.loc[:, columns] = split[list(columns)].fillna(fill_values).fillna(0.0)
    if val.empty or (include_test and test.empty):
        raise ValueError("Validation and requested test splits must not be empty.")
    return train, val, test, fill_values, columns


def _build_split_windows(
    target: pd.DataFrame,
    columns: tuple[str, ...],
    config: ExperimentConfig,
    history: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build canonical sequences while retaining rows with unobserved targets."""

    group_col = config.group_col if config.universe == "multi" else None
    sequences, labels, aligned = build_sequence_dataset_with_history(
        target,
        columns,
        config.context_len,
        history_frame=history,
        group_col=group_col,
        date_col=config.date_col,
        return_aligned_rows=True,
    )
    if not len(sequences):
        raise ValueError("Context windowing produced an empty split.")

    # The runner owns the canonical model boundary. Builders retain time and
    # feature axes; no architecture-specific flattening is allowed here.
    expected_tail = (config.context_len, len(columns))
    if sequences.ndim != 3 or sequences.shape[1:] != expected_tail:
        raise RuntimeError("Sequences do not match expected dimensions (T, F).")
    if labels.ndim != 1 or not (len(labels) == len(sequences) == len(aligned)):
        raise RuntimeError("Sequences, labels and backtest rows are misaligned.")
    if (labels < 0).any() or (labels >= 3).any():
        raise ValueError("Labels must use Sell/Hold/Buy IDs 0, 1 and 2.")
    return sequences, labels, aligned


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


def run_validation_experiment(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    model: ProbabilisticSequenceClassifier | ManualANNClassifier | None = None,
) -> ValidationResult:
    """Fit and calibrate using training/validation only, leaving test untouched."""

    work = _filter_universe(frame, config)
    train, val, test, fill_values, feature_columns = _prepare_splits(work, config)
    X_train_raw, y_train, aligned_train = _build_split_windows(
        train, feature_columns, config
    )
    X_val_raw, y_val, aligned_val = _build_split_windows(
        val, feature_columns, config, train
    )
    train_mask = aligned_train["_label_known"].to_numpy(dtype=bool)
    val_mask = aligned_val["_label_known"].to_numpy(dtype=bool)
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Training and validation require observed label targets.")
    X_train_raw, y_train = X_train_raw[train_mask], y_train[train_mask]
    labels_summary = label_statistics(
        pd.concat([train.loc[train["_label_known"]], val.loc[val["_label_known"]]])
    )

    split_sizes = {
        "train": len(train),
        "val": len(val),
    }

    del train, val, test

    # Fit statistics only on training windows. Validation and test receive the
    # exact same per-feature normalization, without data-dependent refitting.
    print("\n=== SEQUENCE MEMORY ===")

    for name, X in [
        ("train", X_train_raw),
        ("val", X_val_raw),
    ]:
        print(
            f"{name:5s}: "
            f"shape={X.shape}, "
            f"dtype={X.dtype}, "
            f"size={X.nbytes / 1024**3:.3f} GiB"
        )

    print("=======================\n")
    scaler = SequenceStandardizer()
    X_train = scaler.fit_transform(X_train_raw)
    #del X_train_raw
    X_val = scaler.transform(X_val_raw)
    #del X_val_raw

    estimator = _resolve_sequence_estimator(model, config)
    fit_result = estimator.fit(
        X_train, y_train, X_val=X_val[val_mask], y_val=y_val[val_mask]
    )
    if not isinstance(fit_result, FitResult):
        raise TypeError("model.fit() must return a FitResult.")

    train_probabilities = align_probability_columns(
        estimator, estimator.predict_proba(X_train)
    )
    val_probabilities = align_probability_columns(
        estimator, estimator.predict_proba(X_val)
    )
    decision_policy = DecisionPolicy.calibrate(
        val_probabilities[val_mask],
        y_val[val_mask],
        mode=config.decision_mode,
        min_action_rate=config.min_action_rate,
    )
    train_predictions = decision_policy.predict(train_probabilities)
    val_predictions = decision_policy.predict(val_probabilities)
    train_metrics = evaluate_predictions(y_train, train_predictions)
    val_metrics = evaluate_predictions(y_val[val_mask], val_predictions[val_mask])
    val_backtest = evaluate_strategy_vs_buy_hold(
        aligned_val,
        val_predictions,
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
    return ValidationResult(
        bundle=bundle,
        config=config,
        label_stats=labels_summary,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_backtest=val_backtest,
        train_probabilities=train_probabilities,
        val_probabilities=val_probabilities,
        train_predictions=train_predictions,
        val_predictions=val_predictions,
        y_train=y_train,
        y_val=y_val,
        val_label_mask=val_mask,
        aligned_val_frame=aligned_val,
        split_sizes=split_sizes,
    )


def evaluate_experiment_test(
    frame: pd.DataFrame,
    validation: ValidationResult,
) -> ExperimentResult:
    """Evaluate a frozen fitted model/policy once, without training or calibration."""

    if isinstance(validation, ExperimentResult):
        raise ValueError("Final test has already been evaluated for this result.")
    config, bundle = validation.config, validation.bundle
    train, val, test, _, columns = _prepare_splits(
        _filter_universe(frame, config),
        config,
        include_test=True,
        fill_values=bundle.feature_fill_values,
    )
    if columns != bundle.feature_columns:
        raise ValueError("Test feature columns differ from the fitted bundle.")
    X_test, y_test, aligned_test = _build_split_windows(
        test, columns, config, pd.concat([train, val], ignore_index=True)
    )
    test_probabilities = bundle.predict_proba(X_test)
    test_predictions = bundle.decision_policy.predict(test_probabilities)
    label_mask = aligned_test["_label_known"].to_numpy(dtype=bool)
    if not label_mask.any():
        raise ValueError("Final test requires observed label targets.")
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
    return ExperimentResult(
        **{
            **validation.__dict__,
            "split_sizes": {**validation.split_sizes, "test": len(test)},
        },
        test_metrics=evaluate_predictions(
            y_test[label_mask], test_predictions[label_mask]
        ),
        backtest=backtest,
        test_probabilities=test_probabilities,
        test_predictions=test_predictions,
        y_test=y_test,
        test_label_mask=label_mask,
        aligned_test_frame=aligned_test,
    )


def run_experiment(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    model: ProbabilisticSequenceClassifier | ManualANNClassifier | None = None,
) -> ExperimentResult:
    """Run one static experiment, freezing model/policy before final evaluation."""

    validation = run_validation_experiment(frame, config, model)
    return evaluate_experiment_test(frame, validation)


__all__ = [
    "ExperimentResult",
    "ValidationResult",
    "TrainedModelBundle",
    "align_probability_columns",
    "evaluate_experiment_test",
    "run_experiment",
    "run_validation_experiment",
]
