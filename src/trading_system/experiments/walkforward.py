from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from time import perf_counter
from weakref import ref

import numpy as np
import pandas as pd

from trading_system.backtest.engine import evaluate_strategy_vs_buy_hold
from trading_system.data.scaling import SequenceStandardizer
from trading_system.data.splits import (
    chronological_train_val_split,
    chronological_train_val_test_split,
)
from trading_system.data.windows import (
    build_sequence_dataset_with_history,
    build_sequence_features,
)
from trading_system.evaluation.classification import evaluate_predictions
from trading_system.evaluation.thresholds import DecisionPolicy
from trading_system.labels.config import LabelConfig
from trading_system.labels.forward_return import build_forward_return_labels
from trading_system.labels.oracle_dp import build_oracle_labels_train_only
from trading_system.labels.registry import LabelContext, create_default_label_registry
from trading_system.models.base import ProbabilisticSequenceClassifier
from trading_system.models.factory import ModelRegistry, create_default_model_registry
from trading_system.models.manual_ann.manual_nn import (
    ManualANNClassifier,
    ManualANNConfig,
)
from trading_system.models.manual_ann.sequence_adapter import ManualANNSequenceAdapter
from trading_system.models.specs import ModelBuildContext, ModelSelection

from .runner import TrainedModelBundle, _select_label_rows, align_probability_columns
from trading_system.features.fracdiff import (
    FRACDIFF_FEATURE, FracDiffConfig, FracDiffTransformer,
)
from trading_system.training.sample_weighting import SampleWeightConfig, prepare_sample_weights
from trading_system.training.financial_loss import (
    FinancialLossConfig, ReturnPanel, probabilities_to_positions,
)
from trading_system.training.position_trainer import fit_position_model, predict_positions
from trading_system.features.expanded import ExpandedFeatureSelector
from trading_system.training.overfitting import (
    OverfittingControlConfig,
    TrainOnlyFeatureSelector,
    apply_overfitting_profile,
    generalization_diagnostics,
)

# Compatibility hook. New code should pass ModelSelection plus ModelRegistry.
ModelFactory = Callable[[int], ProbabilisticSequenceClassifier]


def derive_chunk_seed(run_seed: int, chunk_id: int) -> int:
    """Stable independent retrain seeds, without touching global RNG state."""

    for name, value in (("run_seed", run_seed), ("chunk_id", chunk_id)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
    return int(np.random.SeedSequence([run_seed, chunk_id]).generate_state(1)[0])


def _label_history(
    frame: pd.DataFrame,
    *,
    label_mode: str,
    price_col: str,
    initial_capital: float,
    oracle_fee_per_trade: float,
    forward_horizon: int,
    forward_buy_threshold: float,
    forward_sell_threshold: float,
    breakout_window: int,
    breakout_buy_buffer: float,
    breakout_sell_buffer: float,
    breakout_alternating: bool,
    volatility_horizon: int,
    volatility_window: int,
    volatility_long_threshold: float,
    volatility_short_threshold: float,
    volatility_exit_threshold: float,
    volatility_min_holding_period: int,
    volatility_cooldown: int,
    volatility_cost_bps: float,
    volatility_position_mode: str,
    triple_barrier_max_holding: int,
    triple_barrier_volatility_window: int,
    triple_barrier_volatility_estimator: str,
    triple_barrier_profit_barrier: float,
    triple_barrier_stop_barrier: float,
    triple_barrier_event_filter: str,
    triple_barrier_cusum_threshold: float,
    triple_barrier_cost_bps: float,
    triple_barrier_between_event_policy: str,
) -> tuple[pd.DataFrame, dict]:
    if label_mode == "oracle_dp":
        return build_oracle_labels_train_only(
            frame,
            price_col=price_col,
            initial_capital=initial_capital,
            fee_per_trade=oracle_fee_per_trade,
        )
    if label_mode == "forward_return":
        label_config = LabelConfig.forward_return(
            horizon=forward_horizon,
            buy_threshold=forward_buy_threshold,
            sell_threshold=forward_sell_threshold,
        )
    elif label_mode == "breakout":
        label_config = LabelConfig.breakout(
            window=breakout_window,
            buy_buffer=breakout_buy_buffer,
            sell_buffer=breakout_sell_buffer,
            alternating=breakout_alternating,
        )
    elif label_mode == "volatility_position":
        label_config = LabelConfig.volatility_position(
            horizon=volatility_horizon,
            volatility_window=volatility_window,
            long_threshold=volatility_long_threshold,
            short_threshold=volatility_short_threshold,
            exit_threshold=volatility_exit_threshold,
            min_holding_period=volatility_min_holding_period,
            cooldown=volatility_cooldown,
            cost_bps=volatility_cost_bps,
            position_mode=volatility_position_mode,
        )
    elif label_mode == "triple_barrier":
        label_config = LabelConfig.triple_barrier(
            max_holding=triple_barrier_max_holding,
            volatility_window=triple_barrier_volatility_window,
            volatility_estimator=triple_barrier_volatility_estimator,
            profit_barrier=triple_barrier_profit_barrier,
            stop_barrier=triple_barrier_stop_barrier,
            event_filter=triple_barrier_event_filter,
            cusum_threshold=triple_barrier_cusum_threshold,
            cost_bps=triple_barrier_cost_bps,
            between_event_policy=triple_barrier_between_event_policy,
        )
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    result = create_default_label_registry().generate(
        frame,
        label_config,
        LabelContext(price_col=price_col, date_col="date"),
    )
    result.frame["_label_known"] = result.known_mask
    counts = result.metadata["class_counts"]
    return result.frame, {
        **result.metadata,
        **{f"n_{name.lower()}": count for name, count in counts.items()},
    }


def _mark_label_segments(
    frame: pd.DataFrame,
    split_at: int,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Freeze a chronological boundary before future-aware label generation."""

    work = frame.sort_values(date_col).reset_index(drop=True).copy()
    if not 0 < split_at < len(work):
        raise ValueError("Label split must leave rows on both sides.")
    work["_experiment_split"] = np.where(
        np.arange(len(work)) < split_at,
        "history",
        "evaluation",
    )
    return work


def fit_labeled_history(
    labeled_history: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    val_ratio: float,
    context_len: int,
    decision_mode: str,
    min_action_rate: float,
    estimator: ProbabilisticSequenceClassifier | ManualANNClassifier,
    model_selection: ModelSelection | None = None,
    date_col: str = "date",
    forward_horizon: int = 0,
    fracdiff_config: FracDiffConfig | None = None,
    sample_weighting: SampleWeightConfig | None = None,
    price_col: str = "adj_close",
    feature_selector: ExpandedFeatureSelector | None = None,
    overfitting_selector: TrainOnlyFeatureSelector | None = None,
) -> tuple[TrainedModelBundle, pd.DataFrame, dict[str, float]]:
    columns = tuple(feature_columns)
    work = labeled_history.sort_values(date_col).reset_index(drop=True).copy()
    work["Label_id"] = pd.to_numeric(work["Label_id"], errors="coerce")
    work.loc[:, columns] = work[list(columns)].apply(pd.to_numeric, errors="coerce")
    work.loc[:, columns] = work[list(columns)].replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["Label_id"]).copy()
    train, val = chronological_train_val_split(
        work, val_ratio=val_ratio, date_col=date_col
    )
    fracdiff_transformer = None
    if fracdiff_config is not None:
        if FRACDIFF_FEATURE in columns:
            raise ValueError("Pass base feature columns; FracDiff is fitted internally.")
        fracdiff_transformer = FracDiffTransformer(
            fracdiff_config, price_col=price_col, date_col=date_col
        ).fit(train)
        transformed = fracdiff_transformer.transform(work)
        train, val = transformed.iloc[:len(train)].copy(), transformed.iloc[len(train):].copy()
        columns = (*columns, FRACDIFF_FEATURE)
    del work
    for split in (train, val):
        if "_label_known" not in split.columns:
            split["_label_known"] = True
        else:
            split["_label_known"] = split["_label_known"].astype(bool)
        if forward_horizon:
            split.loc[split.tail(forward_horizon).index, "_label_known"] = False
    if feature_selector is not None:
        train = train.dropna(subset=["ret_20", *([FRACDIFF_FEATURE] if fracdiff_config else [])]).copy()
    else:
        train = train.dropna(subset=list(columns)).copy()
    if train.empty:
        raise ValueError("Walk-forward training history has no complete feature rows.")
    fill_values = train[list(columns)].median(numeric_only=True).fillna(0.0)
    val.loc[:, columns] = val[list(columns)].fillna(fill_values).fillna(0.0)
    train.loc[:, columns] = train[list(columns)].fillna(fill_values).fillna(0.0)
    X_train_raw, y_train, aligned_train = build_sequence_dataset_with_history(
        train,
        columns,
        context_len,
        group_col=None,
        date_col=date_col,
        return_aligned_rows=True,
    )
    X_val_raw, y_val, aligned_val = build_sequence_dataset_with_history(
        val,
        columns,
        context_len,
        history_frame=train,
        group_col=None,
        date_col=date_col,
        return_aligned_rows=True,
    )
    train_mask = aligned_train["_label_known"].to_numpy(dtype=bool)
    val_mask = aligned_val["_label_known"].to_numpy(dtype=bool)
    X_train_raw = _select_label_rows(X_train_raw, train_mask)
    y_train = _select_label_rows(y_train, train_mask)
    X_val_raw = _select_label_rows(X_val_raw, val_mask)
    y_val = _select_label_rows(y_val, val_mask)
    weight_arguments, sample_weight_state = prepare_sample_weights(
        aligned_train.loc[train_mask], aligned_val.loc[val_mask], sample_weighting,
        date_col=date_col,
    )
    del aligned_train, aligned_val, train_mask, val_mask
    if not len(X_train_raw) or not len(X_val_raw):
        raise ValueError(
            "Walk-forward context windowing produced an empty train or validation set."
        )
    scaler = SequenceStandardizer()
    X_train = scaler.fit_transform(X_train_raw)
    del X_train_raw
    X_val = scaler.transform(X_val_raw)
    del X_val_raw
    if isinstance(estimator, ManualANNClassifier):
        adapter = ManualANNSequenceAdapter(estimator.config)
        adapter.estimator = estimator
        adapter.classes_ = estimator.classes_.copy()
        estimator = adapter
    if not isinstance(estimator, ProbabilisticSequenceClassifier):
        raise TypeError("estimator must implement ProbabilisticSequenceClassifier.")
    fit_result = estimator.fit(X_train, y_train, X_val=X_val, y_val=y_val, **weight_arguments)
    del X_train
    val_probabilities = align_probability_columns(
        estimator, estimator.predict_proba(X_val)
    )
    del X_val
    policy = DecisionPolicy.calibrate(
        val_probabilities,
        y_val,
        mode=decision_mode,
        min_action_rate=min_action_rate,
    )
    val_predictions = policy.predict(val_probabilities)
    metrics = evaluate_predictions(y_val, val_predictions)
    filled_history = (
        pd.concat([train, val], ignore_index=True)
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    bundle = TrainedModelBundle(
        estimator=estimator,
        scaler=scaler,
        feature_columns=columns,
        context_len=context_len,
        decision_policy=policy,
        feature_fill_values=fill_values,
        fit_result=fit_result,
        model_selection=model_selection
        or ModelSelection(getattr(estimator, "model_name", type(estimator).__name__)),
        fracdiff_transformer=fracdiff_transformer,
        sample_weight_state=sample_weight_state,
        feature_selector=feature_selector,
        overfitting_selector=overfitting_selector,
    )
    return bundle, filled_history, metrics


def predict_chunk_with_model(
    bundle: TrainedModelBundle,
    history: pd.DataFrame,
    chunk: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    columns = list(bundle.feature_columns)
    work = chunk.copy().reset_index(drop=True)
    work["_chunk_local_index"] = np.arange(len(work), dtype=np.int64)
    if bundle.fracdiff_transformer is not None:
        source = bundle.fracdiff_transformer.transform(
            pd.concat([history, work], ignore_index=True)
        )
        history = source.iloc[:len(history)].copy()
        work = source.iloc[len(history):].copy().reset_index(drop=True)
        history.loc[:, columns] = history[columns].fillna(bundle.feature_fill_values).fillna(0.0)
    work.loc[:, columns] = work[columns].apply(pd.to_numeric, errors="coerce")
    work.loc[:, columns] = work[columns].fillna(bundle.feature_fill_values).fillna(0.0)
    prefix = history.tail(bundle.context_len - 1)[columns].copy()
    source = pd.concat([prefix, work[columns]], ignore_index=True)
    windows, source_indices = build_sequence_features(
        source,
        columns,
        bundle.context_len,
        target_start=len(prefix),
        return_indices=True,
    )
    if not len(windows):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    local_indices = source_indices - len(prefix)
    valid = (local_indices >= 0) & (local_indices < len(work))
    return bundle.predict(windows[valid]), local_indices[valid].astype(np.int64)


def fit_position_history(
    labeled_history: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    val_ratio: float,
    context_len: int,
    estimator: ProbabilisticSequenceClassifier | ManualANNClassifier,
    loss_config: FinancialLossConfig,
    position_mode: str,
    execution_delay: int,
    initial_capital: float = 10_000.0,
    model_selection: ModelSelection | None = None,
    date_col: str = "date",
    price_col: str = "adj_close",
    fracdiff_config: FracDiffConfig | None = None,
    feature_selector: ExpandedFeatureSelector | None = None,
    overfitting_selector: TrainOnlyFeatureSelector | None = None,
) -> tuple[TrainedModelBundle, pd.DataFrame, dict[str, float]]:
    """Fit one expanding-window position model; validation owns checkpointing."""
    if loss_config.objective == "cross_entropy":
        raise ValueError("Use fit_labeled_history for cross_entropy.")
    columns = tuple(feature_columns)
    work = labeled_history.sort_values(date_col).reset_index(drop=True).copy()
    work.loc[:, columns] = work[list(columns)].apply(pd.to_numeric, errors="coerce")
    work.loc[:, columns] = work[list(columns)].replace([np.inf, -np.inf], np.nan)
    train, val = chronological_train_val_split(work, val_ratio=val_ratio, date_col=date_col)
    fracdiff_transformer = None
    if fracdiff_config is not None:
        if FRACDIFF_FEATURE in columns:
            raise ValueError("Pass base feature columns; FracDiff is fitted internally.")
        fracdiff_transformer = FracDiffTransformer(
            fracdiff_config, price_col=price_col, date_col=date_col
        ).fit(train)
        transformed = fracdiff_transformer.transform(work)
        train = transformed.iloc[:len(train)].copy()
        val = transformed.iloc[len(train):].copy()
        columns = (*columns, FRACDIFF_FEATURE)
    del work
    if feature_selector is not None:
        train = train.dropna(
            subset=["ret_20", *([FRACDIFF_FEATURE] if fracdiff_config else [])]
        ).copy()
    else:
        train = train.dropna(subset=list(columns)).copy()
    if train.empty:
        raise ValueError("Walk-forward position history has no complete feature rows.")
    fills = train[list(columns)].median(numeric_only=True).fillna(0.0)
    for split in (train, val):
        split.loc[:, columns] = split[list(columns)].fillna(fills).fillna(0.0)
    X_train_raw, _, aligned_train = build_sequence_dataset_with_history(
        train, columns, context_len, group_col=None, date_col=date_col,
        return_aligned_rows=True,
    )
    X_val_raw, _, aligned_val = build_sequence_dataset_with_history(
        val, columns, context_len, history_frame=train, group_col=None,
        date_col=date_col, return_aligned_rows=True,
    )
    if not len(X_train_raw) or not len(X_val_raw):
        raise ValueError("Position windowing produced an empty train or validation set.")
    scaler = SequenceStandardizer()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    del X_train_raw, X_val_raw
    if isinstance(estimator, ManualANNClassifier):
        adapter = ManualANNSequenceAdapter(estimator.config)
        adapter.estimator = estimator
        adapter.classes_ = estimator.classes_.copy()
        estimator = adapter
    if not isinstance(estimator, ProbabilisticSequenceClassifier):
        raise TypeError("estimator must implement ProbabilisticSequenceClassifier.")
    train_panel = ReturnPanel(
        aligned_train, price_col=price_col, date_col=date_col,
        execution_delay=execution_delay,
    )
    val_panel = ReturnPanel(
        aligned_val, price_col=price_col, date_col=date_col,
        execution_delay=execution_delay,
    )
    fit = fit_position_model(
        estimator, X_train, train_panel, X_val, val_panel, loss_config, position_mode
    )
    positions = predict_positions(estimator, X_val, position_mode)
    metrics = val_panel.metrics(positions, loss_config, initial_capital)
    filled_history = pd.concat([train, val], ignore_index=True).sort_values(date_col).reset_index(drop=True)
    bundle = TrainedModelBundle(
        estimator, scaler, columns, context_len, DecisionPolicy(mode="argmax"),
        fills.copy(), fit,
        model_selection or ModelSelection(getattr(estimator, "model_name", type(estimator).__name__)),
        fracdiff_transformer, None, feature_selector, overfitting_selector,
    )
    return bundle, filled_history, metrics


def predict_position_chunk(
    bundle: TrainedModelBundle,
    history: pd.DataFrame,
    chunk: pd.DataFrame,
    position_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal continuous positions for one chunk, using frozen preprocessing."""
    columns = list(bundle.feature_columns)
    work = chunk.copy().reset_index(drop=True)
    if bundle.fracdiff_transformer is not None:
        source = bundle.fracdiff_transformer.transform(pd.concat([history, work], ignore_index=True))
        history = source.iloc[:len(history)].copy()
        work = source.iloc[len(history):].copy().reset_index(drop=True)
        history.loc[:, columns] = history[columns].fillna(bundle.feature_fill_values).fillna(0.0)
    work.loc[:, columns] = work[columns].apply(pd.to_numeric, errors="coerce")
    work.loc[:, columns] = work[columns].fillna(bundle.feature_fill_values).fillna(0.0)
    prefix = history.tail(bundle.context_len - 1)[columns].copy()
    windows, indices = build_sequence_features(
        pd.concat([prefix, work[columns]], ignore_index=True), columns,
        bundle.context_len, target_start=len(prefix), return_indices=True,
    )
    local = indices - len(prefix)
    valid = (local >= 0) & (local < len(work))
    if not valid.any():
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    probabilities = bundle.predict_proba(windows[valid])
    return probabilities_to_positions(probabilities, position_mode), local[valid].astype(np.int64)


def walk_forward_classifier(
    full_df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    price_col: str = "adj_close",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    walkforward_step: int = 20,
    oracle_fee_per_trade: float = 2.0,
    label_mode: str = "forward_return",
    forward_horizon: int = 1,
    forward_buy_threshold: float = 0.002,
    forward_sell_threshold: float = 0.002,
    breakout_window: int = 20,
    breakout_buy_buffer: float = 0.0,
    breakout_sell_buffer: float = 0.0,
    breakout_alternating: bool = True,
    volatility_horizon: int = 10,
    volatility_window: int = 20,
    volatility_long_threshold: float = 1.0,
    volatility_short_threshold: float = 1.5,
    volatility_exit_threshold: float = 0.25,
    volatility_min_holding_period: int = 5,
    volatility_cooldown: int = 0,
    volatility_cost_bps: float = 5.0,
    volatility_position_mode: str = "long_flat",
    triple_barrier_max_holding: int = 10,
    triple_barrier_volatility_window: int = 20,
    triple_barrier_volatility_estimator: str = "rolling_std",
    fracdiff_config: FracDiffConfig | None = None,
    sample_weighting: SampleWeightConfig | None = None,
    expanded_min_coverage: float | None = None,
    triple_barrier_profit_barrier: float = 1.0,
    triple_barrier_stop_barrier: float = 1.0,
    triple_barrier_event_filter: str = "all",
    triple_barrier_cusum_threshold: float = 0.5,
    triple_barrier_cost_bps: float = 5.0,
    triple_barrier_between_event_policy: str = "hold",
    decision_mode: str = "argmax",
    min_action_rate: float = 0.0,
    position_mode: str = "long_only",
    strategy_fee_per_trade: float = 0.0,
    initial_capital: float = 10_000.0,
    execution_delay: int = 1,
    context_len: int = 20,
    model_factory: ModelFactory | None = None,
    manual_ann_config: ManualANNConfig | None = None,
    model_selection: ModelSelection | None = None,
    registry: ModelRegistry | None = None,
    seed: int = 1,
    device: str = "auto",
    evaluation_split: str = "test",
    financial_loss: FinancialLossConfig | None = None,
    overfitting_control: OverfittingControlConfig | None = None,
) -> dict[str, object]:
    if model_factory is not None and (model_selection is not None or registry is not None):
        raise ValueError("Legacy model_factory cannot be combined with registry selection.")
    if walkforward_step <= 0:
        raise ValueError("walkforward_step must be positive.")
    if evaluation_split not in ("validation", "test"):
        raise ValueError("evaluation_split must be 'validation' or 'test'.")
    if sample_weighting is not None and label_mode != "triple_barrier":
        raise ValueError("Event sample weighting requires triple_barrier labels.")
    if financial_loss is not None:
        if not isinstance(financial_loss, FinancialLossConfig):
            raise TypeError("financial_loss must be FinancialLossConfig or None.")
        if financial_loss.objective == "cross_entropy":
            financial_loss = None
        elif sample_weighting is not None:
            raise ValueError("Sample weighting cannot weight a chronological financial objective.")
        elif label_mode == "oracle_dp":
            raise ValueError("Oracle labels are not permitted with financial objectives.")
    data = full_df.sort_values("date").reset_index(drop=True).copy()
    if financial_loss is not None and "ticker" in data and data["ticker"].nunique() > 1:
        raise ValueError("Financial walk-forward currently requires one ticker; filter explicitly.")
    if len(data) < context_len + 50:
        raise ValueError(
            "Dataset is too short for configured context and walk-forward evaluation."
        )
    missing = [
        column
        for column in ["date", price_col, *feature_columns]
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Missing walk-forward columns: {missing}")
    initial_train, _, initial_test = chronological_train_val_test_split(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    test_start = len(data) - len(initial_test)
    if evaluation_split == "validation":
        # Truncate before labeling and retraining, not merely before reporting.
        data = data.iloc[:test_start].copy()
        test_start = len(initial_train)
    evaluation_label_source = _mark_label_segments(data, test_start)
    if financial_loss is not None:
        evaluation_labels = evaluation_label_source.assign(
            Label_id=1, _label_known=True
        )
        evaluation_label_report = {
            "method": "unused_for_financial_objective",
            "objective": financial_loss.objective,
        }
    else:
        evaluation_labels, evaluation_label_report = _label_history(
            evaluation_label_source,
            label_mode=label_mode,
            price_col=price_col,
            initial_capital=initial_capital,
            oracle_fee_per_trade=oracle_fee_per_trade,
            forward_horizon=forward_horizon,
            forward_buy_threshold=forward_buy_threshold,
            forward_sell_threshold=forward_sell_threshold,
            breakout_window=breakout_window,
            breakout_buy_buffer=breakout_buy_buffer,
            breakout_sell_buffer=breakout_sell_buffer,
            breakout_alternating=breakout_alternating,
            volatility_horizon=volatility_horizon,
            volatility_window=volatility_window,
            volatility_long_threshold=volatility_long_threshold,
            volatility_short_threshold=volatility_short_threshold,
            volatility_exit_threshold=volatility_exit_threshold,
            volatility_min_holding_period=volatility_min_holding_period,
            volatility_cooldown=volatility_cooldown,
            volatility_cost_bps=volatility_cost_bps,
            volatility_position_mode=volatility_position_mode,
            triple_barrier_max_holding=triple_barrier_max_holding,
            triple_barrier_volatility_window=triple_barrier_volatility_window,
            triple_barrier_volatility_estimator=triple_barrier_volatility_estimator,
            triple_barrier_profit_barrier=triple_barrier_profit_barrier,
            triple_barrier_stop_barrier=triple_barrier_stop_barrier,
            triple_barrier_event_filter=triple_barrier_event_filter,
            triple_barrier_cusum_threshold=triple_barrier_cusum_threshold,
            triple_barrier_cost_bps=triple_barrier_cost_bps,
            triple_barrier_between_event_policy=triple_barrier_between_event_policy,
        )
    y_true_global = evaluation_labels["Label_id"].to_numpy(dtype=np.int64)
    known_labels_global = evaluation_labels.get(
        "_label_known",
        pd.Series(True, index=evaluation_labels.index),
    ).to_numpy(dtype=bool)
    predictions = (
        np.full(len(data), np.nan, dtype=np.float64)
        if financial_loss is not None
        else np.full(len(data), -1, dtype=np.int64)
    )
    retrain_logs: list[dict[str, object]] = []
    if manual_ann_config is not None:
        if model_selection is not None:
            raise ValueError("manual_ann_config cannot be combined with model_selection.")
        parameters = asdict(manual_ann_config)
        seed = int(parameters.pop("seed"))
        parameters.pop("num_classes")
        model_selection = ModelSelection("manual_ann", parameters)
    selection = model_selection or ModelSelection(
        "manual_ann",
        {
            "hidden_size": 64,
            "epochs": 150,
            "batch_size": 64,
            "early_stopping_patience": 30,
        },
    )
    selection = apply_overfitting_profile(selection, overfitting_control)
    model_registry = registry or create_default_model_registry()
    previous_estimators = []

    for chunk_id, start in enumerate(
        range(test_start, len(data), walkforward_step), start=1
    ):
        end = min(start + walkforward_step, len(data))
        history = data.iloc[:start].copy()
        history_train, _ = chronological_train_val_split(
            history,
            val_ratio=val_ratio,
            date_col="date",
        )
        label_source = _mark_label_segments(history, len(history_train))
        if financial_loss is not None:
            labeled_history = label_source.assign(Label_id=1, _label_known=True)
            label_report = {
                "method": "unused_for_financial_objective",
                "objective": financial_loss.objective,
            }
        else:
            labeled_history, label_report = _label_history(
                label_source,
                label_mode=label_mode,
                price_col=price_col,
                initial_capital=initial_capital,
                oracle_fee_per_trade=oracle_fee_per_trade,
                forward_horizon=forward_horizon,
                forward_buy_threshold=forward_buy_threshold,
                forward_sell_threshold=forward_sell_threshold,
                breakout_window=breakout_window,
                breakout_buy_buffer=breakout_buy_buffer,
                breakout_sell_buffer=breakout_sell_buffer,
                breakout_alternating=breakout_alternating,
                volatility_horizon=volatility_horizon,
                volatility_window=volatility_window,
                volatility_long_threshold=volatility_long_threshold,
                volatility_short_threshold=volatility_short_threshold,
                volatility_exit_threshold=volatility_exit_threshold,
                volatility_min_holding_period=volatility_min_holding_period,
                volatility_cooldown=volatility_cooldown,
                volatility_cost_bps=volatility_cost_bps,
                volatility_position_mode=volatility_position_mode,
                triple_barrier_max_holding=triple_barrier_max_holding,
                triple_barrier_volatility_window=triple_barrier_volatility_window,
                triple_barrier_volatility_estimator=triple_barrier_volatility_estimator,
                triple_barrier_profit_barrier=triple_barrier_profit_barrier,
                triple_barrier_stop_barrier=triple_barrier_stop_barrier,
                triple_barrier_event_filter=triple_barrier_event_filter,
                triple_barrier_cusum_threshold=triple_barrier_cusum_threshold,
                triple_barrier_cost_bps=triple_barrier_cost_bps,
                triple_barrier_between_event_policy=triple_barrier_between_event_policy,
            )
        started = perf_counter()
        chunk_seed = derive_chunk_seed(seed, chunk_id)
        feature_selector = None
        overfitting_selector = None
        active_columns = tuple(feature_columns)
        selection_train = None
        if expanded_min_coverage is not None or overfitting_control is not None:
            selection_train, _ = chronological_train_val_split(labeled_history, val_ratio=val_ratio, date_col="date")
        if expanded_min_coverage is not None:
            selection_train = selection_train.dropna(subset=["ret_20"])
            feature_selector = ExpandedFeatureSelector(expanded_min_coverage).fit(selection_train, active_columns)
            active_columns = feature_selector.columns
        if overfitting_control is not None:
            assert selection_train is not None
            overfitting_selector = TrainOnlyFeatureSelector(overfitting_control).fit(
                selection_train,
                active_columns,
                supervised=financial_loss is None,
            )
            active_columns = overfitting_selector.columns
        estimator = (
            model_factory(chunk_id)
            if model_factory is not None
            else model_registry.build(
                selection.name,
                ModelBuildContext(
                    input_size=len(active_columns) + int(fracdiff_config is not None),
                    context_len=context_len,
                    num_classes=3,
                    seed=chunk_seed,
                    device=device,
                ),
                selection.parameters,
            )
        )
        if any(estimator is previous() for previous in previous_estimators):
            raise ValueError(
                "model_factory must return a fresh estimator for each chunk."
            )
        try:
            previous_estimators.append(ref(estimator))
        except TypeError:
            # Preserve the identity guard for slot-only custom classifiers.
            previous_estimators.append(lambda estimator=estimator: estimator)
        if financial_loss is None:
            bundle, filled_history, val_metrics = fit_labeled_history(
                labeled_history,
                active_columns,
                val_ratio=val_ratio,
                context_len=context_len,
                decision_mode=decision_mode,
                min_action_rate=min_action_rate,
                estimator=estimator,
                model_selection=selection,
                fracdiff_config=fracdiff_config,
                sample_weighting=sample_weighting,
                price_col=price_col,
                feature_selector=feature_selector,
                overfitting_selector=overfitting_selector,
                forward_horizon=(
                    forward_horizon
                    if label_mode == "forward_return"
                    else volatility_horizon
                    if label_mode == "volatility_position"
                    else triple_barrier_max_holding
                    if label_mode == "triple_barrier"
                    else 0
                ),
            )
        else:
            bundle, filled_history, val_metrics = fit_position_history(
                labeled_history,
                active_columns,
                val_ratio=val_ratio,
                context_len=context_len,
                estimator=estimator,
                loss_config=financial_loss,
                position_mode=position_mode,
                execution_delay=execution_delay,
                initial_capital=initial_capital,
                model_selection=selection,
                fracdiff_config=fracdiff_config,
                price_col=price_col,
                feature_selector=feature_selector,
                overfitting_selector=overfitting_selector,
            )
        chunk = data.iloc[start:end].copy().reset_index(drop=True)
        predict_history = history if fracdiff_config is not None else filled_history
        if financial_loss is None:
            chunk_predictions, local_indices = predict_chunk_with_model(
                bundle, predict_history, chunk,
            )
        else:
            chunk_predictions, local_indices = predict_position_chunk(
                bundle, predict_history, chunk, position_mode,
            )
        absolute_indices = np.arange(start, end, dtype=np.int64)[local_indices]
        predictions[absolute_indices] = chunk_predictions
        retrain_logs.append(
            {
                "chunk_id": chunk_id,
                "run_seed": seed,
                "seed": chunk_seed,
                "seed_applied_by_registry": model_factory is None,
                "model_name": bundle.model_selection.name,
                "model_parameters": dict(bundle.model_selection.parameters),
                "parameter_count": bundle.parameter_count(),
                "duration_seconds": perf_counter() - started,
                "start_idx": start,
                "end_idx": end,
                "n_hist": len(history),
                "n_pred": len(chunk_predictions),
                "best_epoch": int(bundle.fit_result.best_epoch),
                "loss_objective": financial_loss.objective if financial_loss else "cross_entropy",
                "validation_metrics": dict(val_metrics),
                "val_macro_f1": val_metrics.get("macro_f1"),
                "val_bal_acc": val_metrics.get("bal_acc"),
                "label_hist_info": label_report,
                "sample_weight_state": bundle.sample_weight_state,
                "feature_selection": feature_selector.state_dict() if feature_selector else None,
                "overfitting_feature_selection": (
                    overfitting_selector.state_dict() if overfitting_selector else None
                ),
                "generalization": generalization_diagnostics(bundle.fit_result),
                "fracdiff_state": (
                    bundle.fracdiff_transformer.state_dict()
                    if bundle.fracdiff_transformer is not None else None
                ),
            }
        )

    test_mask = np.arange(len(data)) >= test_start
    evaluation_mask = test_mask & (
        np.isfinite(predictions) if financial_loss is not None else predictions >= 0
    )
    if not evaluation_mask.any():
        raise RuntimeError("Walk-forward evaluation produced no predictions.")
    label_mask = evaluation_mask & known_labels_global
    y_pred = predictions[evaluation_mask]
    aligned_test = data.loc[evaluation_mask].reset_index(drop=True)
    if financial_loss is not None:
        panel = ReturnPanel(
            aligned_test, price_col=price_col, date_col="date",
            execution_delay=execution_delay,
        )
        benchmark = panel.metrics(y_pred, financial_loss, initial_capital)
        buy_hold = panel.metrics(np.ones(len(y_pred)), financial_loss, initial_capital)
        benchmark.update(
            model_pnl=benchmark["net_pnl"],
            buy_hold_pnl=buy_hold["net_pnl"],
            outperformance=benchmark["net_pnl"] - buy_hold["net_pnl"],
            buy_hold_net_return=buy_hold["net_return"],
            buy_hold_regularized_sharpe=buy_hold["regularized_sharpe"],
        )
        test_metrics = {}
    else:
        if not label_mask.any():
            raise ValueError("Evaluation requires observed label targets.")
        test_metrics = evaluate_predictions(
            y_true_global[label_mask], predictions[label_mask]
        )
    label_semantics = "action"
    if label_mode == "volatility_position" or (
        label_mode == "triple_barrier"
        and triple_barrier_between_event_policy != "hold"
    ):
        label_semantics = "target_position"
    effective_position_mode = (
        "long_only"
        if label_mode == "volatility_position"
        and volatility_position_mode == "long_flat"
        else position_mode
    )
    if financial_loss is None:
        benchmark = evaluate_strategy_vs_buy_hold(
            aligned_test,
            y_pred,
            initial_capital=initial_capital,
            price_col=price_col,
            fee_per_trade=strategy_fee_per_trade,
            position_mode=effective_position_mode,
            execution_delay=execution_delay,
            label_semantics=label_semantics,
        )
    shared = {
        "evaluation_split": evaluation_split,
        "n_total_rows": len(data),
        "n_eval_rows": int(evaluation_mask.sum()),
        "n_scored_labels": int(label_mask.sum()),
        "label_eval_report": evaluation_label_report,
        "retrain_logs": retrain_logs,
        "predictions": predictions,
        "evaluation_mask": evaluation_mask,
        "label_mask": label_mask,
        "loss_objective": financial_loss.objective if financial_loss else "cross_entropy",
        "financial_loss": asdict(financial_loss) if financial_loss else None,
    }
    if evaluation_split == "validation":
        return {
            **shared,
            "val_metrics": test_metrics,
            "val_backtest": benchmark,
            "val_start_idx": test_start,
            "n_val_rows": int(test_mask.sum()),
            "n_missing_val_preds": int(test_mask.sum() - evaluation_mask.sum()),
            "aligned_val_frame": aligned_test,
        }
    return {
        **shared,
        "test_metrics": test_metrics,
        "benchmark_comparison": benchmark,
        "test_start_idx": test_start,
        "n_test_rows": int(test_mask.sum()),
        "n_missing_test_preds": int(test_mask.sum() - evaluation_mask.sum()),
        "aligned_test_frame": aligned_test,
    }


def walk_forward_oracle_ann(
    full_df: pd.DataFrame,
    feature_cols: Sequence[str],
    price_col: str = "adj_close",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    walkforward_step: int = 20,
    oracle_fee_per_trade: float = 2.0,
    label_mode: str = "forward_return",
    forward_horizon: int = 1,
    forward_buy_threshold: float = 0.002,
    forward_sell_threshold: float = 0.002,
    breakout_window: int = 20,
    breakout_buy_buffer: float = 0.0,
    breakout_sell_buffer: float = 0.0,
    breakout_alternating: bool = True,
    volatility_horizon: int = 10,
    volatility_window: int = 20,
    volatility_long_threshold: float = 1.0,
    volatility_short_threshold: float = 1.5,
    volatility_exit_threshold: float = 0.25,
    volatility_min_holding_period: int = 5,
    volatility_cooldown: int = 0,
    volatility_cost_bps: float = 5.0,
    volatility_position_mode: str = "long_flat",
    triple_barrier_max_holding: int = 10,
    triple_barrier_volatility_window: int = 20,
    triple_barrier_volatility_estimator: str = "rolling_std",
    fracdiff_config: FracDiffConfig | None = None,
    sample_weighting: SampleWeightConfig | None = None,
    expanded_min_coverage: float | None = None,
    triple_barrier_profit_barrier: float = 1.0,
    triple_barrier_stop_barrier: float = 1.0,
    triple_barrier_event_filter: str = "all",
    triple_barrier_cusum_threshold: float = 0.5,
    triple_barrier_cost_bps: float = 5.0,
    triple_barrier_between_event_policy: str = "hold",
    decision_mode: str = "argmax",
    min_action_rate: float = 0.0,
    position_mode: str = "long_only",
    strategy_fee_per_trade: float = 0.0,
    initial_capital: float = 10_000.0,
    context_len: int = 20,
    epochs: int = 150,
    alpha: float = 1e-3,
    hidden: int = 64,
    batch_size: int = 64,
    do_dropout: bool = False,
    dropout_percent: float = 0.1,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
    seed: int = 1,
    evaluation_split: str = "test",
    execution_delay: int = 1,
    financial_loss: FinancialLossConfig | None = None,
    overfitting_control: OverfittingControlConfig | None = None,
):
    """Compatibility adapter for previous walk-forward ANN API."""

    ann_config = ManualANNConfig(
        hidden_size=hidden,
        learning_rate=alpha,
        epochs=epochs,
        batch_size=batch_size,
        dropout_probability=dropout_percent if do_dropout else 0.0,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        seed=seed,
    )
    return walk_forward_classifier(
        full_df,
        feature_cols,
        price_col=price_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        walkforward_step=walkforward_step,
        oracle_fee_per_trade=oracle_fee_per_trade,
        label_mode=label_mode,
        forward_horizon=forward_horizon,
        forward_buy_threshold=forward_buy_threshold,
        forward_sell_threshold=forward_sell_threshold,
        breakout_window=breakout_window,
        breakout_buy_buffer=breakout_buy_buffer,
        breakout_sell_buffer=breakout_sell_buffer,
        breakout_alternating=breakout_alternating,
        volatility_horizon=volatility_horizon,
        volatility_window=volatility_window,
        volatility_long_threshold=volatility_long_threshold,
        volatility_short_threshold=volatility_short_threshold,
        volatility_exit_threshold=volatility_exit_threshold,
        volatility_min_holding_period=volatility_min_holding_period,
        volatility_cooldown=volatility_cooldown,
        volatility_cost_bps=volatility_cost_bps,
        volatility_position_mode=volatility_position_mode,
        triple_barrier_max_holding=triple_barrier_max_holding,
        triple_barrier_volatility_window=triple_barrier_volatility_window,
        triple_barrier_volatility_estimator=triple_barrier_volatility_estimator,
        fracdiff_config=fracdiff_config,
        sample_weighting=sample_weighting,
        expanded_min_coverage=expanded_min_coverage,
        triple_barrier_profit_barrier=triple_barrier_profit_barrier,
        triple_barrier_stop_barrier=triple_barrier_stop_barrier,
        triple_barrier_event_filter=triple_barrier_event_filter,
        triple_barrier_cusum_threshold=triple_barrier_cusum_threshold,
        triple_barrier_cost_bps=triple_barrier_cost_bps,
        triple_barrier_between_event_policy=triple_barrier_between_event_policy,
        decision_mode=decision_mode,
        min_action_rate=min_action_rate,
        position_mode=position_mode,
        strategy_fee_per_trade=strategy_fee_per_trade,
        initial_capital=initial_capital,
        context_len=context_len,
        manual_ann_config=ann_config,
        evaluation_split=evaluation_split,
        execution_delay=execution_delay,
        financial_loss=financial_loss,
        overfitting_control=overfitting_control,
    )


__all__ = [
    "derive_chunk_seed",
    "fit_labeled_history",
    "fit_position_history",
    "predict_chunk_with_model",
    "predict_position_chunk",
    "walk_forward_classifier",
    "walk_forward_oracle_ann",
]
