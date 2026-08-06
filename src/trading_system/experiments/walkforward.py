from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from trading_system.backtest.engine import evaluate_strategy_vs_buy_hold
from trading_system.data.scaling import Standardizer
from trading_system.data.splits import (
    chronological_train_val_split,
    chronological_train_val_test_split,
)
from trading_system.data.windows import (
    build_context_dataset_with_history,
    build_context_features,
)
from trading_system.evaluation.classification import evaluate_predictions
from trading_system.evaluation.thresholds import DecisionPolicy
from trading_system.labels.breakout import generate_breakout_labels
from trading_system.labels.forward_return import build_forward_return_labels
from trading_system.labels.oracle_dp import build_oracle_labels_train_only
from trading_system.models.base import ProbabilisticClassifier
from trading_system.models.manual_ann.manual_nn import (
    ManualANNClassifier,
    ManualANNConfig,
)

from .runner import TrainedModelBundle, align_probability_columns

# TODO(sequence-walkforward-factory): Replace this chunk-id-only callable with the
# shared registry/factory receiving `ModelBuildContext`, typed parameters and a
# seed derived from `(run_seed, chunk_id)`.
ModelFactory = Callable[[int], ProbabilisticClassifier]


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
) -> tuple[pd.DataFrame, dict]:
    if label_mode == "oracle_dp":
        return build_oracle_labels_train_only(
            frame,
            price_col=price_col,
            initial_capital=initial_capital,
            fee_per_trade=oracle_fee_per_trade,
        )
    if label_mode == "forward_return":
        return build_forward_return_labels(
            frame,
            price_col=price_col,
            horizon=forward_horizon,
            buy_threshold=forward_buy_threshold,
            sell_threshold=forward_sell_threshold,
        )
    if label_mode == "breakout":
        labeled = generate_breakout_labels(frame, breakout_window, price_col=price_col)
        counts = labeled["Label"].value_counts()
        return labeled, {
            "n_buy": int(counts.get("Buy", 0)),
            "n_hold": int(counts.get("Hold", 0)),
            "n_sell": int(counts.get("Sell", 0)),
        }
    raise ValueError(f"Unknown label_mode: {label_mode}")


def fit_labeled_history(
    labeled_history: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    val_ratio: float,
    context_len: int,
    decision_mode: str,
    min_action_rate: float,
    estimator: ProbabilisticClassifier,
    date_col: str = "date",
) -> tuple[TrainedModelBundle, pd.DataFrame, dict[str, float]]:
    # TODO(sequence-walkforward-fit-1): Build 3D train/validation sequences and
    # fit one `SequenceStandardizer` on train only.
    #
    # TODO(sequence-walkforward-fit-2): Train any registry-created sequence model,
    # calibrate the policy on validation only, and persist the same bundle schema
    # as the static runner.
    columns = tuple(feature_columns)
    work = labeled_history.sort_values(date_col).reset_index(drop=True).copy()
    work["Label_id"] = pd.to_numeric(work["Label_id"], errors="coerce")
    work.loc[:, columns] = work[list(columns)].apply(pd.to_numeric, errors="coerce")
    work.loc[:, columns] = work[list(columns)].replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["Label_id"]).copy()
    train, val = chronological_train_val_split(
        work, val_ratio=val_ratio, date_col=date_col
    )
    train = train.dropna(subset=list(columns)).copy()
    if train.empty:
        raise ValueError("Walk-forward training history has no complete feature rows.")
    fill_values = train[list(columns)].median(numeric_only=True).fillna(0.0)
    val.loc[:, columns] = val[list(columns)].fillna(fill_values).fillna(0.0)
    train.loc[:, columns] = train[list(columns)].fillna(fill_values).fillna(0.0)
    X_train_raw, y_train = build_context_dataset_with_history(
        train,
        columns,
        context_len,
        group_col=None,
        date_col=date_col,
    )
    X_val_raw, y_val = build_context_dataset_with_history(
        val,
        columns,
        context_len,
        history_frame=train,
        group_col=None,
        date_col=date_col,
    )
    if not len(X_train_raw) or not len(X_val_raw):
        raise ValueError(
            "Walk-forward context windowing produced an empty train or validation set."
        )
    scaler = Standardizer()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    fit_result = estimator.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    val_probabilities = align_probability_columns(
        estimator, estimator.predict_proba(X_val)
    )
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
    )
    return bundle, filled_history, metrics


def predict_chunk_with_model(
    bundle: TrainedModelBundle,
    history: pd.DataFrame,
    chunk: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    # TODO(sequence-walkforward-predict-1): Replace flat inference windows with
    # `build_sequence_features`, using at most `context_len - 1` earlier rows.
    #
    # TODO(sequence-walkforward-predict-2): Keep source/local index conversion and
    # verify that every prediction maps to a row in the current chunk only.
    columns = list(bundle.feature_columns)
    work = chunk.copy().reset_index(drop=True)
    work["_chunk_local_index"] = np.arange(len(work), dtype=np.int64)
    work.loc[:, columns] = work[columns].apply(pd.to_numeric, errors="coerce")
    work.loc[:, columns] = work[columns].fillna(bundle.feature_fill_values).fillna(0.0)
    prefix = history.tail(bundle.context_len - 1)[columns].copy()
    source = pd.concat([prefix, work[columns]], ignore_index=True)
    windows, source_indices = build_context_features(
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
    decision_mode: str = "argmax",
    min_action_rate: float = 0.0,
    position_mode: str = "long_only",
    strategy_fee_per_trade: float = 0.0,
    initial_capital: float = 10_000.0,
    execution_delay: int = 1,
    context_len: int = 20,
    model_factory: ModelFactory | None = None,
    manual_ann_config: ManualANNConfig | None = None,
) -> dict[str, object]:
    # TODO(sequence-walkforward-run-1): Accept model selection/registry instead of
    # ANN-specific fallback config. Construct a fresh model and scaler per chunk.
    #
    # TODO(sequence-walkforward-run-2): Derive deterministic chunk seeds, forbid
    # implicit warm-start, and record model name/parameters/duration per retrain.
    #
    # TODO(sequence-walkforward-run-3): Preserve the rule that history ends before
    # each chunk and prediction at `t` executes at `t+1` in the common backtest.
    if walkforward_step <= 0:
        raise ValueError("walkforward_step must be positive.")
    data = full_df.sort_values("date").reset_index(drop=True).copy()
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
    _, _, initial_test = chronological_train_val_test_split(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    test_start = len(data) - len(initial_test)
    evaluation_labels, evaluation_label_report = _label_history(
        data,
        label_mode=label_mode,
        price_col=price_col,
        initial_capital=initial_capital,
        oracle_fee_per_trade=oracle_fee_per_trade,
        forward_horizon=forward_horizon,
        forward_buy_threshold=forward_buy_threshold,
        forward_sell_threshold=forward_sell_threshold,
        breakout_window=breakout_window,
    )
    y_true_global = evaluation_labels["Label_id"].to_numpy(dtype=np.int64)
    predictions = np.full(len(data), -1, dtype=np.int64)
    retrain_logs: list[dict[str, object]] = []
    base_ann_config = manual_ann_config or ManualANNConfig(
        hidden_size=64,
        epochs=150,
        batch_size=64,
        early_stopping_patience=30,
    )

    for chunk_id, start in enumerate(
        range(test_start, len(data), walkforward_step), start=1
    ):
        end = min(start + walkforward_step, len(data))
        history = data.iloc[:start].copy()
        labeled_history, label_report = _label_history(
            history,
            label_mode=label_mode,
            price_col=price_col,
            initial_capital=initial_capital,
            oracle_fee_per_trade=oracle_fee_per_trade,
            forward_horizon=forward_horizon,
            forward_buy_threshold=forward_buy_threshold,
            forward_sell_threshold=forward_sell_threshold,
            breakout_window=breakout_window,
        )
        estimator = (
            model_factory(chunk_id)
            if model_factory is not None
            else ManualANNClassifier(
                ManualANNConfig(
                    **{
                        **base_ann_config.__dict__,
                        "seed": base_ann_config.seed + chunk_id - 1,
                    }
                )
            )
        )
        bundle, filled_history, val_metrics = fit_labeled_history(
            labeled_history,
            feature_columns,
            val_ratio=val_ratio,
            context_len=context_len,
            decision_mode=decision_mode,
            min_action_rate=min_action_rate,
            estimator=estimator,
        )
        chunk = data.iloc[start:end].copy().reset_index(drop=True)
        chunk_predictions, local_indices = predict_chunk_with_model(
            bundle,
            filled_history,
            chunk,
        )
        absolute_indices = np.arange(start, end, dtype=np.int64)[local_indices]
        predictions[absolute_indices] = chunk_predictions
        retrain_logs.append(
            {
                "chunk_id": chunk_id,
                "start_idx": start,
                "end_idx": end,
                "n_hist": len(history),
                "n_pred": len(chunk_predictions),
                "best_epoch": int(bundle.fit_result.best_epoch),
                "val_macro_f1": val_metrics["macro_f1"],
                "val_bal_acc": val_metrics["bal_acc"],
                "label_hist_info": label_report,
            }
        )

    test_mask = np.arange(len(data)) >= test_start
    evaluation_mask = test_mask & (predictions >= 0)
    if not evaluation_mask.any():
        raise RuntimeError("Walk-forward evaluation produced no predictions.")
    y_true = y_true_global[evaluation_mask]
    y_pred = predictions[evaluation_mask]
    test_metrics = evaluate_predictions(y_true, y_pred)
    aligned_test = data.loc[evaluation_mask].reset_index(drop=True)
    benchmark = evaluate_strategy_vs_buy_hold(
        aligned_test,
        y_pred,
        initial_capital=initial_capital,
        price_col=price_col,
        fee_per_trade=strategy_fee_per_trade,
        position_mode=position_mode,
        execution_delay=execution_delay,
    )
    return {
        "test_metrics": test_metrics,
        "benchmark_comparison": benchmark,
        "n_total_rows": len(data),
        "test_start_idx": test_start,
        "n_test_rows": int(test_mask.sum()),
        "n_eval_rows": int(evaluation_mask.sum()),
        "n_missing_test_preds": int(test_mask.sum() - evaluation_mask.sum()),
        "label_eval_report": evaluation_label_report,
        "retrain_logs": retrain_logs,
        "predictions": predictions,
        "evaluation_mask": evaluation_mask,
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
        decision_mode=decision_mode,
        min_action_rate=min_action_rate,
        position_mode=position_mode,
        strategy_fee_per_trade=strategy_fee_per_trade,
        initial_capital=initial_capital,
        context_len=context_len,
        manual_ann_config=ann_config,
    )


__all__ = [
    "fit_labeled_history",
    "predict_chunk_with_model",
    "walk_forward_classifier",
    "walk_forward_oracle_ann",
]
