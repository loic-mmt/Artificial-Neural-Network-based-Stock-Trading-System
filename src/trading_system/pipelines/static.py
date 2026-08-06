from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import ExperimentResult, run_experiment
from trading_system.models.base import ProbabilisticClassifier
from trading_system.paths import default_market_dataset_path
from trading_system.reporting.plots import format_experiment_summary


def train_model(
    frame: pd.DataFrame,
    *,
    config: ExperimentConfig,
    model: ProbabilisticClassifier | None = None,
    epochs: int | None = None,
    alpha: float | None = None,
    hidden: int | None = None,
    do_dropout: bool | None = None,
    dropout_percent: float | None = None,
    batch_size: int | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    context_len: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float | None = None,
) -> ExperimentResult:
    """Compatibility adapter from old pipeline kwargs to shared experiment config."""

    # TODO(sequence-static-compatibility): Keep this ANN-specific adapter only for
    # old callers. New neural CLIs must use `ModelSelection` and the shared registry
    # instead of adding RNN/LSTM/GRU/Transformer kwargs to this signature.

    ann = config.manual_ann
    if do_dropout is None:
        dropout_probability = ann.dropout_probability
    elif do_dropout:
        dropout_probability = 0.1 if dropout_percent is None else dropout_percent
    else:
        dropout_probability = 0.0
    ann = replace(
        ann,
        epochs=epochs if epochs is not None else ann.epochs,
        learning_rate=alpha if alpha is not None else ann.learning_rate,
        hidden_size=hidden if hidden is not None else ann.hidden_size,
        dropout_probability=dropout_probability,
        batch_size=batch_size if batch_size is not None else ann.batch_size,
        early_stopping_patience=(
            early_stopping_patience
            if early_stopping_patience is not None
            else ann.early_stopping_patience
        ),
        early_stopping_min_delta=(
            early_stopping_min_delta
            if early_stopping_min_delta is not None
            else ann.early_stopping_min_delta
        ),
    )
    configured = replace(
        config,
        manual_ann=ann,
        train_ratio=train_ratio if train_ratio is not None else config.train_ratio,
        val_ratio=val_ratio if val_ratio is not None else config.val_ratio,
        context_len=context_len if context_len is not None else config.context_len,
    )
    return run_experiment(frame, configured, model=model)


def run_configured_pipeline(
    config: ExperimentConfig,
    data_path: str | Path | None = None,
) -> ExperimentResult:
    # TODO(sequence-static-cli): Accept model selection/config and pass it to the
    # sequence-aware runner once migration is complete. Preserve this function as
    # a thin I/O/reporting wrapper.
    frame = read_parquet_dataset(data_path or default_market_dataset_path())
    result = run_experiment(frame, config)
    print(f"labels={result.label_stats} split={result.split_sizes}")
    print(format_experiment_summary(result.test_metrics, result.backtest))
    return result


__all__ = ["run_configured_pipeline", "train_model"]
