from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import ExperimentResult, run_experiment
from trading_system.models.base import ProbabilisticClassifier
from trading_system.models.manual_ann.manual_nn import ManualANNConfig
from trading_system.models.specs import ModelSelection
from trading_system.paths import default_market_dataset_path
from trading_system.reporting.plots import format_experiment_summary
from trading_system.reporting.warnings import current_universe_warning


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

    if config.model.name != "manual_ann":
        raise ValueError("Legacy ANN keyword adapter requires model='manual_ann'.")
    ann = ManualANNConfig(**{**config.model.parameters, "seed": config.seed})
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
        model=ModelSelection(
            "manual_ann",
            {
                "hidden_size": ann.hidden_size,
                "learning_rate": ann.learning_rate,
                "epochs": ann.epochs,
                "batch_size": ann.batch_size,
                "dropout_probability": ann.dropout_probability,
                "early_stopping_patience": ann.early_stopping_patience,
                "early_stopping_min_delta": ann.early_stopping_min_delta,
            },
        ),
        train_ratio=train_ratio if train_ratio is not None else config.train_ratio,
        val_ratio=val_ratio if val_ratio is not None else config.val_ratio,
        context_len=context_len if context_len is not None else config.context_len,
    )
    return run_experiment(frame, configured, model=model)


def run_configured_pipeline(
    config: ExperimentConfig,
    data_path: str | Path | None = None,
    *,
    artifact_path: str | Path | None = None,
) -> ExperimentResult:
    resolved_path = data_path or default_market_dataset_path()
    warning = current_universe_warning(resolved_path)
    if warning:
        print(warning)
    frame = read_parquet_dataset(resolved_path)
    result = run_experiment(frame, config)
    if artifact_path is not None:
        from trading_system.artifacts.experiment import save_experiment_artifact

        save_experiment_artifact(
            artifact_path,
            frame,
            result,
            dataset_path=resolved_path,
        )
    print(f"labels={result.label_stats} split={result.split_sizes}")
    print(format_experiment_summary(result.test_metrics, result.backtest))
    return result


__all__ = ["run_configured_pipeline", "train_model"]
