"""Replay frozen position checkpoints on their original outer CV folds."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import hash_dataframe
from trading_system.data.io import read_parquet_dataset
from trading_system.data.optional_sources import prepare_feature_sources
from trading_system.experiments.position_objectives import _align_position_calendar, load_position_artifact
from trading_system.experiments.runner import _build_split_windows, _prepare_splits
from trading_system.training.financial_loss import ReturnPanel


def replay_fold(frame: pd.DataFrame, row: dict, *, tolerance: float = 1e-5) -> tuple[pd.DataFrame, dict]:
    """Recompute outer-fold predictions without fitting or opening final holdout."""
    artifact = Path(row["artifact_path"])
    validation = load_position_artifact(artifact)
    config, bundle = validation.config, validation.bundle
    cutoff = pd.Timestamp(row["outer_end"])
    fold_frame = frame.loc[pd.to_datetime(frame[config.date_col], utc=True) <= cutoff].copy()
    manifest = json.loads((artifact / "manifest.json").read_text())
    expected_hash = manifest["experiment_parameters"]["dataset_sha256"]
    if hash_dataframe(fold_frame) != expected_hash:
        raise ValueError(f"Fold {row['fold']} seed {row['seed']}: dataset hash differs from checkpoint.")
    aligned_frame = _align_position_calendar(fold_frame, config)
    train, val, outer, _, columns = _prepare_splits(
        aligned_frame, config, include_test=True,
        fill_values=bundle.feature_fill_values,
        fracdiff_transformer=bundle.fracdiff_transformer,
        feature_selector=bundle.feature_selector,
        overfitting_selector=bundle.overfitting_selector,
        overfitting_supervised=False,
    )
    if columns != bundle.feature_columns:
        raise ValueError("Replay features differ from checkpoint features.")
    windows, _, aligned = _build_split_windows(
        outer, columns, config, pd.concat([train, val], ignore_index=True)
    )
    positions = validation.predict_positions(windows)
    panel = ReturnPanel(aligned, price_col=config.price_col, date_col=config.date_col,
                        group_col=config.group_col if config.universe == "multi" else None,
                        execution_delay=config.execution_delay)
    metrics = panel.metrics(positions, validation.loss_config, config.initial_capital)
    for metric in ("regularized_sharpe", "net_return", "max_drawdown", "turnover"):
        if not np.isclose(metrics[metric], row["outer_metrics"][metric], atol=tolerance, rtol=0):
            raise ValueError(f"Fold {row['fold']} seed {row['seed']}: {metric} did not reproduce "
                             f"({metrics[metric]} versus {row['outer_metrics'][metric]}).")
    required = (config.date_col, config.group_col, "open", "high", "low", "close", config.price_col)
    missing = set(required) - set(aligned)
    if missing:
        raise ValueError(f"Replay lacks price columns: {sorted(missing)}")
    output = aligned.loc[:, list(dict.fromkeys(required))].copy()
    output["position"] = positions
    output["seed"] = int(row["seed"])
    output["fold"] = int(row["fold"])
    output["source_candidate"] = row["candidate"]
    output["signal_timing"] = "after_close_J"
    output["original_execution"] = "close_J_plus_1"
    return output, metrics


def replay_reference(cv_dir: str | Path, data: str | Path, tickers: list[str],
                     output: str | Path, *, folds: tuple[int, ...] | None = None) -> dict:
    """Replay the Sharpe N0 control of GRU normalization benchmark 04."""
    cv_dir, output = Path(cv_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Replay output exists: {output}")
    rows = json.loads((cv_dir / "folds.json").read_text())
    matching = [row for row in rows if row["objective"] == "sharpe" and
                row["status"] == "ok" and
                row["parameters"].get("temporal_pooling") == "attention" and
                row["parameters"].get("input_normalization", "none") == "none"]
    if len(matching) != 9 or len({(row["fold"], row["seed"]) for row in matching}) != 9:
        raise ValueError("Expected exactly nine successful Sharpe N0 checkpoints.")
    if folds is not None:
        matching = [row for row in matching if row["fold"] in folds]
        if not matching:
            raise ValueError("No selected folds found.")
    frame = read_parquet_dataset(data)
    frame = frame.loc[frame.ticker.isin(tickers)].copy()
    frame, _ = prepare_feature_sources(frame, disabled=True)
    frames, checks = [], []
    for row in matching:
        replayed, metrics = replay_fold(frame, row)
        frames.append(replayed)
        checks.append({"seed": row["seed"], "fold": row["fold"],
                       "regularized_sharpe": metrics["regularized_sharpe"],
                       "rows": len(replayed)})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(output, index=False)
    return {"path": str(output), "runs": len(checks), "rows": sum(x["rows"] for x in checks),
            "checks": checks, "final_holdout_opened": False}
