from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_system.data.scaling import SequenceStandardizer, Standardizer
from trading_system.data.splits import chronological_train_val_test_split
from trading_system.experiments.runner import ExperimentResult
from trading_system.reporting.warnings import current_universe_warning

from .serialization import ArtifactManifest, save_model_artifact, stable_config_hash


def _nullable_metadata(value: Any) -> Any:
    """Map unavailable numerical diagnostics to strict-JSON null values."""

    if isinstance(value, np.generic):
        return _nullable_metadata(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.DataFrame):
        return _nullable_metadata(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _nullable_metadata(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _nullable_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nullable_metadata(item) for item in value]
    return value


def hash_dataframe(frame: pd.DataFrame) -> str:
    """Hash ordered values, index, columns and dtypes for rerun identification."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Dataset frame must be a non-empty DataFrame.")
    digest = hashlib.sha256()
    schema = "\n".join(
        f"{column}\t{frame[column].dtype}" for column in frame.columns
    )
    digest.update(schema.encode("utf-8"))
    hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    digest.update(hashed.to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def _date_range(frame: pd.DataFrame, date_col: str) -> dict[str, str]:
    dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError(f"Dataset contains invalid {date_col} values.")
    return {"start": dates.min().isoformat(), "end": dates.max().isoformat()}


def _split_boundaries(frame: pd.DataFrame, result: ExperimentResult) -> dict[str, Any]:
    config = result.config
    work = frame.copy()
    if config.ticker is not None and config.group_col in work:
        work = work[work[config.group_col] == config.ticker].copy()
    group_col = config.group_col if config.universe == "multi" else None
    splits = chronological_train_val_test_split(
        work,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        group_col=group_col,
        date_col=config.date_col,
    )
    output: dict[str, Any] = {}
    for name, split in zip(("train", "val", "test"), splits):
        if group_col is None:
            output[name] = {"rows": len(split), **_date_range(split, config.date_col)}
        else:
            output[name] = {
                str(group): {"rows": len(part), **_date_range(part, config.date_col)}
                for group, part in split.groupby(group_col, sort=True, dropna=False)
            }
    return output


def _runtime_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "torch", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"code_commit": commit, "working_tree_dirty": dirty, "packages": packages}


def build_experiment_manifest(
    frame: pd.DataFrame,
    result: ExperimentResult,
    *,
    dataset_path: str | Path | None = None,
) -> ArtifactManifest:
    if not isinstance(result, ExperimentResult):
        raise TypeError("result must be ExperimentResult.")
    config = result.config
    selection = result.bundle.model_selection
    experiment_parameters = {
        "config": asdict(config),
        "dataset": {
            "path": str(Path(dataset_path).expanduser().resolve())
            if dataset_path is not None
            else None,
            "sha256": hash_dataframe(frame),
            "rows": len(frame),
            "date_range": _date_range(frame, config.date_col),
            "tickers": sorted(
                frame[config.group_col].dropna().astype(str).unique().tolist()
            )
            if config.group_col in frame
            else [],
            "survivor_bias_warning": current_universe_warning(dataset_path),
        },
        "split_boundaries": _split_boundaries(frame, result),
        "split_sizes_after_features": dict(result.split_sizes),
    }
    decision_parameters = asdict(result.bundle.decision_policy)
    canonical = {
        "model_name": selection.name,
        "model_parameters": selection.parameters,
        "experiment_parameters": experiment_parameters,
        "decision_parameters": decision_parameters,
    }
    return ArtifactManifest(
        format_version=1,
        model_name=selection.name,
        model_parameters=dict(selection.parameters),
        experiment_parameters=experiment_parameters,
        config_hash=stable_config_hash(canonical),
        feature_columns=result.bundle.feature_columns,
        context_len=result.bundle.context_len,
        decision_parameters=decision_parameters,
        runtime_metadata=_runtime_metadata(),
    )


def _scaler_state(scaler: Standardizer | SequenceStandardizer) -> dict[str, Any]:
    if isinstance(scaler, SequenceStandardizer):
        return scaler.state_dict()
    if scaler.mean_ is None or scaler.scale_ is None:
        raise RuntimeError("Cannot persist an unfitted scaler.")
    return {"mean": scaler.mean_.copy(), "scale": scaler.scale_.copy()}


def save_experiment_artifact(
    destination: str | Path,
    frame: pd.DataFrame,
    result: ExperimentResult,
    *,
    dataset_path: str | Path | None = None,
    advanced_diagnostics: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist model, preprocessing, run metadata and result diagnostics together."""

    manifest = build_experiment_manifest(frame, result, dataset_path=dataset_path)
    fit = result.bundle.fit_result
    metrics = {
        "classification": {
            "train": result.train_metrics,
            "validation": result.val_metrics,
            "test": result.test_metrics,
        },
        "backtest": {
            "validation": result.val_backtest,
            "test": result.backtest,
        },
        "advanced": _nullable_metadata(dict(advanced_diagnostics or {})),
    }
    return save_model_artifact(
        destination,
        manifest=manifest,
        model_state=result.bundle.estimator.state_dict(),
        scaler_state=_scaler_state(result.bundle.scaler),
        training_history=_nullable_metadata({
            "best_epoch": fit.best_epoch,
            "stop_reason": fit.stop_reason,
            "train_loss": fit.history.train_loss,
            "val_loss": fit.history.val_loss,
            "training_duration_seconds": fit.training_duration_seconds,
            "parameter_count": fit.parameter_count,
            "seed": fit.seed,
            "device": fit.device,
        }),
        metrics=_nullable_metadata(metrics),
        overwrite=overwrite,
    )


__all__ = [
    "build_experiment_manifest",
    "hash_dataframe",
    "save_experiment_artifact",
]
