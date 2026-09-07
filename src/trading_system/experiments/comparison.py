from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from trading_system.artifacts.serialization import stable_config_hash, to_jsonable
from trading_system.models.factory import ModelRegistry
from trading_system.models.specs import ModelSelection, normalize_model_name
from trading_system.training.overfitting import generalization_diagnostics

from .config import ExperimentConfig
from .runner import (
    ExperimentResult,
    ValidationResult,
    evaluate_experiment_test,
    run_experiment,
    run_validation_experiment,
)


@dataclass(frozen=True)
class ComparisonRun:
    """One predeclared model/configuration/seed combination."""

    model_name: str
    model_parameters: dict[str, Any] = field(default_factory=dict)
    seed: int = 1

    def __post_init__(self) -> None:
        selection = ModelSelection(self.model_name, self.model_parameters)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        object.__setattr__(self, "model_name", selection.name)
        object.__setattr__(self, "model_parameters", selection.parameters)


@dataclass
class ComparisonResult:
    """Raw per-run results and grouped scientific summary."""

    runs: pd.DataFrame
    summary: pd.DataFrame
    failures: pd.DataFrame


def build_comparison_runs(
    model_parameter_sets: Mapping[str, Sequence[Mapping[str, Any]]],
    seeds: Sequence[int],
) -> list[ComparisonRun]:
    if not isinstance(model_parameter_sets, Mapping) or not model_parameter_sets:
        raise ValueError("At least one model parameter set is required.")
    seed_list = list(seeds)
    if not seed_list or len(seed_list) != len(set(seed_list)):
        raise ValueError("seeds must be non-empty and unique.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seed_list):
        raise ValueError("seeds must contain non-negative integers.")
    normalized: dict[str, list[Mapping[str, Any]]] = {}
    for raw_name, parameter_sets in model_parameter_sets.items():
        name = normalize_model_name(raw_name)
        if name in normalized:
            raise ValueError(f"Duplicate normalized model name: {name}")
        normalized[name] = list(parameter_sets)
        if not normalized[name]:
            raise ValueError(f"Model {name} has no parameter sets.")
    budgets = {len(parameter_sets) for parameter_sets in normalized.values()}
    if len(budgets) != 1:
        raise ValueError("Every model must receive the same parameter-set budget.")
    return [
        ComparisonRun(name, dict(parameters), seed)
        for name in sorted(normalized)
        for parameters in normalized[name]
        for seed in seed_list
    ]


def validate_fair_comparison(
    experiment_config: ExperimentConfig,
    runs: Sequence[ComparisonRun],
) -> None:
    if not isinstance(experiment_config, ExperimentConfig):
        raise TypeError("experiment_config must be ExperimentConfig.")
    declared = list(runs)
    model_names = sorted({run.model_name for run in declared})
    if len(model_names) < 2:
        raise ValueError("Comparison requires at least two model names.")
    seed_coverage = {
        name: sorted(run.seed for run in declared if run.model_name == name)
        for name in model_names
    }
    first = seed_coverage[model_names[0]]
    if any(coverage != first for coverage in seed_coverage.values()):
        raise ValueError("Every model must have identical seed coverage and budget.")
    if experiment_config.label_mode.startswith("oracle"):
        raise ValueError("Oracle labels cannot be used for model comparison.")
    for run in declared:
        unsafe = [
            key
            for key in run.model_parameters
            if "test" in key.lower() or key.lower() in {"objective", "selection_split"}
        ]
        if unsafe:
            raise ValueError(f"Test-derived/selection parameters are forbidden: {unsafe}")


def flatten_experiment_result(
    result: Any,
    run: ComparisonRun,
    *,
    config_hash: str,
    duration_seconds: float,
) -> dict[str, Any]:
    if not isinstance(result, ExperimentResult):
        raise TypeError("result must be ExperimentResult.")
    row: dict[str, Any] = {
        "status": "ok",
        "model_name": run.model_name,
        "model_parameters": json.dumps(
            to_jsonable(run.model_parameters), sort_keys=True, separators=(",", ":")
        ),
        "seed": run.seed,
        "config_hash": config_hash,
        "parameter_count": result.bundle.parameter_count(),
        "best_epoch": result.bundle.fit_result.best_epoch,
        "stop_reason": result.bundle.fit_result.stop_reason,
        "duration_seconds": float(duration_seconds),
    }
    if result.bundle.fracdiff_transformer is not None:
        row["fracdiff_state"] = json.dumps(
            to_jsonable(result.bundle.fracdiff_transformer.state_dict()),
            sort_keys=True, separators=(",", ":"),
        )
    if result.bundle.sample_weight_state is not None:
        row["sample_weight_state"] = json.dumps(
            to_jsonable(result.bundle.sample_weight_state), sort_keys=True
        )
    if result.bundle.feature_selector is not None:
        row["feature_selection"] = json.dumps(to_jsonable(result.bundle.feature_selector.state_dict()), sort_keys=True)
    if result.bundle.overfitting_selector is not None:
        row["overfitting_feature_selection"] = json.dumps(
            to_jsonable(result.bundle.overfitting_selector.state_dict()), sort_keys=True
        )
    row.update(
        {
            f"generalization_{name}": value
            for name, value in generalization_diagnostics(
                result.bundle.fit_result
            ).items()
        }
    )
    for prefix, values in (
        ("train", result.train_metrics),
        ("val", result.val_metrics),
        ("test", result.test_metrics),
        ("val_backtest", result.val_backtest),
        ("backtest", result.backtest),
        ("split", result.split_sizes),
    ):
        row.update({f"{prefix}_{name}": value for name, value in values.items()})
    return row


def flatten_validation_result(
    result: ValidationResult,
    run: ComparisonRun,
    *,
    config_hash: str,
    duration_seconds: float,
) -> dict[str, Any]:
    """Flatten sealed validation evidence without touching final-test rows."""

    row: dict[str, Any] = {
        "status": "ok",
        "model_name": run.model_name,
        "model_parameters": json.dumps(
            to_jsonable(run.model_parameters), sort_keys=True, separators=(",", ":")
        ),
        "seed": run.seed,
        "config_hash": config_hash,
        "parameter_count": result.bundle.parameter_count(),
        "best_epoch": result.bundle.fit_result.best_epoch,
        "stop_reason": result.bundle.fit_result.stop_reason,
        "duration_seconds": float(duration_seconds),
        "final_test_evaluated": False,
    }
    if result.bundle.fracdiff_transformer is not None:
        row["fracdiff_state"] = json.dumps(
            to_jsonable(result.bundle.fracdiff_transformer.state_dict()),
            sort_keys=True, separators=(",", ":"),
        )
    if result.bundle.sample_weight_state is not None:
        row["sample_weight_state"] = json.dumps(
            to_jsonable(result.bundle.sample_weight_state), sort_keys=True
        )
    if result.bundle.feature_selector is not None:
        row["feature_selection"] = json.dumps(
            to_jsonable(result.bundle.feature_selector.state_dict()), sort_keys=True
        )
    if result.bundle.overfitting_selector is not None:
        row["overfitting_feature_selection"] = json.dumps(
            to_jsonable(result.bundle.overfitting_selector.state_dict()), sort_keys=True
        )
    row.update({f"generalization_{name}": value for name, value in generalization_diagnostics(result.bundle.fit_result).items()})
    for prefix, values in (
        ("train", result.train_metrics),
        ("val", result.val_metrics),
        ("val_backtest", result.val_backtest),
        ("split", result.split_sizes),
    ):
        row.update({f"{prefix}_{name}": value for name, value in values.items()})
    return row


def summarize_comparison_runs(
    runs: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("model_name",),
) -> pd.DataFrame:
    if not isinstance(runs, pd.DataFrame):
        raise TypeError("runs must be a DataFrame.")
    groups = list(group_columns)
    missing = [column for column in [*groups, "status"] if column not in runs]
    if missing:
        raise ValueError(f"Missing comparison columns: {missing}")
    successful = runs[runs["status"] == "ok"].copy()
    if successful.empty:
        return pd.DataFrame(columns=groups)
    numeric = [
        column
        for column in successful.select_dtypes(include=[np.number]).columns
        if column not in groups
    ]
    if not numeric:
        raise ValueError("Comparison runs contain no numeric metrics.")
    summary = (
        successful.groupby(groups, sort=True, dropna=False)[numeric]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    totals = runs.groupby(groups, sort=True, dropna=False).size().rename("run_count")
    successes = (
        successful.groupby(groups, sort=True, dropna=False)
        .size()
        .rename("success_count")
    )
    counts = pd.concat([totals, successes], axis=1).fillna(0).reset_index()
    counts["success_count"] = counts["success_count"].astype(int)
    counts["failure_count"] = counts["run_count"] - counts["success_count"]
    return counts.merge(summary, on=groups, how="left", validate="one_to_one")


def add_stability_flags(
    summary: pd.DataFrame,
    *,
    expected_runs: int,
    max_validation_metric_std: float,
    max_train_validation_gap: float,
    require_all_seeds: bool,
) -> pd.DataFrame:
    """Mark candidates eligible from validation-only multi-seed evidence."""

    output = summary.copy()
    if "success_count" not in output:
        output["success_count"] = pd.Series(dtype=np.int64)
        output["train_validation_macro_f1_gap"] = pd.Series(dtype=np.float64)
        output["stable"] = pd.Series(dtype=bool)
        return output
    metric_std = output.get("val_macro_f1_std", pd.Series(np.nan, index=output.index))
    train_mean = output.get("train_macro_f1_mean", pd.Series(np.nan, index=output.index))
    val_mean = output.get("val_macro_f1_mean", pd.Series(np.nan, index=output.index))
    output["train_validation_macro_f1_gap"] = train_mean - val_mean
    complete = output["success_count"].eq(expected_runs) if require_all_seeds else output["success_count"].gt(0)
    # One successful seed has undefined sample std; treat it as zero dispersion.
    output["stable"] = (
        complete
        & metric_std.fillna(0.0).le(max_validation_metric_std)
        & output["train_validation_macro_f1_gap"].le(max_train_validation_gap)
    )
    return output


def _run_stability_gated_comparison(
    frame: pd.DataFrame,
    experiment_config: ExperimentConfig,
    runs: Sequence[ComparisonRun],
    registry: ModelRegistry,
    *,
    continue_on_error: bool,
    artifact_directory: str | Path | None,
    dataset_path: str | Path | None,
) -> ComparisonResult:
    """Select stable candidates on validation across seeds before opening test."""

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    validations: dict[int, tuple[ValidationResult, ComparisonRun]] = {}
    for run_id, run in enumerate(runs, start=1):
        configured = replace(
            experiment_config,
            model=ModelSelection(run.model_name, run.model_parameters),
            seed=run.seed,
            manual_ann=None,
        )
        config_hash = stable_config_hash(asdict(configured))
        started = perf_counter()
        try:
            validation = run_validation_experiment(frame, configured, registry=registry)
            row = flatten_validation_result(
                validation, run, config_hash=config_hash,
                duration_seconds=perf_counter() - started,
            )
            row["run_id"] = run_id
            validations[run_id] = (validation, run)
            rows.append(row)
        except Exception as error:
            failure = {
                "run_id": run_id, "status": "error", "model_name": run.model_name,
                "model_parameters": json.dumps(run.model_parameters, sort_keys=True, separators=(",", ":")),
                "seed": run.seed, "config_hash": config_hash,
                "duration_seconds": perf_counter() - started,
                "final_test_evaluated": False,
                "error_type": type(error).__name__, "error": str(error),
            }
            failures.append(failure)
            rows.append(failure)
            if not continue_on_error:
                raise

    run_frame = pd.DataFrame(rows).sort_values("run_id", kind="stable").reset_index(drop=True)
    summary = summarize_comparison_runs(
        run_frame, group_columns=("model_name", "model_parameters")
    )
    control = experiment_config.overfitting_control
    assert control is not None
    summary = add_stability_flags(
        summary,
        expected_runs=len({run.seed for run in runs}),
        max_validation_metric_std=control.max_validation_metric_std,
        max_train_validation_gap=control.max_train_validation_gap,
        require_all_seeds=control.require_all_seeds,
    )
    eligible = {
        (row.model_name, row.model_parameters)
        for row in summary.itertuples(index=False)
        if row.stable
    }
    run_frame["stable_candidate"] = [
        (row.get("model_name"), row.get("model_parameters")) in eligible
        for _, row in run_frame.iterrows()
    ]

    for index, row in run_frame.iterrows():
        run_id = int(row["run_id"])
        if not bool(row["stable_candidate"]) or run_id not in validations:
            continue
        validation, run = validations[run_id]
        started = perf_counter()
        try:
            result = evaluate_experiment_test(frame, validation)
            for prefix, values in (("test", result.test_metrics), ("backtest", result.backtest)):
                for name, value in values.items():
                    run_frame.loc[index, f"{prefix}_{name}"] = value
            run_frame.loc[index, "duration_seconds"] += perf_counter() - started
            run_frame.loc[index, "final_test_evaluated"] = True
            if artifact_directory is not None:
                from trading_system.artifacts.experiment import save_experiment_artifact
                artifact_path = Path(artifact_directory) / f"{run_id:04d}-{run.model_name}-seed-{run.seed}"
                save_experiment_artifact(artifact_path, frame, result, dataset_path=dataset_path)
                run_frame.loc[index, "artifact_path"] = str(artifact_path.resolve())
        except Exception as error:
            run_frame.loc[index, "status"] = "error"
            run_frame.loc[index, "error_type"] = type(error).__name__
            run_frame.loc[index, "error"] = str(error)
            failures.append(run_frame.loc[index].to_dict())
            if not continue_on_error:
                raise

    # Recompute numeric aggregates after eligible candidates receive test metrics,
    # then reapply validation-derived eligibility unchanged.
    summary = summarize_comparison_runs(
        run_frame, group_columns=("model_name", "model_parameters")
    )
    summary = add_stability_flags(
        summary,
        expected_runs=len({run.seed for run in runs}),
        max_validation_metric_std=control.max_validation_metric_std,
        max_train_validation_gap=control.max_train_validation_gap,
        require_all_seeds=control.require_all_seeds,
    )
    return ComparisonResult(run_frame, summary, pd.DataFrame(failures))


def run_model_comparison(
    frame: pd.DataFrame,
    experiment_config: ExperimentConfig,
    runs: Sequence[ComparisonRun],
    registry: ModelRegistry,
    *,
    continue_on_error: bool = True,
    artifact_directory: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> ComparisonResult:
    validate_fair_comparison(experiment_config, runs)
    if not isinstance(registry, ModelRegistry):
        raise TypeError("registry must be ModelRegistry.")
    if experiment_config.overfitting_control is not None:
        return _run_stability_gated_comparison(
            frame, experiment_config, runs, registry,
            continue_on_error=continue_on_error,
            artifact_directory=artifact_directory,
            dataset_path=dataset_path,
        )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run_id, run in enumerate(runs, start=1):
        selection = ModelSelection(run.model_name, run.model_parameters)
        configured = replace(
            experiment_config,
            model=selection,
            seed=run.seed,
            manual_ann=None,
        )
        config_hash = stable_config_hash(asdict(configured))
        started = perf_counter()
        try:
            result = run_experiment(frame, configured, registry=registry)
            row = flatten_experiment_result(
                result,
                run,
                config_hash=config_hash,
                duration_seconds=perf_counter() - started,
            )
            row["run_id"] = run_id
            if artifact_directory is not None:
                from trading_system.artifacts.experiment import (
                    save_experiment_artifact,
                )

                artifact_path = Path(artifact_directory) / (
                    f"{run_id:04d}-{run.model_name}-seed-{run.seed}"
                )
                save_experiment_artifact(
                    artifact_path,
                    frame,
                    result,
                    dataset_path=dataset_path,
                )
                row["artifact_path"] = str(artifact_path.resolve())
            rows.append(row)
        except Exception as error:
            failure = {
                "run_id": run_id,
                "status": "error",
                "model_name": run.model_name,
                "model_parameters": json.dumps(
                    run.model_parameters, sort_keys=True, separators=(",", ":")
                ),
                "seed": run.seed,
                "config_hash": config_hash,
                "duration_seconds": perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            rows.append(failure)
            if not continue_on_error:
                raise
    run_frame = pd.DataFrame(rows).sort_values("run_id", kind="stable").reset_index(drop=True)
    failure_frame = pd.DataFrame(failures)
    summary = summarize_comparison_runs(
        run_frame,
        group_columns=("model_name", "model_parameters"),
    )
    control = experiment_config.overfitting_control
    if control is not None:
        expected_runs = len({run.seed for run in runs})
        summary = add_stability_flags(
            summary,
            expected_runs=expected_runs,
            max_validation_metric_std=control.max_validation_metric_std,
            max_train_validation_gap=control.max_train_validation_gap,
            require_all_seeds=control.require_all_seeds,
        )
    return ComparisonResult(run_frame, summary, failure_frame)


def save_comparison_result(
    result: ComparisonResult,
    destination: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write analysis-ready CSV files plus one complete JSON report."""

    if not isinstance(result, ComparisonResult):
        raise TypeError("result must be ComparisonResult.")
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    tables = {
        "runs": result.runs,
        "summary": result.summary,
        "failures": result.failures,
    }
    output_names = {f"{name}.csv" for name in tables} | {"report.json"}
    existing = sorted(name for name in output_names if (target / name).exists())
    if existing:
        raise FileExistsError(f"Comparison outputs already exist: {existing}")
    payload: dict[str, Any] = {"metadata": to_jsonable(dict(metadata or {}))}
    for name, table in tables.items():
        table.to_csv(target / f"{name}.csv", index=False)
        # pandas emits strict JSON nulls for NaN/NaT, unlike Python json.dumps.
        payload[name] = json.loads(table.to_json(orient="records", date_format="iso"))
    (target / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    return target


__all__ = [
    "ComparisonResult",
    "ComparisonRun",
    "build_comparison_runs",
    "add_stability_flags",
    "flatten_experiment_result",
    "run_model_comparison",
    "save_comparison_result",
    "summarize_comparison_runs",
    "validate_fair_comparison",
]
