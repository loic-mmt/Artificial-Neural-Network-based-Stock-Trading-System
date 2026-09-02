from __future__ import annotations

import contextlib
import io
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from weakref import ref

import numpy as np
import pandas as pd

from trading_system.models.base import ProbabilisticClassifier
from trading_system.models.factory import ModelRegistry, create_default_model_registry
from trading_system.models.specs import ModelSelection

from .config import ExperimentConfig
from .runner import (
    ExperimentResult,
    ValidationResult,
    evaluate_experiment_test,
    run_validation_experiment,
)
from .walkforward import walk_forward_classifier, walk_forward_oracle_ann


@dataclass(frozen=True)
class TrialConfig:
    parameters: Mapping[str, Any]


@dataclass
class SearchResult:
    best_parameters: dict[str, Any]
    best_result: ExperimentResult
    trials: pd.DataFrame


SELECTION_OBJECTIVES = frozenset(
    {
        "acc", "bal_acc", "macro_f1", "precision_sell", "recall_sell",
        "precision_hold", "recall_hold", "precision_buy", "recall_buy",
        "model_pnl", "outperformance",
    }
)


def _validate_objective(objective: str) -> None:
    if objective not in SELECTION_OBJECTIVES:
        raise ValueError(f"Unknown validation objective: {objective}")


@dataclass(frozen=True)
class WalkForwardTrialConfig:
    """Compatibility shape for legacy ANN-only walk-forward searches."""
    forward_horizon: int
    forward_buy_threshold: float
    forward_sell_threshold: float
    context_len: int
    walkforward_step: int
    hidden: int
    epochs: int
    learning_rate: float
    batch_size: int
    decision_mode: str
    min_action_rate: float


@dataclass(frozen=True, kw_only=True)
class WalkForwardModelTrialConfig:
    """Model-neutral walk-forward data axes plus typed registry selection."""

    model: ModelSelection
    forward_horizon: int
    forward_buy_threshold: float
    forward_sell_threshold: float
    context_len: int
    walkforward_step: int
    decision_mode: str = "argmax"
    min_action_rate: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelSelection):
            raise TypeError("model must be ModelSelection.")


def make_trial_grid(parameter_grid: Mapping[str, Sequence[Any]]) -> list[TrialConfig]:
    keys = tuple(parameter_grid)
    return [
        TrialConfig(dict(zip(keys, values)))
        for values in itertools.product(*(parameter_grid[key] for key in keys))
    ]


def pick_trials(
    trials: Sequence[TrialConfig],
    max_trials: int | None,
    seed: int = 1,
) -> list[TrialConfig]:
    selected = list(trials)
    if max_trials is None or max_trials >= len(selected):
        return selected
    if max_trials <= 0:
        raise ValueError("max_trials must be positive.")
    indices = np.random.default_rng(seed).choice(
        len(selected), size=max_trials, replace=False
    )
    return [selected[int(index)] for index in sorted(indices)]


def objective_value(result: ValidationResult, objective: str) -> float:
    """Selection can read validation metrics/backtest only, never final test."""

    _validate_objective(objective)
    source = (
        result.val_metrics if objective in result.val_metrics else result.val_backtest
    )
    value = float(source[objective])
    if not np.isfinite(value):
        raise ValueError(f"Validation objective {objective} is not finite.")
    return value


def run_grid_search(
    frame: pd.DataFrame,
    experiment_config: ExperimentConfig,
    parameter_grid: Mapping[str, Sequence[Any]],
    model_factory: Callable[[Mapping[str, Any]], ProbabilisticClassifier] | None = None,
    *,
    model_name: str | None = None,
    registry: ModelRegistry | None = None,
    objective: str = "macro_f1",
    max_trials: int | None = None,
    seed: int = 1,
) -> SearchResult:
    if model_factory is not None and (model_name is not None or registry is not None):
        raise ValueError("Legacy model_factory cannot be combined with registry options.")
    if model_factory is None:
        model_name = model_name or experiment_config.model.name
        registry = registry or create_default_model_registry()
    _validate_objective(objective)
    if experiment_config.label_mode.startswith("oracle"):
        raise ValueError("Oracle labels are diagnostic only and cannot select a model.")
    trials = pick_trials(make_trial_grid(parameter_grid), max_trials, seed)
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_parameters: dict[str, Any] | None = None
    best_result: ValidationResult | None = None
    estimators = []
    for trial_id, trial in enumerate(trials, start=1):
        started = perf_counter()
        row = {
            **trial.parameters, "trial_id": trial_id, "selection_split": "validation"
        }
        try:
            if model_factory is None:
                result = run_validation_experiment(
                    frame,
                    experiment_config,
                    model_selection=ModelSelection(model_name, dict(trial.parameters)),
                    registry=registry,
                )
            else:
                estimator = model_factory(dict(trial.parameters))
                if estimator is None:
                    raise TypeError("model_factory must return a classifier, not None.")
                if any(estimator is previous() for previous in estimators):
                    raise ValueError(
                        "model_factory must return a fresh estimator for each trial."
                    )
                try:
                    estimators.append(ref(estimator))
                except TypeError:
                    estimators.append(lambda estimator=estimator: estimator)
                result = run_validation_experiment(frame, experiment_config, estimator)
            score = objective_value(result, objective)
            row.update(status="ok", objective=score)
            bundle = getattr(result, "bundle", None)
            if bundle is not None:
                row.update(
                    model_name=bundle.model_selection.name,
                    parameter_count=bundle.parameter_count(),
                )
            row.update({f"val_{key}": value for key, value in result.val_metrics.items()})
            row.update({
                f"val_{key}": value for key, value in result.val_backtest.items()
            })
        except Exception as error:
            row.update(status="error", error=str(error), objective=-np.inf)
            row["duration_seconds"] = perf_counter() - started
            rows.append(row)
            continue
        row["duration_seconds"] = perf_counter() - started
        rows.append(row)
        if score > best_score:
            best_score = score
            best_parameters = dict(trial.parameters)
            best_result = result
    if best_result is None or best_parameters is None:
        raise RuntimeError(f"Grid search produced no valid trials: {rows}")
    # Keep the exact winning weights, scaler and calibrated policy. No refit and
    # no fallback to a runner-up after inspecting final-test performance.
    final_result = evaluate_experiment_test(frame, best_result)
    return SearchResult(
        best_parameters=best_parameters,
        best_result=final_result,
        trials=pd.DataFrame(rows)
        .sort_values("objective", ascending=False, kind="stable")
        .reset_index(drop=True),
    )


def make_walkforward_trial_grid(
    *,
    forward_horizons: Sequence[int],
    forward_buy_thresholds: Sequence[float],
    forward_sell_thresholds: Sequence[float],
    context_lengths: Sequence[int],
    walkforward_steps: Sequence[int],
    hidden_sizes: Sequence[int],
    epochs: Sequence[int],
    learning_rates: Sequence[float],
    batch_sizes: Sequence[int],
    decision_modes: Sequence[str],
    min_action_rates: Sequence[float],
) -> list[WalkForwardTrialConfig]:
    return [
        WalkForwardTrialConfig(*values)
        for values in itertools.product(
            forward_horizons,
            forward_buy_thresholds,
            forward_sell_thresholds,
            context_lengths,
            walkforward_steps,
            hidden_sizes,
            epochs,
            learning_rates,
            batch_sizes,
            decision_modes,
            min_action_rates,
        )
    ]


def make_model_walkforward_trial_grid(
    *,
    model_parameter_spaces: Mapping[str, Sequence[Mapping[str, Any]]],
    forward_horizons: Sequence[int],
    forward_buy_thresholds: Sequence[float],
    forward_sell_thresholds: Sequence[float],
    context_lengths: Sequence[int],
    walkforward_steps: Sequence[int],
    decision_modes: Sequence[str],
    min_action_rates: Sequence[float],
) -> list[WalkForwardModelTrialConfig]:
    if not model_parameter_spaces:
        raise ValueError("model_parameter_spaces cannot be empty.")
    spaces = {name: list(values) for name, values in model_parameter_spaces.items()}
    if any(not values for values in spaces.values()):
        raise ValueError("Every model requires at least one parameter set.")
    if len({len(values) for values in spaces.values()}) != 1:
        raise ValueError("Every model must receive the same parameter-set budget.")
    data_axes = itertools.product(
        forward_horizons,
        forward_buy_thresholds,
        forward_sell_thresholds,
        context_lengths,
        walkforward_steps,
        decision_modes,
        min_action_rates,
    )
    axes = list(data_axes)
    return [
        WalkForwardModelTrialConfig(
            model=ModelSelection(name, dict(parameters)),
            forward_horizon=horizon,
            forward_buy_threshold=buy,
            forward_sell_threshold=sell,
            context_len=context,
            walkforward_step=step,
            decision_mode=decision,
            min_action_rate=action_rate,
        )
        for name in sorted(spaces)
        for parameters in spaces[name]
        for horizon, buy, sell, context, step, decision, action_rate in axes
    ]


def run_walkforward_grid_search(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    trials: Sequence[WalkForwardTrialConfig | WalkForwardModelTrialConfig],
    *,
    objective: str = "outperformance",
    seed: int = 1,
    suppress_inner_logs: bool = True,
    common_parameters: Mapping[str, Any] | None = None,
    registry: ModelRegistry | None = None,
) -> pd.DataFrame:
    """Rank validation walks, then evaluate only the frozen winner on final test.

    Rows contain validation results. ``attrs['final_test']`` contains the separate
    winner report, never merged into selection columns.
    """

    _validate_objective(objective)
    common = dict(common_parameters or {})
    if common.get("label_mode", "forward_return").startswith("oracle"):
        raise ValueError("Oracle labels are diagnostic only and cannot select a model.")
    if "evaluation_split" in common:
        raise ValueError("Search owns evaluation_split; it cannot be overridden.")
    if not trials:
        raise ValueError("Walk-forward search requires at least one trial.")
    neutral_models = [
        trial.model.name
        for trial in trials
        if isinstance(trial, WalkForwardModelTrialConfig)
    ]
    if neutral_models:
        budgets = {name: neutral_models.count(name) for name in set(neutral_models)}
        if len(set(budgets.values())) != 1:
            raise ValueError("Every model must receive the same walk-forward budget.")
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_parameters = None
    best_trial_id = None
    validation_retrain_logs: dict[str, list[dict[str, Any]]] = {}

    def execute(
        parameters: Mapping[str, Any],
        split: str,
        selection: ModelSelection | None,
    ):
        output_context = (
            contextlib.redirect_stdout(io.StringIO())
            if suppress_inner_logs
            else contextlib.nullcontext()
        )
        with output_context:
            if selection is not None:
                return walk_forward_classifier(
                    frame,
                    feature_columns,
                    **common,
                    **parameters,
                    model_selection=selection,
                    registry=registry,
                    seed=seed,
                    evaluation_split=split,
                )
            return walk_forward_oracle_ann(
                frame, feature_columns, **common, **parameters, evaluation_split=split
            )

    for trial_id, trial in enumerate(trials, start=1):
        started = perf_counter()
        data_parameters = {
            "forward_horizon": trial.forward_horizon,
            "forward_buy_threshold": trial.forward_buy_threshold,
            "forward_sell_threshold": trial.forward_sell_threshold,
            "context_len": trial.context_len,
            "walkforward_step": trial.walkforward_step,
            "decision_mode": trial.decision_mode,
            "min_action_rate": trial.min_action_rate,
        }
        selection = (
            trial.model if isinstance(trial, WalkForwardModelTrialConfig) else None
        )
        parameters = dict(data_parameters)
        if selection is None:
            parameters.update(
                hidden=trial.hidden,
                epochs=trial.epochs,
                alpha=trial.learning_rate,
                batch_size=trial.batch_size,
                seed=seed,
            )
        row_parameters = dict(parameters)
        if selection is not None:
            row_parameters.update(
                model_name=selection.name,
                model_parameters=json.dumps(
                    selection.parameters, sort_keys=True, separators=(",", ":")
                ),
                seed=seed,
            )
        try:
            result = execute(data_parameters if selection else parameters, "validation", selection)
            metrics = result["val_metrics"]
            backtest = result["val_backtest"]
            combined = {**metrics, **backtest}
            score = float(combined[objective])
            if not np.isfinite(score):
                raise ValueError(f"Validation objective {objective} is not finite.")
            validation_retrain_logs[str(trial_id)] = result["retrain_logs"]
            rows.append(
                {
                    "trial_id": trial_id,
                    **row_parameters,
                    "status": "ok",
                    "selection_split": "validation",
                    **{f"val_{key}": value for key, value in combined.items()},
                    "objective_score": score,
                    "n_val_rows": int(result["n_val_rows"]),
                    "n_eval_rows": int(result["n_eval_rows"]),
                    "n_missing_val_preds": int(result["n_missing_val_preds"]),
                    "duration_seconds": perf_counter() - started,
                }
            )
            if score > best_score:
                best_score = score
                best_parameters = dict(row_parameters)
                best_trial_id = trial_id
                best_execution = (dict(data_parameters if selection else parameters), selection)
        except Exception as error:
            rows.append(
                {
                    "trial_id": trial_id,
                    **row_parameters,
                    "status": "error",
                    "selection_split": "validation",
                    "error": str(error),
                    "objective_score": -np.inf,
                    "duration_seconds": perf_counter() - started,
                }
            )
    results = (
        pd.DataFrame(rows)
        .sort_values("objective_score", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    results.attrs["validation_retrain_logs"] = validation_retrain_logs
    results["selected"] = results["trial_id"] == best_trial_id
    if best_parameters is not None:
        # Retrains inside this one final walk are predeclared. Their results
        # cannot change the winning configuration or trigger a runner-up test.
        execution_parameters, execution_selection = best_execution
        final = execute(execution_parameters, "test", execution_selection)
        results.attrs["best_parameters"] = dict(best_parameters)
        results.attrs["final_test"] = {
            key: final[key]
            for key in (
                "test_metrics", "benchmark_comparison", "n_test_rows",
                "n_eval_rows", "n_missing_test_preds", "retrain_logs",
            )
        }
    return results


__all__ = [
    "SearchResult",
    "TrialConfig",
    "WalkForwardTrialConfig",
    "WalkForwardModelTrialConfig",
    "make_model_walkforward_trial_grid",
    "make_trial_grid",
    "make_walkforward_trial_grid",
    "objective_value",
    "pick_trials",
    "run_grid_search",
    "run_walkforward_grid_search",
]
