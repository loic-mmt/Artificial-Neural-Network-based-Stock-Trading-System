from __future__ import annotations

import contextlib
import io
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from weakref import ref

import numpy as np
import pandas as pd

from trading_system.models.base import ProbabilisticClassifier

from .config import ExperimentConfig
from .runner import (
    ExperimentResult,
    ValidationResult,
    evaluate_experiment_test,
    run_validation_experiment,
)
from .walkforward import walk_forward_oracle_ann


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
    # TODO(model-neutral-search-config): Replace ANN-only `hidden`, epochs and
    # learning-rate fields with model name plus a typed parameter mapping. Keep
    # data/label/decision axes separate from architecture axes.
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
    model_factory: Callable[[Mapping[str, Any]], ProbabilisticClassifier],
    *,
    objective: str = "macro_f1",
    max_trials: int | None = None,
    seed: int = 1,
) -> SearchResult:
    # TODO(model-neutral-grid-search-1): Build models through the shared registry
    # after sequence dimensions are known. Give each architecture the same trial
    # budget and seed policy.
    #
    # TODO(model-neutral-grid-search-2): Record parameter count via the registry.
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
                # Slot-only implementations may not support weak references.
                estimators.append(lambda estimator=estimator: estimator)
            result = run_validation_experiment(frame, experiment_config, estimator)
            score = objective_value(result, objective)
            row.update(status="ok", objective=score)
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
    # TODO(model-neutral-walkforward-grid): Replace this ANN-shaped Cartesian grid
    # with model-specific parameter spaces keyed by registry model name.
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


def run_walkforward_grid_search(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    trials: Sequence[WalkForwardTrialConfig],
    *,
    objective: str = "outperformance",
    seed: int = 1,
    suppress_inner_logs: bool = True,
    common_parameters: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Rank validation walks, then evaluate only the frozen winner on final test.

    Rows contain validation results. ``attrs['final_test']`` contains the separate
    winner report, never merged into selection columns.
    """

    # TODO(model-neutral-walkforward-search-1): Call `walk_forward_classifier`
    # through the registry instead of the ANN compatibility function.
    #
    # TODO(model-neutral-walkforward-search-2): Ensure equal per-model budgets.
    _validate_objective(objective)
    common = dict(common_parameters or {})
    if common.get("label_mode", "forward_return").startswith("oracle"):
        raise ValueError("Oracle labels are diagnostic only and cannot select a model.")
    if "evaluation_split" in common:
        raise ValueError("Search owns evaluation_split; it cannot be overridden.")
    if not trials:
        raise ValueError("Walk-forward search requires at least one trial.")
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_parameters = None
    best_trial_id = None
    validation_retrain_logs: dict[str, list[dict[str, Any]]] = {}

    def execute(parameters: Mapping[str, Any], split: str):
        output_context = (
            contextlib.redirect_stdout(io.StringIO())
            if suppress_inner_logs
            else contextlib.nullcontext()
        )
        with output_context:
            return walk_forward_oracle_ann(
                frame, feature_columns, **common, **parameters, evaluation_split=split
            )

    for trial_id, trial in enumerate(trials, start=1):
        started = perf_counter()
        parameters = {
            "forward_horizon": trial.forward_horizon,
            "forward_buy_threshold": trial.forward_buy_threshold,
            "forward_sell_threshold": trial.forward_sell_threshold,
            "context_len": trial.context_len,
            "walkforward_step": trial.walkforward_step,
            "hidden": trial.hidden,
            "epochs": trial.epochs,
            "alpha": trial.learning_rate,
            "batch_size": trial.batch_size,
            "decision_mode": trial.decision_mode,
            "min_action_rate": trial.min_action_rate,
            # All configurations share the same run/chunk seed policy.
            "seed": seed,
        }
        try:
            result = execute(parameters, "validation")
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
                    **parameters,
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
                best_parameters = dict(parameters)
                best_trial_id = trial_id
        except Exception as error:
            rows.append(
                {
                    "trial_id": trial_id,
                    **parameters,
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
        final = execute(best_parameters, "test")
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
    "make_trial_grid",
    "make_walkforward_trial_grid",
    "objective_value",
    "pick_trials",
    "run_grid_search",
    "run_walkforward_grid_search",
]
