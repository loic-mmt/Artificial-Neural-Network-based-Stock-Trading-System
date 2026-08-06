from __future__ import annotations

import contextlib
import io
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading_system.models.base import ProbabilisticClassifier

from .config import ExperimentConfig
from .runner import ExperimentResult, run_experiment
from .walkforward import walk_forward_oracle_ann


@dataclass(frozen=True)
class TrialConfig:
    parameters: Mapping[str, Any]


@dataclass
class SearchResult:
    best_parameters: dict[str, Any]
    best_result: ExperimentResult
    trials: pd.DataFrame


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


def objective_value(result: ExperimentResult, objective: str) -> float:
    # TODO(validation-only-search-objective): This legacy function currently reads
    # test metrics/backtest and must not be used for final model selection. Change
    # search results to expose validation objectives, then lock test evaluation
    # until the selected configuration is fixed.
    if objective in result.test_metrics:
        return float(result.test_metrics[objective])
    if objective in result.backtest:
        return float(result.backtest[objective])
    raise ValueError(f"Unknown objective: {objective}")


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
    # TODO(model-neutral-grid-search-2): Record failed trials, parameter count and
    # duration rather than aborting the full comparison. Rank on validation only.
    trials = pick_trials(make_trial_grid(parameter_grid), max_trials, seed)
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_parameters: dict[str, Any] | None = None
    best_result: ExperimentResult | None = None
    for trial in trials:
        result = run_experiment(
            frame, experiment_config, model_factory(trial.parameters)
        )
        score = objective_value(result, objective)
        rows.append({**trial.parameters, "objective": score})
        if score > best_score:
            best_score = score
            best_parameters = dict(trial.parameters)
            best_result = result
    if best_result is None or best_parameters is None:
        raise RuntimeError("Grid search executed no trials.")
    return SearchResult(
        best_parameters=best_parameters,
        best_result=best_result,
        trials=pd.DataFrame(rows)
        .sort_values("objective", ascending=False)
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
    # TODO(model-neutral-walkforward-search-1): Call `walk_forward_classifier`
    # through the registry instead of the ANN compatibility function.
    #
    # TODO(model-neutral-walkforward-search-2): Separate validation selection from
    # final test reporting and ensure equal budgets across architectures.
    common = dict(common_parameters or {})
    rows: list[dict[str, Any]] = []
    for trial_id, trial in enumerate(trials, start=1):
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
            "seed": seed + trial_id - 1,
        }
        try:
            if suppress_inner_logs:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = walk_forward_oracle_ann(
                        frame,
                        feature_columns,
                        **common,
                        **parameters,
                    )
            else:
                result = walk_forward_oracle_ann(
                    frame,
                    feature_columns,
                    **common,
                    **parameters,
                )
            metrics = result["test_metrics"]
            backtest = result["benchmark_comparison"]
            combined = {**metrics, **backtest}
            if objective not in combined:
                raise ValueError(f"Unknown walk-forward objective: {objective}")
            rows.append(
                {
                    "trial_id": trial_id,
                    **parameters,
                    "status": "ok",
                    **metrics,
                    **backtest,
                    "objective_score": float(combined[objective]),
                    "n_test_rows": int(result["n_test_rows"]),
                    "n_eval_rows": int(result["n_eval_rows"]),
                    "n_missing_test_preds": int(result["n_missing_test_preds"]),
                }
            )
        except Exception as error:
            rows.append(
                {
                    "trial_id": trial_id,
                    **parameters,
                    "status": "error",
                    "error": str(error),
                    "objective_score": -np.inf,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values("objective_score", ascending=False)
        .reset_index(drop=True)
    )


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
