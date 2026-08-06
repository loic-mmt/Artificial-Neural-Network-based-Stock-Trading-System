from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading_system.models.factory import ModelRegistry

from .config import ExperimentConfig


@dataclass(frozen=True)
class ComparisonRun:
    """One predeclared model/configuration/seed combination."""

    model_name: str
    model_parameters: dict[str, Any] = field(default_factory=dict)
    seed: int = 1

    def __post_init__(self) -> None:
        # TODO(comparison-run-validate): Require a non-empty model name,
        # non-negative seed and JSON-compatible defensive parameter copy.
        raise NotImplementedError


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
    # TODO(comparison-matrix-1): Validate unique non-negative seeds and at least
    # one model/configuration. Build the Cartesian product of each architecture's
    # parameter sets and the exact same seed list.
    #
    # TODO(comparison-matrix-2): Preserve deterministic ordering by sorted model
    # name, declared parameter-set order and seed order. Do not silently give one
    # architecture more trials than another.
    raise NotImplementedError


def validate_fair_comparison(
    experiment_config: ExperimentConfig,
    runs: Sequence[ComparisonRun],
) -> None:
    # TODO(comparison-fairness-1): Require at least two model names and equal seed
    # coverage. Validate that feature, label, split, context, decision, fees and
    # execution settings come only from the one shared experiment config.
    #
    # TODO(comparison-fairness-2): Reject test-derived objectives or parameters.
    # This validation cannot prove research integrity alone, but it should make
    # accidental per-model data settings impossible.
    raise NotImplementedError


def flatten_experiment_result(
    result: Any,
    run: ComparisonRun,
    *,
    config_hash: str,
    duration_seconds: float,
) -> dict[str, Any]:
    # TODO(comparison-flatten-1): Convert classification, backtest, split, model
    # and training diagnostics to one flat row. Prefix ambiguous metric names,
    # e.g. `test_macro_f1` and `backtest_total_return`.
    #
    # TODO(comparison-flatten-2): Include model name, parameters, seed, config
    # hash, parameter count, best epoch, stop reason and duration. Keep values
    # serializable without discarding failed-run context.
    raise NotImplementedError


def summarize_comparison_runs(
    runs: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("model_name",),
) -> pd.DataFrame:
    # TODO(comparison-summary-1): Validate required identifier/metric columns and
    # include only successful runs. Group by the declared identifiers.
    #
    # TODO(comparison-summary-2): For every numeric metric, report count, mean,
    # sample standard deviation, minimum and maximum with stable column names.
    # Never select a winner here from test metrics; this function reports only.
    raise NotImplementedError


def run_model_comparison(
    frame: pd.DataFrame,
    experiment_config: ExperimentConfig,
    runs: Sequence[ComparisonRun],
    registry: ModelRegistry,
    *,
    continue_on_error: bool = True,
) -> ComparisonResult:
    # TODO(comparison-execute-1): Validate fairness once, prepare the shared data
    # deterministically, and compute a stable experiment config hash.
    #
    # TODO(comparison-execute-2): For each declared run, construct a fresh model
    # through the registry after input dimensions are known, execute the neutral
    # runner, time it and flatten its output. Never reuse fitted weights between
    # seeds or architectures.
    #
    # TODO(comparison-execute-3): Record exceptions as structured failure rows.
    # Re-raise immediately only when `continue_on_error=False`.
    #
    # TODO(comparison-execute-4): Build the grouped summary from successful runs
    # and return raw, summary and failure DataFrames in deterministic order.
    raise NotImplementedError


__all__ = [
    "ComparisonResult",
    "ComparisonRun",
    "build_comparison_runs",
    "flatten_experiment_result",
    "run_model_comparison",
    "summarize_comparison_runs",
    "validate_fair_comparison",
]
