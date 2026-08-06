"""Model-neutral experiment configuration, execution, and search."""

from .comparison import (
    ComparisonResult,
    ComparisonRun,
    build_comparison_runs,
    run_model_comparison,
)
from .config import ExperimentConfig
from .runner import ExperimentResult, TrainedModelBundle, run_experiment

__all__ = [
    "ComparisonResult",
    "ComparisonRun",
    "ExperimentConfig",
    "ExperimentResult",
    "TrainedModelBundle",
    "build_comparison_runs",
    "run_experiment",
    "run_model_comparison",
]
