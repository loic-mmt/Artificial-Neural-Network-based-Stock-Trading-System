"""Multi-ticker, technical-feature, long-only experiment."""

from functools import partial

from trading_system.experiments.config import ExperimentConfig

from .static import run_configured_pipeline
from .static import train_model as _train_model

DEFAULT_CONFIG = ExperimentConfig(
    universe="multi",
    feature_set="technical",
    label_mode="breakout",
    position_mode="long_only",
    label_window=30,
)

train_model = partial(_train_model, config=DEFAULT_CONFIG)
main = partial(run_configured_pipeline, DEFAULT_CONFIG)


__all__ = ["DEFAULT_CONFIG", "main", "train_model"]
