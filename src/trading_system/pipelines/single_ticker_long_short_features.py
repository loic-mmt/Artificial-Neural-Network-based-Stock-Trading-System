"""Single-ticker, market-feature, long/short experiment."""

from functools import partial

from trading_system.experiments.config import ExperimentConfig

from .static import run_configured_pipeline
from .static import train_model as _train_model

DEFAULT_CONFIG = ExperimentConfig(
    universe="single",
    ticker="EN.PA",
    feature_set="market",
    label_mode="breakout",
    position_mode="long_short",
    label_window=11,
)

train_model = partial(_train_model, config=DEFAULT_CONFIG)
main = partial(run_configured_pipeline, DEFAULT_CONFIG)


__all__ = ["DEFAULT_CONFIG", "main", "train_model"]
