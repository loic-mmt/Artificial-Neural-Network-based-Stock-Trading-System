"""Single-ticker, market-feature, long-only experiment."""

from functools import partial

from trading_system.experiments.config import ExperimentConfig

from .static import run_configured_pipeline
from .static import train_model as _train_model

DEFAULT_CONFIG = ExperimentConfig(
    universe="single",
    ticker="EN.PA",
    feature_set="market",
    label_mode="oracle_all",
    position_mode="long_only",
    label_window=30,
    oracle_fee_per_trade=2.0,
)

train_model = partial(_train_model, config=DEFAULT_CONFIG)
main = partial(run_configured_pipeline, DEFAULT_CONFIG)


__all__ = ["DEFAULT_CONFIG", "main", "train_model"]
