"""Matched detached/TBPTT study, without changing the S0-S3 execution recipe."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading_system.artifacts.multimodal_study import atomic_write_json
from trading_system.experiments.label_loss_benchmark import _write_table
from trading_system.experiments.trade_state_benchmark import (
    TradeStateBenchmarkConfig, run_trade_state_benchmark,
)


@dataclass(frozen=True)
class TradeStateGradientBenchmarkConfig(TradeStateBenchmarkConfig):
    state_variants: tuple[str, ...] = ("S0", "S3")
    tbptt_windows: tuple[int, ...] = (10, 40)

    def __post_init__(self):
        super().__post_init__()
        if self.state_variants != ("S0", "S3"):
            raise ValueError("The gradient study fixes S0/S3 controls in that order.")
        if "financial" not in self.families:
            raise ValueError("The gradient study requires the financial family.")
        if (not self.tbptt_windows or len(set(self.tbptt_windows)) != len(self.tbptt_windows)
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in self.tbptt_windows)):
            raise ValueError("TBPTT windows must be unique positive session counts.")

    @property
    def schema_version(self):
        return "trade_state_gradients_v1"

    def candidates(self):
        candidates = []
        for variant in self.state_variants:
            candidates.append({"candidate": f"financial-{variant}-detached", "family": "financial",
                "state_variant": variant, "state_gradient_mode": "detached", "tbptt_steps": None,
                "label_method": None, "training_protocol": "overnight"})
        for steps in self.tbptt_windows:
            candidates.append({"candidate": f"financial-S3-tbptt-{steps}", "family": "financial",
                "state_variant": "S3", "state_gradient_mode": "tbptt", "tbptt_steps": steps,
                "label_method": None, "training_protocol": "overnight"})
        if "cross_entropy" in self.families:
            candidates.append({"candidate": "cross_entropy-S0-detached", "family": "cross_entropy",
                "state_variant": "S0", "state_gradient_mode": "detached", "tbptt_steps": None,
                "label_method": "volatility_position", "training_protocol": "overnight"})
        return candidates


def run_trade_state_gradient_benchmark(frame, tickers, destination, *, config=None, **options):
    config = config or TradeStateGradientBenchmarkConfig()
    if not isinstance(config, TradeStateGradientBenchmarkConfig):
        raise TypeError("The gradient study requires TradeStateGradientBenchmarkConfig.")
    report = run_trade_state_benchmark(frame, tickers, destination, config=config, **options)
    root = Path(destination).resolve()
    results = pd.read_csv(root / "results.csv")
    baseline = results[results.candidate == "financial-S3-detached"]
    alternatives = results[results.state_gradient_mode == "tbptt"]
    paired = alternatives.merge(baseline, on=["fold", "seed", "protocol", "decoder"],
                                suffixes=("", "_detached"), validate="many_to_one")
    columns = ["fold", "seed", "protocol", "decoder", "candidate", "tbptt_steps"]
    comparison = paired[columns].copy()
    for metric in ("net_return", "net_sharpe", "max_drawdown", "mean_abs_position",
                   "mean_long_exposure", "mean_short_exposure", "turnover", "cost_return_sum",
                   "trades", "short_pnl_contribution", "training_duration_seconds"):
        comparison[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_detached"]
    _write_table(root / "gradient-paired.csv", comparison)
    report = {**report, "paired_gradient_comparisons": len(comparison),
              "gradient_scope": "partial_tbptt_continuous_channels_only",
              "optimizer_cadence": "one_global_step_per_epoch"}
    atomic_write_json(root / "report.json", report)
    return report
