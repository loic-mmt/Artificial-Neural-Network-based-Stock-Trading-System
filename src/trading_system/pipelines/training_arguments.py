"""Optional event importance weighting for neural training."""

import argparse
from dataclasses import replace

from trading_system.training.sample_weighting import SampleWeightConfig
from trading_system.training.financial_loss import FinancialLossConfig


def add_weight_arguments(parser):
    parser.add_argument("--sample-weighting", choices=("none", "net_return", "volatility", "uniqueness"), default=argparse.SUPPRESS)
    parser.add_argument("--sample-weight-clip-quantile", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--sample-weight-min", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--sample-weight-max", type=float, default=argparse.SUPPRESS)


def add_financial_loss_arguments(parser, *, default="cross_entropy"):
    parser.add_argument(
        "--loss-objective", choices=("cross_entropy", "pnl", "sharpe"),
        default=default,
    )
    parser.add_argument("--loss-cost-bps", type=float, default=5.0)
    parser.add_argument("--loss-annualization", type=int, default=252)
    parser.add_argument("--loss-sharpe-epsilon", type=float, default=1e-4)


def financial_loss_config_from_args(args):
    config = FinancialLossConfig(
        objective=args.loss_objective,
        cost_bps=args.loss_cost_bps,
        annualization=args.loss_annualization,
        sharpe_epsilon=args.loss_sharpe_epsilon,
    )
    return None if config.objective == "cross_entropy" else config


def sample_weight_config_from_args(args, base=None):
    values = vars(args)
    mode = values.get("sample_weighting", base.mode if base else "none")
    fields = {
        "sample_weight_clip_quantile": "clip_quantile",
        "sample_weight_min": "min_weight", "sample_weight_max": "max_weight",
    }
    updates = {target: values[source] for source, target in fields.items() if source in values}
    if mode == "none":
        if updates:
            raise ValueError("Sample-weight limits require enabled --sample-weighting.")
        return None
    return replace(base or SampleWeightConfig(), mode=mode, **updates)


def apply_weight_arguments(config, args):
    return replace(config, sample_weighting=sample_weight_config_from_args(args, config.sample_weighting))
