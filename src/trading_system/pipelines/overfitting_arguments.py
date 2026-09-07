"""Shared CLI arguments for opt-in overfitting controls."""

from __future__ import annotations

import argparse

from trading_system.training.overfitting import OverfittingControlConfig


def add_overfitting_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overfitting-control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable compact regularized defaults and train-only feature selection.",
    )
    parser.add_argument("--overfitting-max-features", type=int, default=32)
    parser.add_argument("--overfitting-min-feature-variance", type=float, default=1e-8)
    parser.add_argument("--overfitting-max-feature-correlation", type=float, default=0.98)
    parser.add_argument("--overfitting-max-seed-std", type=float, default=0.05)
    parser.add_argument("--overfitting-max-train-val-gap", type=float, default=0.15)
    parser.add_argument(
        "--overfitting-require-all-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def overfitting_config_from_args(args: argparse.Namespace) -> OverfittingControlConfig | None:
    if not args.overfitting_control:
        return None
    return OverfittingControlConfig(
        max_features=args.overfitting_max_features,
        min_feature_variance=args.overfitting_min_feature_variance,
        max_feature_correlation=args.overfitting_max_feature_correlation,
        max_validation_metric_std=args.overfitting_max_seed_std,
        max_train_validation_gap=args.overfitting_max_train_val_gap,
        require_all_seeds=args.overfitting_require_all_seeds,
    )


__all__ = ["add_overfitting_arguments", "overfitting_config_from_args"]
