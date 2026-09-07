"""Shared opt-in feature preprocessing arguments."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from trading_system.features.fracdiff import FracDiffConfig


def _order(value):
    if value == "auto":
        return None
    try:
        order = float(value)
        FracDiffConfig(order=order)
        return order
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("FracDiff order must be auto or 0 < d < 1.") from error


def add_feature_arguments(parser):
    add_feature_source_arguments(parser)
    parser.add_argument("--feature-set", choices=("technical", "market", "expanded"), default=argparse.SUPPRESS)
    parser.add_argument("--feature-groups", default=argparse.SUPPRESS, help="Expanded ablation: technical,market,sector,fundamentals,sentiment.")
    parser.add_argument("--feature-min-coverage", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--fracdiff", action="store_true", help="Add causal fractional differentiation of log price.")
    parser.add_argument("--fracdiff-order", type=_order, default=argparse.SUPPRESS, help="auto (train-only ADF) or a fixed order in (0, 1).")
    parser.add_argument("--fracdiff-threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--fracdiff-max-terms", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--fracdiff-min-samples", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--fracdiff-adf-pvalue", type=float, default=argparse.SUPPRESS)


def add_feature_source_arguments(parser):
    parser.add_argument("--fundamentals", type=Path, help="Optional local publication-dated fundamentals CSV/parquet; expanded only.")
    parser.add_argument("--sentiment", type=Path, help="Optional local publication-dated news/earnings CSV/parquet; expanded only.")
    parser.add_argument("--no-external-features", action="store_true", help="Ignore historical fundamentals/sentiment even in an enriched dataset; expanded only.")


def apply_feature_sources(frame, args, config):
    from trading_system.data.optional_sources import prepare_feature_sources

    fundamentals = getattr(args, "fundamentals", None)
    sentiment = getattr(args, "sentiment", None)
    disabled = getattr(args, "no_external_features", False)
    if config.feature_set != "expanded":
        if fundamentals is not None or sentiment is not None or disabled:
            raise ValueError("Historical source arguments require --feature-set expanded.")
        return frame, None
    work, report = prepare_feature_sources(
        frame, fundamentals=fundamentals, sentiment=sentiment, disabled=disabled,
        group_col=config.group_col, date_col=config.date_col,
    )
    print(f"feature_sources={json.dumps(report, sort_keys=True)}")
    return work, report


def fracdiff_config_from_args(args, base=None):
    values = vars(args)
    overrides = {
        name: values[f"fracdiff_{name}"]
        for name in ("order", "threshold", "max_terms", "min_samples", "adf_pvalue")
        if f"fracdiff_{name}" in values
    }
    if not values.get("fracdiff") and base is None:
        if overrides:
            raise ValueError("FracDiff parameters require --fracdiff.")
        return None
    return replace(base or FracDiffConfig(), **overrides)


def apply_feature_arguments(config, args):
    updates = {"fracdiff": fracdiff_config_from_args(args, config.fracdiff)}
    if hasattr(args, "feature_set"):
        updates["feature_set"] = args.feature_set
    if hasattr(args, "feature_groups"):
        updates["expanded_feature_groups"] = tuple(args.feature_groups.split(","))
    if hasattr(args, "feature_min_coverage"):
        updates["expanded_min_coverage"] = args.feature_min_coverage
    if (hasattr(args, "feature_groups") or hasattr(args, "feature_min_coverage")) and updates.get("feature_set", config.feature_set) != "expanded":
        raise ValueError("Feature groups/coverage require --feature-set expanded.")
    return replace(config, **updates)


def build_cli_features(frame, args, *, default="market"):
    from trading_system.experiments.config import ExperimentConfig
    from trading_system.features.expanded import compute_expanded_features, feature_columns
    from trading_system.features.market import compute_market_features, MARKET_FEATURE_COLUMNS
    from trading_system.features.technical import compute_technical_features, TECHNICAL_FEATURE_COLUMNS

    config = apply_feature_arguments(ExperimentConfig(feature_set=default), args)
    frame, _ = apply_feature_sources(frame, args, config)
    if config.feature_set == "expanded":
        return compute_expanded_features(frame), feature_columns(config.expanded_feature_groups), config
    if config.feature_set == "technical":
        return compute_technical_features(frame, group_col="ticker" if "ticker" in frame else None), tuple(TECHNICAL_FEATURE_COLUMNS), config
    return compute_market_features(frame), tuple(MARKET_FEATURE_COLUMNS), config
