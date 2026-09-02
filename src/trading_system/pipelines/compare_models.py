from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import Any

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.comparison import (
    build_comparison_runs,
    run_model_comparison,
    save_comparison_result,
)
from trading_system.models.factory import create_default_model_registry
from trading_system.paths import comparisons_dir, default_market_dataset_path
from trading_system.reporting.warnings import current_universe_warning

from .multi_ticker import DEFAULT_CONFIG as MULTI_TICKER
from .multi_ticker_long_short import DEFAULT_CONFIG as MULTI_TICKER_LONG_SHORT
from .single_ticker import DEFAULT_CONFIG as SINGLE_TICKER
from .single_ticker_features import DEFAULT_CONFIG as SINGLE_TICKER_FEATURES
from .single_ticker_long_short import DEFAULT_CONFIG as SINGLE_TICKER_LONG_SHORT
from .single_ticker_long_short_features import (
    DEFAULT_CONFIG as SINGLE_TICKER_LONG_SHORT_FEATURES,
)

PRESETS = {
    "single_ticker": SINGLE_TICKER,
    "single_ticker_features": SINGLE_TICKER_FEATURES,
    "single_ticker_long_short": SINGLE_TICKER_LONG_SHORT,
    "single_ticker_long_short_features": SINGLE_TICKER_LONG_SHORT_FEATURES,
    "multi_ticker": MULTI_TICKER,
    "multi_ticker_long_short": MULTI_TICKER_LONG_SHORT,
}


def _comma_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def _seeds(value: str) -> list[int]:
    try:
        seeds = [int(item) for item in _comma_list(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be integers.") from error
    if any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("Seeds must be unique and non-negative.")
    return seeds


def _json_source(value: str) -> dict[str, Any]:
    candidate = Path(value).expanduser()
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    text = candidate.read_text(encoding="utf-8") if is_file else value
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Model parameters JSON must be an object.")
    return parsed


def load_ticker_selection(path: Path) -> list[str]:
    """Load and validate a performance-independent benchmark ticker selection."""
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load ticker selection {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("tickers"), list):
        raise ValueError("Ticker selection must be an object containing a 'tickers' list.")
    tickers: list[str] = []
    for entry in payload["tickers"]:
        ticker = entry.get("ticker") if isinstance(entry, dict) else entry
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Every ticker selection entry must be a string or have a ticker string.")
        tickers.append(ticker.strip())
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError("Ticker selection must contain unique tickers and cannot be empty.")
    return tickers


def build_parser() -> argparse.ArgumentParser:
    registry = create_default_model_registry()
    parser = argparse.ArgumentParser(
        description="Compare frozen model configurations over identical seeds."
    )
    parser.add_argument("--data", type=Path, default=default_market_dataset_path())
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default="multi_ticker_long_short"
    )
    parser.add_argument(
        "--models", type=_comma_list, default=list(registry.names())
    )
    parser.add_argument(
        "--seeds", type=_seeds, default=[1, 7, 19, 42, 1337]
    )
    parser.add_argument(
        "--model-parameter-sets",
        type=_json_source,
        help="Inline JSON or JSON file: {model: [parameter_object, ...]}.",
    )
    parser.add_argument(
        "--ticker-selection",
        type=Path,
        help="JSON selection file. Supported only by multi-ticker presets.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--no-run-artifacts",
        action="store_true",
        help="Only save comparison tables, not reloadable artifacts for every run.",
    )
    return parser


def _parameter_sets(
    models: list[str], raw: dict[str, Any] | None
) -> dict[str, list[dict[str, Any]]]:
    if raw is None:
        return {model: [{}] for model in models}
    extras = sorted(set(raw) - set(models))
    missing = sorted(set(models) - set(raw))
    if extras or missing:
        raise ValueError(
            f"Parameter-set models must exactly match --models; missing={missing}, extra={extras}."
        )
    output: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        values = raw[model]
        if not isinstance(values, list) or not values or not all(
            isinstance(value, dict) for value in values
        ):
            raise ValueError(f"{model} parameter sets must be a non-empty list of objects.")
        output[model] = values
    return output


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    registry = create_default_model_registry()
    unknown = sorted(set(args.models) - set(registry.names()))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}; available={list(registry.names())}")
    warning = current_universe_warning(args.data)
    if warning:
        print(warning)
    frame = read_parquet_dataset(args.data)
    config = replace(PRESETS[args.preset], device=args.device)
    selected_tickers = None
    if args.ticker_selection is not None:
        if config.universe != "multi":
            raise ValueError("--ticker-selection requires a multi-ticker preset.")
        selected_tickers = load_ticker_selection(args.ticker_selection)
        if config.group_col not in frame.columns:
            raise ValueError(f"Dataset has no {config.group_col!r} column.")
        available = set(frame[config.group_col].dropna().astype(str))
        missing = sorted(set(selected_tickers) - available)
        if missing:
            raise ValueError(f"Selected tickers missing from dataset: {missing}")
        frame = frame[frame[config.group_col].astype(str).isin(selected_tickers)].copy()
    parameter_sets = _parameter_sets(args.models, args.model_parameter_sets)
    runs = build_comparison_runs(parameter_sets, args.seeds)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or comparisons_dir() / stamp
    result = run_model_comparison(
        frame,
        config,
        runs,
        registry,
        continue_on_error=not args.fail_fast,
        artifact_directory=None if args.no_run_artifacts else output_dir / "runs",
        dataset_path=args.data,
    )
    saved = save_comparison_result(
        result,
        output_dir,
        metadata={
            "data_path": str(args.data.resolve()),
            "preset": args.preset,
            "models": args.models,
            "seeds": args.seeds,
            "device": args.device,
            "ticker_selection": (
                str(args.ticker_selection.resolve())
                if args.ticker_selection is not None
                else None
            ),
            "selected_tickers": selected_tickers,
            "survivor_bias_warning": warning,
        },
    )
    print(f"comparison_runs={len(result.runs)} failures={len(result.failures)}")
    print(f"saved={saved}")
    return result


__all__ = ["PRESETS", "build_parser", "load_ticker_selection", "main"]
