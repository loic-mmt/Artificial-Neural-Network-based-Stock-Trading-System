"""CLI orchestration for expanding-window model evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.walkforward import (
    fit_labeled_history,
    predict_chunk_with_model,
    walk_forward_classifier,
    walk_forward_oracle_ann,
)
from trading_system.features.market import (
    MARKET_FEATURE_COLUMNS,
    compute_market_features,
)
from trading_system.paths import default_market_dataset_path
from trading_system.reporting.plots import format_experiment_summary

# Compatibility alias used by old grid-search code.
features = MARKET_FEATURE_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    # TODO(sequence-walkforward-cli-1): Add `--model` choices from the default
    # registry and a model-config input. Stop exposing ANN-only flags globally;
    # validate parameters against the selected typed model config.
    parser = argparse.ArgumentParser(
        description="Expanding-window probabilistic classifier evaluation."
    )
    parser.add_argument("--data-dir", type=Path, default=default_market_dataset_path())
    parser.add_argument("--ticker", default="EN.PA")
    parser.add_argument("--price-col", default="adj_close")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--walkforward-step", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--do-dropout", action="store_true")
    parser.add_argument("--dropout-percent", type=float, default=0.1)
    parser.add_argument(
        "--label-mode",
        choices=("oracle_dp", "forward_return", "breakout"),
        default="forward_return",
    )
    parser.add_argument("--breakout-window", type=int, default=20)
    parser.add_argument("--forward-horizon", type=int, default=1)
    parser.add_argument("--forward-buy-threshold", type=float, default=0.002)
    parser.add_argument("--forward-sell-threshold", type=float, default=0.002)
    parser.add_argument(
        "--decision-mode", choices=("thresholds", "argmax"), default="argmax"
    )
    parser.add_argument("--min-action-rate", type=float, default=0.02)
    parser.add_argument(
        "--position-mode", choices=("long_short", "long_only"), default="long_only"
    )
    parser.add_argument("--oracle-fee-per-trade", type=float, default=2.0)
    parser.add_argument("--strategy-fee-per-trade", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main() -> None:
    # TODO(sequence-walkforward-cli-2): Build `ModelSelection`, call the neutral
    # walk-forward runner, and include model/config hash in console output.
    args = build_parser().parse_args()
    frame = read_parquet_dataset(args.data_dir)
    if "date" not in frame.columns:
        raise ValueError("Dataset requires date column.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    if args.ticker:
        if "ticker" not in frame.columns:
            raise ValueError("Ticker filtering requires ticker column.")
        frame = frame[frame["ticker"] == args.ticker].copy()
    if frame.empty:
        raise ValueError("No rows remain after ticker filtering.")
    featured = compute_market_features(frame.sort_values("date").reset_index(drop=True))
    result = walk_forward_oracle_ann(
        full_df=featured,
        feature_cols=MARKET_FEATURE_COLUMNS,
        price_col=args.price_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        walkforward_step=args.walkforward_step,
        oracle_fee_per_trade=args.oracle_fee_per_trade,
        label_mode=args.label_mode,
        forward_horizon=args.forward_horizon,
        forward_buy_threshold=args.forward_buy_threshold,
        forward_sell_threshold=args.forward_sell_threshold,
        breakout_window=args.breakout_window,
        decision_mode=args.decision_mode,
        min_action_rate=args.min_action_rate,
        position_mode=args.position_mode,
        strategy_fee_per_trade=args.strategy_fee_per_trade,
        initial_capital=args.capital,
        context_len=args.context_len,
        epochs=args.epochs,
        alpha=args.alpha,
        hidden=args.hidden,
        batch_size=args.batch_size,
        do_dropout=args.do_dropout,
        dropout_percent=args.dropout_percent,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        seed=args.seed,
    )
    print(
        f"coverage={result['n_eval_rows']}/{result['n_test_rows']} "
        f"retrains={len(result['retrain_logs'])}"
    )
    print(
        format_experiment_summary(
            result["test_metrics"], result["benchmark_comparison"]
        )
    )


__all__ = [
    "build_parser",
    "fit_labeled_history",
    "main",
    "predict_chunk_with_model",
    "walk_forward_classifier",
    "walk_forward_oracle_ann",
]


if __name__ == "__main__":
    main()
