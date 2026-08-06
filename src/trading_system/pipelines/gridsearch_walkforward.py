"""CLI configuration for walk-forward hyperparameter search."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.search import (
    make_walkforward_trial_grid,
    pick_trials,
    run_walkforward_grid_search,
)
from trading_system.features.market import (
    MARKET_FEATURE_COLUMNS,
    compute_market_features,
)
from trading_system.paths import default_market_dataset_path, gridsearch_dir


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    # TODO(sequence-grid-cli-1): Add model list/config-space inputs and obtain
    # model names from the registry. Remove ANN-specific hidden/alpha flags after
    # the compatibility period.
    parser = argparse.ArgumentParser(
        description="Grid search over expanding-window experiments."
    )
    parser.add_argument("--data-dir", type=Path, default=default_market_dataset_path())
    parser.add_argument("--ticker", default="EN.PA")
    parser.add_argument("--price-col", default="adj_close")
    parser.add_argument(
        "--objective",
        choices=("outperformance", "model_pnl", "macro_f1", "bal_acc"),
        default="outperformance",
    )
    parser.add_argument(
        "--label-mode",
        choices=("oracle_dp", "forward_return", "breakout"),
        default="forward_return",
    )
    parser.add_argument(
        "--position-mode", choices=("long_only", "long_short"), default="long_only"
    )
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--oracle-fee-per-trade", type=float, default=2.0)
    parser.add_argument("--strategy-fee-per-trade", type=float, default=0.0)
    parser.add_argument("--do-dropout", action="store_true")
    parser.add_argument("--dropout-percent", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--forward-horizons", default="1,3,5")
    parser.add_argument("--forward-buy-thresholds", default="0.001,0.002,0.005")
    parser.add_argument("--forward-sell-thresholds", default="0.001,0.002,0.005")
    parser.add_argument("--context-lengths", default="10,20,40")
    parser.add_argument("--walkforward-steps", default="30,60")
    parser.add_argument("--hiddens", default="64,128")
    parser.add_argument("--epochs-grid", default="300")
    parser.add_argument("--alphas", default="0.001")
    parser.add_argument("--batch-sizes", default="64")
    parser.add_argument("--decision-modes", default="argmax")
    parser.add_argument("--min-action-rates", default="0.02")
    parser.add_argument("--max-trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--show-inner-logs", action="store_true")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    # TODO(sequence-grid-cli-2): Build equal per-model trial matrices, select only
    # on validation, and write raw runs/failures/summary through artifact helpers.
    args = build_parser().parse_args()
    frame = read_parquet_dataset(args.data_dir)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    if args.ticker:
        if "ticker" not in frame.columns:
            raise ValueError("Ticker filtering requires ticker column.")
        frame = frame[frame["ticker"] == args.ticker].copy()
    if frame.empty:
        raise ValueError("No rows remain after ticker filtering.")
    featured = compute_market_features(frame.sort_values("date").reset_index(drop=True))
    trials = make_walkforward_trial_grid(
        forward_horizons=parse_int_list(args.forward_horizons),
        forward_buy_thresholds=parse_float_list(args.forward_buy_thresholds),
        forward_sell_thresholds=parse_float_list(args.forward_sell_thresholds),
        context_lengths=parse_int_list(args.context_lengths),
        walkforward_steps=parse_int_list(args.walkforward_steps),
        hidden_sizes=parse_int_list(args.hiddens),
        epochs=parse_int_list(args.epochs_grid),
        learning_rates=parse_float_list(args.alphas),
        batch_sizes=parse_int_list(args.batch_sizes),
        decision_modes=parse_str_list(args.decision_modes),
        min_action_rates=parse_float_list(args.min_action_rates),
    )
    selected = pick_trials(trials, args.max_trials, args.seed)
    results = run_walkforward_grid_search(
        featured,
        MARKET_FEATURE_COLUMNS,
        selected,
        objective=args.objective,
        seed=args.seed,
        suppress_inner_logs=not args.show_inner_logs,
        common_parameters={
            "price_col": args.price_col,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "oracle_fee_per_trade": args.oracle_fee_per_trade,
            "label_mode": args.label_mode,
            "position_mode": args.position_mode,
            "strategy_fee_per_trade": args.strategy_fee_per_trade,
            "initial_capital": args.capital,
            "do_dropout": args.do_dropout,
            "dropout_percent": args.dropout_percent,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        },
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = (
        args.output_csv or gridsearch_dir() / f"gridsearch_walkforward_{timestamp}.csv"
    ).resolve()
    output_json = (
        args.output_json
        or gridsearch_dir() / f"gridsearch_walkforward_{timestamp}.json"
    ).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    valid = results[results["status"] == "ok"].head(max(1, args.top_k))
    output_json.write_text(
        json.dumps(
            {
                "ticker": args.ticker,
                "objective": args.objective,
                "trials": len(results),
                "valid_trials": len(results[results["status"] == "ok"]),
                "top": valid.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    print(valid.to_string(index=False) if not valid.empty else "No valid trials.")
    print(f"csv={output_csv}\njson={output_json}")


if __name__ == "__main__":
    main()
