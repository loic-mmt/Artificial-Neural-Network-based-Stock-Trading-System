"""CLI configuration for walk-forward hyperparameter search."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.search import (
    make_model_walkforward_trial_grid,
    make_walkforward_trial_grid,
    pick_trials,
    run_walkforward_grid_search,
)
from trading_system.models.factory import create_default_model_registry
from trading_system.features.market import (
    MARKET_FEATURE_COLUMNS,
    compute_market_features,
)
from trading_system.labels.config import normalize_label_method
from trading_system.pipelines.label_arguments import add_barrier_estimator_argument
from trading_system.pipelines.feature_arguments import add_feature_arguments, fracdiff_config_from_args, build_cli_features
from trading_system.pipelines.training_arguments import (
    add_weight_arguments, sample_weight_config_from_args,
    add_financial_loss_arguments, financial_loss_config_from_args,
)
from trading_system.paths import default_market_dataset_path, gridsearch_dir
from trading_system.reporting.warnings import current_universe_warning
from trading_system.pipelines.overfitting_arguments import (
    add_overfitting_arguments,
    overfitting_config_from_args,
)


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid search over expanding-window experiments."
    )
    parser.add_argument("--data-dir", type=Path, default=default_market_dataset_path())
    parser.add_argument("--ticker", default="EN.PA")
    parser.add_argument("--price-col", default="adj_close")
    parser.add_argument(
        "--objective",
        choices=("outperformance", "model_pnl", "macro_f1", "bal_acc", "net_pnl", "net_return", "regularized_sharpe"),
        default="outperformance",
    )
    parser.add_argument(
        "--label-method",
        "--label-mode",
        dest="label_mode",
        choices=(
            "forward_return",
            "forward-return",
            "breakout",
            "triple_barrier",
            "triple-barrier",
            "volatility_position",
            "volatility-position",
        ),
        default="forward-return",
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
    parser.add_argument(
        "--label-horizons",
        "--forward-horizons",
        dest="forward_horizons",
        default="1,3,5,10",
    )
    parser.add_argument(
        "--label-buy-thresholds",
        "--forward-buy-thresholds",
        dest="forward_buy_thresholds",
        default="0.001,0.002,0.005",
    )
    parser.add_argument(
        "--label-sell-thresholds",
        "--forward-sell-thresholds",
        dest="forward_sell_thresholds",
        default="0.001,0.002,0.005",
    )
    parser.add_argument("--label-vol-window", type=int, default=20)
    add_barrier_estimator_argument(parser)
    add_feature_arguments(parser)
    add_weight_arguments(parser)
    add_financial_loss_arguments(parser)
    add_overfitting_arguments(parser)
    parser.add_argument("--label-long-threshold", type=float, default=1.0)
    parser.add_argument("--label-short-threshold", type=float, default=1.5)
    parser.add_argument("--label-exit-threshold", type=float, default=0.25)
    parser.add_argument("--label-min-hold", type=int, default=5)
    parser.add_argument("--label-cooldown", type=int, default=0)
    parser.add_argument("--label-cost-bps", type=float, default=5.0)
    parser.add_argument(
        "--label-position-mode",
        choices=("long_flat", "long_short", "long-flat", "long-short"),
        default="long_flat",
    )
    parser.add_argument("--label-profit-barrier", type=float, default=1.0)
    parser.add_argument("--label-stop-barrier", type=float, default=1.0)
    parser.add_argument(
        "--label-event-filter", choices=("all", "cusum"), default="all"
    )
    parser.add_argument("--label-cusum-threshold", type=float, default=0.5)
    parser.add_argument(
        "--label-between-events",
        choices=("hold", "flat", "carry"),
        default="hold",
    )
    parser.add_argument("--context-lengths", default="10,20,40")
    parser.add_argument("--walkforward-steps", default="30,60")
    parser.add_argument("--hiddens", default="64,128")
    parser.add_argument("--epochs-grid", default="300")
    parser.add_argument("--alphas", default="0.001")
    parser.add_argument("--batch-sizes", default="64")
    parser.add_argument(
        "--models",
        default="manual_ann",
        help="Comma-separated registry model names.",
    )
    parser.add_argument(
        "--model-parameter-spaces",
        help=(
            "JSON mapping model names to equally sized lists of parameter objects."
        ),
    )
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
    args = build_parser().parse_args()
    label_mode = normalize_label_method(args.label_mode)
    label_position_mode = args.label_position_mode.replace("-", "_")
    warning = current_universe_warning(args.data_dir)
    if warning:
        print(warning)
    frame = read_parquet_dataset(args.data_dir)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    if args.ticker:
        if "ticker" not in frame.columns:
            raise ValueError("Ticker filtering requires ticker column.")
        frame = frame[frame["ticker"] == args.ticker].copy()
    if frame.empty:
        raise ValueError("No rows remain after ticker filtering.")
    featured, feature_columns, feature_config = build_cli_features(frame.sort_values("date").reset_index(drop=True), args)
    financial_loss = financial_loss_config_from_args(args)
    model_names = parse_str_list(args.models)
    available = set(create_default_model_registry().names())
    unknown = sorted(set(model_names) - available)
    if not model_names or unknown:
        raise ValueError(f"Invalid model list; unknown={unknown}, available={sorted(available)}")
    neutral = args.model_parameter_spaces is not None or model_names != ["manual_ann"]
    forward_buy_thresholds = (
        [0.0]
        if label_mode in ("volatility_position", "triple_barrier")
        else parse_float_list(args.forward_buy_thresholds)
    )
    forward_sell_thresholds = (
        [0.0]
        if label_mode in ("volatility_position", "triple_barrier")
        else parse_float_list(args.forward_sell_thresholds)
    )
    decision_modes = parse_str_list(args.decision_modes)
    action_rates = parse_float_list(args.min_action_rates)
    if financial_loss is not None:
        # Labels, thresholds and discrete decision calibration do not train a
        # direct position objective. Keep one declared diagnostic label setup
        # instead of multiplying equivalent financial trials.
        forward_buy_thresholds = forward_buy_thresholds[:1]
        forward_sell_thresholds = forward_sell_thresholds[:1]
        forward_horizons = parse_int_list(args.forward_horizons)[:1]
        decision_modes, action_rates = ["argmax"], [0.0]
    else:
        forward_horizons = parse_int_list(args.forward_horizons)
    if neutral:
        if args.model_parameter_spaces:
            try:
                spaces = json.loads(args.model_parameter_spaces)
            except json.JSONDecodeError as error:
                raise ValueError("--model-parameter-spaces must be valid JSON.") from error
            if not isinstance(spaces, dict):
                raise ValueError("--model-parameter-spaces must encode an object.")
        else:
            spaces = {name: [{}] for name in model_names}
        if set(spaces) != set(model_names):
            raise ValueError("Model parameter spaces must exactly match --models.")
        trials = make_model_walkforward_trial_grid(
            model_parameter_spaces=spaces,
            forward_horizons=forward_horizons,
            forward_buy_thresholds=forward_buy_thresholds,
            forward_sell_thresholds=forward_sell_thresholds,
            context_lengths=parse_int_list(args.context_lengths),
            walkforward_steps=parse_int_list(args.walkforward_steps),
            decision_modes=decision_modes,
            min_action_rates=action_rates,
        )
        selected = list(trials)
        if args.max_trials is not None and args.max_trials < len(trials):
            if args.max_trials % len(model_names):
                raise ValueError("--max-trials must be divisible by model count.")
            per_model = args.max_trials // len(model_names)
            selected = []
            for name in sorted(model_names):
                model_trials = [trial for trial in trials if trial.model.name == name]
                selected.extend(pick_trials(model_trials, per_model, args.seed))
    else:
        trials = make_walkforward_trial_grid(
            forward_horizons=forward_horizons,
            forward_buy_thresholds=forward_buy_thresholds,
            forward_sell_thresholds=forward_sell_thresholds,
            context_lengths=parse_int_list(args.context_lengths),
            walkforward_steps=parse_int_list(args.walkforward_steps),
            hidden_sizes=parse_int_list(args.hiddens),
            epochs=parse_int_list(args.epochs_grid),
            learning_rates=parse_float_list(args.alphas),
            batch_sizes=parse_int_list(args.batch_sizes),
            decision_modes=decision_modes,
            min_action_rates=action_rates,
        )
        selected = pick_trials(trials, args.max_trials, args.seed)
    common_parameters = {
        "financial_loss": financial_loss,
        "expanded_min_coverage": feature_config.expanded_min_coverage if feature_config.feature_set == "expanded" else None,
        "sample_weighting": sample_weight_config_from_args(args),
        "fracdiff_config": fracdiff_config_from_args(args),
        "price_col": args.price_col,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "oracle_fee_per_trade": args.oracle_fee_per_trade,
        "label_mode": label_mode,
        "position_mode": args.position_mode,
        "strategy_fee_per_trade": args.strategy_fee_per_trade,
        "initial_capital": args.capital,
    }
    if common_parameters["financial_loss"] is not None:
        if common_parameters["sample_weighting"] is not None:
            raise ValueError("Sample weighting cannot be combined with pnl/sharpe loss.")
        if args.objective in ("macro_f1", "bal_acc"):
            raise ValueError("Financial loss requires a financial grid-search objective.")
    if label_mode == "volatility_position":
        common_parameters.update(
            volatility_window=args.label_vol_window,
            volatility_long_threshold=args.label_long_threshold,
            volatility_short_threshold=args.label_short_threshold,
            volatility_exit_threshold=args.label_exit_threshold,
            volatility_min_holding_period=args.label_min_hold,
            volatility_cooldown=args.label_cooldown,
            volatility_cost_bps=args.label_cost_bps,
            volatility_position_mode=label_position_mode,
        )
    elif label_mode == "triple_barrier":
        common_parameters.update(
            triple_barrier_volatility_window=args.label_vol_window,
            triple_barrier_volatility_estimator=args.label_volatility_estimator,
            triple_barrier_profit_barrier=args.label_profit_barrier,
            triple_barrier_stop_barrier=args.label_stop_barrier,
            triple_barrier_event_filter=args.label_event_filter,
            triple_barrier_cusum_threshold=args.label_cusum_threshold,
            triple_barrier_cost_bps=args.label_cost_bps,
            triple_barrier_between_event_policy=args.label_between_events,
        )
    if not neutral:
        common_parameters.update(
            do_dropout=args.do_dropout,
            dropout_percent=args.dropout_percent,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        )
    common_parameters["overfitting_control"] = overfitting_config_from_args(args)
    results = run_walkforward_grid_search(
        featured,
        feature_columns,
        selected,
        objective=args.objective,
        seed=args.seed,
        suppress_inner_logs=not args.show_inner_logs,
        common_parameters=common_parameters,
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
                "feature_sources": featured.attrs.get("feature_sources"),
                "objective": args.objective,
                "loss_objective": args.loss_objective,
                "financial_loss": (
                    asdict(common_parameters["financial_loss"])
                    if common_parameters["financial_loss"] is not None else None
                ),
                "overfitting_control": (
                    asdict(common_parameters["overfitting_control"])
                    if common_parameters["overfitting_control"] is not None else None
                ),
                "label_method": label_mode,
                "selection_split": "validation",
                "trials": len(results),
                "valid_trials": len(results[results["status"] == "ok"]),
                "models": model_names,
                "top": valid.to_dict(orient="records"),
                "failures": results[results["status"] == "error"].to_dict(
                    orient="records"
                ),
                "best_parameters": results.attrs.get("best_parameters"),
                "validation_retrain_logs": results.attrs["validation_retrain_logs"],
                "final_test": results.attrs.get("final_test"),
            },
            indent=2,
        )
    )
    print("Validation ranking:")
    print(valid.to_string(index=False) if not valid.empty else "No valid trials.")
    if "final_test" in results.attrs:
        print("Frozen winner — final test:")
        print(json.dumps(results.attrs["final_test"], indent=2))
    print(f"csv={output_csv}\njson={output_json}")


if __name__ == "__main__":
    main()
