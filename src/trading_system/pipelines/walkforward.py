"""CLI orchestration for expanding-window model evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.walkforward import (
    fit_labeled_history,
    fit_position_history,
    predict_chunk_with_model,
    predict_position_chunk,
    walk_forward_classifier,
    walk_forward_oracle_ann,
)
from trading_system.artifacts.serialization import stable_config_hash
from trading_system.features.market import (
    MARKET_FEATURE_COLUMNS,
    compute_market_features,
)
from trading_system.labels.config import LabelConfig, normalize_label_method
from trading_system.pipelines.label_arguments import add_barrier_estimator_argument
from trading_system.pipelines.feature_arguments import add_feature_arguments, fracdiff_config_from_args, build_cli_features
from trading_system.pipelines.training_arguments import (
    add_weight_arguments, sample_weight_config_from_args,
    add_financial_loss_arguments, financial_loss_config_from_args,
)
from trading_system.paths import default_market_dataset_path
from trading_system.models.factory import create_default_model_registry
from trading_system.models.specs import ModelSelection
from trading_system.reporting.plots import format_experiment_summary
from trading_system.reporting.warnings import current_universe_warning
from trading_system.pipelines.overfitting_arguments import (
    add_overfitting_arguments,
    overfitting_config_from_args,
)

# Compatibility alias used by old grid-search code.
features = MARKET_FEATURE_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    model_names = create_default_model_registry().names()
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
        "--label-method",
        "--label-mode",
        dest="label_mode",
        choices=(
            "oracle_dp",
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
        "--label-window", "--breakout-window", dest="breakout_window", type=int, default=20
    )
    parser.add_argument("--label-buy-buffer", type=float, default=0.0)
    parser.add_argument("--label-sell-buffer", type=float, default=0.0)
    parser.add_argument(
        "--label-alternating",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--label-horizon",
        "--forward-horizon",
        "--label-max-holding",
        dest="label_horizon",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--label-buy-threshold",
        "--forward-buy-threshold",
        dest="forward_buy_threshold",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--label-sell-threshold",
        "--forward-sell-threshold",
        dest="forward_sell_threshold",
        type=float,
        default=0.002,
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
    parser.add_argument("--model", choices=model_names, default="manual_ann")
    parser.add_argument(
        "--model-config",
        help="JSON object containing parameters for the selected model.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    label_mode = normalize_label_method(args.label_mode)
    label_horizon = args.label_horizon
    if label_horizon is None:
        label_horizon = (
            10
            if label_mode in ("volatility_position", "triple_barrier")
            else 1
        )
    label_position_mode = args.label_position_mode.replace("-", "_")
    warning = current_universe_warning(args.data_dir)
    if warning:
        print(warning)
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
    featured, feature_columns, feature_config = build_cli_features(frame.sort_values("date").reset_index(drop=True), args)
    if args.model_config:
        try:
            model_parameters = json.loads(args.model_config)
        except json.JSONDecodeError as error:
            raise ValueError("--model-config must be valid JSON.") from error
        if not isinstance(model_parameters, dict):
            raise ValueError("--model-config must encode a JSON object.")
    elif args.model == "manual_ann":
        model_parameters = {
            "hidden_size": args.hidden,
            "learning_rate": args.alpha,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dropout_probability": args.dropout_percent if args.do_dropout else 0.0,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        }
    else:
        model_parameters = {}
    selection = ModelSelection(args.model, model_parameters)
    if label_mode == "breakout":
        label_config = LabelConfig.breakout(
            window=args.breakout_window,
            buy_buffer=args.label_buy_buffer,
            sell_buffer=args.label_sell_buffer,
            alternating=args.label_alternating,
        )
    elif label_mode == "forward_return":
        label_config = LabelConfig.forward_return(
            horizon=label_horizon,
            buy_threshold=args.forward_buy_threshold,
            sell_threshold=args.forward_sell_threshold,
        )
    elif label_mode == "volatility_position":
        label_config = LabelConfig.volatility_position(
            horizon=label_horizon,
            volatility_window=args.label_vol_window,
            long_threshold=args.label_long_threshold,
            short_threshold=args.label_short_threshold,
            exit_threshold=args.label_exit_threshold,
            min_holding_period=args.label_min_hold,
            cooldown=args.label_cooldown,
            cost_bps=args.label_cost_bps,
            position_mode=label_position_mode,
        )
    elif label_mode == "triple_barrier":
        label_config = LabelConfig.triple_barrier(
            max_holding=label_horizon,
            volatility_window=args.label_vol_window,
            volatility_estimator=args.label_volatility_estimator,
            profit_barrier=args.label_profit_barrier,
            stop_barrier=args.label_stop_barrier,
            event_filter=args.label_event_filter,
            cusum_threshold=args.label_cusum_threshold,
            cost_bps=args.label_cost_bps,
            between_event_policy=args.label_between_events,
        )
    else:
        label_config = None
    label_payload = (
        asdict(label_config)
        if label_config is not None
        else {"method": label_mode}
    )
    label_config_hash = stable_config_hash(label_payload)
    print(
        f"label_config={json.dumps(label_payload, sort_keys=True)} "
        f"label_config_hash={label_config_hash}"
    )
    result = walk_forward_classifier(
        financial_loss=financial_loss_config_from_args(args),
        overfitting_control=overfitting_config_from_args(args),
        expanded_min_coverage=feature_config.expanded_min_coverage if feature_config.feature_set == "expanded" else None,
        sample_weighting=sample_weight_config_from_args(args),
        fracdiff_config=fracdiff_config_from_args(args),
        full_df=featured,
        feature_columns=feature_columns,
        price_col=args.price_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        walkforward_step=args.walkforward_step,
        oracle_fee_per_trade=args.oracle_fee_per_trade,
        label_mode=label_mode,
        forward_horizon=label_horizon,
        forward_buy_threshold=args.forward_buy_threshold,
        forward_sell_threshold=args.forward_sell_threshold,
        breakout_window=args.breakout_window,
        breakout_buy_buffer=args.label_buy_buffer,
        breakout_sell_buffer=args.label_sell_buffer,
        breakout_alternating=args.label_alternating,
        volatility_horizon=label_horizon,
        volatility_window=args.label_vol_window,
        volatility_long_threshold=args.label_long_threshold,
        volatility_short_threshold=args.label_short_threshold,
        volatility_exit_threshold=args.label_exit_threshold,
        volatility_min_holding_period=args.label_min_hold,
        volatility_cooldown=args.label_cooldown,
        volatility_cost_bps=args.label_cost_bps,
        volatility_position_mode=label_position_mode,
        triple_barrier_max_holding=label_horizon,
        triple_barrier_volatility_window=args.label_vol_window,
        triple_barrier_volatility_estimator=args.label_volatility_estimator,
        triple_barrier_profit_barrier=args.label_profit_barrier,
        triple_barrier_stop_barrier=args.label_stop_barrier,
        triple_barrier_event_filter=args.label_event_filter,
        triple_barrier_cusum_threshold=args.label_cusum_threshold,
        triple_barrier_cost_bps=args.label_cost_bps,
        triple_barrier_between_event_policy=args.label_between_events,
        decision_mode=args.decision_mode,
        min_action_rate=args.min_action_rate,
        position_mode=args.position_mode,
        strategy_fee_per_trade=args.strategy_fee_per_trade,
        initial_capital=args.capital,
        context_len=args.context_len,
        model_selection=selection,
        seed=args.seed,
        device=args.device,
    )
    config_hash = stable_config_hash(
        {
            "model": selection,
            "seed": args.seed,
            "context_len": args.context_len,
            "label": label_payload,
            "features": asdict(feature_config),
            "feature_sources": featured.attrs.get("feature_sources"),
            "financial_loss": (
                result["financial_loss"]
                if result["loss_objective"] != "cross_entropy" else None
            ),
            "overfitting_control": (
                asdict(overfitting_config_from_args(args))
                if overfitting_config_from_args(args) else None
            ),
        }
    )
    print(
        f"model={selection.name} config_hash={config_hash} "
        f"coverage={result['n_eval_rows']}/{result['n_test_rows']} "
        f"retrains={len(result['retrain_logs'])}"
    )
    if result["loss_objective"] == "cross_entropy":
        print(format_experiment_summary(result["test_metrics"], result["benchmark_comparison"]))
    else:
        print(json.dumps(result["benchmark_comparison"], indent=2, sort_keys=True))


__all__ = [
    "build_parser",
    "fit_labeled_history",
    "fit_position_history",
    "main",
    "predict_chunk_with_model",
    "predict_position_chunk",
    "walk_forward_classifier",
    "walk_forward_oracle_ann",
]


if __name__ == "__main__":
    main()
