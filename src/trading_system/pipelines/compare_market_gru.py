"""CLI for matched GRU market-context and gate controls."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.market_gru_ablation import MarketAblationConfig, run_market_gru_ablation
from trading_system.paths import comparisons_dir
from trading_system.pipelines.compare_losses import (
    PRESETS, _loss_configs, build_parser as loss_parser, load_ticker_selection,
)
from trading_system.pipelines.feature_arguments import apply_feature_arguments, apply_feature_sources
from trading_system.pipelines.label_arguments import apply_label_arguments
from trading_system.pipelines.overfitting_arguments import overfitting_config_from_args
from trading_system.pipelines.training_arguments import apply_weight_arguments


def build_parser():
    parser = loss_parser()
    parser.description = "Matched GRU market-context/gating ablations; final holdout sealed."
    parser.set_defaults(models=["gru"], losses=["sharpe"])
    parser.add_argument("--market-close-columns", default="market_close,vix_close",
                        help="Comma-separated close-observed global series; no publication-dated macro inputs.")
    parser.add_argument("--market-transformer-width", type=int, default=32)
    parser.add_argument("--market-transformer-heads", type=int, default=4)
    parser.add_argument("--market-transformer-layers", type=int, default=1)
    parser.add_argument("--date-batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.models != ["gru"] or args.cv_folds is None or args.cv_folds < 2:
        raise ValueError("Choose --models gru and --cv-folds >= 2.")
    if args.cv_score not in (None, "regularized_sharpe") or args.selection_metric != "regularized_sharpe":
        raise ValueError("Market ablation selects by regularized_sharpe.")
    if args.cv_final_test or args.final_test or args.no_run_artifacts:
        raise ValueError("Market ablation requires artifacts and keeps final holdout sealed.")
    if args.position_gate_quantiles is not None or args.position_gate_min_coverage is not None:
        raise ValueError("Position gating is a separate ablation.")
    losses = _loss_configs(args)
    if len(losses) != 1 or losses[0].objective not in ("sharpe", "combined"):
        raise ValueError("Choose one financial loss: sharpe or one combined weight.")
    if args.model_parameter_sets is None or set(args.model_parameter_sets) != {"gru"} or len(args.model_parameter_sets["gru"]) != 1:
        raise ValueError("Supply one frozen GRU configuration via --model-parameter-sets.")
    config = replace(PRESETS[args.preset], device=args.device)
    config = apply_weight_arguments(apply_feature_arguments(apply_label_arguments(config, args), args), args)
    config = replace(config, overfitting_control=overfitting_config_from_args(args))
    config = replace(config, **{name: getattr(args, name) for name in
                               ("context_len", "train_ratio", "val_ratio", "position_mode", "execution_delay")
                               if getattr(args, name) is not None})
    frame = read_parquet_dataset(args.data)
    if args.ticker_selection:
        selected = load_ticker_selection(args.ticker_selection)
        missing = set(selected) - set(frame[config.group_col])
        if missing:
            raise ValueError(f"Selected tickers missing from dataset: {sorted(missing)}")
        frame = frame.loc[frame[config.group_col].isin(selected)].copy()
    frame, _ = apply_feature_sources(frame, args, config)
    ablation = MarketAblationConfig(
        close_columns=tuple(item.strip() for item in args.market_close_columns.split(",")),
        date_batch_size=args.date_batch_size,
        transformer_width=args.market_transformer_width,
        transformer_heads=args.market_transformer_heads,
        transformer_layers=args.market_transformer_layers,
    )
    target = args.output_dir or comparisons_dir() / ("market-gru-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    return run_market_gru_ablation(
        frame, config, losses[0], args.model_parameter_sets["gru"][0], args.seeds,
        target, ablation=ablation, n_splits=args.cv_folds,
        initial_train_fraction=args.cv_initial_train_fraction,
        inner_val_fraction=args.cv_inner_val_fraction,
        gap_bars=args.cv_gap_bars, embargo_bars=args.cv_embargo_bars,
        dataset_path=args.data, resume=args.resume,
    )


__all__ = ["build_parser", "main"]
