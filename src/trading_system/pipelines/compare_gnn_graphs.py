"""CLI for the standalone GNN graph controls and matched GRU reference."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.graph_ablation import GraphAblationConfig, run_graph_ablation
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
    parser.description = "Matched GRU versus independent GNN graph controls; final holdout remains sealed."
    parser.set_defaults(models=["gru"], losses=["sharpe"])
    parser.add_argument("--graph-lookback", type=int, default=252)
    parser.add_argument("--graph-threshold", type=float, default=0.7)
    parser.add_argument("--graph-weight-mode", choices=("positive", "absolute"), default="positive")
    parser.add_argument("--gnn-hidden-size", type=int, default=32)
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--gnn-dropout", type=float, default=0.0)
    parser.add_argument("--date-batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Resume matching completed folds in --output-dir.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.models != ["gru"]:
        raise ValueError("Graph ablation requires exactly --models gru.")
    if args.cv_folds is None or args.cv_folds < 2:
        raise ValueError("Graph ablation requires --cv-folds >= 2.")
    if args.cv_score not in (None, "regularized_sharpe"):
        raise ValueError("Graph ablation selects by regularized_sharpe.")
    if args.selection_metric != "regularized_sharpe":
        raise ValueError("Graph ablation selects by regularized_sharpe.")
    if args.cv_final_test or args.final_test:
        raise ValueError("Graph ablation keeps final holdout sealed.")
    if args.no_run_artifacts:
        raise ValueError("Graph ablation requires run artifacts for graph provenance and resume.")
    if args.position_gate_quantiles is not None or args.position_gate_min_coverage is not None:
        raise ValueError("Position gate is a separate ablation.")
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
    ablation = GraphAblationConfig(
        graph_lookback=args.graph_lookback, graph_threshold=args.graph_threshold,
        graph_weight_mode=args.graph_weight_mode, gnn_hidden_size=args.gnn_hidden_size,
        gnn_layers=args.gnn_layers, gnn_dropout=args.gnn_dropout,
        date_batch_size=args.date_batch_size,
    )
    target = args.output_dir or comparisons_dir() / ("gnn-graphs-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    return run_graph_ablation(
        frame, config, losses[0], args.model_parameter_sets["gru"][0], args.seeds,
        target, ablation=ablation, n_splits=args.cv_folds,
        initial_train_fraction=args.cv_initial_train_fraction,
        inner_val_fraction=args.cv_inner_val_fraction,
        gap_bars=args.cv_gap_bars, embargo_bars=args.cv_embargo_bars,
        dataset_path=args.data, resume=args.resume,
    )


__all__ = ["build_parser", "main"]
