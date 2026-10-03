"""Replay all graph controls, preserving a sealed final holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from trading_system.artifacts.experiment import _nullable_metadata
from trading_system.artifacts.multimodal_study import atomic_write_json
from trading_system.data.io import read_parquet_dataset
from trading_system.data.purged_cv import PurgedSplit
from trading_system.experiments.graph_ablation import MODES, _dates
from trading_system.experiments.graph_information import paired_graph_information
from trading_system.experiments.graph_replay import config_from_metadata, replay_graph_ablation
from trading_system.experiments.market_gru_ablation import build_close_market_frame
from trading_system.experiments.position_objectives import _align_position_calendar
from trading_system.experiments.runner import _filter_universe
from trading_system.pipelines.compare_models import load_ticker_selection
from trading_system.pipelines.feature_arguments import apply_feature_sources
from trading_system.training.financial_loss import FinancialLossConfig


def _volatility_regime(fold_frame, outer, config, fold, *, window=20):
    """Current/past-only market volatility; threshold fitted before inner validation."""
    raw = fold_frame[[config.date_col, config.group_col, config.price_col]].copy()
    raw[config.date_col] = _dates(raw, config.date_col)
    prices = raw.pivot(index=config.date_col, columns=config.group_col,
                       values=config.price_col).sort_index()
    market_return = prices.pct_change(fill_method=None).mean(axis=1)
    volatility = market_return.rolling(window, min_periods=window).std()
    train_volatility = volatility.loc[volatility.index < pd.Timestamp(fold["split"].validation_start)].dropna()
    if train_volatility.empty:
        raise ValueError("Not enough pre-validation dates to fit the volatility-regime threshold.")
    threshold = float(train_volatility.median())
    outer_volatility = volatility.reindex(_dates(outer, config.date_col))
    if outer_volatility.isna().any():
        raise ValueError("Outer volatility regime has unavailable past returns.")
    return outer_volatility.to_numpy() > threshold, threshold


def _replay_inputs(metadata, data, ticker_selection, *, no_external_features=False,
                   fundamentals=None, sentiment=None, market_context_data=None,
                   market_frame_data=None):
    config = config_from_metadata(metadata)
    source = read_parquet_dataset(data)
    selected = load_ticker_selection(ticker_selection)
    if set(selected) - set(source[config.group_col]):
        raise ValueError("Selected tickers are missing from the dataset.")
    source = source.loc[source[config.group_col].isin(selected)].copy()
    source, _ = apply_feature_sources(source, SimpleNamespace(
        fundamentals=fundamentals, sentiment=sentiment,
        no_external_features=no_external_features,
    ), config)
    context = None
    context_path = market_context_data or metadata.get("graph_context_path") or metadata.get("market_context_path")
    if context_path:
        context = read_parquet_dataset(context_path)
    market = None
    if market_frame_data:
        market = read_parquet_dataset(market_frame_data)
    elif metadata.get("market_context_sha256") is not None:
        audit = metadata.get("market_audit")
        if not audit or not audit.get("close_columns") or "cross_sectional_features" not in audit:
            raise ValueError("Market replay requires saved feature construction metadata or explicit --market-frame-data.")
        windows = {int(name.rsplit("_", 1)[1]) for name in audit["features"]
                   if name.startswith("broad_realized_vol_")}
        if len(windows) > 1:
            raise ValueError("Saved market realized-volatility windows are ambiguous.")
        # Without cross-sectional features this argument is unused; otherwise
        # the actual window is encoded in the saved derived feature name.
        if audit["cross_sectional_features"] and not windows:
            has_broad = any(name in audit["features"] for name in ("spy_close_ret_1", "market_close_ret_1"))
            if has_broad:
                raise ValueError("Saved market realized-volatility construction metadata is missing.")
        market, _ = build_close_market_frame(
            source, config.date_col, tuple(audit["close_columns"]), context_frame=context,
            include_cross_section=audit["cross_sectional_features"],
            realized_vol_window=next(iter(windows)) if windows else 20,
            price_col=config.price_col, ticker_col=config.group_col,
        )
    return config, source, context, market


def run_information_tests(
    run_dir, output_dir, data, ticker_selection, *,
    no_external_features=False, fundamentals=None, sentiment=None,
    modes=None, folds=None, seeds=None, device=None, block_length=20,
    bootstrap_samples=1000, initial_train_fraction=None, inner_val_fraction=None,
    gap_bars=None, embargo_bars=None, allow_hash_mismatch=False,
    allow_provenance_mismatch=False, market_context_data=None,
    market_frame_data=None, replay_only=False, restore_checkpoint_feature_order=False,
):
    """Export inner/outer logits and daily paths; test outer information only.

    Legacy CV fractions must be explicit. Market, top-k and residual graph
    candidates use the exact same model and dataset builders as training.
    """
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output_dir}")
    report = json.loads((run_dir / "report.json").read_text())
    metadata = report["metadata"]
    config, source, context, market = _replay_inputs(
        metadata, data, ticker_selection, no_external_features=no_external_features,
        fundamentals=fundamentals, sentiment=sentiment,
        market_context_data=market_context_data, market_frame_data=market_frame_data,
    )
    available = tuple(metadata["ablation"].get("candidates", MODES))
    if replay_only:
        candidates = available if modes is None else tuple(modes)
        if not candidates or len(set(candidates)) != len(candidates) or not set(candidates) <= set(available):
            raise ValueError("Choose distinct candidates present in the graph-ablation run.")
    else:
        modes = tuple(mode for mode in available if mode != "gru") if modes is None else tuple(modes)
        if ("gru" not in available or not modes or len(set(modes)) != len(modes)
                or "gru" in modes or not set(modes) <= set(available)):
            raise ValueError("Information tests require GRU and distinct non-reference candidates in the source run.")
        candidates = ("gru", *modes)
    result = replay_graph_ablation(
        run_dir, output_dir, source, graph_context=context, market_frame=market,
        candidates=candidates, folds=folds, seeds=seeds, device=device,
        initial_train_fraction=initial_train_fraction, inner_val_fraction=inner_val_fraction,
        gap_bars=gap_bars, embargo_bars=embargo_bars,
        allow_hash_mismatch=allow_hash_mismatch,
        allow_provenance_mismatch=allow_provenance_mismatch,
        restore_checkpoint_feature_order=restore_checkpoint_feature_order,
    )
    if replay_only:
        return result
    loss = FinancialLossConfig(**metadata["loss_config"])
    work = _align_position_calendar(_filter_universe(source, config), config)
    statistics = []
    for fold in result["cv_folds"]:
        fold_id = fold["fold"]
        tasks = [task for task in result["tasks"] if task["fold"] == fold_id]
        if not tasks:
            continue
        fold_frame = work.loc[_dates(work, config.date_col) <= pd.Timestamp(fold["end"])].copy()
        parsed_fold = {**fold, "split": PurgedSplit(**fold["split"])}
        for seed in sorted({task["seed"] for task in tasks}):
            group = pd.read_parquet(output_dir / "predictions.parquet", filters=[
                ("partition", "=", "outer"), ("fold", "=", fold_id), ("seed", "=", seed),
            ])
            base = group.loc[group.candidate.eq("gru")].copy()
            high_volatility, threshold = _volatility_regime(
                fold_frame, base.rename(columns={"date": config.date_col}), config, parsed_fold,
            )
            base["high_volatility"] = high_volatility
            base = base.rename(columns={"position": "gru_position"})
            for mode_index, mode in enumerate(modes):
                graph = group.loc[group.candidate.eq(mode)].rename(columns={
                    "position": "gnn_position", "graph_degree": "gnn_degree",
                })
                paired = base[["date", "ticker", "adj_close", "gru_position", "high_volatility"]].merge(
                    graph[["date", "ticker", "gnn_position", "gnn_degree"]],
                    on=["date", "ticker"], validate="one_to_one",
                )
                if len(paired) != len(base):
                    raise ValueError("GRU/candidate outer predictions are not ticker/date aligned.")
                values = paired_graph_information(
                    paired, loss, execution_delay=config.execution_delay,
                    block_length=block_length, samples=bootstrap_samples,
                    seed=seed + 1000 * fold_id + 10000 * mode_index,
                )
                statistics.append({"candidate": mode, "fold": fold_id, "seed": seed,
                                   "training_volatility_median": threshold, **values})
    result = {**result, "bootstrap_block_length": block_length,
              "bootstrap_samples": bootstrap_samples, "statistics": statistics}
    atomic_write_json(output_dir / "statistics.json", _nullable_metadata(result))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ticker-selection", type=Path, required=True)
    parser.add_argument("--no-external-features", action="store_true")
    parser.add_argument("--fundamentals", type=Path)
    parser.add_argument("--sentiment", type=Path)
    parser.add_argument("--market-context-data", type=Path)
    parser.add_argument("--market-frame-data", type=Path,
                        help="Already-constructed market frame when legacy construction provenance is missing.")
    parser.add_argument("--modes", nargs="+", help="Defaults to every non-reference candidate in the source run.")
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                        help="Defaults to the original training device; changing it requires an exploratory provenance override.")
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--cv-initial-train-fraction", type=float)
    parser.add_argument("--cv-inner-val-fraction", type=float)
    parser.add_argument("--cv-gap-bars", type=int)
    parser.add_argument("--cv-embargo-bars", type=int)
    parser.add_argument("--replay-only", action="store_true", help="Export logits and daily paths without information tests.")
    parser.add_argument("--allow-hash-mismatch", action="store_true",
                        help="Exploratory export only; checkpoints must still reproduce every saved metric.")
    parser.add_argument("--allow-provenance-mismatch", action="store_true",
                        help="Exploratory export only; source/runtime mismatch prevents reuse certification.")
    parser.add_argument("--restore-checkpoint-feature-order", action="store_true",
                        help="Legacy diagnostic only: restore recorded order for the identical feature set; never skip scaler checks.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_information_tests(
        args.run_dir, args.output_dir, args.data, args.ticker_selection,
        no_external_features=args.no_external_features,
        fundamentals=args.fundamentals, sentiment=args.sentiment,
        modes=args.modes, folds=args.folds, seeds=args.seeds,
        device=args.device, block_length=args.block_length,
        bootstrap_samples=args.bootstrap_samples,
        initial_train_fraction=args.cv_initial_train_fraction,
        inner_val_fraction=args.cv_inner_val_fraction,
        gap_bars=args.cv_gap_bars, embargo_bars=args.cv_embargo_bars,
        allow_hash_mismatch=args.allow_hash_mismatch,
        allow_provenance_mismatch=args.allow_provenance_mismatch,
        market_context_data=args.market_context_data,
        market_frame_data=args.market_frame_data, replay_only=args.replay_only,
        restore_checkpoint_feature_order=args.restore_checkpoint_feature_order,
    )


__all__ = ["build_parser", "main", "run_information_tests"]
