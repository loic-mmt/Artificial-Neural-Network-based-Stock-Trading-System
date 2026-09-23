"""Replay graph-ablation checkpoints for sealed-holdout information tests."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import hash_dataframe, _nullable_metadata
from trading_system.data.io import read_parquet_dataset
from trading_system.data.purged_cv import expanding_calendar_folds
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.graph_ablation import (
    GraphAblationConfig, MODES, _complete, _dataset, _dates, _graphs,
    _positions, _prepare, _scale,
)
from trading_system.experiments.graph_information import paired_graph_information
from trading_system.experiments.position_objectives import _align_position_calendar
from trading_system.experiments.runner import _filter_universe, _prepare_splits
from trading_system.models.multimodal_branches import GRUBranch, GNNBranch
from trading_system.models.neural.config import GRUConfig
from trading_system.models.neural.trainer import resolve_device
from trading_system.models.specs import ModelSelection
from trading_system.pipelines.compare_models import load_ticker_selection
from trading_system.pipelines.feature_arguments import apply_feature_sources
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel


def _checkpoint(path, torch):
    # Existing graph-ablation checkpoints contain NumPy scaler arrays. Allow
    # only their concrete NumPy types rather than unpickling arbitrary objects.
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:  # NumPy 1.x
        numpy_core = np.core
    allowed = [numpy_core.multiarray._reconstruct, np.ndarray, np.dtype,
               type(np.dtype("float32")), type(np.dtype("float64"))]
    from torch.serialization import safe_globals
    with safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def _outer_fold(fold_frame, config, ablation, fold):
    fold_config = replace(config, purged_split=fold["split"])
    prepared = _prepare(fold_frame, fold_config, ablation)
    raw_train, raw_val, raw_outer, _, columns = _prepare_splits(
        fold_frame, fold_config, include_test=True, fill_values=prepared.fills,
        fracdiff_transformer=prepared.fracdiff,
        feature_selector=prepared.selector,
        overfitting_selector=prepared.overfitting_selector,
        overfitting_supervised=False,
    )
    if columns != prepared.columns:
        raise ValueError("Replayed outer features differ from the frozen inner-training columns.")
    outer = _complete(raw_outer, config, prepared.tickers)
    history = pd.concat((raw_train, raw_val), ignore_index=True)
    return prepared, _scale(outer, columns, prepared.scaler), _scale(history, columns, prepared.scaler)


def _degrees(graphs, outer, tickers, mode):
    if mode in ("gru", "identity"):
        return np.zeros(len(outer), dtype=np.int64)
    dates = _dates(outer).unique().sort_values()
    if len(graphs) != len(dates) or any(graph.session != day for graph, day in zip(graphs, dates)):
        raise ValueError("Replayed graph dates do not match outer predictions.")
    return np.stack([np.bincount(graph.edge_index[1], minlength=len(tickers))
                     for graph in graphs]).reshape(-1)


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


def run_information_tests(
    run_dir, output_dir, data, ticker_selection, *,
    no_external_features=False, fundamentals=None, sentiment=None,
    modes=("identity", "sector", "rolling_pearson"), folds=None, seeds=None,
    device="auto", block_length=20, bootstrap_samples=1000,
    initial_train_fraction=.5, inner_val_fraction=.2,
    allow_hash_mismatch=False,
):
    """Replay outer-fold predictions only; never train or open final holdout."""
    import torch

    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output_dir}")
    report = json.loads((run_dir / "report.json").read_text())
    metadata = report["metadata"]
    if report.get("final_test") != [] or metadata.get("final_holdout_opened") is not False:
        raise ValueError("This diagnostic requires a sealed final holdout.")
    if any(mode not in MODES[1:] for mode in modes) or not modes or len(set(modes)) != len(modes):
        raise ValueError("Choose distinct GNN candidates from the graph-ablation run.")
    config_data = dict(metadata["config"])
    config_data["model"] = ModelSelection(**config_data["model"])
    config = ExperimentConfig(**config_data)
    loss = FinancialLossConfig(**metadata["loss_config"])
    ablation = GraphAblationConfig(**metadata["ablation"])
    source = read_parquet_dataset(data)
    selected = load_ticker_selection(ticker_selection)
    if set(selected) - set(source[config.group_col]):
        raise ValueError("Selected tickers are missing from the dataset.")
    source = source.loc[source[config.group_col].isin(selected)].copy()
    source, _ = apply_feature_sources(source, SimpleNamespace(
        fundamentals=fundamentals, sentiment=sentiment,
        no_external_features=no_external_features,
    ), config)
    work = _align_position_calendar(_filter_universe(source, config), config)
    replay_hash = hash_dataframe(work)
    if replay_hash != metadata["dataset_sha256"] and not allow_hash_mismatch:
        raise ValueError("Input data or feature-source choices differ from the benchmark.")
    saved = {(row["candidate"], row["fold"], row["seed"]): row for row in report["folds"]}
    if len(saved) != len(report["folds"]) or not all(row["status"] == "ok" for row in report["folds"]):
        raise ValueError("Graph-ablation rows are duplicate or incomplete.")
    cv_folds, final_split = expanding_calendar_folds(
        work, n_splits=metadata["n_splits"],
        initial_train_fraction=initial_train_fraction, inner_val_fraction=inner_val_fraction,
        final_test_fraction=1 - config.train_ratio - config.val_ratio,
        gap_bars=metadata["final_split"]["gap_bars"],
        embargo_bars=metadata["final_split"]["embargo_bars"], date_col=config.date_col,
    )
    if asdict(final_split) != metadata["final_split"]:
        raise ValueError("Replayed CV boundaries differ from benchmark metadata.")
    wanted_folds = set(range(metadata["n_splits"])) if folds is None else set(folds)
    wanted_seeds = set(metadata["seeds"]) if seeds is None else set(seeds)
    if not wanted_folds or not wanted_folds <= set(range(metadata["n_splits"])):
        raise ValueError("Requested folds are not in the benchmark.")
    if not wanted_seeds or not wanted_seeds <= set(metadata["seeds"]):
        raise ValueError("Requested seeds are not in the benchmark.")
    device = resolve_device(device, torch)
    predictions, statistics = [], []
    for fold in cv_folds:
        fold_id = fold["fold"]
        if fold_id not in wanted_folds:
            continue
        if pd.Timestamp(fold["end"]) >= pd.Timestamp(final_split.test_start):
            raise ValueError("A requested outer fold reaches the final holdout.")
        fold_frame = work.loc[_dates(work, config.date_col) <= pd.Timestamp(fold["end"])].copy()
        prepared, outer, history = _outer_fold(fold_frame, config, ablation, fold)
        high_volatility, volatility_threshold = _volatility_regime(
            fold_frame, outer, config, fold,
        )
        panel = ReturnPanel(outer, price_col=config.price_col, date_col=config.date_col,
                            group_col=config.group_col, execution_delay=config.execution_delay)
        per_mode = {}
        for mode in ("gru", *modes):
            graphs, _ = _graphs(fold_frame, config, prepared, ablation, mode, outer)
            dataset = _dataset(outer, history, prepared.columns, config, graphs)
            degree = _degrees(graphs, outer, prepared.tickers, mode)
            for seed in sorted(wanted_seeds):
                saved_row = saved.get((mode, fold_id, seed))
                if saved_row is None:
                    raise ValueError(f"Missing completed checkpoint: {mode} fold={fold_id} seed={seed}.")
                checkpoint = _checkpoint(run_dir / f"fold-{fold_id}-{mode}-seed-{seed}.pt", torch)
                if (tuple(checkpoint["feature_columns"]) != prepared.columns
                        or checkpoint["mode"] != mode or checkpoint["seed"] != seed
                        or not np.allclose(checkpoint["scaler_mean"], prepared.scaler.mean_)
                        or not np.allclose(checkpoint["scaler_scale"], prepared.scaler.scale_)):
                    raise ValueError(f"Checkpoint/preprocessing mismatch: {mode} fold={fold_id} seed={seed}.")
                if mode == "gru":
                    training = GRUConfig(**{**metadata["gru_parameters"], "seed": seed})
                    model = GRUBranch(len(prepared.columns), config.context_len, training).to(device)
                else:
                    model = GNNBranch(len(prepared.columns), hidden_size=ablation.gnn_hidden_size,
                                      num_layers=ablation.gnn_layers, dropout=ablation.gnn_dropout,
                                      graph_mode="identity" if mode == "identity" else "provided").to(device)
                model.load_state_dict(checkpoint["model_state"], strict=True)
                model.eval()
                with torch.no_grad():
                    positions = _positions(model, dataset, mode=mode, config=config,
                                           batch_dates=ablation.date_batch_size, torch=torch)
                replayed = panel.metrics(positions, loss, config.initial_capital)
                if any(not np.isclose(replayed[key], saved_row["outer_metrics"][key], atol=5e-4, rtol=5e-4)
                       for key in ("regularized_sharpe", "net_return", "max_drawdown")):
                    raise ValueError(f"Replayed metrics differ from run 06: {mode} fold={fold_id} seed={seed}.")
                current = pd.DataFrame({
                    "date": pd.to_datetime(outer[config.date_col], utc=True),
                    "ticker": outer[config.group_col].to_numpy(),
                    "adj_close": outer[config.price_col].to_numpy(dtype=np.float64),
                    "position": positions, "graph_degree": degree,
                    "high_volatility": high_volatility,
                    "candidate": mode, "fold": fold_id, "seed": seed,
                })
                predictions.append(current)
                per_mode[(mode, seed)] = current
                print(f"replayed {mode} seed={seed} fold={fold_id}", flush=True)
        for mode in modes:
            for seed in sorted(wanted_seeds):
                base = per_mode[("gru", seed)].rename(columns={"position": "gru_position"})
                graph = per_mode[(mode, seed)].rename(columns={
                    "position": "gnn_position", "graph_degree": "gnn_degree",
                })
                paired = base[["date", "ticker", "adj_close", "gru_position", "high_volatility"]].merge(
                    graph[["date", "ticker", "gnn_position", "gnn_degree"]],
                    on=["date", "ticker"], validate="one_to_one",
                )
                if len(paired) != len(base):
                    raise ValueError("GRU/GNN outer predictions are not date/ticker aligned.")
                result = paired_graph_information(
                    paired, loss, execution_delay=config.execution_delay,
                    block_length=block_length, samples=bootstrap_samples,
                    seed=seed + 1000 * fold_id + 10000 * MODES.index(mode),
                )
                statistics.append({"candidate": mode, "fold": fold_id, "seed": seed,
                                   "training_volatility_median": volatility_threshold, **result})
    output_dir.mkdir(parents=True)
    pd.concat(predictions, ignore_index=True).to_parquet(output_dir / "predictions.parquet", index=False)
    result = {"source_run": str(run_dir), "source_dataset_sha256": metadata["dataset_sha256"],
              "replay_dataset_sha256": replay_hash,
              "hash_mismatch_override": replay_hash != metadata["dataset_sha256"],
              "final_holdout_opened": False, "exploratory": True,
              "bootstrap_block_length": block_length, "bootstrap_samples": bootstrap_samples,
              "statistics": statistics}
    (output_dir / "statistics.json").write_text(json.dumps(_nullable_metadata(result), indent=2, allow_nan=False))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ticker-selection", type=Path, required=True)
    parser.add_argument("--no-external-features", action="store_true")
    parser.add_argument("--fundamentals", type=Path)
    parser.add_argument("--sentiment", type=Path)
    parser.add_argument("--modes", nargs="+", default=["identity", "sector", "rolling_pearson"],
                        choices=MODES[1:])
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--cv-initial-train-fraction", type=float, default=.5)
    parser.add_argument("--cv-inner-val-fraction", type=float, default=.2)
    parser.add_argument("--allow-hash-mismatch", action="store_true",
                        help="Cross-platform recovery only; every replayed checkpoint must still match saved metrics.")
    args = parser.parse_args(argv)
    return run_information_tests(
        args.run_dir, args.output_dir, args.data, args.ticker_selection,
        no_external_features=args.no_external_features,
        fundamentals=args.fundamentals, sentiment=args.sentiment,
        modes=tuple(args.modes), folds=args.folds, seeds=args.seeds,
        device=args.device, block_length=args.block_length,
        bootstrap_samples=args.bootstrap_samples,
        initial_train_fraction=args.cv_initial_train_fraction,
        inner_val_fraction=args.cv_inner_val_fraction,
        allow_hash_mismatch=args.allow_hash_mismatch,
    )


__all__ = ["main", "run_information_tests"]
