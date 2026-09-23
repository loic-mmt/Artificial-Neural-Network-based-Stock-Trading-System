"""Matched, sealed-holdout GRU versus independent GNN graph ablation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import _nullable_metadata, hash_dataframe
from trading_system.data.causal_graphs import GraphBuildConfig, build_graph_snapshots, graph_diagnostics
from trading_system.data.multimodal import build_multimodal_dataset
from trading_system.data.purged_cv import expanding_calendar_folds
from trading_system.data.scaling import Standardizer
from trading_system.evaluation.classification import evaluate_predictions
from trading_system.features.expanded import ExpandedFeatureSelector
from trading_system.features.fracdiff import FracDiffTransformer
from trading_system.models.multimodal_branches import GRUBranch, GNNBranch
from trading_system.models.neural.config import GRUConfig
from trading_system.models.neural.trainer import resolve_device, seed_torch_run
from trading_system.models.specs import ModelSelection
from trading_system.reporting.warnings import current_universe_warning
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel, position_coefficients
from trading_system.training.overfitting import TrainOnlyFeatureSelector
from .position_objectives import _align_position_calendar
from .runner import _filter_universe, _prepare_splits


MODES = ("gru", "identity", "sector", "train_pearson", "rolling_pearson")


@dataclass(frozen=True)
class GraphAblationConfig:
    graph_lookback: int = 252
    graph_threshold: float = 0.7
    graph_weight_mode: str = "positive"
    gnn_hidden_size: int = 32
    gnn_layers: int = 1
    gnn_dropout: float = 0.0
    date_batch_size: int = 32

    def __post_init__(self) -> None:
        GraphBuildConfig("train_pearson", self.graph_lookback,
                         self.graph_threshold, self.graph_weight_mode)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (self.gnn_hidden_size, self.gnn_layers, self.date_batch_size)):
            raise ValueError("GNN width, layers and date batch size must be positive integers.")
        if not np.isfinite(self.gnn_dropout) or not 0 <= self.gnn_dropout < 1:
            raise ValueError("gnn_dropout must be in [0, 1).")


def _dates(frame, date_col):
    return pd.DatetimeIndex(pd.to_datetime(frame[date_col], utc=True)).normalize()


def _complete(frame, config, tickers):
    if set(frame[config.group_col]) != set(tickers):
        raise ValueError("A prepared split lost one or more selected tickers.")
    return _align_position_calendar(frame, config).sort_values(
        [config.date_col, config.group_col]
    ).reset_index(drop=True)


def _scale(frame, columns, scaler):
    result = frame.copy()
    result.loc[:, list(columns)] = scaler.transform(result.loc[:, list(columns)].to_numpy(dtype=np.float32))
    return result


@dataclass
class _Prepared:
    train: pd.DataFrame
    validation: pd.DataFrame
    history_train: pd.DataFrame
    history_validation: pd.DataFrame
    columns: tuple[str, ...]
    scaler: Standardizer
    fills: pd.Series
    selector: ExpandedFeatureSelector | None
    overfitting_selector: TrainOnlyFeatureSelector | None
    fracdiff: FracDiffTransformer | None
    purging: dict | None
    tickers: tuple[str, ...]


def _prepare(frame, config, ablation):
    tickers = tuple(sorted(frame[config.group_col].unique()))
    selector = ExpandedFeatureSelector(config.expanded_min_coverage) if config.feature_set == "expanded" else None
    overfitting = TrainOnlyFeatureSelector(config.overfitting_control) if config.overfitting_control else None
    fracdiff = (FracDiffTransformer(config.fracdiff, price_col=config.price_col,
                                   date_col=config.date_col, group_col=config.group_col)
                if config.fracdiff else None)
    train, val, _, fills, columns = _prepare_splits(
        frame, config, feature_selector=selector, overfitting_selector=overfitting,
        fracdiff_transformer=fracdiff, overfitting_supervised=False,
    )
    purging = train.attrs.get("purging")
    all_dates = _dates(train, config.date_col).unique().sort_values()
    warmup = max(config.context_len - 1, ablation.graph_lookback)
    if len(all_dates) <= warmup + config.execution_delay + 4:
        raise ValueError("Training fold is too short after shared graph/window warmup.")
    start = all_dates[warmup]
    training = train.loc[_dates(train, config.date_col) > start].copy()
    if "_cv_gap" in training:
        training = training.loc[~training["_cv_gap"]].copy()
    validation = val.loc[~val["_cv_gap"]].copy() if "_cv_gap" in val else val.copy()
    training, validation = _complete(training, config, tickers), _complete(validation, config, tickers)
    first_train = _dates(training, config.date_col).min()
    history_train = train.loc[_dates(train, config.date_col) < first_train].copy()
    scaler = Standardizer().fit(train.loc[train["_fit_eligible"], list(columns)].to_numpy(dtype=np.float32))
    return _Prepared(
        _scale(training, columns, scaler), _scale(validation, columns, scaler),
        _scale(history_train, columns, scaler), _scale(train, columns, scaler),
        columns, scaler, fills, selector, overfitting, fracdiff, purging, tickers,
    )


def _graphs(frame, config, prepared, ablation, mode, target):
    if mode in ("gru", "identity"):
        return (), {"sessions": len(_dates(target, config.date_col).unique()),
                    "mean_density": 0.0,
                    "mean_isolated": float(len(prepared.tickers)) if mode == "identity" else None,
                    "mean_degree": 0.0, "mean_edge_turnover": 0.0}
    graph_config = GraphBuildConfig(mode, ablation.graph_lookback,
                                     ablation.graph_threshold, ablation.graph_weight_mode)
    sessions = _dates(target, config.date_col).unique().sort_values()
    train_end = _dates(prepared.history_validation, config.date_col).max()
    snapshots = build_graph_snapshots(
        frame, tickers=sorted(frame[config.group_col].unique()),
        prediction_sessions=sessions, training_end=train_end, config=graph_config,
        date_col=config.date_col, ticker_col=config.group_col, price_col=config.price_col,
    )
    return snapshots, graph_diagnostics(snapshots, frame[config.group_col].nunique())


def _dataset(target, history, columns, config, graphs):
    dataset = build_multimodal_dataset(
        target, tickers=sorted(target[config.group_col].unique()), context_len=config.context_len,
        temporal_columns=columns, node_columns=columns, history_frame=history,
        graphs=graphs, date_col=config.date_col, ticker_col=config.group_col,
    )
    for batch in dataset.iter_batches(128):
        if not batch.asset_mask.all() or not batch.temporal_mask.all() or not batch.node_mask.all():
            raise ValueError("Graph ablation requires identical complete GRU/GNN ticker-date rows.")
    return dataset


def _positions(model, dataset, *, mode, config, batch_dates, torch, backward=None,
               return_probabilities=False):
    values = np.zeros(len(dataset) * len(dataset.tickers), dtype=np.float64)
    probabilities = np.zeros((len(values), 3), dtype=np.float64) if return_probabilities else None
    coefficients = torch.as_tensor(position_coefficients(config.resolved_backtest_position_mode()),
                                   dtype=torch.float32, device=next(model.parameters()).device)
    for batch in dataset.iter_batches(batch_dates):
        output = model(batch)
        available = output.availability
        if not bool(available.all()):
            raise ValueError(f"{mode} has unavailable rows in the matched panel.")
        classes = torch.softmax(output.logits, dim=-1)
        positions = classes @ coefficients
        rows = batch.row_positions.reshape(-1)
        if backward is None:
            values[rows] = positions.reshape(-1).detach().cpu().numpy()
            if probabilities is not None:
                probabilities[rows] = classes.reshape(-1, 3).detach().cpu().numpy()
        else:
            gradient = torch.as_tensor(backward[rows].reshape(positions.shape),
                                       dtype=positions.dtype, device=positions.device)
            positions.backward(gradient)
    return (values, probabilities) if probabilities is not None else values


def _classification(frame, probabilities):
    known = frame["_label_known"].to_numpy(dtype=bool)
    if not known.any():
        return None
    labels = frame.loc[known, "Label_id"].to_numpy(dtype=np.int64)
    probs = probabilities[known]
    predicted = probs.argmax(axis=1)
    metrics = evaluate_predictions(labels, predicted)
    metrics["nll"] = float(-np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-12)).mean())
    confidence = probs.max(axis=1)
    correct = predicted == labels
    ece = 0.0
    for lower in np.linspace(0, 1, 11)[:-1]:
        upper = lower + .1
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    metrics["ece_10"] = float(ece)
    metrics["labeled_rows"] = int(known.sum())
    return metrics


def _fit(model, train_ds, val_ds, train_panel, val_panel, loss, config, training, ablation, torch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate,
                                  weight_decay=training.weight_decay)
    best, best_state, best_epoch, stale = np.inf, None, 0, 0
    started = perf_counter()
    for epoch in range(training.epochs):
        epoch_seed = (training.seed + epoch + 1) % (2**32)
        seed_torch_run(epoch_seed, training.deterministic, torch)
        model.train()
        with torch.no_grad():
            train_positions = _positions(model, train_ds, mode=type(model).__name__,
                                         config=config, batch_dates=ablation.date_batch_size, torch=torch)
        _, gradient = train_panel.loss_and_gradient(train_positions, loss)
        optimizer.zero_grad(set_to_none=True)
        seed_torch_run(epoch_seed, training.deterministic, torch)
        model.train()
        _positions(model, train_ds, mode=type(model).__name__, config=config,
                   batch_dates=ablation.date_batch_size, torch=torch, backward=gradient)
        if training.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters()):
            raise FloatingPointError("Non-finite graph-ablation gradient.")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_positions = _positions(model, val_ds, mode=type(model).__name__,
                                       config=config, batch_dates=ablation.date_batch_size, torch=torch)
        val_loss, _ = val_panel.loss_and_gradient(val_positions, loss)
        if val_loss < best - training.early_stopping_min_delta:
            best, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= training.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("No finite graph-ablation checkpoint.")
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "epochs_run": epoch + 1,
            "seconds": perf_counter() - started,
            "parameter_count": sum(p.numel() for p in model.parameters())}


def _write_graphs(path, graphs):
    if not graphs:
        return None
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump([{
            "session": graph.session.isoformat(),
            "source_start": graph.source_start.isoformat() if graph.source_start is not None else None,
            "source_end": graph.source_end.isoformat(),
            "edge_index": graph.edge_index.tolist(),
            "edge_weight": graph.edge_weight.tolist(),
        } for graph in graphs], stream)
    return str(path)


def run_graph_ablation(frame, config, loss, gru_parameters, seeds, destination, *,
                       ablation=GraphAblationConfig(), n_splits=3,
                       initial_train_fraction=.5, inner_val_fraction=.2,
                       gap_bars=5, embargo_bars=0, dataset_path=None,
                       resume=False):
    """Evaluate G0-G4 on matched outer folds; final holdout stays unopened."""
    import torch

    if config.universe != "multi" or config.evaluation_mode != "static" or config.execution_delay < 1:
        raise ValueError("Graph ablation needs static multi-asset evaluation and delayed execution.")
    if config.label_mode.startswith("oracle") or loss.objective == "cross_entropy":
        raise ValueError("Graph ablation needs non-oracle labels and a financial objective.")
    if config.sample_weighting is not None:
        raise ValueError("Graph ablation does not support classification sample weights.")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be non-empty and unique.")
    config = replace(config, model=ModelSelection("gru", dict(gru_parameters)))
    target = Path(destination).expanduser().resolve()
    if target.exists() and not resume:
        raise FileExistsError(f"Graph ablation output already exists: {target}")
    if resume and not target.is_dir():
        raise FileNotFoundError(f"Cannot resume missing graph ablation: {target}")
    work = _align_position_calendar(_filter_universe(frame, config), config)
    folds, final_split = expanding_calendar_folds(
        work, n_splits=n_splits, initial_train_fraction=initial_train_fraction,
        inner_val_fraction=inner_val_fraction,
        final_test_fraction=1 - config.train_ratio - config.val_ratio,
        gap_bars=gap_bars, embargo_bars=embargo_bars, date_col=config.date_col,
    )
    training_template = GRUConfig(**gru_parameters)
    metadata = {"config": asdict(config), "loss_config": asdict(loss),
                "gru_parameters": gru_parameters, "ablation": asdict(ablation),
                "seeds": list(seeds), "n_splits": n_splits,
                "dataset_sha256": hash_dataframe(work),
                "dataset_path": str(Path(dataset_path).resolve()) if dataset_path else None,
                "survivor_bias_warning": current_universe_warning(dataset_path),
                "final_split": asdict(final_split), "final_holdout_opened": False,
                "protocol": "matched_purged_graph_ablation"}
    metadata = _nullable_metadata(metadata)
    if resume:
        saved = json.loads((target / "metadata.json").read_text())
        if saved != metadata:
            raise ValueError("Resume metadata does not match data, folds or settings.")
        rows = json.loads((target / "folds.json").read_text()) if (target / "folds.json").exists() else []
        completed = {(row["candidate"], row["seed"], row["fold"]) for row in rows if row["status"] == "ok"}
        if len(completed) != len(rows):
            raise ValueError("Resume folds contain duplicates or incomplete rows.")
    else:
        target.mkdir(parents=True)
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
        rows, completed = [], set()
    for fold in folds:
        fold_frame = work.loc[pd.to_datetime(work[config.date_col], utc=True) <= pd.Timestamp(fold["end"])].copy()
        fold_config = replace(config, purged_split=fold["split"])
        prepared = _prepare(fold_frame, fold_config, ablation)
        train_panel = ReturnPanel(prepared.train, price_col=config.price_col,
                                  date_col=config.date_col, group_col=config.group_col,
                                  execution_delay=config.execution_delay)
        val_panel = ReturnPanel(prepared.validation, price_col=config.price_col,
                                date_col=config.date_col, group_col=config.group_col,
                                execution_delay=config.execution_delay)
        graphs_by_mode = {}
        for mode in MODES:
            if all((mode, seed, fold["fold"]) in completed for seed in seeds):
                continue
            graphs_train, diag_train = _graphs(fold_frame, config, prepared, ablation, mode, prepared.train)
            graphs_val, diag_val = _graphs(fold_frame, config, prepared, ablation, mode, prepared.validation)
            graph_path = _write_graphs(target / f"fold-{fold['fold']}-{mode}-graphs.json.gz",
                                       (*graphs_train, *graphs_val))
            train_ds = _dataset(prepared.train, prepared.history_train,
                                prepared.columns, config, graphs_train)
            val_ds = _dataset(prepared.validation, prepared.history_validation,
                              prepared.columns, config, graphs_val)
            for seed in seeds:
                if (mode, seed, fold["fold"]) in completed:
                    continue
                seed_torch_run(seed, training_template.deterministic, torch)
                device = resolve_device(config.device, torch)
                training = replace(training_template, seed=seed, device=config.device)
                if mode == "gru":
                    model = GRUBranch(len(prepared.columns), config.context_len, training).to(device)
                else:
                    model = GNNBranch(len(prepared.columns), hidden_size=ablation.gnn_hidden_size,
                                      num_layers=ablation.gnn_layers, dropout=ablation.gnn_dropout,
                                      graph_mode="identity" if mode == "identity" else "provided").to(device)
                fitted = _fit(model, train_ds, val_ds, train_panel, val_panel,
                              loss, config, training, ablation, torch)
                model.eval()
                with torch.no_grad():
                    positions = _positions(model, val_ds, mode=mode, config=config,
                                           batch_dates=ablation.date_batch_size, torch=torch)
                inner = val_panel.metrics(positions, loss, config.initial_capital)
                # Outer validation is prepared only after the checkpoint is fixed.
                raw_train, raw_val, raw_outer, _, outer_columns = _prepare_splits(
                    fold_frame, fold_config, include_test=True, fill_values=prepared.fills,
                    fracdiff_transformer=prepared.fracdiff,
                    feature_selector=prepared.selector,
                    overfitting_selector=prepared.overfitting_selector,
                    overfitting_supervised=False,
                )
                if outer_columns != prepared.columns:
                    raise ValueError("Outer features differ from frozen inner-train columns.")
                outer = _complete(raw_outer, config, prepared.tickers)
                history_outer = pd.concat((raw_train, raw_val), ignore_index=True)
                outer_scaled = _scale(outer, prepared.columns, prepared.scaler)
                history_scaled = _scale(history_outer, prepared.columns, prepared.scaler)
                graphs_outer, diag_outer = _graphs(fold_frame, config, prepared, ablation, mode, outer_scaled)
                outer_ds = _dataset(outer_scaled, history_scaled, prepared.columns, config, graphs_outer)
                outer_panel = ReturnPanel(outer_scaled, price_col=config.price_col,
                                          date_col=config.date_col, group_col=config.group_col,
                                          execution_delay=config.execution_delay)
                with torch.no_grad():
                    outer_positions, outer_probabilities = _positions(
                        model, outer_ds, mode=mode, config=config,
                        batch_dates=ablation.date_batch_size, torch=torch,
                        return_probabilities=True,
                    )
                outer_metrics = outer_panel.metrics(outer_positions, loss, config.initial_capital)
                classification = _classification(outer, outer_probabilities)
                outer_graph_path = _write_graphs(
                    target / f"fold-{fold['fold']}-{mode}-outer-graphs.json.gz", graphs_outer
                ) if seed == seeds[0] else str(target / f"fold-{fold['fold']}-{mode}-outer-graphs.json.gz") if graphs_outer else None
                row = {"candidate": mode, "seed": seed, "fold": fold["fold"],
                       "status": "ok", "inner_metrics": inner, "outer_metrics": outer_metrics,
                       "classification": classification,
                       "score": outer_metrics["regularized_sharpe"], "fit": fitted,
                       "graph_train": diag_train, "graph_inner": diag_val,
                       "graph_outer": diag_outer, "graph_artifact": graph_path,
                       "outer_graph_artifact": outer_graph_path,
                       "purging": prepared.purging, "feature_columns": prepared.columns,
                       "train_dates": len(train_ds), "inner_dates": len(val_ds),
                       "outer_dates": len(outer_ds)}
                model_path = target / f"fold-{fold['fold']}-{mode}-seed-{seed}.pt"
                torch.save({"model_state": model.state_dict(),
                            "scaler_mean": prepared.scaler.mean_,
                            "scaler_scale": prepared.scaler.scale_,
                            "feature_columns": prepared.columns,
                            "mode": mode, "seed": seed}, model_path)
                row["model_artifact"] = str(model_path)
                rows.append(row)
                (target / "folds.json").write_text(json.dumps(_nullable_metadata(rows), indent=2, allow_nan=False))
                print(f"graph_cv={len(rows)}/{len(MODES)*len(seeds)*len(folds)} {mode} seed={seed} fold={fold['fold']} score={row['score']:.4f}", flush=True)
    summary = []
    for mode in MODES:
        selected = [row for row in rows if row["candidate"] == mode]
        scores = np.asarray([row["score"] for row in selected])
        summary.append({"candidate": mode, "mean": float(scores.mean()),
                        "std": float(scores.std()), "min": float(scores.min()),
                        "mean_net_return": float(np.mean([row["outer_metrics"]["net_return"] for row in selected])),
                        "mean_max_drawdown": float(np.mean([row["outer_metrics"]["max_drawdown"] for row in selected])),
                        "worst_max_drawdown": float(np.min([row["outer_metrics"]["max_drawdown"] for row in selected])),
                        "mean_macro_f1": float(np.mean([row["classification"]["macro_f1"] for row in selected
                                                    if row["classification"] is not None]))
                        if any(row["classification"] is not None for row in selected) else None,
                        "complete": len(selected) == len(seeds) * len(folds)})
    reference = {(row["seed"], row["fold"]): row for row in rows if row["candidate"] == "gru"}
    paired = []
    for mode in MODES[1:]:
        selected = [row for row in rows if row["candidate"] == mode]
        score_delta = [row["score"] - reference[row["seed"], row["fold"]]["score"] for row in selected]
        paired.append({"candidate": mode, "mean_score_delta": float(np.mean(score_delta)),
                       "sharpe_wins": int(np.count_nonzero(np.asarray(score_delta) > 0)),
                       "return_wins": sum(row["outer_metrics"]["net_return"] >
                                          reference[row["seed"], row["fold"]]["outer_metrics"]["net_return"]
                                          for row in selected),
                       "drawdown_wins": sum(row["outer_metrics"]["max_drawdown"] >
                                            reference[row["seed"], row["fold"]]["outer_metrics"]["max_drawdown"]
                                            for row in selected)})
    winner = max(summary, key=lambda row: row["mean"])["candidate"]
    report = _nullable_metadata({"metadata": metadata, "folds": rows, "summary": summary,
                                 "paired_vs_gru": paired, "selected": winner,
                                 "final_test": []})
    (target / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


__all__ = ["GraphAblationConfig", "run_graph_ablation"]
