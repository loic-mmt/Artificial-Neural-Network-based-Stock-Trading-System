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
from trading_system.artifacts.multimodal_study import (
    atomic_torch_save, atomic_write_json, completion_manifest, file_record,
    preprocessing_state, runtime_provenance, stable_digest, training_signature,
    validate_completion,
)
from trading_system.data.causal_graphs import (
    GRAPH_MODES, GraphBuildConfig, build_graph_snapshots, graph_diagnostics,
)
from trading_system.data.multimodal import build_multimodal_dataset
from trading_system.data.purged_cv import expanding_calendar_folds
from trading_system.data.scaling import Standardizer
from trading_system.evaluation.classification import evaluate_predictions
from trading_system.features.expanded import ExpandedFeatureSelector
from trading_system.features.fracdiff import FracDiffTransformer
from trading_system.models.multimodal_branches import GRUBranch, GNNBranch
from trading_system.models.market_gnn import MarketGNNControl
from trading_system.models.market_gru_ablation import MarketGRUControl
from trading_system.models.neural.config import GRUConfig, TransformerConfig
from trading_system.models.neural.trainer import resolve_device, seed_torch_run
from trading_system.models.specs import ModelSelection
from trading_system.reporting.warnings import current_universe_warning
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel, position_coefficients
from trading_system.training.overfitting import TrainOnlyFeatureSelector
from .position_objectives import _align_position_calendar
from .runner import _filter_universe, _prepare_splits


MODES = ("gru", "identity", "sector", "train_pearson", "rolling_pearson")


def _candidate_parts(candidate: str) -> tuple[str, bool]:
    market = candidate.endswith("_market")
    base = candidate[:-7] if market else candidate
    if base != "gru" and base not in GRAPH_MODES:
        raise ValueError(f"Unsupported graph candidate: {candidate}")
    return base, market


@dataclass(frozen=True)
class GraphAblationConfig:
    graph_lookback: int = 252
    graph_threshold: float = 0.7
    graph_weight_mode: str = "positive"
    graph_neighbors: int = 5
    graph_rebalance_bars: int = 1
    gnn_hidden_size: int = 32
    gnn_layers: int = 1
    gnn_dropout: float = 0.0
    date_batch_size: int = 32
    candidates: tuple[str, ...] = MODES
    market_transformer_width: int = 32
    market_transformer_heads: int = 4
    market_transformer_layers: int = 1
    market_gate_temperature: float = 1.0

    def __post_init__(self) -> None:
        GraphBuildConfig(
            "train_pearson", self.graph_lookback, self.graph_threshold,
            self.graph_weight_mode, self.graph_neighbors, self.graph_rebalance_bars,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (
                   self.gnn_hidden_size, self.gnn_layers, self.date_batch_size,
                   self.market_transformer_width, self.market_transformer_heads,
                   self.market_transformer_layers,
               )):
            raise ValueError("GNN width, layers and date batch size must be positive integers.")
        if not np.isfinite(self.gnn_dropout) or not 0 <= self.gnn_dropout < 1:
            raise ValueError("gnn_dropout must be in [0, 1).")
        if (not self.candidates or len(self.candidates) != len(set(self.candidates))
                or any(not isinstance(item, str) for item in self.candidates)):
            raise ValueError("Graph candidates must be unique non-empty names.")
        for candidate in self.candidates:
            _candidate_parts(candidate)
        if self.market_transformer_width % self.market_transformer_heads:
            raise ValueError("Market Transformer width must be divisible by head count.")
        if not np.isfinite(self.market_gate_temperature) or self.market_gate_temperature <= 0:
            raise ValueError("Market gate temperature must be positive.")


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


def _graphs(
    frame, config, prepared, ablation, candidate, target, *,
    graph_context=None, sector_context_columns=None,
):
    mode, _ = _candidate_parts(candidate)
    if mode in ("gru", "identity"):
        return (), {"sessions": len(_dates(target, config.date_col).unique()),
                    "mean_density": 0.0,
                    "mean_isolated": float(len(prepared.tickers)) if mode == "identity" else None,
                    "mean_degree": 0.0, "mean_edge_turnover": 0.0}
    graph_config = GraphBuildConfig(
        mode, ablation.graph_lookback, ablation.graph_threshold,
        ablation.graph_weight_mode, ablation.graph_neighbors,
        ablation.graph_rebalance_bars,
    )
    sessions = _dates(target, config.date_col).unique().sort_values()
    train_end = _dates(prepared.history_validation, config.date_col).max()
    snapshots = build_graph_snapshots(
        frame, tickers=sorted(frame[config.group_col].unique()),
        prediction_sessions=sessions, training_end=train_end, config=graph_config,
        date_col=config.date_col, ticker_col=config.group_col, price_col=config.price_col,
        context_frame=graph_context,
        sector_context_columns=sector_context_columns,
    )
    return snapshots, graph_diagnostics(snapshots, frame[config.group_col].nunique())


def _dataset(
    target, history, columns, config, graphs, *, market=None,
    market_columns=(), require_market=False,
):
    dataset = build_multimodal_dataset(
        target, tickers=sorted(target[config.group_col].unique()), context_len=config.context_len,
        temporal_columns=columns, node_columns=columns, history_frame=history,
        graphs=graphs, date_col=config.date_col, ticker_col=config.group_col,
        market_frame=market,
        market_close_columns=market_columns,
        market_context_len=config.context_len,
    )
    for batch in dataset.iter_batches(128):
        if not batch.asset_mask.all() or not batch.temporal_mask.all() or not batch.node_mask.all():
            raise ValueError("Graph ablation requires identical complete GRU/GNN ticker-date rows.")
        if require_market and not batch.market_sequence_mask.all():
            first = next(str(day) for day, ok in zip(batch.sessions, batch.market_sequence_mask) if not ok)
            raise ValueError(f"Incomplete market context at {first}.")
    return dataset


def _scaled_market_for_fold(market, columns, train, config):
    if market is None or not columns:
        return market, None
    fit_dates = set(_dates(train.loc[train["_fit_eligible"]], config.date_col))
    fit = market.loc[market[config.date_col].isin(fit_dates), list(columns)].dropna()
    if fit.empty:
        raise ValueError("No complete train-only market observations for scaling.")
    scaler = Standardizer().fit(fit.to_numpy(dtype=np.float32))
    scaled = market.copy()
    valid = scaled[list(columns)].notna().all(axis=1)
    scaled.loc[valid, list(columns)] = scaler.transform(
        scaled.loc[valid, list(columns)].to_numpy(dtype=np.float32)
    )
    return scaled, scaler


def _make_model(sample, candidate, config, gru_parameters, ablation, seed, torch):
    """Construct exactly the independently trained branch, including its gate."""
    mode, market_gate = _candidate_parts(candidate)
    stock_width = sample.temporal.shape[-1]
    market_width = sample.market_sequence.shape[-1]
    training = GRUConfig(**{**gru_parameters, "seed": seed, "device": config.device})
    transformer = TransformerConfig(
        d_model=ablation.market_transformer_width,
        n_heads=ablation.market_transformer_heads,
        num_layers=ablation.market_transformer_layers,
        dim_feedforward=2 * ablation.market_transformer_width,
        dropout=0.0, pooling="last", causal_attention=True,
    )
    if mode == "gru" and market_gate:
        model = MarketGRUControl(
            "market_gate_transformer", stock_width, market_width,
            config.context_len, training, transformer,
        )
    elif mode == "gru":
        model = GRUBranch(stock_width, config.context_len, training)
    elif market_gate:
        model = MarketGNNControl(
            stock_width, market_width, config.context_len, transformer,
            hidden_size=ablation.gnn_hidden_size, num_layers=ablation.gnn_layers,
            dropout=ablation.gnn_dropout,
            graph_mode="identity" if mode == "identity" else "provided",
            market_gate=True, gate_temperature=ablation.market_gate_temperature,
        )
    else:
        model = GNNBranch(
            stock_width, hidden_size=ablation.gnn_hidden_size,
            num_layers=ablation.gnn_layers, dropout=ablation.gnn_dropout,
            graph_mode="identity" if mode == "identity" else "provided",
        )
    return model.to(resolve_device(config.device, torch))


def _daily_paths(panel, positions, loss):
    """The exact executable ReturnPanel path, including terminal liquidation."""
    net, executed, _, turnover, costs = panel.path(positions, loss)
    buy_hold_net, *_ = panel.path(np.ones(panel.rows), loss)
    return pd.DataFrame({
        "date": pd.to_datetime(panel.dates, utc=True),
        "net_return": net,
        "gross_return": (executed * panel.returns).mean(axis=0),
        "cost": costs.mean(axis=0),
        "turnover": turnover.mean(axis=0),
        "gross_exposure": np.abs(executed).mean(axis=0),
        "net_exposure": executed.mean(axis=0),
        "buy_hold_return": buy_hold_net,
        "buy_hold_gross_return": panel.returns.mean(axis=0),
    })


def _atomic_parquet(frame, path):
    import os
    import tempfile
    path = Path(path)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _export_predictions(model, dataset, frame, path, *, candidate, fold, seed,
                        partition, config, ablation, torch):
    """Stream actual logits to Parquet; only small aligned metric arrays remain."""
    import os
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    positions = np.empty(len(frame), dtype=np.float64)
    probabilities = np.empty((len(frame), 3), dtype=np.float64)
    seen = np.zeros(len(frame), dtype=bool)
    coefficients = torch.as_tensor(
        position_coefficients(config.resolved_backtest_position_mode()),
        dtype=torch.float32, device=next(model.parameters()).device,
    )
    writer = None
    try:
        for batch in dataset.iter_batches(ablation.date_batch_size):
            output = model(batch)
            if not bool(output.availability.all()):
                raise ValueError("Matched prediction export has unavailable rows.")
            rows = batch.row_positions.reshape(-1)
            if (rows < 0).any() or (rows >= len(frame)).any() or seen[rows].any():
                raise ValueError("Prediction/backtest row keys are duplicate or invalid.")
            logits = output.logits.reshape(-1, 3).detach().cpu().numpy()
            classes = torch.softmax(output.logits, dim=-1)
            probs = classes.reshape(-1, 3).detach().cpu().numpy()
            values = (classes @ coefficients).reshape(-1).detach().cpu().numpy()
            if not np.isfinite(logits).all() or not np.isfinite(probs).all():
                raise FloatingPointError("Non-finite prediction export.")
            positions[rows], probabilities[rows], seen[rows] = values, probs, True
            part = frame.iloc[rows]
            days = pd.to_datetime(part[config.date_col], utc=True).reset_index(drop=True)
            tickers = part[config.group_col].astype(str).reset_index(drop=True)
            known = part["_label_known"].to_numpy(dtype=bool)
            labels = part["Label_id"].to_numpy(dtype=np.int64)
            result = pd.DataFrame({
                "variant_id": candidate, "candidate": candidate,
                "fold": fold, "seed": seed, "partition": partition,
                "date": days, "ticker": tickers,
                "backtest_key": days.map(lambda value: value.isoformat()) + "|" + tickers,
                "row_position": rows, "available": True, "label_known": known,
                "adj_close": part[config.price_col].to_numpy(dtype=np.float64),
                "label": np.where(known, labels, -1),
                "logit_sell": logits[:, 0], "logit_hold": logits[:, 1], "logit_buy": logits[:, 2],
                "p_sell": probs[:, 0], "p_hold": probs[:, 1], "p_buy": probs[:, 2],
                "position": values,
            })
            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
            writer.write_table(table)
        if not seen.all():
            raise ValueError("Prediction export did not cover the complete backtest.")
        writer.close()
        writer = None
        os.replace(temporary, path)
    finally:
        if writer is not None:
            writer.close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    return positions, probabilities


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
    import os
    import tempfile
    path = Path(path)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as binary:
            # Stable gzip headers keep existing seed manifests valid when an
            # interrupted seed rewrites the identical shared sparse graphs.
            with gzip.GzipFile(fileobj=binary, mode="wb", mtime=0, filename="") as stream:
                stream.write(b"[")
                for index, graph in enumerate(graphs):
                    if index:
                        stream.write(b",")
                    record = {
                        "session": graph.session.isoformat(),
                        "source_start": graph.source_start.isoformat() if graph.source_start is not None else None,
                        "source_end": graph.source_end.isoformat(),
                        "edge_index": graph.edge_index.tolist(),
                        "edge_weight": graph.edge_weight.tolist(),
                    }
                    stream.write(json.dumps(record, allow_nan=False).encode("utf-8"))
                stream.write(b"]")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(path)


def _run_context(frame, config, loss, gru_parameters, seeds, ablation, *,
                 n_splits, initial_train_fraction, inner_val_fraction,
                 gap_bars, embargo_bars, dataset_path=None,
                 graph_context=None, graph_context_path=None,
                 sector_context_columns=None, market_frame=None,
                 market_columns=(), market_audit=None, market_context_path=None):
    if config.universe != "multi" or config.evaluation_mode != "static" or config.execution_delay < 1:
        raise ValueError("Graph ablation needs static multi-asset evaluation and delayed execution.")
    if config.label_mode.startswith("oracle") or loss.objective == "cross_entropy":
        raise ValueError("Graph ablation needs non-oracle labels and a financial objective.")
    if config.sample_weighting is not None:
        raise ValueError("Graph ablation does not support classification sample weights.")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be non-empty and unique.")
    if any(_candidate_parts(item)[1] for item in ablation.candidates) and (
        market_frame is None or not market_columns
    ):
        raise ValueError("Market-gated candidates require a prepared market frame.")
    if any(_candidate_parts(item)[0] == "rolling_residual_topk" for item in ablation.candidates) and graph_context is None:
        raise ValueError("Residual top-k candidates require ETF graph context.")
    config = replace(config, model=ModelSelection("gru", dict(gru_parameters)))
    GRUConfig(**gru_parameters)
    work = _align_position_calendar(_filter_universe(frame, config), config)
    cv_spec = {
        "n_splits": n_splits, "initial_train_fraction": initial_train_fraction,
        "inner_val_fraction": inner_val_fraction,
        "final_test_fraction": 1 - config.train_ratio - config.val_ratio,
        "gap_bars": gap_bars, "embargo_bars": embargo_bars,
    }
    folds, final_split = expanding_calendar_folds(work, **cv_spec, date_col=config.date_col)
    sessions = [day.isoformat() for day in _dates(work, config.date_col).unique().sort_values()]
    tickers = sorted(work[config.group_col].unique())
    metadata = {
        "schema_version": 2, "config": asdict(config), "loss_config": asdict(loss),
        "gru_parameters": dict(gru_parameters), "ablation": asdict(ablation),
        "seeds": list(seeds), "n_splits": n_splits, "cv_spec": cv_spec,
        "cv_folds": [{"fold": item["fold"], "split": asdict(item["split"]), "end": item["end"]}
                     for item in folds],
        "calendar": {"tickers": tickers, "sessions": sessions,
                     "sha256": stable_digest({"tickers": tickers, "sessions": sessions})},
        "dataset_sha256": hash_dataframe(work),
        "dataset_path": str(Path(dataset_path).resolve()) if dataset_path else None,
        "graph_context_path": str(Path(graph_context_path).resolve()) if graph_context_path else None,
        "graph_context_sha256": hash_dataframe(graph_context) if graph_context is not None else None,
        "sector_context_columns": dict(sector_context_columns or {}),
        "market_context_path": str(Path(market_context_path).resolve()) if market_context_path else None,
        "market_context_sha256": hash_dataframe(market_frame) if market_frame is not None else None,
        "market_columns": list(market_columns), "market_audit": market_audit,
        "survivor_bias_warning": current_universe_warning(dataset_path),
        "final_split": asdict(final_split), "final_holdout_opened": False,
        "class_names": ["Sell", "Hold", "Buy"],
        "export_contract": {
            "schema_version": 1, "timezone": "UTC",
            "backtest_key": "date.isoformat()|ticker",
            "unknown_label": -1, "label_availability_mask": "label_known",
            "signal_availability_mask": "available",
            "logits": "unscaled class scores", "probabilities": "softmax, range [0,1]",
            "position": ("P(Buy)-P(Sell), range [-1,1]" if config.position_mode == "long_short"
                         else "P(Buy), range [0,1]"),
            "adj_close": f"Backtest price from {config.price_col}; not a standardized input feature",
            "returns_and_costs": "fractional daily returns, not percent or currency PnL",
            "turnover": "mean absolute change in executed position across tickers, including final liquidation",
            "exposure": "mean absolute/signed executed position across tickers",
            "price_availability": "session J close; delayed execution only",
            "graph_availability": "source_end <= J UTC session-end bound; close-J data for delayed execution",
            "external_features_availability": "available_at strictly before J midnight UTC",
            "execution_delay_sessions": config.execution_delay,
            "execution_convention": "signal at close J executed at close J+delay, then earns next close-to-close return",
        },
        "protocol": "matched_purged_graph_ablation", "provenance": runtime_provenance(),
    }
    return work, config, folds, _nullable_metadata(metadata)


def _task_spec(metadata, prepared, fold, candidate, seed, ablation, market_scaler):
    mode, market_gate = _candidate_parts(candidate)
    snapshot = preprocessing_state(
        prepared, market_scaler if market_gate else None,
        metadata["market_columns"] if market_gate else (),
    )
    effective_model = {"candidate": candidate, "width": len(prepared.columns)}
    if mode != "gru":
        effective_model.update(gnn_hidden_size=ablation.gnn_hidden_size,
                               gnn_layers=ablation.gnn_layers, gnn_dropout=ablation.gnn_dropout)
    if market_gate:
        effective_model.update(
            market_transformer_width=ablation.market_transformer_width,
            market_transformer_heads=ablation.market_transformer_heads,
            market_transformer_layers=ablation.market_transformer_layers,
            # The legacy GRU path uses the gate default. Record what runs, not
            # a CLI value that this candidate does not currently consume.
            market_gate_temperature=(1.0 if mode == "gru" else ablation.market_gate_temperature),
        )
    graph = None if mode in ("gru", "identity") else {
        "mode": mode, "lookback": ablation.graph_lookback,
        "threshold": ablation.graph_threshold, "weight_mode": ablation.graph_weight_mode,
        "neighbors": ablation.graph_neighbors, "rebalance_bars": ablation.graph_rebalance_bars,
        "context_sha256": metadata["graph_context_sha256"] if mode == "rolling_residual_topk" else None,
        "sector_context_columns": metadata["sector_context_columns"] if mode == "rolling_residual_topk" else None,
    }
    device = resolve_device(metadata["config"]["device"], __import__("torch"))
    return _nullable_metadata({
        "schema_version": 2, "candidate": candidate, "fold": fold["fold"], "seed": seed,
        "config": metadata["config"], "loss_config": metadata["loss_config"],
        "gru_parameters": metadata["gru_parameters"], "model": effective_model,
        "dataset_sha256": metadata["dataset_sha256"], "calendar": metadata["calendar"],
        "cv_spec": metadata["cv_spec"],
        "fold_boundaries": {"split": asdict(fold["split"]), "end": fold["end"]},
        "eligible_sessions": {
            name: [day.isoformat() for day in _dates(value, metadata["config"]["date_col"]).unique().sort_values()]
            for name, value in (("train", prepared.train), ("inner", prepared.validation))
        },
        "preprocessing": snapshot, "graph": graph,
        "shared_graph_warmup": ablation.graph_lookback,
        "market_context_sha256": metadata["market_context_sha256"] if market_gate else None,
        "date_batch_size": ablation.date_batch_size,
        "resolved_device": str(device), "precision": "float32",
        "provenance": metadata["provenance"],
    })


def plan_run_graph_ablation(frame, config, loss, gru_parameters, seeds, *,
                            ablation=GraphAblationConfig(), n_splits=3,
                            initial_train_fraction=.5, inner_val_fraction=.2,
                            gap_bars=5, embargo_bars=0, dataset_path=None,
                            graph_context=None, graph_context_path=None,
                            sector_context_columns=None, market_frame=None,
                            market_columns=(), market_audit=None, market_context_path=None):
    """Exact train-only task signatures without model fitting or artifact writes."""
    work, config, folds, metadata = _run_context(
        frame, config, loss, gru_parameters, seeds, ablation,
        n_splits=n_splits, initial_train_fraction=initial_train_fraction,
        inner_val_fraction=inner_val_fraction, gap_bars=gap_bars, embargo_bars=embargo_bars,
        dataset_path=dataset_path, graph_context=graph_context,
        graph_context_path=graph_context_path, sector_context_columns=sector_context_columns,
        market_frame=market_frame, market_columns=market_columns,
        market_audit=market_audit, market_context_path=market_context_path,
    )
    tasks = []
    for fold in folds:
        frame_fold = work.loc[_dates(work, config.date_col) <= pd.Timestamp(fold["end"])].copy()
        prepared = _prepare(frame_fold, replace(config, purged_split=fold["split"]), ablation)
        _, scaler = _scaled_market_for_fold(market_frame, market_columns, prepared.history_validation, config)
        for candidate in ablation.candidates:
            for seed in seeds:
                spec = _task_spec(metadata, prepared, fold, candidate, seed, ablation, scaler)
                tasks.append({"candidate": candidate, "seed": seed, "fold": fold["fold"],
                              "signature": training_signature(spec), "spec": spec})
    return {"metadata": metadata, "task_specs": tasks}


def _resume_metadata(metadata):
    # A portable run may move between directories. Numerical runtime/code and
    # all actual protocol settings remain strict; only location/warning text is
    # informational. Unlike reuse, candidate/seed lists are part of resume.
    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items() if key != "observed_torch_state"}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value
    result = normalize(metadata)
    return {key: value for key, value in result.items()
            if not key.endswith("_path") and key != "survivor_bias_warning"}


def _row_root(root, row, _seen=None):
    from pathlib import PureWindowsPath
    reference = row.get("artifact_root", ".")
    if (not isinstance(reference, str) or not reference or "\\" in reference
            or Path(reference).is_absolute() or PureWindowsPath(reference).drive):
        raise ValueError("Artifact roots must be portable relative paths.")
    if reference != "." and not row.get("reuse", {}).get("verified"):
        raise ValueError("External artifact roots require verified reuse provenance.")
    physical = (Path(root) / reference).resolve()
    if reference != ".":
        source_reference = row["reuse"].get("source_run")
        if (not isinstance(source_reference, str) or not source_reference or "\\" in source_reference
                or Path(source_reference).is_absolute() or PureWindowsPath(source_reference).drive):
            raise ValueError("Reuse source must be a portable relative run reference.")
        seen = set() if _seen is None else set(_seen)
        current = Path(root).resolve()
        if current in seen:
            raise ValueError("Cyclic reuse provenance.")
        seen.add(current)
        source = (current / source_reference).resolve()
        source_rows = json.loads((source / "folds.json").read_text())
        matching = [item for item in source_rows if all(item.get(key) == row.get(key)
                    for key in ("candidate", "fold", "seed", "task_signature"))]
        if len(matching) != 1 or _row_root(source, matching[0], seen) != physical:
            raise ValueError("Artifact root does not match its declared source task.")
    return physical


def _validated_row(row, root, signature=None):
    """Read immutable metrics, not an editable folds.json status or score."""
    from trading_system.artifacts.multimodal_study import artifact_path
    physical = _row_root(root, row)
    expected = signature or row.get("task_signature")
    files = validate_completion(row.get("completion_manifest"), physical, expected)
    result_path = artifact_path(row.get("result_artifact"), physical)
    if result_path not in files:
        raise ValueError("Completed task lacks its immutable result artifact.")
    immutable = json.loads(result_path.read_text())
    for key in ("candidate", "fold", "seed", "task_signature"):
        if immutable.get(key) != row.get(key):
            raise ValueError("Result artifact identity does not match the task row.")
    required = [immutable.get("model_artifact"), immutable.get("result_artifact")]
    required.extend(immutable.get("prediction_artifacts", {}).values())
    required.extend(immutable.get("daily_path_artifacts", {}).values())
    required.extend(value for value in (immutable.get("graph_artifact"), immutable.get("outer_graph_artifact")) if value)
    if (set(immutable.get("prediction_artifacts", {})) != {"inner", "outer"}
            or set(immutable.get("daily_path_artifacts", {})) != {"inner", "outer"}
            or any(artifact_path(value, physical) not in files for value in required)):
        raise ValueError("Completed task lacks a required checkpoint or prediction/path artifact.")
    immutable["completion_manifest"] = row["completion_manifest"]
    for key in ("artifact_root", "reuse"):
        if key in row:
            immutable[key] = row[key]
    return immutable


def _reuse_rows(reuse_from):
    result = {}
    for source in reuse_from or ():
        source = Path(source).resolve()
        try:
            metadata = json.loads((source / "metadata.json").read_text())
            path = source / "folds.json"
            rows = json.loads(path.read_text()) if path.is_file() else []
            if not isinstance(metadata, dict) or not isinstance(rows, list):
                raise ValueError("Invalid reuse metadata/folds schema.")
            if metadata.get("schema_version") != 2:
                continue  # Missing legacy provenance cannot be certified retroactively.
        except (OSError, ValueError, TypeError, KeyError):
            continue  # Missing/interrupted references are not completed controls.
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("status") != "ok" or not row.get("task_signature"):
                continue
            try:
                row = _validated_row(row, source)
            except (OSError, ValueError, TypeError, KeyError):
                continue  # Never reuse a corrupt task; orchestration reports incompatibilities.
            root = _row_root(source, row)
            result.setdefault(row["task_signature"], (source, root, row))
    return result


def run_graph_ablation(frame, config, loss, gru_parameters, seeds, destination, *,
                       ablation=GraphAblationConfig(), n_splits=3,
                       initial_train_fraction=.5, inner_val_fraction=.2,
                       gap_bars=5, embargo_bars=0, dataset_path=None,
                       graph_context: pd.DataFrame | None = None,
                       graph_context_path=None,
                       sector_context_columns=None,
                       market_frame: pd.DataFrame | None = None,
                       market_columns=(), market_audit=None,
                       market_context_path=None, resume=False, task_filter=None,
                       reuse_from=(), progress_callback=None):
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
    candidates = ablation.candidates
    needs_market = any(_candidate_parts(candidate)[1] for candidate in candidates)
    needs_residual = any(
        _candidate_parts(candidate)[0] == "rolling_residual_topk"
        for candidate in candidates
    )
    market_columns = tuple(market_columns)
    if needs_market and (market_frame is None or not market_columns):
        raise ValueError("Market-gated candidates require a prepared market frame.")
    if needs_residual and graph_context is None:
        raise ValueError("Residual top-k candidates require ETF graph context.")
    config = replace(config, model=ModelSelection("gru", dict(gru_parameters)))
    target = Path(destination).expanduser().resolve()
    if target.exists() and not resume:
        raise FileExistsError(f"Graph ablation output already exists: {target}")
    if resume and not target.is_dir():
        raise FileNotFoundError(f"Cannot resume missing graph ablation: {target}")
    work, config, folds, metadata = _run_context(
        frame, config, loss, gru_parameters, seeds, ablation,
        n_splits=n_splits, initial_train_fraction=initial_train_fraction,
        inner_val_fraction=inner_val_fraction, gap_bars=gap_bars, embargo_bars=embargo_bars,
        dataset_path=dataset_path, graph_context=graph_context,
        graph_context_path=graph_context_path, sector_context_columns=sector_context_columns,
        market_frame=market_frame, market_columns=market_columns,
        market_audit=market_audit, market_context_path=market_context_path,
    )
    training_template = GRUConfig(**gru_parameters)
    all_tasks = {(candidate, seed, fold["fold"]) for candidate in candidates for seed in seeds for fold in folds}
    wanted = all_tasks if task_filter is None else set(task_filter)
    if not wanted or not wanted <= all_tasks:
        raise ValueError("Task filter must select existing candidate/seed/fold tasks.")
    reusable = _reuse_rows(reuse_from)
    if resume:
        saved = json.loads((target / "metadata.json").read_text())
        if _resume_metadata(saved) != _resume_metadata(metadata):
            raise ValueError("Resume metadata does not match data, folds or settings.")
        rows = json.loads((target / "folds.json").read_text()) if (target / "folds.json").exists() else []
        keys = [(row["candidate"], row["seed"], row["fold"]) for row in rows]
        if len(set(keys)) != len(keys) or not set(keys) <= all_tasks:
            raise ValueError("Resume folds contain duplicate or unexpected tasks.")
        valid, invalid = [], []
        for row in rows:
            try:
                if row.get("status") != "ok":
                    raise ValueError("Task is not completed.")
                valid.append(_validated_row(row, target))
            except (ValueError, OSError, TypeError, KeyError) as error:
                invalid.append({"candidate": row["candidate"], "seed": row["seed"],
                                "fold": row["fold"], "error": str(error)})
        rows = valid
        completed = {(row["candidate"], row["seed"], row["fold"]) for row in rows}
        if invalid:
            atomic_write_json(target / "recovery.json", invalid)
            atomic_write_json(target / "folds.json", rows)
    else:
        target.mkdir(parents=True)
        atomic_write_json(target / "metadata.json", metadata)
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
        fold_market, market_scaler = _scaled_market_for_fold(
            market_frame, market_columns, prepared.history_validation, config,
        )
        graphs_by_mode = {}
        for candidate in candidates:
            candidate_keys = {(candidate, seed, fold["fold"]) for seed in seeds} & wanted
            if not candidate_keys or candidate_keys <= completed:
                continue
            mode, market_gate = _candidate_parts(candidate)
            for seed in seeds:
                key = (candidate, seed, fold["fold"])
                if key not in wanted or key in completed:
                    continue
                spec = _task_spec(metadata, prepared, fold, candidate, seed, ablation, market_scaler)
                signature = training_signature(spec)
                if signature in reusable:
                    import os
                    source, source_root, source_row = reusable[signature]
                    row = deepcopy(source_row)
                    row["artifact_root"] = Path(os.path.relpath(source_root, target)).as_posix()
                    row["reuse"] = {"source_run": Path(os.path.relpath(source, target)).as_posix(),
                                    "source_task_signature": signature, "verified": True}
                    rows.append(row)
                    completed.add(key)
                    atomic_write_json(target / "folds.json", _nullable_metadata(rows))
                    if progress_callback:
                        progress_callback(row, len(wanted))
            if candidate_keys <= completed:
                continue
            if mode not in graphs_by_mode:
                graphs_by_mode[mode] = (
                    *_graphs(
                        fold_frame, config, prepared, ablation, candidate, prepared.train,
                        graph_context=graph_context,
                        sector_context_columns=sector_context_columns,
                    ),
                    *_graphs(
                        fold_frame, config, prepared, ablation, candidate, prepared.validation,
                        graph_context=graph_context,
                        sector_context_columns=sector_context_columns,
                    ),
                )
            graphs_train, diag_train, graphs_val, diag_val = graphs_by_mode[mode]
            graph_path = _write_graphs(target / f"fold-{fold['fold']}-{candidate}-graphs.json.gz",
                                       (*graphs_train, *graphs_val))
            train_ds = _dataset(prepared.train, prepared.history_train,
                                prepared.columns, config, graphs_train,
                                market=fold_market, market_columns=market_columns,
                                require_market=market_gate)
            val_ds = _dataset(prepared.validation, prepared.history_validation,
                              prepared.columns, config, graphs_val,
                              market=fold_market, market_columns=market_columns,
                              require_market=market_gate)
            for seed in seeds:
                if (candidate, seed, fold["fold"]) in completed or (candidate, seed, fold["fold"]) not in wanted:
                    continue
                seed_torch_run(seed, training_template.deterministic, torch)
                training = replace(training_template, seed=seed, device=config.device)
                spec = _task_spec(metadata, prepared, fold, candidate, seed, ablation, market_scaler)
                signature = training_signature(spec)
                model = _make_model(next(train_ds.iter_batches(1)), candidate, config,
                                    gru_parameters, ablation, seed, torch)
                fitted = _fit(model, train_ds, val_ds, train_panel, val_panel,
                              loss, config, training, ablation, torch)
                model.eval()
                stem = f"fold-{fold['fold']}-{candidate}-seed-{seed}"
                inner_prediction = target / f"{stem}-inner-predictions.parquet"
                with torch.no_grad():
                    positions, _ = _export_predictions(
                        model, val_ds, prepared.validation, inner_prediction,
                        candidate=candidate, fold=fold["fold"], seed=seed,
                        partition="inner", config=config, ablation=ablation, torch=torch,
                    )
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
                outer_key = (mode, "outer")
                if outer_key not in graphs_by_mode:
                    graphs_by_mode[outer_key] = _graphs(
                        fold_frame, config, prepared, ablation, candidate, outer_scaled,
                        graph_context=graph_context,
                        sector_context_columns=sector_context_columns,
                    )
                graphs_outer, diag_outer = graphs_by_mode[outer_key]
                outer_ds = _dataset(
                    outer_scaled, history_scaled, prepared.columns, config, graphs_outer,
                    market=fold_market, market_columns=market_columns,
                    require_market=market_gate,
                )
                outer_panel = ReturnPanel(outer_scaled, price_col=config.price_col,
                                          date_col=config.date_col, group_col=config.group_col,
                                          execution_delay=config.execution_delay)
                outer_prediction = target / f"{stem}-outer-predictions.parquet"
                with torch.no_grad():
                    outer_positions, outer_probabilities = _export_predictions(
                        model, outer_ds, outer_scaled, outer_prediction,
                        candidate=candidate, fold=fold["fold"], seed=seed,
                        partition="outer", config=config, ablation=ablation, torch=torch,
                    )
                outer_metrics = outer_panel.metrics(outer_positions, loss, config.initial_capital)
                classification = _classification(outer, outer_probabilities)
                outer_graph_path = _write_graphs(
                    target / f"fold-{fold['fold']}-{candidate}-outer-graphs.json.gz", graphs_outer
                )
                row = {"candidate": candidate, "seed": seed, "fold": fold["fold"],
                       "status": "ok", "inner_metrics": inner, "outer_metrics": outer_metrics,
                       "classification": classification,
                       "score": outer_metrics["regularized_sharpe"], "fit": fitted,
                       "graph_train": diag_train, "graph_inner": diag_val,
                       "graph_outer": diag_outer,
                       "graph_artifact": Path(graph_path).name if graph_path else None,
                       "outer_graph_artifact": Path(outer_graph_path).name if outer_graph_path else None,
                       "purging": prepared.purging, "feature_columns": prepared.columns,
                       "train_dates": len(train_ds), "inner_dates": len(val_ds),
                       "outer_dates": len(outer_ds)}
                model_path = target / f"{stem}.pt"
                snapshot = spec["preprocessing"]
                atomic_torch_save(model_path, {"model_state": model.state_dict(),
                            "scaler_mean": prepared.scaler.mean_,
                            "scaler_scale": prepared.scaler.scale_,
                            "feature_columns": prepared.columns,
                            "mode": candidate, "seed": seed,
                            "market_columns": market_columns,
                            "market_scaler_mean": (
                                market_scaler.mean_ if market_scaler is not None else None
                            ),
                            "market_scaler_scale": (
                                market_scaler.scale_ if market_scaler is not None else None
                            ), "schema_version": 2, "preprocessing": snapshot,
                            "model_spec": spec["model"], "training_spec": spec,
                            "task_signature": signature, "fold": fold["fold"]})
                row.update(
                    model_artifact=model_path.name, task_signature=signature,
                    preprocessing_signature=stable_digest(snapshot),
                    prediction_artifacts={"inner": inner_prediction.name, "outer": outer_prediction.name},
                    daily_path_artifacts={}, result_artifact=f"{stem}-result.json",
                    eligible_sessions={
                        **spec["eligible_sessions"],
                        "outer": [day.isoformat() for day in _dates(outer, config.date_col).unique().sort_values()],
                    },
                )
                for partition, panel, predicted in (("inner", val_panel, positions),
                                                    ("outer", outer_panel, outer_positions)):
                    daily_path = target / f"{stem}-{partition}-daily.parquet"
                    daily = _daily_paths(panel, predicted, loss)
                    daily.insert(0, "partition", partition)
                    daily.insert(0, "seed", seed)
                    daily.insert(0, "fold", fold["fold"])
                    daily.insert(0, "variant_id", candidate)
                    _atomic_parquet(daily, daily_path)
                    row["daily_path_artifacts"][partition] = daily_path.name
                row = _nullable_metadata(row)
                atomic_write_json(target / row["result_artifact"], row)
                artifacts = [model_path, inner_prediction, outer_prediction,
                             target / row["result_artifact"],
                             *(target / value for value in row["daily_path_artifacts"].values()),
                             *(Path(value) for value in (graph_path, outer_graph_path) if value)]
                row["completion_manifest"] = completion_manifest(signature, [file_record(value, target) for value in artifacts])
                rows.append(row)
                completed.add((candidate, seed, fold["fold"]))
                atomic_write_json(target / "folds.json", _nullable_metadata(rows))
                if progress_callback:
                    progress_callback(row, len(wanted))
                else:
                    print(f"graph_cv={len(rows)}/{len(all_tasks)} {candidate} seed={seed} fold={fold['fold']} score={row['score']:.4f}", flush=True)
    summary = []
    for mode in candidates:
        selected = [row for row in rows if row["candidate"] == mode]
        if not selected:
            summary.append({"candidate": mode, "mean": None, "complete": False})
            continue
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
    for mode in candidates:
        if mode == "gru":
            continue
        selected = [row for row in rows if row["candidate"] == mode]
        selected = [row for row in selected if (row["seed"], row["fold"]) in reference]
        if not selected:
            continue
        score_delta = [row["score"] - reference[row["seed"], row["fold"]]["score"] for row in selected]
        paired.append({"candidate": mode, "mean_score_delta": float(np.mean(score_delta)),
                       "sharpe_wins": int(np.count_nonzero(np.asarray(score_delta) > 0)),
                       "return_wins": sum(row["outer_metrics"]["net_return"] >
                                          reference[row["seed"], row["fold"]]["outer_metrics"]["net_return"]
                                          for row in selected),
                       "drawdown_wins": sum(row["outer_metrics"]["max_drawdown"] >
                                            reference[row["seed"], row["fold"]]["outer_metrics"]["max_drawdown"]
                                            for row in selected)})
    winner = (max(summary, key=lambda row: row["mean"])["candidate"]
              if all(row["complete"] for row in summary) else None)
    report = _nullable_metadata({"metadata": metadata, "folds": rows, "summary": summary,
                                 "paired_vs_gru": paired, "selected": winner,
                                 "final_test": []})
    atomic_write_json(target / "report.json", report)
    return report


__all__ = ["GraphAblationConfig", "run_graph_ablation", "plan_run_graph_ablation"]
