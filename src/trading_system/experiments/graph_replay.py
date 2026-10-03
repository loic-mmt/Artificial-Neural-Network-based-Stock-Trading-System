"""Read-only checkpoint replay into a new, sealed-holdout diagnostic export.

Legacy checkpoints may be explored, but are never certified as reusable inputs:
matching metrics is not a substitute for missing training provenance.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import _nullable_metadata, hash_dataframe
from trading_system.data.purged_cv import expanding_calendar_folds
from trading_system.data.scaling import Standardizer
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.graph_ablation import (
    GraphAblationConfig, _candidate_parts, _classification, _complete, _dataset,
    _dates, _graphs, _prepare, _scale, _scaled_market_for_fold,
)
from trading_system.experiments.position_objectives import _align_position_calendar
from trading_system.experiments.runner import _filter_universe, _prepare_splits
from trading_system.models.neural.trainer import resolve_device
from trading_system.models.specs import ModelSelection
from trading_system.training.financial_loss import (
    FinancialLossConfig, ReturnPanel, position_coefficients,
)


PREDICTION_KEYS = ("candidate", "fold", "seed", "partition", "date", "ticker")


def load_graph_checkpoint(path, torch):
    """Allow concrete NumPy scaler types without enabling arbitrary pickle code."""
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:
        numpy_core = np.core
    allowed = [numpy_core.multiarray._reconstruct, np.ndarray, np.dtype,
               type(np.dtype("float32")), type(np.dtype("float64"))]
    from torch.serialization import safe_globals
    with safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def config_from_metadata(metadata):
    values = dict(metadata["config"])
    values["model"] = ModelSelection(**values["model"])
    return ExperimentConfig(**values)


def _outer_fold(fold_frame, config, ablation, fold):
    fold_config = replace(config, purged_split=fold["split"])
    prepared = _prepare(fold_frame, fold_config, ablation)
    train, validation, outer, _, columns = _prepare_splits(
        fold_frame, fold_config, include_test=True, fill_values=prepared.fills,
        fracdiff_transformer=prepared.fracdiff, feature_selector=prepared.selector,
        overfitting_selector=prepared.overfitting_selector, overfitting_supervised=False,
    )
    if columns != prepared.columns:
        raise ValueError("Replayed outer features differ from frozen inner-training columns.")
    outer = _complete(outer, config, prepared.tickers)
    return (prepared, _scale(outer, columns, prepared.scaler),
            _scale(pd.concat((train, validation), ignore_index=True), columns, prepared.scaler))


def _resolve_cv(metadata, *, initial_train_fraction=None, inner_val_fraction=None,
                gap_bars=None, embargo_bars=None):
    saved = metadata.get("cv_spec")
    overrides = {"initial_train_fraction": initial_train_fraction,
                 "inner_val_fraction": inner_val_fraction, "gap_bars": gap_bars,
                 "embargo_bars": embargo_bars}
    if saved is not None:
        for key, value in overrides.items():
            if value is not None and value != saved[key]:
                raise ValueError(f"Explicit CV setting differs from benchmark metadata: {key}.")
        return dict(saved)
    # Older reports stored only the final boundary, which does not identify the
    # outer folds. Never infer either fraction from an old CLI default.
    if initial_train_fraction is None or inner_val_fraction is None:
        raise ValueError("Legacy replay requires explicit initial_train_fraction and inner_val_fraction.")
    return {"n_splits": metadata["n_splits"],
            "initial_train_fraction": initial_train_fraction,
            "inner_val_fraction": inner_val_fraction,
            "final_test_fraction": 1 - metadata["config"]["train_ratio"] - metadata["config"]["val_ratio"],
            "gap_bars": metadata["final_split"]["gap_bars"] if gap_bars is None else gap_bars,
            "embargo_bars": metadata["final_split"]["embargo_bars"] if embargo_bars is None else embargo_bars}


def _restore_feature_order(prepared, checkpoint):
    """Restore a legacy checkpoint's named order, never its feature selection.

    The prepared frames are already scaled by column name. Permuting scaler
    state and the explicit tensor-column order therefore leaves each named
    feature untouched. Different feature sets or statistics remain errors.
    """
    original, frozen = tuple(prepared.columns), tuple(checkpoint["feature_columns"])
    if (not frozen or len(set(original)) != len(original) or len(set(frozen)) != len(frozen)
            or set(original) != set(frozen)):
        raise ValueError("Checkpoint feature-order restoration requires the identical named feature set.")
    permutation = [original.index(name) for name in frozen]
    reordered = {}
    for suffix in ("mean", "scale"):
        current = np.asarray(getattr(prepared.scaler, f"{suffix}_"))
        stored = np.asarray(checkpoint[f"scaler_{suffix}"])
        if (current.shape != (1, len(original)) or stored.shape != current.shape
                or not np.isfinite(current).all() or not np.isfinite(stored).all()
                or suffix == "scale" and ((current <= 0).any() or (stored <= 0).any())):
            raise ValueError("Invalid scaler state for checkpoint feature-order restoration.")
        value = current[:, permutation].copy()
        if not np.allclose(value, stored):
            raise ValueError("Named feature scalers differ from the legacy checkpoint.")
        reordered[suffix] = value
    prepared.columns = frozen
    prepared.scaler = Standardizer(mean_=reordered["mean"], scale_=reordered["scale"])
    return {"restored": original != frozen, "reconstructed_columns": list(original),
            "checkpoint_columns": list(frozen), "permutation": permutation}


def _metric_match(actual, expected, *, candidate, fold, seed, partition,
                  atol=5e-4, rtol=5e-4):
    """Check every saved backtest key, not just a favorable summary metric."""
    def equal(value, reference):
        if isinstance(value, dict) and isinstance(reference, dict):
            return set(value) == set(reference) and all(equal(value[key], reference[key]) for key in value)
        if isinstance(value, (list, tuple, np.ndarray)) and isinstance(reference, (list, tuple, np.ndarray)):
            return len(value) == len(reference) and all(equal(left, right) for left, right in zip(value, reference))
        if value is None or reference is None:
            return value is None and reference is None
        if isinstance(value, (bool, int, str, np.integer)):
            return value == reference
        if isinstance(value, (float, np.floating)) and isinstance(reference, (float, int)):
            return bool(np.isclose(value, reference, atol=atol, rtol=rtol))
        return False
    if not equal(actual, expected):
        raise ValueError(f"Replayed {partition} metrics differ: {candidate} fold={fold} seed={seed}.")


def _degrees(graphs, frame, tickers, candidate, date_col):
    mode, _ = _candidate_parts(candidate)
    if mode in ("gru", "identity"):
        return np.zeros(len(frame), dtype=np.int64)
    dates = _dates(frame, date_col).unique().sort_values()
    if len(graphs) != len(dates) or any(graph.session != day for graph, day in zip(graphs, dates)):
        raise ValueError("Replayed graph dates do not match prediction rows.")
    return np.stack([np.bincount(graph.edge_index[1], minlength=len(tickers))
                     for graph in graphs]).reshape(-1)


def prediction_frame(model, dataset, frame, *, candidate, fold, seed, partition,
                     config, ablation, torch, graphs=()):
    """Preserve actual pre-softmax logits and complete backtest identity keys."""
    logits = np.empty((len(frame), 3), dtype=np.float64)
    probabilities = np.empty_like(logits)
    positions = np.empty(len(frame), dtype=np.float64)
    coefficients = torch.as_tensor(position_coefficients(config.resolved_backtest_position_mode()),
                                   dtype=torch.float32, device=next(model.parameters()).device)
    seen = np.zeros(len(frame), dtype=bool)
    for batch in dataset.iter_batches(ablation.date_batch_size):
        output = model(batch)
        if not bool(output.availability.all()):
            raise ValueError(f"{candidate} has unavailable rows in the matched panel.")
        rows = batch.row_positions.reshape(-1)
        if (rows < 0).any() or (rows >= len(frame)).any() or seen[rows].any() or len(np.unique(rows)) != len(rows):
            raise ValueError("Prediction dataset repeats or omits backtest rows.")
        logits[rows] = output.logits.reshape(-1, 3).detach().cpu().numpy()
        classes = torch.softmax(output.logits, dim=-1)
        probabilities[rows] = classes.reshape(-1, 3).detach().cpu().numpy()
        positions[rows] = (classes @ coefficients).reshape(-1).detach().cpu().numpy()
        seen[rows] = True
    if not seen.all() or not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError("Prediction export requires finite logits for every backtest row.")
    dates = pd.to_datetime(frame[config.date_col], utc=True).reset_index(drop=True)
    tickers = frame[config.group_col].astype(str).reset_index(drop=True)
    known = frame["_label_known"].to_numpy(dtype=bool)
    labels = frame["Label_id"].to_numpy(dtype=np.int64)
    result = pd.DataFrame({
        "date": dates, "ticker": tickers,
        "adj_close": frame[config.price_col].to_numpy(dtype=np.float64),
        "variant_id": candidate, "candidate": candidate, "fold": fold, "seed": seed, "partition": partition,
        "backtest_key": dates.map(lambda value: value.isoformat()) + "|" + tickers,
        "row_position": np.arange(len(frame)), "available": True,
        "position": positions,
        "label_known": known, "label": np.where(known, labels, -1),
        "graph_degree": _degrees(graphs, frame, dataset.tickers, candidate, config.date_col),
    })
    for index, name in enumerate(("sell", "hold", "buy")):
        result[f"logit_{name}"] = logits[:, index]
        result[f"p_{name}"] = probabilities[:, index]
    return result


def daily_path_frame(panel, positions, loss, *, candidate, fold, seed, partition,
                     initial_capital=10000.):
    from trading_system.experiments.graph_ablation import _daily_paths

    result = _daily_paths(panel, positions, loss)
    result["candidate"], result["fold"], result["seed"], result["partition"] = candidate, fold, seed, partition
    result["equity"] = initial_capital * np.cumprod(1 + result.net_return.to_numpy())
    return result


def _replay_graph_ablation(run_dir, output_dir, frame, *, export, graph_context=None,
                          market_frame=None, candidates=None, folds=None, seeds=None,
                          initial_train_fraction=None, inner_val_fraction=None,
                          gap_bars=None, embargo_bars=None, device=None,
                          allow_hash_mismatch=False, allow_provenance_mismatch=False,
                          restore_checkpoint_feature_order=False):
    """Replay inner and outer partitions, never train or mutate source artifacts.

    ``frame`` must already have the same ticker selection and feature-source
    attachment as the benchmark. All contexts are hashed before any checkpoint
    is opened. An override always makes the resulting export non-reusable.
    """
    import torch
    from trading_system.experiments.graph_ablation import _make_model

    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Replay output already exists: {output_dir}")
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("Replay output must be outside the source benchmark directory.")
    report = json.loads((run_dir / "report.json").read_text())
    metadata = report["metadata"]
    if restore_checkpoint_feature_order and metadata.get("schema_version", 1) >= 2:
        raise ValueError("Checkpoint feature-order restoration is an explicit legacy-only diagnostic.")
    if report.get("final_test") != [] or metadata.get("final_holdout_opened") is not False:
        raise ValueError("Replay requires a sealed final holdout.")
    config = config_from_metadata(metadata)
    loss = FinancialLossConfig(**metadata["loss_config"])
    ablation = GraphAblationConfig(**metadata["ablation"])
    candidates = ablation.candidates if candidates is None else tuple(candidates)
    if not candidates or len(set(candidates)) != len(candidates) or not set(candidates) <= set(ablation.candidates):
        raise ValueError("Requested candidates must be distinct candidates present in the benchmark.")
    wanted_folds = set(range(metadata["n_splits"])) if folds is None else set(folds)
    wanted_seeds = set(metadata["seeds"]) if seeds is None else set(seeds)
    if not wanted_folds or not wanted_folds <= set(range(metadata["n_splits"])):
        raise ValueError("Requested folds are not in the benchmark.")
    if not wanted_seeds or not wanted_seeds <= set(metadata["seeds"]):
        raise ValueError("Requested seeds are not in the benchmark.")
    work = _align_position_calendar(_filter_universe(frame, config), config)
    replay_hash = hash_dataframe(work)
    mismatches = []
    for name, current, expected in (
        ("dataset", replay_hash, metadata["dataset_sha256"]),
        ("graph_context", hash_dataframe(graph_context) if graph_context is not None else None,
         metadata.get("graph_context_sha256")),
        ("market_context", hash_dataframe(market_frame) if market_frame is not None else None,
         metadata.get("market_context_sha256")),
    ):
        if current != expected:
            mismatches.append(name)
    if mismatches and not allow_hash_mismatch:
        raise ValueError(f"Input data or feature-source choices differ from benchmark: {', '.join(mismatches)}.")
    cv_spec = _resolve_cv(metadata, initial_train_fraction=initial_train_fraction,
                          inner_val_fraction=inner_val_fraction, gap_bars=gap_bars,
                          embargo_bars=embargo_bars)
    cv_folds, final_split = expanding_calendar_folds(work, **cv_spec, date_col=config.date_col)
    serialized_folds = [{"fold": fold["fold"], "split": asdict(fold["split"]), "end": fold["end"]}
                        for fold in cv_folds]
    if asdict(final_split) != metadata["final_split"]:
        raise ValueError("Replayed CV boundaries differ from benchmark metadata.")
    if metadata.get("cv_folds") is not None and serialized_folds != metadata["cv_folds"]:
        raise ValueError("Replayed outer fold boundaries differ from benchmark metadata.")
    rows = report["folds"]
    saved = {(row["candidate"], row["fold"], row["seed"]): row for row in rows}
    if len(saved) != len(rows):
        raise ValueError("Graph-ablation rows contain duplicate tasks.")
    # New-format provenance checks and full preprocessing validation are kept
    # separate from metric checks so legacy reports cannot become certified.
    provenance = _replay_provenance(metadata, allow_provenance_mismatch)
    verified = []
    market_columns = tuple(metadata.get("market_columns", ()))
    resolved_device = resolve_device(config.device if device is None else device, torch)
    try:
        training_device = resolve_device(config.device, torch)
    except RuntimeError:
        if not allow_provenance_mismatch:
            raise
        training_device = None
    if training_device is None or str(resolved_device) != str(training_device):
        if not allow_provenance_mismatch:
            raise ValueError("Replay device differs from benchmark numerical provenance.")
        provenance = {**provenance, "verified": False, "reason": "Replay numerical device override."}
    model_config = replace(config, device=str(resolved_device))
    feature_order_audits = []
    for fold in cv_folds:
        fold_id = fold["fold"]
        if fold_id not in wanted_folds:
            continue
        if pd.Timestamp(fold["end"]) >= pd.Timestamp(final_split.test_start):
            raise ValueError("A requested outer fold reaches the final holdout.")
        fold_frame = work.loc[_dates(work, config.date_col) <= pd.Timestamp(fold["end"])].copy()
        prepared, outer, outer_history = _outer_fold(fold_frame, config, ablation, fold)
        if restore_checkpoint_feature_order:
            canonical_path = run_dir / f"fold-{fold_id}-{candidates[0]}-seed-{min(wanted_seeds)}.pt"
            canonical = load_graph_checkpoint(canonical_path, torch)
            feature_order_audits.append({"fold": fold_id,
                                        **_restore_feature_order(prepared, canonical)})
        fold_market, market_scaler = _scaled_market_for_fold(
            market_frame, market_columns, prepared.history_validation, config,
        )
        for candidate in candidates:
            _, market_gate = _candidate_parts(candidate)
            datasets = {}
            for partition, target, history in (
                ("inner", prepared.validation, prepared.history_validation),
                ("outer", outer, outer_history),
            ):
                graphs, _ = _graphs(fold_frame, config, prepared, ablation, candidate, target,
                                     graph_context=graph_context,
                                     sector_context_columns=metadata.get("sector_context_columns"))
                dataset = _dataset(target, history, prepared.columns, config, graphs,
                                   market=fold_market, market_columns=market_columns,
                                   require_market=market_gate)
                datasets[partition] = (target, dataset, graphs)
            for seed in sorted(wanted_seeds):
                key = (candidate, fold_id, seed)
                row = saved.get(key)
                if row is None or row.get("status") != "ok":
                    raise ValueError(f"Missing completed checkpoint: {candidate} fold={fold_id} seed={seed}.")
                if metadata.get("schema_version", 1) >= 2:
                    from trading_system.artifacts.multimodal_study import artifact_path, training_signature
                    from trading_system.experiments.graph_ablation import _row_root, _task_spec, _validated_row

                    row = _validated_row(row, run_dir)
                    checkpoint_path = artifact_path(row["model_artifact"], _row_root(run_dir, row))
                else:
                    checkpoint_path = run_dir / f"fold-{fold_id}-{candidate}-seed-{seed}.pt"
                checkpoint = load_graph_checkpoint(checkpoint_path, torch)
                if metadata.get("schema_version", 1) >= 2:
                    try:
                        spec = _task_spec(metadata, prepared, fold, candidate, seed, ablation, market_scaler)
                    except RuntimeError:
                        if not allow_provenance_mismatch:
                            raise
                        fallback = {**metadata, "config": {**metadata["config"], "device": str(resolved_device)}}
                        spec = _task_spec(fallback, prepared, fold, candidate, seed, ablation, market_scaler)
                        spec["config"] = metadata["config"]
                    if not provenance["verified"]:
                        # The signed original device remains part of training
                        # identity; an explicitly different replay runtime is
                        # exploratory only, not a newly certified training task.
                        spec["resolved_device"] = checkpoint.get("training_spec", {}).get("resolved_device")
                    expected_signature = training_signature(spec)
                    if row["task_signature"] != expected_signature:
                        raise ValueError("Completed task signature differs from reconstructed replay inputs.")
                _verify_preprocessing(checkpoint, prepared, market_scaler, market_columns,
                                      candidate, fold_id, seed, row, metadata)
                if metadata.get("schema_version", 1) >= 2:
                    if (training_signature(checkpoint.get("training_spec", {})) != expected_signature
                            or checkpoint.get("model_spec") != spec["model"]):
                        raise ValueError("Checkpoint effective training/model signature differs from reconstructed task.")
                sample = next(datasets["inner"][1].iter_batches(1))
                model = _make_model(sample, candidate, model_config, metadata["gru_parameters"],
                                    ablation, seed, torch).to(resolved_device)
                model.load_state_dict(checkpoint["model_state"], strict=True)
                model.eval()
                task_metrics = {}
                for partition, (target, dataset, graphs) in datasets.items():
                    with torch.no_grad():
                        current = prediction_frame(model, dataset, target, candidate=candidate,
                                                   fold=fold_id, seed=seed, partition=partition,
                                                   config=config, ablation=ablation, torch=torch,
                                                   graphs=graphs)
                    positions = current.position.to_numpy()
                    panel = ReturnPanel(target, price_col=config.price_col,
                                        date_col=config.date_col, group_col=config.group_col,
                                        execution_delay=config.execution_delay)
                    metrics = panel.metrics(positions, loss, config.initial_capital)
                    _metric_match(metrics, row[f"{partition}_metrics"], candidate=candidate,
                                  fold=fold_id, seed=seed, partition=partition)
                    if partition == "outer" and row.get("classification") is not None:
                        classification = _classification(target, current[["p_sell", "p_hold", "p_buy"]].to_numpy())
                        _metric_match(classification, row["classification"], candidate=candidate,
                                      fold=fold_id, seed=seed, partition="classification")
                    export.append(current, daily_path_frame(
                        panel, positions, loss, candidate=candidate, fold=fold_id,
                        seed=seed, partition=partition, initial_capital=config.initial_capital,
                    ))
                    task_metrics[partition] = metrics
                verified.append({"candidate": candidate, "fold": fold_id, "seed": seed,
                                 "task_signature": row.get("task_signature"),
                                 "preprocessing_signature": row.get("preprocessing_signature"),
                                 "metrics": task_metrics})
                print(f"replayed {candidate} seed={seed} fold={fold_id} inner+outer", flush=True)
    result = {"schema_version": 2, "source_run": str(run_dir),
              "source_dataset_sha256": metadata["dataset_sha256"],
              "replay_dataset_sha256": replay_hash, "hash_mismatch_override": bool(mismatches),
              "mismatches": mismatches, "source_provenance": provenance,
              "final_holdout_opened": False,
              "reusable": provenance["verified"] and not mismatches,
              "exploratory": True, "cv_spec": cv_spec, "cv_folds": serialized_folds,
              "checkpoint_feature_order_requested": bool(restore_checkpoint_feature_order),
              "checkpoint_feature_order_audits": feature_order_audits,
              "final_split": asdict(final_split), "config": metadata["config"],
              "loss_config": metadata["loss_config"], "tasks": verified}
    export.publish(result)
    return result


def _replay_provenance(metadata, allow_mismatch):
    from trading_system.artifacts.multimodal_study import runtime_provenance, training_signature
    if metadata.get("schema_version", 1) < 2:
        return {"verified": False, "reason": "Legacy run lacks complete CV, source and preprocessing provenance."}
    current = runtime_provenance()
    saved = metadata.get("provenance")
    if saved is None or training_signature(saved) != training_signature(current):
        if not allow_mismatch:
            raise ValueError("Replay source/runtime fingerprint differs from benchmark provenance.")
        return {"verified": False, "reason": "Source/runtime fingerprint override.",
                "saved": saved, "current": current}
    return {"verified": True, "saved": saved, "current": current}


def _verify_preprocessing(checkpoint, prepared, market_scaler, market_columns,
                          candidate, fold_id, seed, row, metadata):
    if (tuple(checkpoint["feature_columns"]) != prepared.columns
            or checkpoint["mode"] != candidate or checkpoint["seed"] != seed
            or not np.allclose(checkpoint["scaler_mean"], prepared.scaler.mean_)
            or not np.allclose(checkpoint["scaler_scale"], prepared.scaler.scale_)
            or tuple(checkpoint.get("market_columns", ())) != market_columns):
        raise ValueError(f"Checkpoint/preprocessing mismatch: {candidate} fold={fold_id} seed={seed}.")
    for suffix in ("mean", "scale"):
        expected = getattr(market_scaler, f"{suffix}_", None)
        stored = checkpoint.get(f"market_scaler_{suffix}")
        if ((expected is None) != (stored is None)
                or expected is not None and not np.allclose(stored, expected)):
            raise ValueError("Checkpoint market scaler differs from reconstructed train-only state.")
    if metadata.get("schema_version", 1) >= 2:
        from trading_system.artifacts.multimodal_study import preprocessing_state, stable_digest
        _, market_gate = _candidate_parts(candidate)
        reconstructed = preprocessing_state(prepared,
                                              market_scaler=market_scaler if market_gate else None,
                                              market_columns=market_columns if market_gate else ())
        if checkpoint.get("preprocessing") != reconstructed:
            raise ValueError("Checkpoint complete preprocessing state differs from reconstructed fold.")
        if (checkpoint.get("task_signature") != row.get("task_signature")
                or row.get("preprocessing_signature") != stable_digest(reconstructed)
                or checkpoint.get("fold") != fold_id):
            raise ValueError("Checkpoint task/fold signature differs from completed task.")


class _ReplayExport:
    """Stream one task partition at a time; publish only after all checks pass."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir).resolve()
        self.temporary = None
        self.stage = None
        self.writers = {}
        self.tasks = set()

    def __enter__(self):
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix=f".{self.output_dir.name}-", dir=self.output_dir.parent,
        )
        self.stage = Path(self.temporary.name) / "export"
        self.stage.mkdir()
        return self

    def append(self, predictions, daily):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if predictions.duplicated(list(PREDICTION_KEYS)).any():
            raise ValueError("Replay export contains duplicate prediction keys.")
        identity = tuple(predictions[column].iat[0] for column in PREDICTION_KEYS[:4])
        if identity in self.tasks:
            raise ValueError("Replay export repeats a task partition.")
        self.tasks.add(identity)
        for name, frame in (("predictions.parquet", predictions), ("daily_paths.parquet", daily)):
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if name not in self.writers:
                self.writers[name] = pq.ParquetWriter(self.stage / name, table.schema)
            self.writers[name].write_table(table)

    def close(self):
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()

    def publish(self, result):
        from trading_system.artifacts.multimodal_study import atomic_write_json, file_record

        self.close()
        result["files"] = [file_record(self.stage / name, root=self.stage)
                           for name in ("predictions.parquet", "daily_paths.parquet")]
        atomic_write_json(self.stage / "replay.json", _nullable_metadata(result))
        if self.output_dir.exists():
            raise FileExistsError(f"Replay output already exists: {self.output_dir}")
        self.stage.rename(self.output_dir)

    def __exit__(self, *_):
        self.close()
        if self.temporary is not None:
            self.temporary.cleanup()


def replay_graph_ablation(run_dir, output_dir, frame, **kwargs):
    """Export verified inner/outer predictions and daily paths into a new folder.

    Legacy runs require explicit CV fractions and always remain non-reusable.
    Data/context or provenance overrides likewise prevent certification.
    """
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Replay output already exists: {output_dir}")
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("Replay output must be outside the source benchmark directory.")
    with _ReplayExport(output_dir) as export:
        return _replay_graph_ablation(run_dir, output_dir, frame, export=export, **kwargs)


__all__ = ["PREDICTION_KEYS", "config_from_metadata", "load_graph_checkpoint",
           "prediction_frame", "daily_path_frame", "replay_graph_ablation"]
