"""Matched post-open GRU/StockMixer/asset-attention financial ablation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd
import torch

from trading_system.artifacts.experiment import hash_dataframe
from trading_system.data.io import read_parquet_dataset
from trading_system.data.optional_sources import prepare_feature_sources
from trading_system.data.post_open import build_post_open_frame
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.position_objectives import _align_position_calendar
from trading_system.experiments.post_open_pilot import prepare_fold, execution_sensitivity
from trading_system.models.neural.config import GRUConfig
from trading_system.models.neural.trainer import resolve_device, seed_torch_run
from trading_system.models.specs import ModelSelection
from trading_system.models.stock_mixer_gru import MIXERS, AssetMixer, StockMixerGRU
from trading_system.training.financial_loss import (
    FinancialLossConfig, ReturnPanel, position_coefficients,
)


@dataclass(frozen=True)
class DateGroupedSequences:
    values: np.ndarray
    mask: np.ndarray
    rows: np.ndarray
    row_count: int
    tickers: tuple[str, ...]


def group_sequences(values: np.ndarray, aligned: pd.DataFrame,
                    tickers: tuple[str, ...]) -> DateGroupedSequences:
    """Group complete causal windows by session, retaining original row indices."""
    if values.ndim != 3 or len(values) != len(aligned) or not np.isfinite(values).all():
        raise ValueError("Expected finite windows aligned to rows.")
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError("Ticker slots must be unique and non-empty.")
    sessions = pd.to_datetime(aligned.date, utc=True)
    dates = pd.DatetimeIndex(sessions.unique()).sort_values()
    date_lookup = {date: i for i, date in enumerate(dates)}
    asset_lookup = {ticker: i for i, ticker in enumerate(tickers)}
    grid = np.zeros((len(dates), len(tickers), *values.shape[1:]), dtype=np.float32)
    rows = np.full((len(dates), len(tickers)), -1, dtype=np.int64)
    for row, (date, ticker) in enumerate(zip(sessions, aligned.ticker, strict=True)):
        if ticker not in asset_lookup:
            raise ValueError(f"Unexpected ticker: {ticker}")
        location = (date_lookup[date], asset_lookup[ticker])
        if rows[location] != -1:
            raise ValueError("Duplicate session/ticker window.")
        grid[location] = values[row]
        rows[location] = row
    return DateGroupedSequences(grid, rows >= 0, rows, len(values), tickers)


def _positions(model: StockMixerGRU, grouped: DateGroupedSequences,
               coefficients: torch.Tensor, device: torch.device,
               batch_dates: int) -> np.ndarray:
    result = np.empty(grouped.row_count, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(grouped.values), batch_dates):
            stop = start + batch_dates
            mask = torch.as_tensor(grouped.mask[start:stop], device=device)
            sequence = torch.as_tensor(grouped.values[start:stop], device=device)
            predicted = torch.softmax(model(sequence, mask), dim=-1) @ coefficients
            valid = grouped.mask[start:stop]
            result[grouped.rows[start:stop][valid]] = predicted.cpu().numpy()[valid]
    return result


def fit_grouped_position_model(model: StockMixerGRU, train: DateGroupedSequences,
                               train_panel: ReturnPanel, inner: DateGroupedSequences,
                               inner_panel: ReturnPanel, loss: FinancialLossConfig,
                               config: GRUConfig, position_mode: str,
                               device: torch.device) -> dict:
    """Exact whole-path Sharpe gradient, replayed in date-batched model passes."""
    if train.row_count != train_panel.rows or inner.row_count != inner_panel.rows:
        raise ValueError("Grouped sequences and financial panels are not aligned.")
    started = perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    coefficients = torch.as_tensor(position_coefficients(position_mode),
                                   dtype=torch.float32, device=device)
    batch_dates = max(1, config.batch_size // len(train.tickers))
    best_loss, best_epoch, best_state, stale = np.inf, 0, None, 0
    train_losses: list[float] = []
    inner_losses: list[float] = []
    stop_reason = "max_epochs"
    for epoch in range(config.epochs):
        replay_seed = (config.seed + epoch + 1) % (2**32)
        seed_torch_run(replay_seed, config.deterministic, torch)
        model.train()
        first = np.empty(train.row_count, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(train.values), batch_dates):
                stop = start + batch_dates
                mask = torch.as_tensor(train.mask[start:stop], device=device)
                sequence = torch.as_tensor(train.values[start:stop], device=device)
                output = torch.softmax(model(sequence, mask), dim=-1) @ coefficients
                valid = train.mask[start:stop]
                first[train.rows[start:stop][valid]] = output.cpu().numpy()[valid]
        _, gradient = train_panel.loss_and_gradient(first, loss)
        optimizer.zero_grad(set_to_none=True)
        seed_torch_run(replay_seed, config.deterministic, torch)
        model.train()
        for start in range(0, len(train.values), batch_dates):
            stop = start + batch_dates
            mask = torch.as_tensor(train.mask[start:stop], device=device)
            sequence = torch.as_tensor(train.values[start:stop], device=device)
            output = torch.softmax(model(sequence, mask), dim=-1) @ coefficients
            valid = train.mask[start:stop]
            indices = train.rows[start:stop][valid]
            # Scatter the upstream gradient in NumPy, not differentiable
            # tensor indexing: the latter has a non-deterministic MPS backward.
            upstream = np.zeros(valid.shape, dtype=np.float32)
            upstream[valid] = gradient[indices]
            output.backward(torch.as_tensor(upstream, dtype=output.dtype, device=device))
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters()):
            raise FloatingPointError("Non-finite cross-asset financial gradient.")
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        train_loss, _ = train_panel.loss_and_gradient(
            _positions(model, train, coefficients, device, batch_dates), loss)
        inner_loss, _ = inner_panel.loss_and_gradient(
            _positions(model, inner, coefficients, device, batch_dates), loss)
        train_losses.append(float(train_loss))
        inner_losses.append(float(inner_loss))
        if inner_loss < best_loss - config.early_stopping_min_delta:
            best_loss, best_epoch, stale = inner_loss, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.early_stopping_patience:
            stop_reason = "early_stopping"
            break
    if best_state is None:
        raise RuntimeError("No finite cross-asset checkpoint.")
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "stop_reason": stop_reason,
            "train_loss": train_losses, "inner_loss": inner_losses,
            "training_seconds": perf_counter() - started,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters())}


def run_stock_mixer_ablation(data: str | Path, tickers: list[str],
                             reference_cv: str | Path, destination: str | Path,
                             *, seeds: tuple[int, ...] = (1, 7, 19),
                             folds: tuple[int, ...] = (0, 1, 2),
                             candidates: tuple[AssetMixer, ...] = MIXERS,
                             market_states: int = 3, attention_heads: int = 1,
                             draws: int = 250, device: str = "auto",
                             resume: bool = False, max_epochs: int | None = None) -> dict:
    """Run candidate models against the sealed, matched post-open protocol."""
    if (not seeds or len(set(seeds)) != len(seeds) or not folds or
        len(set(folds)) != len(folds) or not candidates or
        len(set(candidates)) != len(candidates) or any(c not in MIXERS for c in candidates) or
        draws < 1 or market_states < 1 or attention_heads < 1 or
        (max_epochs is not None and max_epochs < 1)):
        raise ValueError("Invalid cross-asset ablation selection or budget.")
    destination = Path(destination).resolve()
    reference_cv = Path(reference_cv).resolve()
    rows = json.loads((reference_cv / "folds.json").read_text())
    reference = {(row["fold"], row["seed"]): row for row in rows
                 if row["objective"] == "sharpe" and row["status"] == "ok" and
                 row["parameters"].get("temporal_pooling") == "attention" and
                 row["parameters"].get("input_normalization", "none") == "none"}
    if any((fold, seed) not in reference for fold in folds for seed in seeds):
        raise ValueError("Reference Sharpe N0 fold/seed is missing.")
    raw = read_parquet_dataset(data)
    raw = raw.loc[raw.ticker.isin(tickers)].copy()
    raw, _ = prepare_feature_sources(raw, disabled=True)
    report = json.loads((reference_cv / "report.json").read_text())
    expected_hash = report["metadata"]["dataset_sha256"]
    if hash_dataframe(raw) != expected_hash:
        raise ValueError("Input dataset differs from benchmark 04.")
    template = reference[(folds[0], seeds[0])]
    manifest = json.loads((Path(template["artifact_path"]) / "manifest.json").read_text())
    config_values = dict(manifest["experiment_parameters"]["config"])
    config_values["model"] = ModelSelection(**config_values["model"])
    config = replace(ExperimentConfig(**config_values), device=device)
    loss = FinancialLossConfig(**manifest["experiment_parameters"]["loss_config"])
    aligned_raw = _align_position_calendar(raw, config)
    featured, _ = build_post_open_frame(aligned_raw, groups=config.expanded_feature_groups)
    holdout_start = pd.Timestamp(report["metadata"]["final_split"]["test_start"])
    featured = featured.loc[pd.to_datetime(featured.date, utc=True) < holdout_start].copy()
    slots = tuple(sorted(tickers))
    if market_states >= len(slots) and "stock_mixer" in candidates:
        raise ValueError("Market states must be fewer than ticker slots.")
    metadata = {"source_cv": str(reference_cv), "dataset_sha256": expected_hash,
                "feature_contract": "completed daily bars through J-1, J open gap only",
                "price_contract": "adjusted open proxy J to J+1; not actual fills",
                "seeds": list(seeds), "folds": list(folds), "candidates": list(candidates),
                "ticker_slots": list(slots), "market_states": market_states,
                "attention_heads": attention_heads, "draws": draws,
                "device": device, "max_epochs": max_epochs,
                "final_holdout_opened": False}
    if destination.exists():
        if not resume or json.loads((destination / "metadata.json").read_text()) != metadata:
            raise FileExistsError(f"Existing incompatible output: {destination}")
        results_path = destination / "results.json"
        results = json.loads(results_path.read_text()) if results_path.exists() else []
    else:
        destination.mkdir(parents=True)
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2))
        results = []
    done = {(row["fold"], row["seed"], row["candidate"]) for row in results}
    resolved_device = resolve_device(device, torch)
    for fold in folds:
        exemplar = reference[(fold, seeds[0])]
        prepared, _ = prepare_fold(featured, exemplar["split"], exemplar["outer_end"],
                                   context_len=config.context_len, max_features=31)
        parts = prepared["open_gap"]
        grouped = {name: group_sequences(*parts[name], slots)
                   for name in ("train", "inner", "outer")}
        panels = {name: ReturnPanel(parts[name][1], price_col="adj_open_target",
                                   group_col="ticker", execution_delay=0,
                                   allow_same_session=True)
                  for name in grouped}
        for seed in seeds:
            for candidate in candidates:
                if (fold, seed, candidate) in done:
                    continue
                parameters = dict(template["parameters"])
                if max_epochs is not None:
                    parameters["epochs"] = max_epochs
                train_config = GRUConfig(**parameters, seed=seed, device=device)
                seed_torch_run(seed, train_config.deterministic, torch)
                model = StockMixerGRU(
                    input_size=len(parts["columns"]), context_len=config.context_len,
                    assets=len(slots), config=train_config, mixer=candidate,
                    market_states=market_states, attention_heads=attention_heads,
                ).to(resolved_device)
                fitted = fit_grouped_position_model(
                    model, grouped["train"], panels["train"], grouped["inner"],
                    panels["inner"], loss, train_config,
                    config.resolved_backtest_position_mode(), resolved_device,
                )
                coefficients = torch.as_tensor(
                    position_coefficients(config.resolved_backtest_position_mode()),
                    dtype=torch.float32, device=resolved_device,
                )
                batch_dates = max(1, train_config.batch_size // len(slots))
                positions = _positions(model, grouped["outer"], coefficients,
                                       resolved_device, batch_dates)
                metrics = panels["outer"].metrics(positions, loss, config.initial_capital)
                stress = execution_sensitivity(
                    panels["outer"], parts["outer"][1], positions, loss,
                    draws=draws, seed=10_000 + fold,
                )
                name = f"fold-{fold}-seed-{seed}-{candidate}"
                exported = parts["outer"][1][
                    ["date", "ticker", "open", "high", "low", "close", "adj_close",
                     "adj_open_target", "night_pct"]
                ].copy()
                exported["position"] = positions
                exported.to_parquet(destination / f"{name}-positions.parquet", index=False)
                torch.save(model.state_dict(), destination / f"{name}-model.pt")
                results.append({"fold": fold, "seed": seed, "candidate": candidate,
                                "fit": fitted, "metrics": metrics, "stress": stress,
                                "feature_columns": parts["columns"]})
                (destination / "results.json").write_text(json.dumps(results, indent=2))
                print(f"stock_mixer={len(results)}/{len(folds)*len(seeds)*len(candidates)} "
                      f"{name} sharpe={metrics['regularized_sharpe']:.4f}", flush=True)
    return {"path": str(destination), "completed": len(results),
            "expected": len(folds) * len(seeds) * len(candidates),
            "final_holdout_opened": False}
