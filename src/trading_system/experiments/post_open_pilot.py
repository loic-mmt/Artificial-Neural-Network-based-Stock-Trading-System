"""Paired GRU pilot using only completed bars and the current opening gap."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import hash_dataframe
from trading_system.data.io import read_parquet_dataset
from trading_system.data.optional_sources import prepare_feature_sources
from trading_system.data.post_open import build_post_open_frame
from trading_system.data.scaling import SequenceStandardizer
from trading_system.data.windows import build_sequence_dataset_with_history
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.position_objectives import _align_position_calendar
from trading_system.experiments.runner import _resolve_sequence_estimator
from trading_system.features.expanded import ExpandedFeatureSelector
from trading_system.models.specs import ModelSelection
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel
from trading_system.training.overfitting import OverfittingControlConfig, TrainOnlyFeatureSelector
from trading_system.training.position_trainer import fit_position_model, predict_positions


def _drop_gap(aligned: pd.DataFrame, X: np.ndarray, source: pd.DataFrame,
              gap_bars: int) -> tuple[pd.DataFrame, np.ndarray]:
    if gap_bars <= 0:
        return aligned, X
    dates = pd.DatetimeIndex(pd.to_datetime(source.date, utc=True)).unique().sort_values()
    forbidden = dates[-gap_bars:]
    keep = ~pd.to_datetime(aligned.date, utc=True).isin(forbidden)
    return aligned.loc[keep].copy().reset_index(drop=True), X[np.asarray(keep)]


def _windows(part: pd.DataFrame, history: pd.DataFrame | None,
             columns: tuple[str, ...], context_len: int) -> tuple[np.ndarray, pd.DataFrame]:
    X, _, aligned = build_sequence_dataset_with_history(
        part, columns, context_len, history_frame=history,
        group_col="ticker", date_col="date", return_aligned_rows=True)
    if not len(X):
        raise ValueError("Post-open fold has no sequence windows.")
    return X, aligned


def prepare_fold(frame: pd.DataFrame, split: dict, outer_end: str,
                 *, context_len: int, max_features: int) -> tuple[dict, tuple[str, ...]]:
    """Fit selectors and fills on train only; return paired causal windows."""
    until = pd.Timestamp(outer_end)
    work = frame.loc[pd.to_datetime(frame.date, utc=True) <= until].copy()
    val_start, outer_start = pd.Timestamp(split["validation_start"]), pd.Timestamp(split["test_start"])
    dates = pd.to_datetime(work.date, utc=True)
    train = work.loc[dates < val_start].copy()
    inner = work.loc[(dates >= val_start) & (dates < outer_start)].copy()
    outer = work.loc[dates >= outer_start].copy()
    if any(part.empty for part in (train, inner, outer)):
        raise ValueError("Post-open fold contains an empty partition.")
    gap = int(split["gap_bars"])
    train_dates = pd.DatetimeIndex(pd.to_datetime(train.date, utc=True)).unique().sort_values()
    inner_dates = pd.DatetimeIndex(pd.to_datetime(inner.date, utc=True)).unique().sort_values()
    fit_train = train.loc[~pd.to_datetime(train.date, utc=True).isin(train_dates[-gap:])].copy() if gap else train
    fit_inner = inner.loc[~pd.to_datetime(inner.date, utc=True).isin(inner_dates[-gap:])].copy() if gap else inner
    if fit_train.empty or fit_inner.empty:
        raise ValueError("Gap removed all train or inner-validation rows.")
    base_columns = tuple(column for column in work if column.startswith("lag1_"))
    coverage = ExpandedFeatureSelector(0.5).fit(fit_train, base_columns)
    selector_config = OverfittingControlConfig(max_features=max_features,
                                               max_feature_correlation=0.95)
    selector = TrainOnlyFeatureSelector(selector_config).fit(
        fit_train, coverage.columns, supervised=False)
    columns = selector.columns
    all_columns = (*columns, "night_pct")
    fills = fit_train[list(all_columns)].median().fillna(0.0)
    for part in (train, inner, outer):
        part[list(all_columns)] = part[list(all_columns)].fillna(fills)
        part["Label_id"] = 1  # Window API contract; never used as a training target.
    common = {}
    for name, selected_columns in (("lagged", columns), ("open_gap", all_columns)):
        X_train, a_train = _windows(train, None, selected_columns, context_len)
        X_inner, a_inner = _windows(inner, train, selected_columns, context_len)
        X_outer, a_outer = _windows(outer, pd.concat([train, inner], ignore_index=True),
                                     selected_columns, context_len)
        a_train, X_train = _drop_gap(a_train, X_train, train, gap)
        a_inner, X_inner = _drop_gap(a_inner, X_inner, inner, gap)
        scaler = SequenceStandardizer().fit(X_train)
        common[name] = {
            "train": (scaler.transform(X_train), a_train),
            "inner": (scaler.transform(X_inner), a_inner),
            "outer": (scaler.transform(X_outer), a_outer),
            "columns": selected_columns,
            "scaler": scaler,
            "fill_values": fills.loc[list(selected_columns)].copy(),
        }
    return common, columns


def execution_sensitivity(panel: ReturnPanel, aligned: pd.DataFrame,
                          positions: np.ndarray, loss: FinancialLossConfig,
                          *, draws: int, seed: int) -> dict:
    """OHLC-bounded price stress, not an observed intraday fill model."""
    base, executed, delta, _, _ = panel.path(positions, loss)
    grouped = [part.sort_values("date") for _, part in aligned.groupby("ticker", sort=True)]
    opens = np.stack([part.open.to_numpy(dtype=float) for part in grouped])
    highs = np.stack([part.high.to_numpy(dtype=float) for part in grouped])
    lows = np.stack([part.low.to_numpy(dtype=float) for part in grouped])
    if (opens <= 0).any() or (highs < opens).any() or (lows > opens).any():
        raise ValueError("Invalid OHLC bounds for execution stress.")
    def stats(fractions: np.ndarray, terminal_fraction: np.ndarray) -> dict:
        fill = lows[:, :-1] + fractions * (highs[:, :-1] - lows[:, :-1])
        trade_cost = delta * (fill - opens[:, :-1]) / opens[:, :-1]
        path = base - trade_cost.mean(axis=0)
        final_fill = lows[:, -1] + terminal_fraction * (highs[:, -1] - lows[:, -1])
        final_delta = -executed[:, -1]
        path[-1] -= float((final_delta * (final_fill - opens[:, -1]) /
                           opens[:, -1]).mean())
        wealth = np.r_[1., np.cumprod(1 + path)]
        std = float(path.std(ddof=0))
        return {
            "net_return": float(wealth[-1] - 1),
            "regularized_sharpe": float(np.sqrt(loss.annualization) * path.mean() /
                                         np.sqrt(std ** 2 + loss.sharpe_epsilon ** 2)),
            "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
        }

    worst = np.where(delta > 0, 1.0, 0.0)
    best = 1.0 - worst
    terminal_worst = np.where(executed[:, -1] < 0, 1.0, 0.0)
    deterministic = {
        "open_proxy": panel.metrics(positions, loss),
        "high_low_worst": stats(worst, terminal_worst),
        "ohlc_midpoint": stats(np.full_like(delta, 0.5),
                               np.full_like(terminal_worst, 0.5)),
        "high_low_best": stats(best, 1.0 - terminal_worst),
    }
    rng = np.random.default_rng(seed)
    samples = [stats(rng.random(delta.shape), rng.random(terminal_worst.shape))
               for _ in range(draws)]
    quantiles = {metric: {str(q): float(np.quantile([row[metric] for row in samples], q))
                          for q in (0.1, 0.5, 0.9)}
                 for metric in ("net_return", "regularized_sharpe", "max_drawdown")}
    return {"deterministic": deterministic, "uniform_ohlc_range_quantiles": quantiles,
            "draws": draws, "not_actual_fills": True}


def run_pilot(data: str | Path, tickers: list[str], reference_cv: str | Path,
              destination: str | Path, *, seeds: tuple[int, ...] = (1, 7, 19),
              folds: tuple[int, ...] = (0, 1, 2), draws: int = 250,
              device: str = "auto", resume: bool = False,
              max_epochs: int | None = None, denoising_config=None,
              recurrent_ablation: bool = False) -> dict:
    if (draws < 1 or not seeds or len(seeds) != len(set(seeds)) or
        not folds or len(folds) != len(set(folds)) or
        (max_epochs is not None and max_epochs < 1)):
        raise ValueError("Draw count and unique seeds must be positive.")
    candidates = ("lagged", "open_gap")
    if recurrent_ablation:
        if denoising_config is not None:
            raise ValueError("Cell and denoising ablations must stay independent.")
        from trading_system.models.neural.attention_reset_gru import (
            ManualGRUClassifier, AttentionResetGRUClassifier,
        )
        from trading_system.models.neural.config import GRUConfig
        from trading_system.models.specs import ModelBuildContext
        candidates = ("native", "manual", "attention_reset")
    if denoising_config is not None:
        from trading_system.models.denoising import DenoisingConfig, fit_denoiser
        if not isinstance(denoising_config, DenoisingConfig):
            raise TypeError("Expected DenoisingConfig.")
        candidates = ("raw", "dae", "attention_dae")
    destination = Path(destination).resolve()
    rows = json.loads((Path(reference_cv) / "folds.json").read_text())
    reference = {(row["fold"], row["seed"]): row for row in rows
                 if row["objective"] == "sharpe" and row["status"] == "ok" and
                 row["parameters"].get("temporal_pooling") == "attention" and
                 row["parameters"].get("input_normalization", "none") == "none"}
    if any((fold, seed) not in reference for fold in folds for seed in seeds):
        raise ValueError("Reference Sharpe N0 fold/seed is missing.")
    raw = read_parquet_dataset(data)
    raw = raw.loc[raw.ticker.isin(tickers)].copy()
    raw, _ = prepare_feature_sources(raw, disabled=True)
    expected_hash = json.loads((Path(reference_cv) / "report.json").read_text())["metadata"]["dataset_sha256"]
    if hash_dataframe(raw) != expected_hash:
        raise ValueError("Input dataset differs from benchmark 04; paired comparison is invalid.")
    template = reference[(folds[0], seeds[0])]
    validation = json.loads((Path(template["artifact_path"]) / "manifest.json").read_text())
    config_values = dict(validation["experiment_parameters"]["config"])
    config_values["model"] = ModelSelection(**config_values["model"])
    config = ExperimentConfig(**config_values)
    config = replace(config, device=device)
    loss = FinancialLossConfig(**validation["experiment_parameters"]["loss_config"])
    aligned_raw = _align_position_calendar(raw, config)
    featured, _ = build_post_open_frame(aligned_raw, groups=config.expanded_feature_groups)
    report = json.loads((Path(reference_cv) / "report.json").read_text())
    holdout_start = pd.Timestamp(report["metadata"]["final_split"]["test_start"])
    featured = featured.loc[pd.to_datetime(featured.date, utc=True) < holdout_start].copy()
    metadata = {"source_cv": str(Path(reference_cv).resolve()), "dataset_sha256": expected_hash,
                "seeds": list(seeds), "folds": list(folds), "draws": draws,
                "device": device, "max_epochs": max_epochs,
                "feature_contract": "completed daily bars through J-1, J open gap only",
                "training_price": "adjusted open proxy J to J+1",
                "execution_stress": "uniform OHLC bounds plus high/low extremes, not actual fills",
                "final_holdout_opened": False}
    if denoising_config is not None:
        metadata.update({"denoising": asdict(denoising_config),
                         "candidates": list(candidates), "protocol": "post_open_denoising_v1"})
    if recurrent_ablation:
        metadata.update({"candidates": list(candidates), "protocol": "post_open_cell_v1",
                         "reset_attention": "softmax(q(h_prev)*k(x)/sqrt(H), dim=hidden)",
                         "reset_gate": "sigmoid(H*alpha*v(x)); MCI-GRU-inspired, not a replica"})
    if destination.exists():
        if not resume or json.loads((destination / "metadata.json").read_text()) != metadata:
            raise FileExistsError(f"Existing incompatible post-open output: {destination}")
        results_path = destination / "results.json"
        results = json.loads(results_path.read_text()) if results_path.exists() else []
    else:
        destination.mkdir(parents=True)
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2))
        results = []
    done = {(row["fold"], row["seed"], row["candidate"]) for row in results}
    for fold in folds:
        exemplar = reference[(fold, seeds[0])]
        prepared, base_columns = prepare_fold(featured, exemplar["split"], exemplar["outer_end"],
                                              context_len=config.context_len, max_features=31)
        for seed in seeds:
            for candidate in candidates:
                if (fold, seed, candidate) in done:
                    continue
                parts = prepared["open_gap" if denoising_config is not None or recurrent_ablation else candidate]
                denoiser = None
                if denoising_config is not None and candidate != "raw":
                    print(f"denoiser fold={fold} seed={seed} candidate={candidate}", flush=True)
                    denoiser = fit_denoiser(
                        parts["train"][0], parts["inner"][0], parts["columns"],
                        config=denoising_config, attention=candidate == "attention_dae",
                        seed=seed, device=device,
                    )
                    parts = {**parts, **{
                        name: (denoiser.transform(parts[name][0]), parts[name][1])
                        for name in ("train", "inner", "outer")
                    }}
                parameters = dict(template["parameters"])
                if max_epochs is not None:
                    parameters["epochs"] = max_epochs
                configured = replace(config, seed=seed,
                                     model=ModelSelection("gru", parameters))
                if recurrent_ablation and candidate != "native":
                    context = ModelBuildContext(input_size=len(parts["columns"]),
                                                context_len=config.context_len,
                                                seed=seed, device=device)
                    training = GRUConfig(**parameters, seed=seed, device=device)
                    cls = ManualGRUClassifier if candidate == "manual" else AttentionResetGRUClassifier
                    model = cls(context, training)
                else:
                    model, _ = _resolve_sequence_estimator(None, configured, len(parts["columns"]))
                train_X, train_rows = parts["train"]
                inner_X, inner_rows = parts["inner"]
                outer_X, outer_rows = parts["outer"]
                panels = [ReturnPanel(part, price_col="adj_open_target", group_col="ticker",
                                      execution_delay=0, allow_same_session=True)
                          for part in (train_rows, inner_rows, outer_rows)]
                fitted = fit_position_model(model, train_X, panels[0], inner_X, panels[1],
                                            loss, configured.resolved_backtest_position_mode())
                positions = predict_positions(model, outer_X, configured.resolved_backtest_position_mode())
                metrics = panels[2].metrics(positions, loss, config.initial_capital)
                stress = execution_sensitivity(panels[2], outer_rows, positions, loss,
                                               draws=draws, seed=10_000 + fold)
                record = {"fold": fold, "seed": seed, "candidate": candidate,
                          "best_epoch": fitted.best_epoch, "metrics": metrics,
                          "stress": stress, "feature_columns": parts["columns"],
                          "reference_close_sharpe": reference[(fold, seed)]["score"]}
                name = f"fold-{fold}-seed-{seed}-{candidate}"
                if denoising_config is not None:
                    record["denoising"] = denoiser.report if denoiser else None
                    record["gru_training_seconds"] = fitted.training_duration_seconds
                    model.torch.save(model.state_dict(), destination / f"{name}-model.pt")
                    if denoiser:
                        model.torch.save(denoiser.checkpoint(), destination / f"{name}-denoiser.pt")
                if recurrent_ablation:
                    record.update({"gru_training_seconds": fitted.training_duration_seconds,
                                   "parameter_count": model.parameter_count(),
                                   "stop_reason": fitted.stop_reason,
                                   "train_loss": fitted.history.train_loss,
                                   "inner_loss": fitted.history.val_loss})
                    model.torch.save(model.state_dict(), destination / f"{name}-model.pt")
                exported = outer_rows[["date", "ticker", "open", "high", "low", "close",
                                       "adj_close", "adj_open_target", "night_pct"]].copy()
                exported["position"] = positions
                exported.to_parquet(destination / f"{name}-positions.parquet", index=False)
                results.append(record)
                pending = destination / "results.json.tmp"
                pending.write_text(json.dumps(results, indent=2))
                pending.replace(destination / "results.json")
                print(f"post_open={len(results)}/{len(folds)*len(seeds)*len(candidates)} "
                      f"{name} sharpe={metrics['regularized_sharpe']:.4f}", flush=True)
    return {"path": str(destination), "completed": len(results), "expected": len(folds)*len(seeds)*len(candidates),
            "final_holdout_opened": False}
