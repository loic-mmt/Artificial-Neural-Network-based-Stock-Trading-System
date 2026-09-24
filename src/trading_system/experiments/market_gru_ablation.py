"""Matched market-context GRU ablations with a sealed final holdout."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import _nullable_metadata, hash_dataframe
from trading_system.data.multimodal import build_multimodal_dataset
from trading_system.data.purged_cv import expanding_calendar_folds
from trading_system.data.scaling import Standardizer
from trading_system.models.market_gru_ablation import CANDIDATES, MarketGRUControl
from trading_system.models.neural.config import GRUConfig, TransformerConfig
from trading_system.models.neural.trainer import resolve_device, seed_torch_run
from trading_system.models.specs import ModelSelection
from trading_system.reporting.warnings import current_universe_warning
from trading_system.training.financial_loss import ReturnPanel
from .graph_ablation import GraphAblationConfig, _classification, _complete, _dates, _fit, _positions, _prepare, _scale
from .position_objectives import _align_position_calendar
from .runner import _filter_universe, _prepare_splits


@dataclass(frozen=True)
class MarketAblationConfig:
    close_columns: tuple[str, ...] = ("market_close", "vix_close")
    date_batch_size: int = 32
    transformer_width: int = 32
    transformer_heads: int = 4
    transformer_layers: int = 1

    def __post_init__(self):
        if not self.close_columns or len(set(self.close_columns)) != len(self.close_columns):
            raise ValueError("Choose unique market close columns.")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in (self.date_batch_size, self.transformer_width,
                             self.transformer_heads, self.transformer_layers)):
            raise ValueError("Batch size and Transformer dimensions must be positive integers.")
        if self.transformer_width % self.transformer_heads:
            raise ValueError("Transformer width must be divisible by head count.")


def build_close_market_frame(frame: pd.DataFrame, date_col: str,
                             close_columns: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    """Derive causal close-session features; never pretend macro releases are close data."""
    allowed = {"market_close", "vix_close", "dxy_close", "oil_close", "gold_close"}
    if set(close_columns) - allowed:
        raise ValueError(f"Only audited close-series names are allowed: {sorted(set(close_columns) - allowed)}")
    missing = set(close_columns) - set(frame)
    if missing:
        raise ValueError(f"Market close columns absent from input: {sorted(missing)}")
    work = frame[[date_col, *close_columns]].copy()
    work[date_col] = pd.to_datetime(work[date_col], utc=True).dt.normalize()
    grouped = work.groupby(date_col, sort=True)
    conflicts = grouped[list(close_columns)].nunique(dropna=True).gt(1)
    if conflicts.any().any():
        bad = conflicts.stack().loc[lambda values: values].index[0]
        raise ValueError(f"Conflicting global market value at {bad[0]} for {bad[1]}.")
    daily = grouped[list(close_columns)].first().reset_index()
    features = {}
    for column in close_columns:
        series = pd.to_numeric(daily[column], errors="coerce")
        if (series.dropna() <= 0).any():
            raise ValueError(f"Market close series {column} contains nonpositive values.")
        if column == "vix_close":
            features["vix_level"] = series
        features[f"{column}_ret_1"] = series.pct_change(fill_method=None)
    result = pd.DataFrame({date_col: daily[date_col], **features})
    # This is a conservative session-end bound, not a vendor publication timestamp.
    # Predictions use close-J data only for execution on J+1 or later.
    result["source_end"] = daily[date_col] + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    columns = tuple(features)
    report = {"rows": len(result), "features": columns,
              "feature_coverage": {name: float(result[name].notna().mean()) for name in columns},
              "source_end_policy": "close-J assumed available for delayed execution; synthetic session-end upper bound"}
    return result, report


def _scaled_market(market, columns, train, config):
    train_dates = set(_dates(train.loc[train["_fit_eligible"]], config.date_col))
    fit = market.loc[market[config.date_col].isin(train_dates), list(columns)].dropna()
    if fit.empty:
        raise ValueError("No complete train-only market observations for scaling.")
    scaler = Standardizer().fit(fit.to_numpy(dtype=np.float32))
    result = market.copy()
    valid = result[list(columns)].notna().all(axis=1)
    result.loc[valid, list(columns)] = scaler.transform(
        result.loc[valid, list(columns)].to_numpy(dtype=np.float32)
    )
    return result, scaler


def _dataset(target, history, columns, market, market_columns, config):
    dataset = build_multimodal_dataset(
        target, tickers=tuple(sorted(target[config.group_col].unique())),
        context_len=config.context_len, temporal_columns=columns,
        history_frame=history, market_frame=market,
        market_close_columns=market_columns, market_context_len=config.context_len,
        date_col=config.date_col, ticker_col=config.group_col,
    )
    for batch in dataset.iter_batches(128):
        if not batch.asset_mask.all() or not batch.temporal_mask.all():
            raise ValueError("Market benchmark requires complete matched ticker/date GRU windows.")
        if not batch.market_sequence_mask.all():
            first = next(str(day) for day, ok in zip(batch.sessions, batch.market_sequence_mask) if not ok)
            raise ValueError(f"Incomplete market context at {first}; select complete close series or repair source.")
    return dataset


def run_market_gru_ablation(frame, config, loss, gru_parameters, seeds, destination, *,
                            ablation=MarketAblationConfig(), n_splits=3,
                            initial_train_fraction=.5, inner_val_fraction=.2,
                            gap_bars=5, embargo_bars=0, dataset_path=None,
                            resume=False):
    import torch

    if config.universe != "multi" or config.evaluation_mode != "static" or config.execution_delay < 1:
        raise ValueError("Market ablation needs static multi-asset evaluation and delayed execution.")
    if config.label_mode.startswith("oracle") or loss.objective not in ("sharpe", "combined"):
        raise ValueError("Market ablation needs non-oracle labels and Sharpe or combined loss.")
    if config.sample_weighting is not None or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("No sample weighting; seeds must be nonempty and unique.")
    config = replace(config, model=ModelSelection("gru", dict(gru_parameters)))
    target = Path(destination).expanduser().resolve()
    if target.exists() and not resume:
        raise FileExistsError(f"Market ablation output already exists: {target}")
    if resume and not target.is_dir():
        raise FileNotFoundError(f"Cannot resume missing market ablation: {target}")
    work = _align_position_calendar(_filter_universe(frame, config), config)
    market_raw, market_audit = build_close_market_frame(work, config.date_col, ablation.close_columns)
    market_columns = tuple(market_audit["features"])
    folds, final_split = expanding_calendar_folds(
        work, n_splits=n_splits, initial_train_fraction=initial_train_fraction,
        inner_val_fraction=inner_val_fraction,
        final_test_fraction=1 - config.train_ratio - config.val_ratio,
        gap_bars=gap_bars, embargo_bars=embargo_bars, date_col=config.date_col,
    )
    template = GRUConfig(**gru_parameters)
    transformer = TransformerConfig(
        d_model=ablation.transformer_width, n_heads=ablation.transformer_heads,
        num_layers=ablation.transformer_layers,
        dim_feedforward=2 * ablation.transformer_width, dropout=0.0,
        pooling="last", causal_attention=True,
    )
    metadata = _nullable_metadata({
        "config": asdict(config), "loss_config": asdict(loss),
        "gru_parameters": gru_parameters, "ablation": asdict(ablation),
        "seeds": list(seeds), "n_splits": n_splits,
        "dataset_sha256": hash_dataframe(work),
        "dataset_path": str(Path(dataset_path).resolve()) if dataset_path else None,
        "market_audit": market_audit,
        "survivor_bias_warning": current_universe_warning(dataset_path),
        "final_split": asdict(final_split), "final_holdout_opened": False,
        "protocol": "matched_purged_market_gru_ablation",
    })
    if resume:
        if json.loads((target / "metadata.json").read_text()) != metadata:
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
        if all((mode, seed, fold["fold"]) in completed for mode in CANDIDATES for seed in seeds):
            continue
        fold_frame = work.loc[_dates(work, config.date_col) <= pd.Timestamp(fold["end"])].copy()
        fold_config = replace(config, purged_split=fold["split"])
        # Reuse the same split preparation and train-only stock scaler as graph CV;
        # Minimum valid graph lookback is three; context windows dominate it.
        prepared = _prepare(fold_frame, fold_config, GraphAblationConfig(graph_lookback=3))
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
        outer_scaled = _scale(outer, prepared.columns, prepared.scaler)
        history_outer = _scale(pd.concat((raw_train, raw_val), ignore_index=True),
                               prepared.columns, prepared.scaler)
        market_scaled, market_scaler = _scaled_market(
            market_raw, market_columns, raw_train, config,
        )
        train_ds = _dataset(prepared.train, prepared.history_train, prepared.columns,
                            market_scaled, market_columns, config)
        inner_ds = _dataset(prepared.validation, prepared.history_validation, prepared.columns,
                            market_scaled, market_columns, config)
        outer_ds = _dataset(outer_scaled, history_outer, prepared.columns,
                            market_scaled, market_columns, config)
        train_panel = ReturnPanel(prepared.train, price_col=config.price_col,
                                  date_col=config.date_col, group_col=config.group_col,
                                  execution_delay=config.execution_delay)
        inner_panel = ReturnPanel(prepared.validation, price_col=config.price_col,
                                  date_col=config.date_col, group_col=config.group_col,
                                  execution_delay=config.execution_delay)
        outer_panel = ReturnPanel(outer_scaled, price_col=config.price_col,
                                  date_col=config.date_col, group_col=config.group_col,
                                  execution_delay=config.execution_delay)
        for mode in CANDIDATES:
            for seed in seeds:
                if (mode, seed, fold["fold"]) in completed:
                    continue
                seed_torch_run(seed, template.deterministic, torch)
                device = resolve_device(config.device, torch)
                training = replace(template, seed=seed, device=config.device)
                model = MarketGRUControl(mode, len(prepared.columns), len(market_columns),
                                         config.context_len, training, transformer).to(device)
                fitted = _fit(model, train_ds, inner_ds, train_panel, inner_panel,
                              loss, config, training, ablation, torch)
                model.eval()
                with torch.no_grad():
                    inner_positions = _positions(model, inner_ds, mode=mode, config=config,
                                                 batch_dates=ablation.date_batch_size, torch=torch)
                    outer_positions, outer_probabilities = _positions(
                        model, outer_ds, mode=mode, config=config,
                        batch_dates=ablation.date_batch_size, torch=torch,
                        return_probabilities=True,
                    )
                inner_metrics = inner_panel.metrics(inner_positions, loss, config.initial_capital)
                outer_metrics = outer_panel.metrics(outer_positions, loss, config.initial_capital)
                row = {"candidate": mode, "seed": seed, "fold": fold["fold"],
                       "status": "ok", "fit": fitted,
                       "inner_metrics": inner_metrics, "outer_metrics": outer_metrics,
                       "classification": _classification(outer, outer_probabilities),
                       "score": outer_metrics["regularized_sharpe"],
                       "feature_columns": prepared.columns, "market_columns": market_columns,
                       "purging": prepared.purging,
                       "train_dates": len(train_ds), "inner_dates": len(inner_ds),
                       "outer_dates": len(outer_ds)}
                model_path = target / f"fold-{fold['fold']}-{mode}-seed-{seed}.pt"
                torch.save({"model_state": model.state_dict(), "mode": mode, "seed": seed,
                            "stock_scaler_mean": prepared.scaler.mean_,
                            "stock_scaler_scale": prepared.scaler.scale_,
                            "market_scaler_mean": market_scaler.mean_,
                            "market_scaler_scale": market_scaler.scale_,
                            "feature_columns": prepared.columns,
                            "market_columns": market_columns}, model_path)
                row["model_artifact"] = str(model_path)
                rows.append(row)
                (target / "folds.json").write_text(json.dumps(_nullable_metadata(rows), indent=2, allow_nan=False))
                print(f"market_cv={len(rows)}/{len(CANDIDATES)*len(seeds)*len(folds)} "
                      f"{mode} seed={seed} fold={fold['fold']} score={row['score']:.4f}", flush=True)
    summary = []
    reference = {(row["seed"], row["fold"]): row for row in rows if row["candidate"] == "gru"}
    for mode in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == mode]
        scores = np.asarray([row["score"] for row in selected])
        deltas = [row["score"] - reference[row["seed"], row["fold"]]["score"]
                  for row in selected] if mode != "gru" else []
        summary.append({"candidate": mode, "mean": float(scores.mean()),
                        "std": float(scores.std()), "min": float(scores.min()),
                        "mean_net_return": float(np.mean([row["outer_metrics"]["net_return"] for row in selected])),
                        "mean_max_drawdown": float(np.mean([row["outer_metrics"]["max_drawdown"] for row in selected])),
                        "mean_turnover": float(np.mean([row["outer_metrics"]["turnover"] for row in selected])),
                        "mean_abs_position": float(np.mean([row["outer_metrics"]["mean_abs_position"] for row in selected])),
                        "mean_score_delta_vs_gru": float(np.mean(deltas)) if deltas else None,
                        "sharpe_wins_vs_gru": int(np.count_nonzero(np.asarray(deltas) > 0)) if deltas else None,
                        "complete": len(selected) == len(seeds) * len(folds)})
    report = _nullable_metadata({"metadata": metadata, "folds": rows, "summary": summary,
                                 "selected": max(summary, key=lambda item: item["mean"])["candidate"],
                                 "final_test": []})
    (target / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


__all__ = ["MarketAblationConfig", "build_close_market_frame", "run_market_gru_ablation"]
