"""Frozen post-open label/loss comparison, independent of the historical runner.

Outer folds are reporting only. Checkpoints and all preprocessing use inner
training/validation, and the final holdout is never used to construct features,
labels, or financial paths. Each completed fit is an integrity-checked task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import json
import math
import os

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import hash_dataframe
from trading_system.artifacts.multimodal_study import (
    _atomic_write, atomic_torch_save, atomic_write_json, file_record,
    runtime_provenance, stable_digest, validate_files,
)
from trading_system.data.post_open import build_post_open_frame
from trading_system.data.scaling import SequenceStandardizer
from trading_system.data.windows import build_sequence_dataset_with_history
from trading_system.features.expanded import ExpandedFeatureSelector
from trading_system.models.neural.gru import create_gru_classifier
from trading_system.models.specs import ModelBuildContext
from trading_system.training.financial_loss import FinancialLossConfig
from trading_system.training.overfitting import OverfittingControlConfig, TrainOnlyFeatureSelector

METHODS = ("intraday_return", "forward_return", "volatility_position")
PROTOCOLS = ("intraday", "overnight")
DECODERS = ("continuous", "sign", "argmax")
FAMILIES = ("cross_entropy", "financial", "hybrid")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LabelLossBenchmarkConfig:
    context_len: int = 60
    max_features: int = 32
    n_folds: int = 3
    folds: tuple[int, ...] | None = None
    seeds: tuple[int, ...] = (1, 7, 19)
    label_methods: tuple[str, ...] = METHODS
    families: tuple[str, ...] = FAMILIES
    protocols: tuple[str, ...] = PROTOCOLS
    decoders: tuple[str, ...] = DECODERS
    holdout_start: str = "2023-06-22"
    start: str = "2005-01-03"
    end: str | None = None
    initial_train_fraction: float = 0.5
    inner_val_fraction: float = 0.2
    gap_bars: int = 5
    label_horizon: int = 10
    forward_threshold: float = 0.002
    volatility_window: int = 20
    long_threshold: float = 1.0
    short_threshold: float = 1.5
    exit_threshold: float = 0.25
    min_holding_period: int = 5
    feature_groups: tuple[str, ...] = ("technical", "market", "sector")
    base_epochs: int = 100
    epoch_multiplier: int = 3
    batch_size: int = 256
    hidden_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20
    min_delta: float = 1e-4
    hybrid_ce_weight: float = 0.5
    cost_bps: float = 5.0
    annualization: int = 252
    sharpe_epsilon: float = 1e-4
    combined_pnl_weight: float = 0.25
    combined_pnl_scale: float = 1e-4
    initial_capital: float = 10000.0
    device: str = "auto"

    def __post_init__(self):
        for name in ("context_len", "max_features", "n_folds", "label_horizon",
                     "volatility_window", "base_epochs", "epoch_multiplier",
                     "batch_size", "hidden_size", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_features < 2:
            raise ValueError("max_features must leave room for a lagged feature and night_pct.")
        for name in ("gap_bars", "min_holding_period"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name, choices in (("label_methods", METHODS), ("families", FAMILIES),
                              ("protocols", PROTOCOLS), ("decoders", DECODERS)):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values) or set(values) - set(choices):
                raise ValueError(f"{name} must contain unique choices from {choices}.")
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.seeds)):
            raise ValueError("seeds must be unique non-negative integers.")
        if self.folds is not None and (not self.folds or len(set(self.folds)) != len(self.folds)
                or any(isinstance(f, bool) or not isinstance(f, int) or not 0 <= f < self.n_folds for f in self.folds)):
            raise ValueError("folds must be unique indices inside n_folds.")
        for name in ("initial_train_fraction", "inner_val_fraction"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be in (0, 1).")
        if not math.isfinite(self.hybrid_ce_weight) or not 0 <= self.hybrid_ce_weight <= 1:
            raise ValueError("hybrid_ce_weight must be in [0, 1].")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("Invalid device.")
        for name in ("holdout_start", "start", "end"):
            value = getattr(self, name)
            if value is not None and pd.isna(pd.to_datetime(value, utc=True, errors="raise")):
                raise ValueError(f"Invalid {name}.")
        if pd.to_datetime(self.start, utc=True) >= pd.to_datetime(self.holdout_start, utc=True):
            raise ValueError("start must precede the closed holdout.")
        if self.end is not None and pd.to_datetime(self.end, utc=True) < pd.to_datetime(self.start, utc=True):
            raise ValueError("end must not precede start.")
        # Validate financial and model parameters before creating any output.
        self.financial_config()
        from trading_system.models.neural.config import GRUConfig
        GRUConfig(**self.model_parameters(), seed=1, device=self.device)
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be finite and positive.")

    def model_parameters(self):
        return {"hidden_size": self.hidden_size, "num_layers": 1,
                "temporal_pooling": "attention", "epochs": self.base_epochs * self.epoch_multiplier,
                "batch_size": self.batch_size, "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay, "early_stopping_patience": self.patience,
                "early_stopping_min_delta": self.min_delta}

    def financial_config(self):
        return FinancialLossConfig(objective="combined", cost_bps=self.cost_bps,
            annualization=self.annualization, sharpe_epsilon=self.sharpe_epsilon,
            combined_pnl_weight=self.combined_pnl_weight, combined_pnl_scale=self.combined_pnl_scale)


def candidate_grid(config):
    """Deduplicate pure finance by protocol and CE across execution protocols."""
    candidates = []
    for family in config.families:
        methods = (None,) if family == "financial" else config.label_methods
        protocols = (None,) if family == "cross_entropy" else config.protocols
        for method in methods:
            for protocol in protocols:
                name = "-".join(str(v) for v in (family, method, protocol) if v is not None)
                candidates.append({"candidate": name, "family": family,
                                   "label_method": method, "training_protocol": protocol})
    return candidates


def benchmark_counts(config):
    grid = candidate_grid(config)
    repeats = len(config.folds if config.folds is not None else range(config.n_folds)) * len(config.seeds)
    paths = sum(len(config.protocols) if c["family"] == "cross_entropy" else 1 for c in grid)
    return {"unique_configurations": len(grid), "fits": len(grid) * repeats,
            "evaluation_paths": paths * len(config.decoders) * repeats}


def calendar_folds(frame, config):
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date, utc=True)).unique().sort_values()
    if dates.isna().any():
        raise ValueError("Dates cannot be missing.")
    if len(dates) and dates.max() >= pd.to_datetime(config.holdout_start, utc=True):
        raise ValueError("Development frame contains closed-holdout dates.")
    initial = int(len(dates) * config.initial_train_fraction)
    if initial < config.context_len + config.gap_bars + 3 or len(dates) - initial < config.n_folds * 3:
        raise ValueError("Insufficient dates for context and requested calendar folds.")
    folds = []
    for fold, block in enumerate(np.array_split(np.arange(initial, len(dates)), config.n_folds)):
        outer_start = int(block[0])
        inner_start = int(outer_start * (1 - config.inner_val_fraction))
        train_last = inner_start - config.gap_bars - 1
        inner_last = outer_start - config.gap_bars - 1
        if train_last < config.context_len or inner_last <= inner_start:
            raise ValueError("Gap leaves insufficient train/inner dates.")
        folds.append({"fold": fold, "train_start": dates[0].isoformat(),
            "train_end": dates[train_last].isoformat(), "inner_start": dates[inner_start].isoformat(),
            "inner_end": dates[inner_last].isoformat(), "outer_start": dates[outer_start].isoformat(),
            "outer_end": dates[int(block[-1])].isoformat(), "gap_bars": config.gap_bars})
    return folds


def prepare_common_fold(featured, fold, config, *, price_frame=None):
    """One label-independent selector, imputer and scaler shared by every fit."""
    dates = pd.to_datetime(featured.date, utc=True)
    until = featured.loc[dates <= pd.Timestamp(fold["outer_end"])].copy()
    price_source = featured if price_frame is None else price_frame
    price_source = price_source.loc[pd.to_datetime(price_source.date, utc=True)
                                    <= pd.Timestamp(fold["outer_end"])].copy()
    fit = until.loc[pd.to_datetime(until.date, utc=True) <= pd.Timestamp(fold["train_end"])].copy()
    columns = tuple(c for c in until if c.startswith("lag1_"))
    coverage = ExpandedFeatureSelector(0.5).fit(fit, columns)
    selector = TrainOnlyFeatureSelector(OverfittingControlConfig(
        max_features=config.max_features - 1, max_feature_correlation=0.95)).fit(
            fit, coverage.columns, supervised=False)
    columns = (*selector.columns, "night_pct")
    fills = fit[list(columns)].replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    until[list(columns)] = until[list(columns)].replace([np.inf, -np.inf], np.nan).fillna(fills)
    until["Label_id"] = 1  # Window API placeholder, never a supervision target.
    # A feature constructor may omit a missing open. It must never be allowed
    # to erase that real date from the execution or context calendar.
    calendar = pd.DatetimeIndex(pd.to_datetime(price_source.date, utc=True)).unique().sort_values()
    valid_keys = set()
    for ticker, group in until.groupby("ticker", sort=False):
        loc = calendar.get_indexer(pd.to_datetime(group.date, utc=True))
        if config.context_len == 1:
            continuous = np.ones(len(group), dtype=bool)
        else:
            # A T-row window needs its final T-1 links to be consecutive global sessions.
            continuous = pd.Series(np.r_[True, np.diff(loc) == 1]).rolling(
                config.context_len - 1, min_periods=config.context_len - 1).sum().eq(
                    config.context_len - 1).to_numpy(copy=True)
            continuous[:config.context_len - 1] = False
        valid_keys.update(zip([ticker] * int(continuous.sum()),
                              pd.to_datetime(group.loc[continuous, "date"], utc=True)))
    parts = {}
    for name, start, end in (("train", "train_start", "train_end"),
                             ("inner", "inner_start", "inner_end"),
                             ("outer", "outer_start", "outer_end")):
        all_dates = pd.to_datetime(until.date, utc=True)
        target = until.loc[all_dates.between(pd.Timestamp(fold[start]), pd.Timestamp(fold[end]))].copy()
        history = until.loc[all_dates < pd.Timestamp(fold[start])].copy()
        X, _, aligned = build_sequence_dataset_with_history(target, columns, config.context_len,
            history_frame=history, group_col="ticker", return_aligned_rows=True)
        if not len(X):
            raise ValueError(f"No sequence windows in {name}.")
        # Missing bars may not masquerade as adjacent sessions inside a context.
        keep = np.array([(t, d) in valid_keys for t, d in zip(aligned.ticker, pd.to_datetime(aligned.date, utc=True))])
        if not keep.all():
            X, aligned = X[keep], aligned.loc[keep].reset_index(drop=True)
        if not len(X):
            raise ValueError(f"No complete-calendar context windows in {name}.")
        prices = price_source.loc[pd.to_datetime(price_source.date, utc=True).between(
            pd.Timestamp(fold[start]), pd.Timestamp(fold[end]))].copy()
        parts[name] = {"X": X, "aligned": aligned, "prices": prices,
                       "calendar": calendar[(calendar >= pd.Timestamp(fold[start])) &
                                             (calendar <= pd.Timestamp(fold[end]))]}
    scaler = SequenceStandardizer().fit(parts["train"]["X"])
    # Scale in place on newly owned windows to avoid a second full window tensor.
    for part in parts.values():
        np.subtract(part["X"], scaler.mean_, out=part["X"])
        np.divide(part["X"], scaler.scale_, out=part["X"])
    preprocessing = {"columns": list(columns), "fill_values": fills.to_dict(),
        "scaler": {k: v.tolist() for k, v in scaler.state_dict().items()},
        "coverage": coverage.state_dict(), "selection": selector.state_dict(),
        "label_independent": True, "fit_end": fold["train_end"],
        "calendar_source": "original_prices" if price_frame is not None else "supplied_frame"}
    return parts, preprocessing


def _write_table(path, frame):
    if str(path).endswith(".parquet"):
        _atomic_write(path, lambda stream: frame.to_parquet(stream, index=False))
    else:
        frame = frame.copy()
        from pandas.api.types import is_object_dtype
        for column in (name for name in frame.columns if is_object_dtype(frame[name].dtype)):
            frame[column] = frame[column].map(lambda value: json.dumps(_json_safe(value), allow_nan=False)
                if isinstance(value, (dict, list, tuple)) else value)
        _atomic_write(path, lambda stream: stream.write(frame.to_csv(index=False).encode("utf-8")))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _supervision(featured, part, method, fold, phase, config):
    from trading_system.labels.post_open_benchmark import build_post_open_labels
    # Labels need prices, not a copy of the entire wide feature table. Persistence
    # resets at the phase boundary; only the preceding volatility context is needed.
    calendar = pd.DatetimeIndex(pd.to_datetime(featured.date, utc=True)).unique().sort_values()
    start = pd.Timestamp(fold[f"{phase}_start"])
    prefix = max(0, int(calendar.searchsorted(start)) - config.volatility_window - 1)
    required = ["date", "ticker", "open", "close", "adj_close"]
    if "adj_open_target" in featured:
        required.append("adj_open_target")
    price_source = featured.loc[pd.to_datetime(featured.date, utc=True).between(
        calendar[prefix], pd.Timestamp(fold[f"{phase}_end"])),
        required]
    labeled = build_post_open_labels(price_source, method, horizon=config.label_horizon,
        volatility_window=config.volatility_window, forward_threshold=config.forward_threshold,
        long_threshold=config.long_threshold, short_threshold=config.short_threshold,
        exit_threshold=config.exit_threshold, min_holding_period=config.min_holding_period,
        cost_bps=config.cost_bps, partition_start=fold[f"{phase}_start"],
        partition_end=fold[f"{phase}_end"])
    aligned = part["aligned"][["date", "ticker"]].merge(labeled[
        ["date", "ticker", "Label_id", "_label_known", "label_score", "label_end_date"]],
        on=["date", "ticker"], how="left", validate="one_to_one")
    if aligned["_label_known"].isna().any():
        raise RuntimeError("Supervision is missing aligned window keys.")
    return aligned


def _evaluate_fit(directory, probabilities, part, panels, labels, candidate, fold, seed, config, *, plots):
    from trading_system.analysis.label_loss_evaluation import (
        classification_metrics, decode_probabilities, evaluate_positions, plot_evaluation,
    )
    classification = []
    for method, target in labels.items():
        classification.append({"candidate": candidate["candidate"], "fold": fold, "seed": seed, "phase": "outer",
            "label_method": method, **classification_metrics(target.Label_id.to_numpy(), probabilities,
                target._label_known.to_numpy(dtype=bool), majority_class=target.attrs.get("train_majority_class"))})
    evaluations = []
    protocols = config.protocols if candidate["family"] == "cross_entropy" else (candidate["training_protocol"],)
    for protocol in protocols:
        for decoder in config.decoders:
            identity = {"candidate": candidate["candidate"], "family": candidate["family"],
                "label_method": candidate["label_method"], "fold": fold, "seed": seed,
                "protocol": protocol, "decoder": decoder}
            result = evaluate_positions(panels[protocol], decode_probabilities(probabilities, decoder),
                config.financial_config(), initial_capital=config.initial_capital, keys=identity)
            name = f"{protocol}-{decoder}"
            for key, suffix in (("daily_paths", "portfolio"), ("position_records", "positions"),
                                ("per_ticker", "tickers"), ("trades", "trades")):
                _write_table(directory / f"{name}-{suffix}.parquet", pd.DataFrame(result[key]))
            atomic_write_json(directory / f"{name}-metrics.json", _json_safe({
                **identity, "metrics": result["metrics"], "exposure_controls": result["exposure_controls"]}))
            evaluations.append({**identity, **result["metrics"], "artifact": name})
            if plots:
                plot_evaluation(directory / f"{name}-plots", result["daily_paths"],
                                position_frame=pd.DataFrame(result["position_records"]), prices=part["prices"])
    _write_table(directory / "classification.csv", pd.DataFrame(classification))
    atomic_write_json(directory / "classification.json", _json_safe(classification))
    atomic_write_json(directory / "evaluations.json", _json_safe(evaluations))
    return evaluations


def _classification_observations(model, parts, supervision, candidate, fold, seed, config):
    """TRAIN/inner diagnostics after restoration, never checkpoint inputs."""
    from trading_system.analysis.label_loss_evaluation import classification_metrics
    from trading_system.training.position_trainer import predict_probabilities
    records = []
    for phase in ("train", "inner"):
        probabilities = predict_probabilities(model, parts[phase]["X"])
        for method, target in supervision[phase].items():
            records.append({"candidate": candidate["candidate"], "fold": fold, "seed": seed,
                "phase": phase, "label_method": method, **classification_metrics(
                    target.Label_id.to_numpy(), probabilities, target._label_known.to_numpy(dtype=bool),
                    majority_class=target.attrs["train_majority_class"])})
    return records


def common_exposure_comparison(positions, config):
    """Reporting-only de-leveraging to the lowest mean exposure in one cohort.

    Scaling an entire target path by a constant scales both gross returns and
    absolute-turnover costs. No prediction is retrained and no leverage is used.
    Timing remains different; a fully flat candidate makes this control degenerate.
    """
    frame = pd.DataFrame(positions)
    means = frame.groupby("candidate").position.apply(lambda q: float(q.abs().mean()))
    common = float(means.min())
    records = []
    for candidate, group in frame.groupby("candidate", sort=True):
        scale = common / means[candidate] if means[candidate] else 0.0
        net = group.groupby("date", sort=True).net_return.mean().to_numpy() * scale
        if (net <= -1).any():
            raise ValueError("Exposure control became insolvent.")
        wealth = np.r_[1., np.cumprod(1 + net)]
        std = float(net.std(ddof=0))
        records.append({"candidate": candidate, "original_mean_exposure": means[candidate],
            "common_mean_exposure": common, "scale": float(scale), "degenerate_cash_control": common == 0,
            "net_return": float(wealth[-1] - 1), "net_pnl": float(config.initial_capital * (wealth[-1] - 1)),
            "net_sharpe": float(np.sqrt(config.annualization) * net.mean() / std) if std > 0 else 0. if np.all(net == 0) else None,
            "regularized_sharpe": float(np.sqrt(config.annualization) * net.mean() /
                                         np.sqrt(std**2 + config.sharpe_epsilon**2)),
            "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
            "turnover": float(group.groupby("date").turnover.mean().sum() * scale),
            "cost_return_sum": float(group.groupby("date").cost.mean().sum() * scale),
            "matching_scope": "mean_only_not_exposure_timing", "selection_use": "outer_reporting_only"})
    return records


def _comparisons(root, records, config, *, prices=None, plots=False):
    """Bound comparison memory to one fold/seed/protocol/decoder at a time."""
    from trading_system.analysis.label_loss_evaluation import compare_opposite_positions
    summary = []
    exposure_records = []
    table = pd.DataFrame(records)
    if table.empty:
        return summary
    for (fold, seed, protocol, decoder), group in table.groupby(["fold", "seed", "protocol", "decoder"]):
        paths = [root / row["fit_directory"] / f"{protocol}-{decoder}-positions.parquet"
                 for row in group.to_dict("records")]
        if len(paths) < 2:
            continue
        positions = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        result = compare_opposite_positions(positions)
        exposure_records.extend({"fold": int(fold), "seed": int(seed), "protocol": protocol,
                                 "decoder": decoder, **row}
                                for row in common_exposure_comparison(positions, config))
        directory = root / "oppositions" / f"fold-{fold}-seed-{seed}" / f"{protocol}-{decoder}"
        # The evaluator returns a summary and contiguous disagreement episodes.
        summaries, episodes = result["summary"], result["episodes"]
        _write_table(directory / "summary.csv", pd.DataFrame(summaries))
        _write_table(directory / "episodes.parquet", pd.DataFrame(episodes))
        if plots:
            from trading_system.analysis.label_loss_evaluation import plot_evaluation, plot_oppositions
            portfolios = pd.concat([pd.read_parquet(root / row["fit_directory"] /
                f"{protocol}-{decoder}-portfolio.parquet") for row in group.to_dict("records")], ignore_index=True)
            plot_evaluation(directory / "plots", portfolios)
            if prices is not None:
                plot_oppositions(directory / "plots", positions, prices)
        summary.extend(pd.DataFrame(summaries).to_dict("records"))
        del positions
    _write_table(root / "oppositions.csv", pd.DataFrame(summary))
    _write_table(root / "common-exposure.csv", pd.DataFrame(exposure_records))
    return summary


def run_label_loss_benchmark(frame, tickers, destination, *, config=None, resume=False, plots=True, progress=True):
    """Run only selected development folds; never open or refit final holdout."""
    from trading_system.training.label_loss_trainer import fit_label_loss_model
    from trading_system.training.post_open_panel import PostOpenReturnPanel
    from trading_system.training.position_trainer import predict_probabilities
    from tqdm.auto import tqdm

    config = config or LabelLossBenchmarkConfig()
    # CUDA's deterministic GEMMs require this before their first context/handle.
    # Keep an explicit user setting; CPU/MPS recipes are unaffected.
    if config.device in ("auto", "cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("An explicit unique ticker universe is required.")
    work = frame.loc[frame.ticker.isin(tickers)].copy()
    missing = set(tickers) - set(work.ticker)
    if missing:
        raise ValueError(f"Tickers not found: {sorted(missing)}")
    work["date"] = pd.to_datetime(work.date, utc=True, errors="raise")
    if work.date.isna().any() or work.duplicated(["date", "ticker"]).any():
        raise ValueError("Unique non-missing ticker/date keys required.")
    work = work.loc[(work.date < pd.to_datetime(config.holdout_start, utc=True)) &
                    (work.date >= pd.to_datetime(config.start, utc=True))].copy()
    if config.end is not None:
        work = work.loc[work.date <= pd.to_datetime(config.end, utc=True)].copy()
    if work.empty or set(tickers) - set(work.ticker):
        raise ValueError("Development range must contain every requested ticker.")
    work = work.sort_values(["ticker", "date"]).reset_index(drop=True)
    raw_hash = hash_dataframe(work)
    # Only requested causal groups enter the model. No sentiment/fundamental API.
    if set(config.feature_groups) - {"technical", "market", "sector"}:
        raise ValueError("This benchmark freezes technical, market and sector inputs only.")
    featured, _ = build_post_open_frame(work, groups=config.feature_groups)
    featured["date"] = pd.to_datetime(featured.date, utc=True)
    folds = calendar_folds(work, config)
    selected = [f for f in folds if config.folds is None or f["fold"] in config.folds]
    provenance = runtime_provenance()
    identity = {"schema_version": SCHEMA_VERSION, "config": asdict(config), "tickers": list(tickers),
        "development_sha256": raw_hash, "source_sha256": provenance["source_sha256"],
        "packages": provenance["runtime"]["packages"], "folds": folds,
        "cuda_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if config.device in ("auto", "cuda") else None}
    digest = stable_digest(identity)
    root = Path(destination).resolve()
    metadata_path = root / "metadata.json"
    if root.exists() and any(root.iterdir()):
        if not resume or not metadata_path.is_file():
            raise FileExistsError("Nonempty output requires --resume and matching metadata.")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("identity_sha256") != digest:
            raise ValueError("Resume inputs, code, package versions, folds or recipe differ.")
    else:
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(metadata_path, _json_safe({**identity, "identity_sha256": digest,
            "runtime": provenance["runtime"], "counts": benchmark_counts(config),
            "final_holdout_opened": False, "checkpoint_selection": "inner_validation_only",
            "feature_contract": "J-1 completed features + gap_open J; no unfinished J OHLCV",
            "label_contract": "post_open_v1; target_position; H sessions J..J+H-1 inclusive",
            "execution": "open proxy, not guaranteed executable fill after observing open",
            "allocation": "fixed ticker slots q/N; no leverage, unused slots cash",
            "training": "one global AdamW step per epoch; TRAIN class weights; no outer selection"}))
    results = []
    grid = candidate_grid(config)
    with tqdm(total=benchmark_counts(config)["fits"], desc="Label/loss GRU", unit="fit", disable=not progress) as bar:
        for fold in selected:
            parts, preprocessing = prepare_common_fold(featured, fold, config, price_frame=work)
            fold_directory = root / f"fold-{fold['fold']}"
            atomic_write_json(fold_directory / "preprocessing.json", _json_safe(preprocessing))
            supervision = {phase: {method: _supervision(work, part, method, fold, phase, config)
                           for method in config.label_methods} for phase, part in parts.items()}
            panels = {phase: {protocol: PostOpenReturnPanel(part["prices"], protocol=protocol,
                       tickers=tickers, calendar=part["calendar"], signal_frame=part["aligned"])
                       for protocol in config.protocols} for phase, part in parts.items()}
            atomic_write_json(fold_directory / "execution_contracts.json", _json_safe({
                phase: {protocol: panel.metadata for protocol, panel in group.items()}
                for phase, group in panels.items()}))
            # Perfect labels are explicitly retrospective, with unknowns cash.
            oracle_rows = []
            for method in config.label_methods:
                train_target = supervision["train"][method]
                counts = np.bincount(train_target.loc[train_target._label_known, "Label_id"], minlength=3)
                for phase in supervision:
                    supervision[phase][method].attrs["train_majority_class"] = int(counts.argmax())
                target = supervision["outer"][method]
                q = np.where(target._label_known, target.Label_id - 1, 0).astype(float)
                _write_table(fold_directory / f"{method}-labels.parquet", target)
                for protocol in config.protocols:
                    oracle = evaluate_oracle(panels["outer"][protocol], q, config)
                    oracle_rows.append({"label_method": method, "protocol": protocol,
                        "retrospective_not_realizable": True, **oracle})
            atomic_write_json(fold_directory / "label_oracles.json", _json_safe(oracle_rows))
            for seed in config.seeds:
                for candidate in grid:
                    name = candidate["candidate"]
                    directory = fold_directory / f"seed-{seed}" / name
                    completion = directory / "complete.json"
                    bar.set_postfix(fold=fold["fold"], seed=seed, candidate=name, refresh=False)
                    if completion.is_file():
                        completed = json.loads(completion.read_text())
                        if completed.get("identity_sha256") != digest:
                            raise ValueError(f"Incompatible completed task {directory}.")
                        validate_files(completed["files"], root)
                        evaluations = json.loads((directory / "evaluations.json").read_text())
                    else:
                        # An interrupted fit is restarted, while all completed fits are reused.
                        context = ModelBuildContext(input_size=len(preprocessing["columns"]),
                            context_len=config.context_len, seed=seed, device=config.device)
                        model = create_gru_classifier(context, config.model_parameters())
                        method = candidate["label_method"]
                        train_target = supervision["train"].get(method)
                        inner_target = supervision["inner"].get(method)
                        training_protocol = candidate["training_protocol"]
                        epoch_elapsed = [0.0]
                        def epoch_progress(record):
                            epoch_elapsed[0] += record["duration_seconds"]
                            budget = config.base_epochs * config.epoch_multiplier
                            remaining = epoch_elapsed[0] / record["epoch"] * (budget - record["epoch"])
                            observations = {"epoch": f"{record['epoch']}/{budget}",
                                "val": f"{record['validation']['total_loss']:.4f}",
                                "epoch_cap_eta": f"{remaining / 60:.1f}m"}
                            if str(model.device).startswith("cuda"):
                                observations["VRAM"] = f"{model.torch.cuda.memory_allocated(model.device) / 1024**3:.2f}GiB"
                            bar.set_postfix(fold=fold["fold"], seed=seed, candidate=name,
                                **observations, refresh=True)
                        fit = fit_label_loss_model(model, parts["train"]["X"],
                            None if train_target is None else train_target.Label_id.to_numpy(),
                            None if train_target is None else train_target._label_known.to_numpy(dtype=bool),
                            parts["inner"]["X"],
                            None if inner_target is None else inner_target.Label_id.to_numpy(),
                            None if inner_target is None else inner_target._label_known.to_numpy(dtype=bool),
                            objective=candidate["family"],
                            train_panel=panels["train"].get(training_protocol),
                            val_panel=panels["inner"].get(training_protocol),
                            loss_config=config.financial_config(), hybrid_ce_weight=config.hybrid_ce_weight,
                            progress_callback=epoch_progress)
                        probabilities = predict_probabilities(model, parts["outer"]["X"])
                        # The fit checkpoint is saved before any reporting on OUTER.
                        atomic_torch_save(directory / "checkpoint.pt", model.state_dict())
                        atomic_write_json(directory / "learning_diagnostics.json", _json_safe(model.learning_diagnostics_))
                        if plots:
                            from trading_system.analysis.label_loss_evaluation import plot_learning_trace
                            plot_learning_trace(directory / "learning-plots", model.learning_trace_)
                        prediction_table = parts["outer"]["aligned"][["date", "ticker"]].copy()
                        prediction_table[["p_short", "p_flat", "p_long"]] = probabilities
                        _write_table(directory / "probabilities.parquet", prediction_table)
                        evaluations = _evaluate_fit(directory, probabilities, parts["outer"], panels["outer"],
                            supervision["outer"], candidate, fold["fold"], seed, config, plots=plots)
                        classification = json.loads((directory / "classification.json").read_text())
                        classification += _classification_observations(model, parts, supervision,
                            candidate, fold["fold"], seed, config)
                        _write_table(directory / "classification.csv", pd.DataFrame(classification))
                        atomic_write_json(directory / "classification.json", _json_safe(classification))
                        atomic_write_json(directory / "fit.json", _json_safe({**candidate,
                            "fold": fold["fold"], "seed": seed, "best_epoch": fit.best_epoch,
                            "stop_reason": fit.stop_reason, "epochs_ran": len(fit.history.train_loss),
                            "training_duration_seconds": fit.training_duration_seconds,
                            "parameter_count": fit.parameter_count, "device": fit.device,
                            "budget_insufficient": model.learning_diagnostics_["budget_insufficient"],
                            "preprocessing_sha256": stable_digest(preprocessing)}))
                        files = [file_record(p, root) for p in sorted(directory.rglob("*"))
                                 if p.is_file() and p.name != "complete.json"]
                        atomic_write_json(completion, {"identity_sha256": digest, "files": files})
                        del model, probabilities
                    relative = directory.relative_to(root).as_posix()
                    fit_metadata = json.loads((directory / "fit.json").read_text())
                    run_fields = {key: fit_metadata[key] for key in ("best_epoch", "stop_reason", "epochs_ran",
                                  "training_duration_seconds", "parameter_count", "budget_insufficient")}
                    results.extend({**row, **run_fields, "fit_directory": relative} for row in evaluations)
                    _write_table(root / "results.csv", pd.DataFrame(results))
                    bar.update(1)
            del parts, supervision, panels
            gc.collect()
    opposition = _comparisons(root, results, config, prices=featured, plots=plots)
    result_table = pd.DataFrame(results)
    group_columns = ["candidate", "family", "protocol", "decoder"]
    metrics = ["net_return", "net_pnl", "net_sharpe", "regularized_sharpe", "max_drawdown",
               "mean_abs_position", "turnover", "outperformance_vs_always_long",
               "outperformance_vs_exposure_matched_long"]
    aggregate = result_table.groupby(group_columns)[metrics].agg(["count", "mean", "std", "min", "max"])
    aggregate.columns = [f"{metric}_{statistic}" for metric, statistic in aggregate.columns]
    _write_table(root / "summary.csv", aggregate.reset_index())
    classification = pd.concat([pd.read_csv(root / directory / "classification.csv")
        .assign(fit_directory=directory) for directory in sorted(result_table.fit_directory.unique())], ignore_index=True)
    _write_table(root / "classification.csv", classification)
    report = {"schema_version": SCHEMA_VERSION, "identity_sha256": digest,
        "counts": benchmark_counts(config), "completed_fits": len({r["fit_directory"] for r in results}),
        "completed_evaluation_paths": len(results), "opposition_comparisons": len(opposition),
        "budget_insufficient_fits": int(result_table.drop_duplicates("fit_directory").budget_insufficient.sum()),
        "final_holdout_opened": False, "selection_performed_on_outer": False}
    atomic_write_json(root / "report.json", report)
    return report


def evaluate_oracle(panel, positions, config):
    return panel.metrics(positions, config.financial_config(), initial_capital=config.initial_capital)


__all__ = ["LabelLossBenchmarkConfig", "candidate_grid", "benchmark_counts", "calendar_folds",
           "prepare_common_fold", "common_exposure_comparison", "run_label_loss_benchmark"]
