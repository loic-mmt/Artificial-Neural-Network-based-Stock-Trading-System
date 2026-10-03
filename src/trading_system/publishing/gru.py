"""Publish GRU probabilities and out-of-sample portfolio returns.

The model stays outside the web server. Only validated JSON leaves this module.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import hash_dataframe, save_experiment_artifact
from trading_system.artifacts.serialization import load_model_artifact
from trading_system.backtest.positions import apply_execution_delay, labels_to_positions, position_turnover
from trading_system.data.scaling import SequenceStandardizer
from trading_system.data.windows import build_sequence_features
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import align_probability_columns, run_experiment
from trading_system.features.technical import TECHNICAL_FEATURE_COLUMNS, compute_technical_features
from trading_system.models.neural.gru import create_gru_classifier
from trading_system.models.specs import ModelBuildContext, ModelSelection


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data/processed/cac40_daily_clean.parquet"
UNIVERSE_PATH = ROOT / "configs/benchmark/cac40_diversified_10.json"
LABELS = ("sell", "hold", "buy")


def read_universe(path: Path = UNIVERSE_PATH) -> tuple[str, ...]:
    rows = json.loads(path.read_text(encoding="utf-8"))["tickers"]
    tickers = tuple(row if isinstance(row, str) else row["ticker"] for row in rows)
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Invalid benchmark universe")
    return tickers


def read_market_data(path: Path = DATA_PATH, universe: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Use only requested parquet; reject missing, duplicate or unpriced assets."""
    tickers = universe or read_universe()
    frame = pd.read_parquet(path)
    needed = {"date", "ticker", "company", "open", "high", "low", "close", "adj_close", "volume"}
    if missing := needed - set(frame):
        raise ValueError(f"Market data missing columns: {sorted(missing)}")
    frame = frame.loc[frame["ticker"].isin(tickers)].sort_values(["ticker", "date"]).reset_index(drop=True)
    if set(frame["ticker"]) != set(tickers):
        raise ValueError("Market data does not cover complete benchmark universe")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Market data has duplicate ticker/date rows")
    if frame["date"].isna().any() or frame["adj_close"].isna().any() or (frame["adj_close"] <= 0).any():
        raise ValueError("Market data has invalid dates or adjusted prices")
    return frame


def default_config(*, epochs: int = 12, device: str = "cpu") -> ExperimentConfig:
    return ExperimentConfig(
        universe="multi", feature_set="technical", label_mode="forward_return",
        position_mode="long_only", decision_mode="argmax", forward_horizon=1,
        context_len=20, train_ratio=0.70, val_ratio=0.15, device=device,
        model=ModelSelection("gru", {"epochs": epochs, "hidden_size": 32, "batch_size": 256,
                                     "early_stopping_patience": 4}),
    )


def _features(frame: pd.DataFrame, columns: tuple[str, ...], fill: dict[str, float]) -> pd.DataFrame:
    if tuple(columns) != tuple(TECHNICAL_FEATURE_COLUMNS):
        raise ValueError("This publisher supports only the technical GRU feature schema")
    featured = compute_technical_features(frame)
    numeric = featured[list(columns)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    featured.loc[:, list(columns)] = numeric.fillna(pd.Series(fill)).fillna(0.0)
    return featured


def _predict_rows(frame: pd.DataFrame, model, scaler: SequenceStandardizer, columns: tuple[str, ...], context_len: int, fill: dict[str, float]) -> pd.DataFrame:
    featured = _features(frame, columns, fill)
    parts = []
    for _, group in featured.groupby("ticker", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        windows, indices = build_sequence_features(group, columns, context_len, return_indices=True)
        if len(windows) == 0:
            raise ValueError(f"Insufficient GRU history for {group['ticker'].iloc[0]}")
        probabilities = align_probability_columns(model, model.predict_proba(scaler.transform(windows)))
        aligned = group.iloc[indices][["date", "ticker", "company", "adj_close"]].copy().reset_index(drop=True)
        aligned[["sell", "hold", "buy"]] = probabilities
        aligned["label_id"] = probabilities.argmax(axis=1)
        parts.append(aligned)
    return pd.concat(parts, ignore_index=True)


def _predict_latest_rows(frame: pd.DataFrame, model, scaler: SequenceStandardizer, columns: tuple[str, ...], context_len: int, fill: dict[str, float]) -> pd.DataFrame:
    """Build one sequence per ticker for daily publication."""
    featured = _features(frame, columns, fill)
    parts = []
    for ticker, group in featured.groupby("ticker", sort=False):
        recent = group.sort_values("date").tail(context_len).reset_index(drop=True)
        windows, indices = build_sequence_features(recent, columns, context_len, return_indices=True)
        if len(windows) != 1:
            raise ValueError(f"Insufficient GRU history for {ticker}")
        probabilities = align_probability_columns(model, model.predict_proba(scaler.transform(windows)))
        row = recent.iloc[indices][["date", "ticker", "company", "adj_close"]].copy().reset_index(drop=True)
        row[["sell", "hold", "buy"]] = probabilities
        row["label_id"] = probabilities.argmax(axis=1)
        parts.append(row)
    return pd.concat(parts, ignore_index=True)


def backtest_days(predictions: pd.DataFrame, test_starts: dict[str, str], *, cost_bps: float = 5.0) -> list[dict[str, float | str]]:
    """Equal initial capital per ticker, next-bar execution, fee per turnover."""
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    curves = []
    for ticker, group in predictions.groupby("ticker", sort=False):
        boundary = pd.Timestamp(test_starts[ticker]).tz_localize(None)
        group = group.loc[pd.to_datetime(group["date"]).dt.tz_localize(None) >= boundary].sort_values("date")
        if len(group) < 3:
            raise ValueError(f"Insufficient out-of-sample prices for {ticker}")
        dates = pd.to_datetime(group["date"]).to_numpy()
        prices = group["adj_close"].to_numpy(dtype=np.float64)
        positions = labels_to_positions(group["label_id"].to_numpy(dtype=np.int64), position_mode="long_only", label_semantics="target_position")
        executed = apply_execution_delay(positions, 1)
        turnover = position_turnover(executed)
        # A signal observed at close t executes after a one-bar delay. The
        # position held over t -> t+1 determines the return dated t+1.
        gross = executed[:-1] * (prices[1:] / prices[:-1] - 1.0)
        net = gross - turnover[:-1] * cost_bps / 10_000.0
        capital = np.cumprod(1.0 + net)
        if not np.isfinite(capital).all() or (capital < 0).any():
            raise ValueError("Backtest produced invalid equity")
        curves.append(pd.Series(capital, index=pd.to_datetime(dates[1:]), name=ticker))
    equity = pd.concat(curves, axis=1).sort_index().ffill().fillna(1.0).mean(axis=1)
    prior = equity.shift(1).fillna(1.0)
    returns = (equity / prior - 1.0) * 100.0
    return [{"date": day.date().isoformat(), "return_pct": float(value)} for day, value in returns.items()]


def build_signal_snapshot(predictions: pd.DataFrame, universe: tuple[str, ...], *, market: str = "EURONEXT PARIS") -> dict:
    latest = predictions.sort_values("date").groupby("ticker", sort=False).tail(1).set_index("ticker")
    if set(latest.index) != set(universe):
        raise ValueError("Predictions do not cover full universe")
    rows = []
    for ticker in universe:
        row = latest.loc[ticker]
        probabilities = {label: float(row[label]) for label in LABELS}
        label = LABELS[int(row["label_id"])]
        rows.append({"ticker": ticker, "name": str(row["company"]), "market": market,
                     "signal": label, "probabilities": probabilities,
                     "as_of": pd.Timestamp(row["date"]).date().isoformat()})
    if len({row["as_of"] for row in rows}) != 1:
        raise ValueError("The complete universe must have one common latest market date")
    latest_date = min(pd.Timestamp(row["date"]) for _, row in latest.iterrows())
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    signals = {"generated_at": generated_at, "data_as_of": latest_date.date().isoformat(), "signals": rows}
    return signals


def build_snapshots(predictions: pd.DataFrame, universe: tuple[str, ...], test_starts: dict[str, str], *, cost_bps: float = 5.0, market: str = "EURONEXT PARIS") -> tuple[dict, dict]:
    signals = build_signal_snapshot(predictions, universe, market=market)
    days = backtest_days(predictions, test_starts, cost_bps=cost_bps)
    performance = {"start_date": days[0]["date"], "end_date": days[-1]["date"], "days": days,
                   "methodology": "Out-of-sample, equal initial capital per asset, long or cash, one-bar execution delay, turnover cost in basis points",
                   "cost_bps": cost_bps}
    return signals, performance


def train(frame: pd.DataFrame, artifact_dir: Path, *, epochs: int = 12, device: str = "cpu", data_path: Path = DATA_PATH) -> tuple[object, SequenceStandardizer, tuple[str, ...], int, dict[str, float], dict[str, str]]:
    result = run_experiment(frame, default_config(epochs=epochs, device=device))
    save_experiment_artifact(artifact_dir, frame, result, dataset_path=data_path)
    starts = {ticker: pd.Timestamp(group["date"].min()).isoformat() for ticker, group in result.aligned_test_frame.groupby("ticker")}
    bundle = result.bundle
    return bundle.estimator, bundle.scaler, bundle.feature_columns, bundle.context_len, bundle.feature_fill_values.to_dict(), starts


def load(frame: pd.DataFrame, artifact_dir: Path, *, data_path: Path = DATA_PATH, universe_path: Path = UNIVERSE_PATH) -> tuple[object, SequenceStandardizer, tuple[str, ...], int, dict[str, float], dict[str, str]]:
    manifest, model_state, scaler_state, _ = load_model_artifact(artifact_dir)
    if manifest.model_name != "gru" or tuple(manifest.class_names) != ("Sell", "Hold", "Buy"):
        raise ValueError("Artifact is not a Sell/Hold/Buy GRU")
    metadata = manifest.experiment_parameters
    dataset = metadata["dataset"]
    if Path(dataset["path"]).resolve() != data_path.resolve() or set(dataset["tickers"]) != set(read_universe(universe_path)):
        raise ValueError("Artifact was trained on a different dataset or universe")
    original_end = pd.Timestamp(dataset["date_range"]["end"]).tz_localize(None)
    original = frame.loc[pd.to_datetime(frame["date"]).dt.tz_localize(None) <= original_end].reset_index(drop=True)
    if len(original) != dataset["rows"] or hash_dataframe(original) != dataset["sha256"]:
        raise ValueError("Training data changed; retrain before refreshing")
    config = metadata["config"]
    if config["feature_set"] != "technical" or config["decision_mode"] != "argmax":
        raise ValueError("Unsupported GRU publishing configuration")
    context = ModelBuildContext(len(manifest.feature_columns), manifest.context_len, seed=config["seed"], device="cpu")
    parameters = dict(manifest.model_parameters)
    parameters.pop("device", None)
    model = create_gru_classifier(context, parameters)
    model.load_state_dict(model_state)
    scaler = SequenceStandardizer.from_state_dict(scaler_state)
    fill = metadata.get("feature_fill_values")
    if not isinstance(fill, dict) or set(fill) != set(manifest.feature_columns):
        raise ValueError("Artifact has no complete feature fill state")
    starts = {ticker: split["start"] for ticker, split in metadata["split_boundaries"]["test"].items()}
    return model, scaler, manifest.feature_columns, manifest.context_len, fill, starts


def generate(artifact_dir: Path, *, epochs: int = 12, device: str = "cpu", train_first: bool = False, cost_bps: float = 5.0, data_path: Path = DATA_PATH, universe_path: Path = UNIVERSE_PATH, market: str = "EURONEXT PARIS", include_backtest: bool = True) -> tuple[dict, dict | None]:
    universe = read_universe(universe_path)
    frame = read_market_data(data_path, universe=universe)
    state = train(frame, artifact_dir, epochs=epochs, device=device, data_path=data_path) if train_first else load(frame, artifact_dir, data_path=data_path, universe_path=universe_path)
    if include_backtest:
        predictions = _predict_rows(frame, *state[:5])
        return build_snapshots(predictions, universe, state[5], cost_bps=cost_bps, market=market)
    predictions = _predict_latest_rows(frame, *state[:5])
    return build_signal_snapshot(predictions, universe, market=market), None
