"""Point-in-time sector and Pearson graphs for date-aligned stock batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .multimodal import GraphSnapshot


GraphMode = Literal["identity", "sector", "train_pearson", "rolling_pearson"]


@dataclass(frozen=True)
class GraphBuildConfig:
    mode: GraphMode
    lookback: int = 252
    threshold: float = 0.7
    weight_mode: Literal["positive", "absolute"] = "positive"

    def __post_init__(self) -> None:
        if self.mode not in ("identity", "sector", "train_pearson", "rolling_pearson"):
            raise ValueError("Unsupported graph mode.")
        if isinstance(self.lookback, bool) or not isinstance(self.lookback, int) or self.lookback < 3:
            raise ValueError("Graph lookback must be an integer >= 3.")
        if not np.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("Graph threshold must be in (0, 1).")
        if self.weight_mode not in ("positive", "absolute"):
            raise ValueError("Graph weight_mode must be positive or absolute.")


def _edges(similarity: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    accepted = np.isfinite(similarity) & (similarity >= threshold)
    np.fill_diagonal(accepted, False)
    source, destination = np.nonzero(accepted)
    return np.stack((source, destination)).astype(np.int64), similarity[source, destination].astype(np.float32)


def _pearson_edges(returns: np.ndarray, config: GraphBuildConfig) -> tuple[np.ndarray, np.ndarray]:
    if returns.shape[0] != config.lookback or not np.isfinite(returns).all():
        raise ValueError("Pearson graph requires a complete trailing return window.")
    centered = returns - returns.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.square(centered).sum(axis=0))
    denominator = norm[:, None] * norm[None, :]
    correlation = np.divide(
        centered.T @ centered, denominator,
        out=np.zeros_like(denominator), where=denominator > 1e-12,
    )
    similarity = np.maximum(correlation, 0) if config.weight_mode == "positive" else np.abs(correlation)
    return _edges(similarity, config.threshold)


def build_graph_snapshots(
    frame: pd.DataFrame,
    *,
    tickers: Sequence[str],
    prediction_sessions: Sequence[object],
    training_end: object,
    config: GraphBuildConfig,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "adj_close",
    sector_col: str = "sector",
) -> tuple[GraphSnapshot, ...]:
    """Build graphs from observed dates only; static Pearson uses an early train prefix.

    `training_end` is the last inner-training session. The first `lookback`
    returns calibrate a static graph, so no prediction may precede its source.
    Rolling graphs use returns through their prediction session's close.
    """
    names = tuple(tickers)
    if not names or len(names) != len(set(names)):
        raise ValueError("Graph tickers must be non-empty and unique.")
    required = {date_col, ticker_col, price_col}
    if config.mode == "sector":
        required.add(sector_col)
    if required - set(frame):
        raise ValueError(f"Missing graph columns: {sorted(required - set(frame))}")
    work = frame.loc[frame[ticker_col].isin(names)].copy()
    work[date_col] = pd.to_datetime(work[date_col], utc=True, errors="raise").dt.normalize()
    if work.duplicated([date_col, ticker_col]).any():
        raise ValueError("Graph input contains duplicate session/ticker rows.")
    prices = work.pivot(index=date_col, columns=ticker_col, values=price_col)
    prices = prices.reindex(columns=names).sort_index().astype(float)
    if prices.empty or not np.isfinite(prices.to_numpy()).all() or (prices.to_numpy() <= 0).any():
        raise ValueError("Graph prices require a complete positive panel.")
    dates = prices.index
    requested = pd.DatetimeIndex(pd.to_datetime(prediction_sessions, utc=True, errors="raise")).normalize()
    if requested.empty or requested.has_duplicates or not requested.isin(dates).all():
        raise ValueError("Prediction sessions must be unique observed graph dates.")
    train_end = pd.Timestamp(training_end)
    if pd.isna(train_end) or train_end.tzinfo is None:
        raise ValueError("training_end must be timezone-aware.")
    train_end = train_end.tz_convert("UTC").normalize()
    if train_end not in dates:
        raise ValueError("Graph data must contain the training end session.")
    if config.mode == "identity":
        return ()
    returns = prices.pct_change(fill_method=None).to_numpy(dtype=np.float64)
    anchor = config.lookback
    if config.mode in ("train_pearson", "rolling_pearson"):
        if anchor >= len(dates) or dates[anchor] >= train_end or requested.min() <= dates[anchor]:
            raise ValueError("Graph calibration prefix must precede all prediction sessions and training end.")
    static_edges = _pearson_edges(returns[1:anchor + 1], config) if config.mode == "train_pearson" else None
    snapshots = []
    index_by_date = {day: index for index, day in enumerate(dates)}
    for day in requested:
        index = index_by_date[day]
        if config.mode == "sector":
            categories = work.loc[work[date_col].eq(day), [ticker_col, sector_col]].set_index(ticker_col)[sector_col].reindex(names)
            if categories.isna().any():
                raise ValueError("Sector graph requires sector for every active ticker.")
            values = categories.astype(str).to_numpy()
            same = (values[:, None] == values[None, :]) & (values[:, None] != "unknown")
            edge_index, edge_weight = _edges(same.astype(float), 0.5)
            source_start = source_end = day + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        elif static_edges is not None:
            edge_index, edge_weight = static_edges
            source_start = dates[0]
            source_end = dates[anchor] + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        else:
            if index < config.lookback:
                raise ValueError("Rolling graph lacks a complete trailing return window.")
            edge_index, edge_weight = _pearson_edges(
                returns[index - config.lookback + 1:index + 1], config,
            )
            source_start = dates[index - config.lookback]
            source_end = day + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        snapshots.append(GraphSnapshot(
            session=day, source_start=source_start, source_end=source_end,
            edge_index=edge_index, edge_weight=edge_weight,
        ))
    return tuple(snapshots)


def graph_diagnostics(graphs: Sequence[GraphSnapshot], assets: int) -> dict:
    """Compact structural report, excluding GCN self-loops."""
    if not graphs or assets < 2:
        return {"sessions": len(graphs), "mean_density": 0.0, "mean_isolated": float(assets)}
    density, isolated, average_degree, turnover = [], [], [], []
    previous_edges = None
    for graph in graphs:
        degrees = np.bincount(graph.edge_index[1], minlength=assets)
        density.append(graph.edge_index.shape[1] / (assets * (assets - 1)))
        isolated.append(int(np.count_nonzero(degrees == 0)))
        average_degree.append(float(degrees.mean()))
        edges = set(map(tuple, graph.edge_index.T))
        if previous_edges is not None:
            union = edges | previous_edges
            turnover.append(len(edges ^ previous_edges) / len(union) if union else 0.0)
        previous_edges = edges
    return {"sessions": len(graphs), "mean_density": float(np.mean(density)),
            "mean_isolated": float(np.mean(isolated)),
            "mean_degree": float(np.mean(average_degree)),
            "mean_edge_turnover": float(np.mean(turnover)) if turnover else 0.0}


__all__ = ["GraphBuildConfig", "build_graph_snapshots", "graph_diagnostics"]
