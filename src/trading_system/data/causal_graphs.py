"""Point-in-time sector and Pearson graphs for date-aligned stock batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .multimodal import GraphSnapshot


GraphMode = Literal[
    "identity",
    "sector",
    "train_pearson",
    "rolling_pearson",
    "train_topk",
    "rolling_topk",
    "rolling_residual_topk",
]


GRAPH_MODES = (
    "identity",
    "sector",
    "train_pearson",
    "rolling_pearson",
    "train_topk",
    "rolling_topk",
    "rolling_residual_topk",
)


@dataclass(frozen=True)
class GraphBuildConfig:
    mode: GraphMode
    lookback: int = 252
    threshold: float = 0.7
    weight_mode: Literal["positive", "absolute"] = "positive"
    neighbors: int = 5
    rebalance_bars: int = 1

    def __post_init__(self) -> None:
        if self.mode not in GRAPH_MODES:
            raise ValueError("Unsupported graph mode.")
        if isinstance(self.lookback, bool) or not isinstance(self.lookback, int) or self.lookback < 3:
            raise ValueError("Graph lookback must be an integer >= 3.")
        if not np.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("Graph threshold must be in (0, 1).")
        if self.weight_mode not in ("positive", "absolute"):
            raise ValueError("Graph weight_mode must be positive or absolute.")
        for value, name in ((self.neighbors, "neighbors"),
                            (self.rebalance_bars, "rebalance_bars")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Graph {name} must be a positive integer.")


def _edges(similarity: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    accepted = np.isfinite(similarity) & (similarity >= threshold)
    np.fill_diagonal(accepted, False)
    source, destination = np.nonzero(accepted)
    return np.stack((source, destination)).astype(np.int64), similarity[source, destination].astype(np.float32)


def _pearson_similarity(returns: np.ndarray, config: GraphBuildConfig) -> np.ndarray:
    if returns.shape[0] != config.lookback or not np.isfinite(returns).all():
        raise ValueError("Pearson graph requires a complete trailing return window.")
    centered = returns - returns.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.square(centered).sum(axis=0))
    denominator = norm[:, None] * norm[None, :]
    correlation = np.divide(
        centered.T @ centered, denominator,
        out=np.zeros_like(denominator), where=denominator > 1e-12,
    )
    return np.maximum(correlation, 0) if config.weight_mode == "positive" else np.abs(correlation)


def _topk_edges(similarity: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    assets = similarity.shape[0]
    if similarity.shape != (assets, assets) or assets < 2:
        raise ValueError("Top-k graph requires a square similarity matrix with >= 2 assets.")
    count = min(neighbors, assets - 1)
    # Correlation is symmetric.  Use the undirected union of each node's k-NN
    # selection and store both directions for message passing.
    selected_edges: dict[tuple[int, int], float] = {}
    for destination in range(assets):
        values = similarity[:, destination].copy()
        values[destination] = -np.inf
        order = np.argsort(-values, kind="stable")
        selected = [int(source) for source in order if np.isfinite(values[source]) and values[source] > 0][:count]
        for source in selected:
            weight = float(values[source])
            selected_edges[(source, destination)] = weight
            selected_edges[(destination, source)] = weight
    ordered = sorted(selected_edges)
    if not ordered:
        return np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float32)
    edge_index = np.asarray(ordered, dtype=np.int64).T
    weights = np.asarray([selected_edges[edge] for edge in ordered], dtype=np.float32)
    return edge_index, weights


def _pearson_edges(returns: np.ndarray, config: GraphBuildConfig) -> tuple[np.ndarray, np.ndarray]:
    similarity = _pearson_similarity(returns, config)
    if config.mode in ("train_topk", "rolling_topk", "rolling_residual_topk"):
        return _topk_edges(similarity, config.neighbors)
    return _edges(similarity, config.threshold)


def _residualize(
    returns: np.ndarray,
    factor_returns: np.ndarray,
    *,
    sectors: Sequence[str],
    factor_columns: Sequence[str],
    sector_columns: Mapping[str, str],
    market_column: str,
) -> np.ndarray:
    """Remove broad-market and sector-ETF exposure using only the supplied window."""

    if not np.isfinite(factor_returns).all():
        raise ValueError("Residual graph requires a complete trailing factor window.")
    column_index = {name: index for index, name in enumerate(factor_columns)}
    if market_column not in column_index:
        raise ValueError(f"Residual graph context lacks {market_column!r}.")
    residuals = np.empty_like(returns, dtype=np.float64)
    market = factor_returns[:, column_index[market_column]]
    for asset, sector in enumerate(sectors):
        columns = [np.ones(len(returns), dtype=np.float64), market]
        sector_column = sector_columns.get(str(sector))
        if sector_column is not None and sector_column in column_index:
            columns.append(factor_returns[:, column_index[sector_column]])
        design = np.column_stack(columns)
        coefficients, *_ = np.linalg.lstsq(design, returns[:, asset], rcond=None)
        residuals[:, asset] = returns[:, asset] - design @ coefficients
    return residuals


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
    context_frame: pd.DataFrame | None = None,
    context_date_col: str = "date",
    market_context_column: str = "spy_close",
    sector_context_columns: Mapping[str, str] | None = None,
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
    if config.mode in ("sector", "rolling_residual_topk"):
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
    factor_returns = None
    factor_columns: tuple[str, ...] = ()
    sector_context_columns = dict(sector_context_columns or {})
    if config.mode == "rolling_residual_topk":
        if context_frame is None:
            raise ValueError("Residual graph requires a close-observed context frame.")
        factor_columns = tuple(dict.fromkeys(
            (market_context_column, *sector_context_columns.values())
        ))
        missing_context = sorted({context_date_col, *factor_columns} - set(context_frame))
        if missing_context:
            raise ValueError(f"Residual graph context is missing columns: {missing_context}")
        context = context_frame[[context_date_col, *factor_columns]].copy()
        context[context_date_col] = pd.to_datetime(
            context[context_date_col], utc=True, errors="raise"
        ).dt.normalize()
        if context[context_date_col].duplicated().any():
            raise ValueError("Residual graph context requires one row per session.")
        context = context.set_index(context_date_col).reindex(dates)
        factors = context.loc[:, factor_columns].apply(pd.to_numeric, errors="coerce")
        if (factors <= 0).any().any():
            raise ValueError("Residual graph context closes must be positive when observed.")
        factor_returns = factors.pct_change(fill_method=None).to_numpy(dtype=np.float64)
    anchor = config.lookback
    if config.mode in (
        "train_pearson", "rolling_pearson", "train_topk", "rolling_topk",
        "rolling_residual_topk",
    ):
        if anchor >= len(dates) or dates[anchor] >= train_end or requested.min() <= dates[anchor]:
            raise ValueError("Graph calibration prefix must precede all prediction sessions and training end.")
    static_edges = (
        _pearson_edges(returns[1:anchor + 1], config)
        if config.mode in ("train_pearson", "train_topk") else None
    )
    snapshots = []
    index_by_date = {day: index for index, day in enumerate(dates)}
    cached: tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp] | None = None
    cached_index: int | None = None
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
            should_refresh = cached is None or cached_index is None or (
                index - cached_index >= config.rebalance_bars
            )
            if should_refresh:
                if index < config.lookback:
                    raise ValueError("Rolling graph lacks a complete trailing return window.")
                trailing = returns[index - config.lookback + 1:index + 1]
                if config.mode == "rolling_residual_topk":
                    assert factor_returns is not None
                    categories = work.loc[
                        work[date_col].eq(day), [ticker_col, sector_col]
                    ].set_index(ticker_col)[sector_col].reindex(names)
                    if categories.isna().any():
                        raise ValueError("Residual graph requires sector for every active ticker.")
                    trailing = _residualize(
                        trailing,
                        factor_returns[index - config.lookback + 1:index + 1],
                        sectors=categories.astype(str),
                        factor_columns=factor_columns,
                        sector_columns=sector_context_columns,
                        market_column=market_context_column,
                    )
                edge_index, edge_weight = _pearson_edges(trailing, config)
                source_start = dates[index - config.lookback]
                source_end = day + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
                cached = edge_index, edge_weight, source_start, source_end
                cached_index = index
            else:
                edge_index, edge_weight, source_start, source_end = cached
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


__all__ = ["GRAPH_MODES", "GraphBuildConfig", "build_graph_snapshots", "graph_diagnostics"]
