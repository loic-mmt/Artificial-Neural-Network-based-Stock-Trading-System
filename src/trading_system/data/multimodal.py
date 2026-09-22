"""Date-aligned, framework-neutral inputs for independent model branches.

This module builds no graph and trains no model. Price-derived features may use
the close of their session for later execution; publication-dated sources must
have been available strictly before that session's UTC midnight.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .windows import build_sequence_dataset_with_history


def _sessions(values: Any, name: str) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(
        pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    )
    if parsed.isna().any():
        raise ValueError(f"{name} contains invalid dates.")
    return parsed.normalize()


def _columns(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of column names.")
    columns = tuple(values)
    if any(not isinstance(column, str) or not column for column in columns):
        raise TypeError(f"{name} must contain non-empty strings.")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{name} must not contain duplicates.")
    return columns


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> np.ndarray:
    missing = sorted(set(columns) - set(frame))
    if missing:
        raise ValueError(f"Missing {name} columns: {missing}")
    if not columns:
        return np.empty((len(frame), 0), dtype=np.float32)
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )


def _prepare_frame(
    frame: pd.DataFrame, *, date_col: str, ticker_col: str, name: str
) -> pd.DataFrame:
    missing = [column for column in (date_col, ticker_col) if column not in frame]
    if missing:
        raise ValueError(f"Missing {name} columns: {missing}")
    work = frame.reset_index(drop=True).copy()
    if not work[ticker_col].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError(f"{name} tickers must be non-empty strings.")
    work["_multimodal_session"] = _sessions(work[date_col], name)
    if work.duplicated(["_multimodal_session", ticker_col]).any():
        raise ValueError(f"{name} has duplicate session/ticker rows.")
    return work


def _check_publication_cutoff(frame: pd.DataFrame, name: str) -> None:
    if "fund_available_at" not in frame:
        return
    values = frame["fund_available_at"]
    published = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    if (values.notna() & published.isna()).any():
        raise ValueError(f"{name} has invalid fund_available_at values.")
    cutoff = frame["_multimodal_session"]
    if (published.notna() & published.ge(cutoff)).any():
        raise ValueError(f"{name} fundamentals are not available before session midnight.")


def validate_calendar_partitions(
    *frames: pd.DataFrame, date_col: str = "date"
) -> None:
    """Reject a session appearing in more than one train/validation/test part."""

    seen: set[pd.Timestamp] = set()
    for part, frame in enumerate(frames):
        if date_col not in frame:
            raise ValueError(f"Partition {part} has no {date_col!r} column.")
        sessions = set(_sessions(frame[date_col], f"partition {part}"))
        if seen.intersection(sessions):
            raise ValueError("Calendar partitions overlap on a session date.")
        seen.update(sessions)


@dataclass(frozen=True, eq=False)
class GraphSnapshot:
    """Sparse graph supplied by a future causal graph builder, not made here."""

    session: object
    source_end: object
    edge_index: np.ndarray
    edge_weight: np.ndarray

    def __post_init__(self) -> None:
        session = _sessions([self.session], "graph session")[0]
        source_end = pd.Timestamp(self.source_end)
        if source_end.tzinfo is None or pd.isna(source_end):
            raise ValueError("Graph source_end must be a timezone-aware timestamp.")
        source_end = source_end.tz_convert("UTC")
        # Source may include close-J prices, but never observations from J+1.
        if source_end >= session + pd.Timedelta(days=1):
            raise ValueError("Graph uses data after its prediction session.")
        edges = np.asarray(self.edge_index)
        weights = np.asarray(self.edge_weight, dtype=np.float32)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("Graph edge_index must have shape (2, edges).")
        if not np.issubdtype(edges.dtype, np.integer) or (edges < 0).any():
            raise ValueError("Graph edge_index must contain non-negative integers.")
        if weights.shape != (edges.shape[1],) or not np.isfinite(weights).all():
            raise ValueError("Graph edge_weight must be finite and match edge_index.")
        edges = edges.astype(np.int64, copy=True)
        weights = weights.copy()
        edges.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "source_end", source_end)
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "edge_weight", weights)


@dataclass(frozen=True, eq=False)
class MultimodalBatch:
    """Logical [batch-date, fixed-ticker-slot, ...] model input."""

    sessions: tuple[pd.Timestamp, ...]
    tickers: tuple[str, ...]
    temporal: np.ndarray
    node: np.ndarray
    market: np.ndarray
    market_sequence: np.ndarray
    sentiment: np.ndarray
    labels: np.ndarray
    row_positions: np.ndarray
    asset_mask: np.ndarray
    temporal_mask: np.ndarray
    node_mask: np.ndarray
    market_mask: np.ndarray
    market_sequence_mask: np.ndarray
    sentiment_mask: np.ndarray
    label_mask: np.ndarray
    graph_mask: np.ndarray
    graphs: tuple[GraphSnapshot | None, ...]


class MultimodalDataset:
    """Store flat per-row features; pad only requested date batches."""

    def __init__(
        self,
        *,
        sessions: pd.DatetimeIndex,
        tickers: tuple[str, ...],
        row_grid: np.ndarray,
        window_grid: np.ndarray,
        windows: np.ndarray,
        node_values: np.ndarray,
        node_available: np.ndarray,
        market_values: np.ndarray,
        market_available: np.ndarray,
        market_sequence_values: np.ndarray,
        market_sequence_available: np.ndarray,
        sentiment_values: np.ndarray,
        sentiment_available: np.ndarray,
        labels: np.ndarray,
        label_known: np.ndarray,
        graphs: dict[pd.Timestamp, GraphSnapshot],
    ) -> None:
        self.sessions = sessions
        self.tickers = tickers
        self._row_grid = row_grid
        self._window_grid = window_grid
        self._windows = windows
        self._node_values = node_values
        self._node_available = node_available
        self._market_values = market_values
        self._market_available = market_available
        self._market_sequence_values = market_sequence_values
        self._market_sequence_available = market_sequence_available
        self._sentiment_values = sentiment_values
        self._sentiment_available = sentiment_available
        self._labels = labels
        self._label_known = label_known
        self._graphs = graphs

    def __len__(self) -> int:
        return len(self.sessions)

    def batch(self, date_indices: Sequence[int]) -> MultimodalBatch:
        indices = np.asarray(date_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("date_indices must be a 1D integer sequence.")
        if not len(indices) or (indices < 0).any() or (indices >= len(self)).any():
            raise ValueError("date_indices must be non-empty and in range.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("date_indices must not contain duplicates.")
        sessions = tuple(self.sessions[indices])
        rows = self._row_grid[indices]
        window_rows = self._window_grid[indices]
        asset_mask = rows >= 0
        temporal_mask = window_rows >= 0
        row_safe = np.maximum(rows, 0)
        window_safe = np.maximum(window_rows, 0)
        batch_dates, assets = rows.shape
        temporal = np.zeros(
            (batch_dates, assets, *self._windows.shape[1:]), dtype=np.float32
        )
        if temporal_mask.any():
            temporal[temporal_mask] = self._windows[window_safe[temporal_mask]]
        node = np.zeros(
            (batch_dates, assets, self._node_values.shape[1]), dtype=np.float32
        )
        sentiment = np.zeros(
            (batch_dates, assets, self._sentiment_values.shape[1]), dtype=np.float32
        )
        node_mask = asset_mask & self._node_available[row_safe]
        sentiment_mask = asset_mask & self._sentiment_available[row_safe]
        if node_mask.any():
            node[node_mask] = self._node_values[row_safe[node_mask]]
        if sentiment_mask.any():
            sentiment[sentiment_mask] = self._sentiment_values[
                row_safe[sentiment_mask]
            ]
        label_mask = asset_mask & self._label_known[row_safe]
        labels = np.full(rows.shape, -1, dtype=np.int64)
        labels[label_mask] = self._labels[row_safe[label_mask]]
        market = self._market_values[indices].copy()
        market_sequence = self._market_sequence_values[indices].copy()
        graphs = tuple(self._graphs.get(session) for session in sessions)
        graph_mask = node_mask & np.asarray(
            [graph is not None for graph in graphs], dtype=bool
        )[:, None]
        return MultimodalBatch(
            sessions=sessions,
            tickers=self.tickers,
            temporal=temporal,
            node=node,
            market=market,
            market_sequence=market_sequence,
            sentiment=sentiment,
            labels=labels,
            row_positions=rows.copy(),
            asset_mask=asset_mask,
            temporal_mask=temporal_mask,
            node_mask=node_mask,
            market_mask=self._market_available[indices].copy(),
            market_sequence_mask=self._market_sequence_available[indices].copy(),
            sentiment_mask=sentiment_mask,
            label_mask=label_mask,
            graph_mask=graph_mask,
            graphs=graphs,
        )

    def iter_batches(self, batch_dates: int) -> Iterator[MultimodalBatch]:
        if isinstance(batch_dates, bool) or not isinstance(batch_dates, int) or batch_dates <= 0:
            raise ValueError("batch_dates must be a positive integer.")
        for start in range(0, len(self), batch_dates):
            yield self.batch(np.arange(start, min(start + batch_dates, len(self))))


def build_multimodal_dataset(
    target_frame: pd.DataFrame,
    *,
    tickers: Sequence[str],
    context_len: int,
    temporal_columns: Sequence[str] = (),
    node_columns: Sequence[str] = (),
    market_columns: Sequence[str] = (),
    market_publication_columns: Sequence[str] = (),
    market_close_columns: Sequence[str] = (),
    sentiment_columns: Sequence[str] = (),
    history_frame: pd.DataFrame | None = None,
    market_frame: pd.DataFrame | None = None,
    market_context_len: int | None = None,
    sentiment_frame: pd.DataFrame | None = None,
    graphs: Sequence[GraphSnapshot] | None = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    label_col: str = "Label_id",
    label_known_col: str = "_label_known",
) -> MultimodalDataset:
    """Align prepared features to fixed ticker slots without fitting statistics."""

    if isinstance(context_len, bool) or not isinstance(context_len, int) or context_len <= 0:
        raise ValueError("context_len must be a positive integer.")
    names = _columns(tickers, "tickers")
    if not names:
        raise ValueError("tickers must not be empty.")
    temporal_cols = _columns(temporal_columns, "temporal_columns")
    node_cols = _columns(node_columns, "node_columns")
    market_cols = _columns(market_columns, "market_columns")
    market_publication_cols = _columns(
        market_publication_columns, "market_publication_columns"
    )
    market_close_cols = _columns(market_close_columns, "market_close_columns")
    market_sequence_cols = (*market_publication_cols, *market_close_cols)
    if len(market_sequence_cols) != len(set(market_sequence_cols)):
        raise ValueError("Market sequence feature groups must not overlap.")
    if set(temporal_cols).intersection(market_sequence_cols):
        raise ValueError("GRU temporal and Transformer market features must be separate.")
    sentiment_cols = _columns(sentiment_columns, "sentiment_columns")
    if not any((temporal_cols, node_cols, market_cols, market_sequence_cols, sentiment_cols)):
        raise ValueError("At least one feature group is required.")
    if label_col in (*temporal_cols, *node_cols, *market_cols, *market_sequence_cols, *sentiment_cols):
        raise ValueError("Labels cannot be model features.")
    if market_sequence_cols and market_frame is None:
        raise ValueError("market_frame is required for a Transformer market sequence.")
    if market_frame is not None and not market_sequence_cols:
        raise ValueError("Market sequence columns are required with market_frame.")
    if market_context_len is None:
        market_context_len = context_len
    if isinstance(market_context_len, bool) or not isinstance(market_context_len, int) or market_context_len <= 0:
        raise ValueError("market_context_len must be a positive integer.")

    target = _prepare_frame(target_frame, date_col=date_col, ticker_col=ticker_col, name="target")
    if target.empty:
        raise ValueError("target_frame cannot be empty.")
    history = (
        None if history_frame is None else _prepare_frame(
            history_frame, date_col=date_col, ticker_col=ticker_col, name="history"
        )
    )
    allowed = set(names)
    for frame_name, frame in (("target", target), ("history", history)):
        if frame is None:
            continue
        extra = sorted(set(frame[ticker_col]) - allowed)
        if extra:
            raise ValueError(f"{frame_name} includes tickers outside selection: {extra}")
        _check_publication_cutoff(frame, frame_name)
    if history is not None and not history.empty:
        validate_calendar_partitions(history, target, date_col="_multimodal_session")
        if history["_multimodal_session"].max() >= target["_multimodal_session"].min():
            raise ValueError("History must precede all target sessions globally.")

    sessions = pd.DatetimeIndex(target["_multimodal_session"].unique()).sort_values()
    date_to_index = {day: index for index, day in enumerate(sessions)}
    ticker_to_index = {ticker: index for index, ticker in enumerate(names)}
    row_grid = np.full((len(sessions), len(names)), -1, dtype=np.int64)
    for row, (day, ticker) in enumerate(zip(target["_multimodal_session"], target[ticker_col])):
        row_grid[date_to_index[day], ticker_to_index[ticker]] = row

    if temporal_cols:
        _numeric(target, temporal_cols, "temporal")
        if history is not None:
            _numeric(history, temporal_cols, "history temporal")
        window_target = target.copy()
        window_target[date_col] = window_target["_multimodal_session"]
        window_target[label_col] = 0
        window_history = None if history is None else history.copy()
        if window_history is not None:
            window_history[date_col] = window_history["_multimodal_session"]
        windows, _, aligned = build_sequence_dataset_with_history(
            window_target,
            temporal_cols,
            context_len,
            history_frame=window_history,
            group_col=ticker_col,
            date_col=date_col,
            label_col=label_col,
            return_aligned_rows=True,
        )
        window_grid = np.full_like(row_grid, -1)
        for index, (day, ticker) in enumerate(
            zip(aligned["_multimodal_session"], aligned[ticker_col])
        ):
            window_grid[date_to_index[day], ticker_to_index[ticker]] = index
    else:
        windows = np.empty((0, context_len, 0), dtype=np.float32)
        window_grid = np.full_like(row_grid, -1)

    node_values = _numeric(target, node_cols, "node")
    node_available = (
        np.isfinite(node_values).all(axis=1)
        if node_cols
        else np.zeros(len(target), bool)
    )
    node_values = np.where(np.isfinite(node_values), node_values, 0).astype(np.float32)

    market_source = _numeric(target, market_cols, "market")
    market_values = np.zeros((len(sessions), len(market_cols)), dtype=np.float32)
    market_available = np.zeros(len(sessions), dtype=bool)
    if market_cols:
        for day, group in target.groupby("_multimodal_session", sort=False):
            values = market_source[group.index.to_numpy()]
            if np.isfinite(values).all():
                if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
                    raise ValueError(f"Conflicting market features for session {day}.")
                index = date_to_index[day]
                market_values[index] = values[0]
                market_available[index] = True

    market_sequence_values = np.zeros(
        (len(sessions), market_context_len if market_sequence_cols else 0,
         len(market_sequence_cols)), dtype=np.float32
    )
    market_sequence_available = np.zeros(len(sessions), dtype=bool)
    if market_frame is not None:
        if date_col not in market_frame:
            raise ValueError(f"Missing market_frame date column {date_col!r}.")
        market_source = market_frame.reset_index(drop=True).copy()
        if market_source.empty:
            raise ValueError("market_frame cannot be empty with market sequence columns.")
        market_source["_multimodal_session"] = _sessions(
            market_source[date_col], "market_frame"
        )
        if market_source.duplicated("_multimodal_session").any():
            raise ValueError("market_frame needs one row per session.")
        for timestamp_col, required, cutoff in (
            ("available_at", bool(market_publication_cols), "publication"),
            ("source_end", bool(market_close_cols), "close"),
        ):
            if not required:
                continue
            if timestamp_col not in market_source:
                raise ValueError(f"market_frame needs {timestamp_col} for {cutoff} features.")
            parsed = []
            for row, value in enumerate(market_source[timestamp_col]):
                try:
                    stamp = pd.Timestamp(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"Invalid market {timestamp_col} row {row}.") from exc
                if pd.isna(stamp) or stamp.tzinfo is None:
                    raise ValueError(f"Market {timestamp_col} needs a timezone-aware value.")
                parsed.append(stamp.tz_convert("UTC"))
            timestamps = pd.DatetimeIndex(parsed)
            session_dates = market_source["_multimodal_session"]
            limit = session_dates if cutoff == "publication" else session_dates + pd.Timedelta(days=1)
            if (timestamps >= limit).any():
                raise ValueError(f"Market {cutoff} features use future information.")
        market_source = market_source.sort_values("_multimodal_session").reset_index(drop=True)
        numeric = _numeric(market_source, market_sequence_cols, "market sequence")
        source_available = np.isfinite(numeric).all(axis=1)
        calendar = pd.DatetimeIndex(sorted(set(sessions).union(
            set(history["_multimodal_session"]) if history is not None else set(),
            set(market_source["_multimodal_session"]),
        )))
        observations = {
            day: (numeric[index], source_available[index])
            for index, day in enumerate(market_source["_multimodal_session"])
        }
        calendar_index = {day: index for index, day in enumerate(calendar)}
        for date_index, day in enumerate(sessions):
            end = calendar_index[day] + 1
            start = end - market_context_len
            if start < 0:
                continue
            window = [observations.get(session) for session in calendar[start:end]]
            if any(item is None or not item[1] for item in window):
                continue
            market_sequence_values[date_index] = np.stack([item[0] for item in window])
            market_sequence_available[date_index] = True

    sentiment_values = np.zeros((len(target), len(sentiment_cols)), dtype=np.float32)
    sentiment_available = np.zeros(len(target), dtype=bool)
    if sentiment_frame is not None:
        if not sentiment_cols:
            raise ValueError("sentiment_columns are required with sentiment_frame.")
        source = _prepare_frame(
            sentiment_frame, date_col=date_col, ticker_col=ticker_col, name="sentiment"
        )
        if "source_available" not in source or "available_at" not in source:
            raise ValueError("Sentiment requires source_available and available_at.")
        if not source["source_available"].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise TypeError("source_available must contain booleans.")
        published = pd.to_datetime(
            source["available_at"], utc=True, errors="coerce", format="mixed"
        )
        available = source["source_available"].to_numpy(dtype=bool)
        missing_timestamp = available & published.isna().to_numpy()
        if missing_timestamp.any():
            # A fully covered window with no articles is known information,
            # although no contributing article timestamp exists. The audited
            # upstream coverage status is its point-in-time evidence.
            count = pd.to_numeric(
                source.get("news_count", pd.Series(index=source.index, dtype=float)),
                errors="coerce",
            )
            covered_empty = (
                source.get("coverage_status", pd.Series(index=source.index, dtype=object))
                .eq("covered").to_numpy(dtype=bool)
                & count.eq(0).to_numpy(dtype=bool)
            )
            if (missing_timestamp & ~covered_empty).any():
                raise ValueError("Available sentiment requires available_at or covered zero-news evidence.")
        if "coverage_status" in source:
            statuses = source["coverage_status"]
            if not statuses.isin(("covered", "incomplete", "unknown")).all():
                raise ValueError("Invalid sentiment coverage_status.")
            if not np.array_equal(available, statuses.eq("covered").to_numpy(dtype=bool)):
                raise ValueError("source_available and coverage_status disagree.")
        if (published.notna() & published.ge(source["_multimodal_session"])).any():
            raise ValueError("Sentiment is not available before session midnight.")
        values = _numeric(source, sentiment_cols, "sentiment")
        if (available & ~np.isfinite(values).all(axis=1)).any():
            raise ValueError("Available sentiment requires finite feature values.")
        key_to_row = {
            (day, ticker): row
            for row, (day, ticker) in enumerate(
                zip(target["_multimodal_session"], target[ticker_col])
            )
        }
        for index, (day, ticker) in enumerate(
            zip(source["_multimodal_session"], source[ticker_col])
        ):
            key = (day, ticker)
            if key not in key_to_row:
                raise ValueError(f"Sentiment row has no target session/ticker: {key}")
            row = key_to_row[key]
            sentiment_available[row] = available[index]
            if available[index]:
                sentiment_values[row] = values[index]

    labels = np.full(len(target), -1, dtype=np.int64)
    label_known = np.zeros(len(target), dtype=bool)
    if label_known_col in target:
        if not target[label_known_col].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise TypeError(f"{label_known_col} must contain booleans.")
        label_known = target[label_known_col].to_numpy(dtype=bool)
    elif label_col in target:
        label_known = target[label_col].notna().to_numpy()
    if label_known.any():
        if label_col not in target:
            raise ValueError("Known labels require Label_id values.")
        numeric = pd.to_numeric(
            target.loc[label_known, label_col], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or not np.isin(numeric, [0, 1, 2]).all():
            raise ValueError("Known labels must be class IDs 0, 1 or 2.")
        labels[label_known] = numeric.astype(np.int64)

    graph_lookup: dict[pd.Timestamp, GraphSnapshot] = {}
    for graph in graphs or ():
        if not isinstance(graph, GraphSnapshot):
            raise TypeError("graphs must contain GraphSnapshot values.")
        if graph.session not in date_to_index or graph.session in graph_lookup:
            raise ValueError("Graph sessions must be unique target sessions.")
        if graph.edge_index.size and (graph.edge_index >= len(names)).any():
            raise ValueError("Graph edge_index exceeds ticker slots.")
        day_index = date_to_index[graph.session]
        rows = row_grid[day_index]
        active = (rows >= 0) & node_available[np.maximum(rows, 0)]
        if graph.edge_index.size and not active[graph.edge_index].all():
            raise ValueError("Graph edges cannot reference absent or invalid nodes.")
        graph_lookup[graph.session] = graph

    return MultimodalDataset(
        sessions=sessions,
        tickers=names,
        row_grid=row_grid,
        window_grid=window_grid,
        windows=windows,
        node_values=node_values,
        node_available=node_available,
        market_values=market_values,
        market_available=market_available,
        market_sequence_values=market_sequence_values,
        market_sequence_available=market_sequence_available,
        sentiment_values=sentiment_values,
        sentiment_available=sentiment_available,
        labels=labels,
        label_known=label_known,
        graphs=graph_lookup,
    )


__all__ = [
    "GraphSnapshot",
    "MultimodalBatch",
    "MultimodalDataset",
    "build_multimodal_dataset",
    "validate_calendar_partitions",
]
