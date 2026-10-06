"""Read persisted results through projected, filtered Parquet queries.

This module deliberately has no dependency on training or backtest engines.
Missing paths are capabilities unavailable to the UI, never an invitation to
retrain a model or silently reconstruct its execution.
"""

from pathlib import Path
from typing import Any
import json
import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from .schemas import RunData, RunRecord


IDENTITY_COLUMNS = (
    "candidate", "fold", "seed", "partition", "method", "decoder",
    "retrain_every_sessions", "source_candidate", "variant_id",
)
EQUITY_COLUMNS = (
    "date", "timestamp", "__index_level_0__", "equity", "model_equity",
    "benchmark_equity", "buy_hold_equity", "drawdown", "net_return",
    "gross_return", "buy_hold_return", "cost", "turnover", "gross_exposure",
    "net_exposure", "cash", "halted",
)
POSITION_COLUMNS = (
    "date", "timestamp", "__index_level_0__", "ticker", "position",
    "decoder_position", "weight", "quantity", "stop_price", "take_profit",
    "blocked_direction", "holding_bars", "trade_id", "adj_close", "close",
    "p_buy", "p_hold", "p_sell", "available", "open", "high", "low",
    "adj_open_target", "position_raw", "raw_position", "executed_position",
    "sign_position", "position_close", "prob_buy", "prob_hold", "prob_sell",
)
MARKET_COLUMNS = ("date", "timestamp", "__index_level_0__", "ticker", "open", "high", "low", "close", "adj_close", "adj_open_target")
ORDER_COLUMNS = (
    "date", "timestamp", "ticker", "trade_id", "quantity", "quantity_before",
    "quantity_after", "base_price", "execution_price", "notional", "fee",
    "slippage_cost", "turnover", "reason", "phase", "earliest_time", "latest_time",
)
TRADE_COLUMNS = (
    "ticker", "trade_id", "side", "direction", "entry_time", "exit_time",
    "entry_price", "exit_price", "entry_price_base", "exit_price_base",
    "entry_reason", "exit_reason", "exit_phase", "entry_notional", "notional",
    "net_pnl", "fees", "total_fees", "entry_fee", "exit_fee", "dividends",
    "return_pct", "duration_bars", "duration_hours", "holding_bars", "is_winner",
)


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def parquet_columns(path: Path) -> list[str]:
    """Read schema only; do not materialize a results table."""
    return list(ds.dataset(path, format="parquet").schema.names)


def parquet_identities(path: Path) -> list[dict[str, Any]]:
    """Project only the small identity columns of a daily-path table."""
    dataset = ds.dataset(path, format="parquet")
    keys = [key for key in IDENTITY_COLUMNS if key in dataset.schema.names]
    if not keys:
        return [{}]
    # Use batches so catalogs do not retain an entire table in memory.
    identities: dict[tuple[Any, ...], dict[str, Any]] = {}
    for batch in dataset.scanner(columns=keys, batch_size=65536).to_batches():
        for row in batch.to_pandas().drop_duplicates().to_dict("records"):
            if any(pd.isna(value) for value in row.values()):
                raise ValueError("Missing daily-path identity value")
            identity = tuple(row[key] for key in keys)
            identities[identity] = row
    return list(identities.values())


def _read_parquet(path: Path, selectors: dict, columns: tuple[str, ...], *, start=None, end=None, trades=False, tickers=None) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    schema = dataset.schema
    names = set(schema.names)
    expression = None
    for key, value in selectors.items():
        if key not in names:
            continue  # A per-run table may encode identity in its directory.
        condition = ds.field(key) == value
        expression = condition if expression is None else expression & condition
    if tickers is not None and "ticker" in names:
        condition = ds.field("ticker").isin([str(ticker) for ticker in tickers])
        expression = condition if expression is None else expression & condition
    date_key = next((key for key in ("date", "timestamp", "__index_level_0__") if key in names), None)
    if date_key is not None and pa.types.is_timestamp(schema.field(date_key).type):
        dtype = schema.field(date_key).type
        for value, operator in ((start, ">="), (end, "<=")):
            if value is None:
                continue
            bound = _utc(value)
            if dtype.tz is None:
                bound = bound.tz_localize(None)
            scalar = pa.scalar(bound.to_pydatetime(warn=False), type=dtype)
            condition = ds.field(date_key) >= scalar if operator == ">=" else ds.field(date_key) <= scalar
            expression = condition if expression is None else expression & condition
    if trades:
        for key, value, operator in (("exit_time", start, ">="), ("entry_time", end, "<=")):
            if value is None or key not in names or not pa.types.is_timestamp(schema.field(key).type):
                continue
            dtype = schema.field(key).type
            bound = _utc(value)
            if dtype.tz is None:
                bound = bound.tz_localize(None)
            scalar = pa.scalar(bound.to_pydatetime(warn=False), type=dtype)
            condition = (ds.field(key).is_null() | (ds.field(key) >= scalar)) if operator == ">=" else ds.field(key) <= scalar
            expression = condition if expression is None else expression & condition
    selected = [key for key in (*columns, *IDENTITY_COLUMNS) if key in names]
    # Physical timestamp columns must remain columns: legacy pandas metadata
    # otherwise restores timestamp as an index and hides it from the adapter.
    frame = dataset.to_table(columns=list(dict.fromkeys(selected)), filter=expression).to_pandas(ignore_metadata=True)
    if not trades and date_key:
        frame["date"] = pd.to_datetime(frame[date_key], utc=True, errors="coerce")
        frame = frame.dropna(subset=["date"])
        if start is not None:
            frame = frame.loc[frame.date.ge(_utc(start))]
        if end is not None:
            frame = frame.loc[frame.date.le(_utc(end))]
        frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    if trades:
        for key in ("entry_time", "exit_time"):
            if key in frame:
                frame[key] = pd.to_datetime(frame[key], utc=True, errors="coerce")
        # Include trades overlapping the selected period, including open trades.
        if start is not None and "exit_time" in frame:
            frame = frame.loc[frame.exit_time.isna() | frame.exit_time.ge(_utc(start))]
        if end is not None and "entry_time" in frame:
            frame = frame.loc[frame.entry_time.le(_utc(end))]
    return frame.reset_index(drop=True)


@lru_cache(maxsize=64)
def _file_digest(path: str, size: int, modified: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _universe(record: RunRecord, tickers=None):
    value = record.metadata.get("tickers", record.metadata.get("selected_tickers", record.metadata.get("symbol")))
    declared = [value] if isinstance(value, str) else list(value) if isinstance(value, (list, tuple)) else None
    if tickers is not None:
        requested = [tickers] if isinstance(tickers, str) else list(tickers)
        if declared is not None:
            allowed = {str(ticker) for ticker in declared}
            return [ticker for ticker in requested if str(ticker) in allowed]
        return requested
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _with_ticker(frame: pd.DataFrame, record: RunRecord) -> pd.DataFrame:
    universe = _universe(record)
    if "ticker" not in frame and universe and len(universe) == 1:
        frame = frame.copy()
        frame["ticker"] = str(universe[0])
    return frame


def _market(record: RunRecord, equity: pd.DataFrame, positions: pd.DataFrame, tickers, warnings: list) -> pd.DataFrame:
    path = record.tables.get("market")
    calendar = equity.date if "date" in equity and not equity.empty else None
    if path is not None:
        try:
            if calendar is None:
                raise ValueError("Equity evaluation calendar unavailable; external market source not loaded")
            expected = record.metadata.get("market_sha256")
            if record.metadata.get("market_requires_hash"):
                if not isinstance(expected, str):
                    raise ValueError("Market source hash unavailable")
                stat = path.stat()
                if _file_digest(str(path), stat.st_size, stat.st_mtime_ns) != expected:
                    raise ValueError("Market source SHA256 mismatch")
            universe = _universe(record, tickers)
            if "positions" in record.tables and positions.empty and tickers is not None:
                raise ValueError("Selected assets absent from persisted run positions")
            if universe is None and "ticker" in positions:
                universe = sorted(positions.ticker.dropna().astype(str).unique())
            if universe is None and "ticker" in parquet_columns(path):
                raise ValueError("Evaluated asset universe unavailable; external market source not loaded")
            frame = _read_parquet(path, {}, MARKET_COLUMNS, start=calendar.min().normalize(), end=calendar.max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1), tickers=universe)
            frame = _with_ticker(frame, record)
            # The selected run's exact session calendar prevents loading prices
            # from a held-out period present in a larger dataset file.
            frame = frame.loc[frame.date.dt.normalize().isin(calendar.dt.normalize().unique())]
            if "ticker" in positions:
                keys = positions[["date", "ticker"]].copy()
                keys["_session"] = pd.to_datetime(keys.date, utc=True).dt.normalize()
                frame["_session"] = frame.date.dt.normalize()
                frame = frame.merge(keys[["_session", "ticker"]].drop_duplicates(), on=["_session", "ticker"], how="inner", validate="many_to_one").drop(columns="_session")
            if not frame.empty:
                frame.attrs["price_provenance"] = str(path)
                return frame.reset_index(drop=True)
        except (OSError, ValueError, TypeError, KeyError, pa.ArrowException) as error:
            warnings.append(f"Market prices unavailable: {error}")
    if not positions.empty and any(key in positions for key in ("close", "adj_close")):
        columns = [key for key in MARKET_COLUMNS if key in positions]
        frame = positions[columns].copy()
        frame.attrs["price_provenance"] = "Persisted position/prediction export"
        return frame
    warnings.append("Market price series unavailable; execution markers remain available when exported.")
    return pd.DataFrame()


def _isolated(frame: pd.DataFrame, selectors: dict) -> None:
    for key in IDENTITY_COLUMNS:
        if key in frame and frame[key].nunique(dropna=False) > 1:
            raise ValueError(f"Table mixes {key}; select one identity before plotting")
        if key in frame and frame[key].isna().any():
            raise ValueError(f"Table contains missing {key} identity")


def _canonical_equity(frame: pd.DataFrame, record: RunRecord) -> tuple[pd.DataFrame, list[str]]:
    warnings = []
    if frame.empty:
        return pd.DataFrame(), warnings
    _isolated(frame, record.selectors)
    if "date" not in frame:
        raise ValueError("Equity table lacks dated observations")
    if frame.date.duplicated().any():
        raise ValueError("Equity table has duplicate dates for this identity")
    frame = frame.rename(columns={"model_equity": "equity", "buy_hold_equity": "benchmark_equity"})
    capital = record.metadata.get("initial_capital", record.metrics.get("initial_capital"))
    if "equity" not in frame and "net_return" in frame:
        if capital is None:
            warnings.append("Initial capital unavailable; return index starts at 1.")
            capital = 1.0
        frame["equity"] = float(capital) * (1 + pd.to_numeric(frame.net_return)).cumprod()
    if "equity" not in frame:
        raise ValueError("Equity table has neither equity nor net_return")
    frame["equity"] = pd.to_numeric(frame.equity, errors="coerce")
    if frame.equity.isna().any() or not np.isfinite(frame.equity).all():
        raise ValueError("Equity table contains invalid capital values")
    if "benchmark_equity" not in frame and "buy_hold_return" in frame:
        if capital is None and "net_return" in frame:
            denominator = 1 + float(frame.net_return.iloc[0])
            if denominator > 0:
                capital = float(frame.equity.iloc[0]) / denominator
        if capital is not None:
            frame["benchmark_equity"] = float(capital) * (1 + pd.to_numeric(frame.buy_hold_return)).cumprod()
        else:
            warnings.append("Benchmark path unavailable: initial capital unknown.")
    if "drawdown" not in frame:
        frame["drawdown"] = frame.equity / frame.equity.cummax() - 1
    keep = [key for key in EQUITY_COLUMNS if key in frame and key not in ("timestamp", "__index_level_0__", "model_equity", "buy_hold_equity")]
    return frame[list(dict.fromkeys(keep))].reset_index(drop=True), warnings


def _slice(frame: pd.DataFrame, start, end, *, trades=False) -> pd.DataFrame:
    attrs = frame.attrs.copy()
    if trades:
        if start is not None and "exit_time" in frame:
            frame = frame.loc[frame.exit_time.isna() | frame.exit_time.ge(_utc(start))]
        if end is not None and "entry_time" in frame:
            frame = frame.loc[frame.entry_time.le(_utc(end))]
    elif "date" in frame:
        if start is not None:
            frame = frame.loc[frame.date.ge(_utc(start))]
        if end is not None:
            frame = frame.loc[frame.date.le(_utc(end))]
    frame = frame.reset_index(drop=True)
    frame.attrs = attrs
    return frame


def _legacy_benchmark(data: RunData, warnings: list) -> None:
    record = data.record
    if record.metadata.get("benchmark_convention") != "legacy_close_buy_hold_no_fees" or data.equity.empty or "benchmark_equity" in data.equity:
        return
    market = data.market
    if "close" not in market:
        warnings.append("Legacy buy-and-hold unavailable: raw close prices missing.")
        return
    if "ticker" in market and market.ticker.nunique() != 1:
        warnings.append("Legacy buy-and-hold unavailable: ambiguous asset series.")
        return
    quotes = market[["date", "close"]].drop_duplicates()
    if quotes.date.duplicated().any():
        warnings.append("Legacy buy-and-hold unavailable: conflicting close prices.")
        return
    aligned = data.equity[["date"]].merge(quotes, on="date", how="left", validate="one_to_one")
    capital = record.metadata.get("initial_capital")
    if capital is None or aligned.close.isna().any() or not aligned.close.gt(0).all():
        warnings.append("Legacy buy-and-hold unavailable: capital or aligned close prices missing.")
        return
    data.equity["benchmark_equity"] = float(capital) * aligned.close / float(aligned.close.iloc[0])
    data.equity.attrs["benchmark_provenance"] = "Stored raw close; legacy buy-and-hold formula without costs"


def load_run(record: RunRecord, *, start=None, end=None, include_details=False, tickers=None) -> RunData:
    """Load one frozen run. Dates filter views, never change reported metrics."""
    if start is not None and end is not None and _utc(start) > _utc(end):
        raise ValueError("start must be on or before end")
    warnings = list(record.warnings)
    data = RunData(record=record)
    equity_path = record.tables.get("equity")
    if equity_path is not None:
        try:
            # Return paths must compound from their original origin before a
            # UI date slice; recomputing from the slice changes its capital.
            raw = _read_parquet(equity_path, record.selectors, EQUITY_COLUMNS)
            data.equity, extra = _canonical_equity(raw, record)
            warnings.extend(extra)
        except (OSError, ValueError, TypeError, KeyError, pa.ArrowException) as error:
            warnings.append(f"Equity unavailable: {error}")
    if include_details:
        detail_start = data.equity.date.min().normalize() if not data.equity.empty else None
        detail_end = data.equity.date.max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1) if not data.equity.empty else None
        selected_tickers = _universe(record, tickers)
        for name, columns in (("positions", POSITION_COLUMNS), ("trades", TRADE_COLUMNS), ("orders", ORDER_COLUMNS)):
            path = record.tables.get(name)
            if path is None:
                continue
            try:
                frame = _read_parquet(path, record.selectors, columns, start=detail_start, end=detail_end, trades=name == "trades", tickers=selected_tickers)
                _isolated(frame, record.selectors)
                frame = _with_ticker(frame, record)
                setattr(data, name, frame)
            except (OSError, ValueError, TypeError, KeyError, pa.ArrowException) as error:
                warnings.append(f"{name.title()} unavailable: {error}")
        path = record.tables.get("history")
        if path is not None:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    data.history = value
            except (OSError, ValueError) as error:
                warnings.append(f"Training history unavailable: {error}")
        if not data.history and isinstance(record.metadata.get("history"), dict):
            data.history = dict(record.metadata["history"])
    if include_details or record.metadata.get("benchmark_convention") == "legacy_close_buy_hold_no_fees":
        data.market = _market(record, data.equity, data.positions, tickers, warnings)
        _legacy_benchmark(data, warnings)
    if include_details:
        from .positions import asset_names, position_events
        events = []
        for ticker in asset_names(data):
            events.extend(position_events(data, ticker).to_dict("records"))
        for name in ("orders", "trades", "positions"):
            frame = getattr(data, name)
            frame.attrs.update(position_events=events, events_complete=True)
        for name in ("market", "positions", "orders", "trades"):
            setattr(data, name, _slice(getattr(data, name), start, end, trades=name == "trades"))
    data.equity = _slice(data.equity, start, end)
    data.warnings = tuple(dict.fromkeys(warnings))
    return data
