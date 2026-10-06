"""Asset prices and explicitly sourced execution or signal markers."""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from functools import lru_cache
import pyarrow as pa
import pyarrow.dataset as ds

from .schemas import RunData


EVENT_COLUMNS = ["date", "price", "side", "kind", "source", "ticker"]


@lru_cache(maxsize=256)
def _projected_assets(path: str, selectors: str, size: int, modified: int) -> tuple[str, ...]:
    dataset = ds.dataset(path, format="parquet")
    if "ticker" not in dataset.schema.names:
        return ()
    expression = None
    for key, value in json.loads(selectors).items():
        if key in dataset.schema.names:
            condition = ds.field(key) == value
            expression = condition if expression is None else expression & condition
    names = set()
    for batch in dataset.scanner(columns=["ticker"], filter=expression, batch_size=65536).to_batches():
        names.update(str(value) for value in batch.column(0).unique().to_pylist() if value is not None)
    return tuple(sorted(names))


def _single_ticker(data: RunData) -> str | None:
    metadata = data.record.metadata
    symbol = metadata.get("symbol", metadata.get("ticker"))
    if isinstance(symbol, str):
        return symbol
    tickers = metadata.get("tickers")
    if isinstance(tickers, (tuple, list)) and len(tickers) == 1:
        return str(tickers[0])
    return None


def asset_names(data: RunData) -> list[str]:
    names = set()
    for frame in (data.market, data.positions, data.orders, data.trades):
        if "ticker" in frame:
            names.update(frame.ticker.dropna().astype(str))
    if names:
        return sorted(names)
    tickers = data.record.metadata.get("tickers", data.record.metadata.get("selected_tickers"))
    if isinstance(tickers, (list, tuple)):
        return sorted({str(ticker) for ticker in tickers})
    symbol = _single_ticker(data)
    if symbol:
        return [symbol]
    # For a metrics-only initial view, inspect a ticker projection for this
    # exact run rather than loading millions of prediction feature rows.
    for table in ("positions", "orders", "trades"):
        path = data.record.tables.get(table)
        if path is None:
            continue
        try:
            stat = Path(path).stat()
            names = _projected_assets(str(path), json.dumps(data.record.selectors, sort_keys=True, default=str), stat.st_size, stat.st_mtime_ns)
            if names:
                return list(names)
        except (OSError, ValueError, TypeError, pa.ArrowException):
            continue
    return []


def _asset(data: RunData, frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "ticker" in frame:
        return frame.loc[frame.ticker.astype(str).eq(str(ticker))].copy()
    return frame.copy() if _single_ticker(data) == str(ticker) else frame.iloc[:0].copy()


def _dated(frame: pd.DataFrame, *, orders=False) -> pd.DataFrame:
    key = next((key for key in (("timestamp", "date") if orders else ("date", "timestamp")) if key in frame), None)
    if key is None:
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.copy()
            frame["date"] = frame.index
        else:
            return pd.DataFrame()
    else:
        frame["date"] = frame[key]
    frame["date"] = pd.to_datetime(frame.date, utc=True, errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date", kind="stable").reset_index(drop=True)


def asset_frame(data: RunData, ticker: str) -> pd.DataFrame:
    """Prefer a provenance-checked market table, then prices in signal exports."""
    basis = str(data.record.metadata.get("price_basis", ""))
    raw = data.record.family == "backtest" or basis in ("raw", "split_adjusted", "unadjusted")
    candidates = ("close", "adj_close", "price") if raw else ("adj_close", "close", "price")
    for source, frame in (("market", data.market), ("positions", data.positions)):
        frame = _dated(_asset(data, frame, ticker))
        if frame.empty:
            continue
        key = next((key for key in candidates if key in frame), None)
        if key is None:
            continue
        columns = ["date", *[key for key in ("open", "high", "low", "close") if key in frame]]
        result = frame[columns].copy()
        result["price"] = pd.to_numeric(frame[key], errors="coerce")
        result = result.loc[result.price.notna() & np.isfinite(result.price) & result.price.gt(0)]
        if result.date.duplicated().any():
            # Repeated identical rows are harmless; conflicting series must not
            # be reduced into a fictional asset path.
            if result.groupby("date").price.nunique().gt(1).any():
                return pd.DataFrame(columns=["date", "price"])
            result = result.drop_duplicates("date")
        result = result[["date", "price", *[name for name in result if name not in ("date", "price")]]].reset_index(drop=True)
        result.attrs.update(source=source, price_column=key, price_basis=basis or key)
        return result
    return pd.DataFrame(columns=["date", "price"])


def _event_price(prices: pd.DataFrame, date) -> float | None:
    if prices.empty:
        return None
    stamp = pd.Timestamp(date)
    # Signal prices are observations on the exact same bar, never an as-of
    # lookup into a preceding or future quote.
    values = prices.loc[prices.date.eq(stamp), "price"]
    return float(values.iloc[0]) if len(values) == 1 else None


def _orders(data: RunData, ticker: str) -> pd.DataFrame:
    frame = _dated(_asset(data, data.orders, ticker), orders=True)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    events = []
    for row in frame.to_dict("records"):
        delta = row.get("quantity")
        if delta is None and row.get("quantity_before") is not None and row.get("quantity_after") is not None:
            delta = row["quantity_after"] - row["quantity_before"]
        price = row.get("execution_price", row.get("price"))
        if delta is None or price is None or not np.isfinite(float(delta)) or not np.isfinite(float(price)) or float(delta) == 0 or float(price) <= 0:
            continue
        before, after = row.get("quantity_before"), row.get("quantity_after")
        kind = "fill"
        if before is not None and after is not None and np.isfinite(float(before)) and np.isfinite(float(after)):
            prior, current = np.sign(float(before)), np.sign(float(after))
            if prior == 0 and current != 0:
                kind = "entry_long" if current > 0 else "entry_short"
            elif current == 0 and prior != 0:
                kind = "exit_long" if prior > 0 else "exit_short"
            elif prior != current:
                kind = "flip_long_to_short" if prior > 0 else "flip_short_to_long"
            else:
                kind = "rebalance_long" if current > 0 else "rebalance_short"
        events.append({"date": row["date"], "price": float(price), "side": "buy" if float(delta) > 0 else "sell", "kind": kind, "source": "executed_orders", "ticker": str(ticker), "quantity": abs(float(delta)), "quantity_delta": float(delta)})
    return pd.DataFrame(events, columns=[*EVENT_COLUMNS, "quantity", "quantity_delta"])


def _trades(data: RunData, ticker: str) -> pd.DataFrame:
    frame = _asset(data, data.trades, ticker)
    events = []
    for row in frame.to_dict("records"):
        side = str(row.get("side", "")).lower()
        if side not in ("long", "short") and row.get("direction") not in (-1, 1):
            continue
        short = side == "short" or row.get("direction") == -1
        for phase in ("entry", "exit"):
            date, price = row.get(f"{phase}_time"), row.get(f"{phase}_price")
            if date is None or price is None or pd.isna(date) or pd.isna(price) or not np.isfinite(float(price)) or float(price) <= 0:
                continue
            buy = (phase == "entry" and not short) or (phase == "exit" and short)
            events.append({"date": pd.to_datetime(date, utc=True), "price": float(price), "side": "buy" if buy else "sell", "kind": f"{phase}_{'short' if short else 'long'}", "source": "executed_trades", "ticker": str(ticker)})
    return pd.DataFrame(events, columns=EVENT_COLUMNS)


def _signals(data: RunData, ticker: str) -> pd.DataFrame:
    frame = _dated(_asset(data, data.positions, ticker))
    key = next((key for key in ("decoder_position", "executed_position", "sign_position", "position_close", "position", "weight", "quantity") if key in frame), None)
    if key is None or frame.empty or frame.date.duplicated().any():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    values = pd.to_numeric(frame[key], errors="coerce")
    if "available" in frame:
        values = values.where(frame.available.eq(True))
    signs = np.sign(values)
    prior = signs.shift(1)
    prices = asset_frame(data, ticker)
    source = "position_signals"
    if key in ("executed_position", "position_close", "quantity", "weight") and data.record.family in ("backtest", "trading"):
        source = "observed_positions"
    events = []
    for index, row in frame.iterrows():
        old, new = prior.iloc[index], signs.iloc[index]
        if pd.isna(old) or pd.isna(new) or old == new:
            continue  # First observation is an existing state, not an entry.
        price = _event_price(prices, row.date)
        if price is None:
            continue
        changes = []
        if old != 0:
            changes.append(("sell" if old > 0 else "buy", "exit_long" if old > 0 else "exit_short"))
        if new != 0:
            changes.append(("buy" if new > 0 else "sell", "entry_long" if new > 0 else "entry_short"))
        for side, kind in changes:
            events.append({"date": row.date, "price": price, "side": side, "kind": kind, "source": source, "ticker": str(ticker)})
    return pd.DataFrame(events, columns=EVENT_COLUMNS)


def position_events(data: RunData, ticker: str) -> pd.DataFrame:
    """Actual fills first. Signals are labeled and use sign transitions only."""
    for frame in (data.orders, data.trades, data.positions):
        cached = frame.attrs.get("position_events")
        if isinstance(cached, list):
            result = pd.DataFrame(cached)
            if result.empty and frame.attrs.get("events_complete"):
                return pd.DataFrame(columns=EVENT_COLUMNS)
            if not result.empty and "ticker" in result:
                result = result.loc[result.ticker.astype(str).eq(str(ticker))]
                result["date"] = pd.to_datetime(result.date, utc=True)
                if not data.equity.empty and "date" in data.equity:
                    dates = pd.to_datetime(data.equity.date, utc=True)
                    result = result.loc[result.date.ge(dates.min().normalize()) & result.date.lt(dates.max().normalize() + pd.Timedelta(days=1))]
                return result.sort_values("date", kind="stable").reset_index(drop=True)
    # An exported execution table is authoritative even if the chosen asset
    # has no valid fills; do not replace that fact with model target changes.
    if not _asset(data, data.orders, ticker).empty:
        result = _orders(data, ticker)
    elif not _asset(data, data.trades, ticker).empty:
        result = _trades(data, ticker)
    else:
        result = _signals(data, ticker)
    if result.empty:
        return result
    return result.sort_values("date", kind="stable").reset_index(drop=True)
