"""Audited close-observed US market context for relational experiments.

The securities below are context factors, not tradable targets.  Their closes
may be used for a decision after session J and execution on J+1 or later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


US_MARKET_CONTEXT_TICKERS: dict[str, str] = {
    "SPY": "spy_close",
    "QQQ": "qqq_close",
    "IWM": "iwm_close",
    "XLB": "xlb_close",
    # XLC and XLRE have short histories.  VOX and VNQ are stable, older
    # point-in-time proxies for communication services and real estate.
    "VOX": "vox_close",
    "XLE": "xle_close",
    "XLF": "xlf_close",
    "XLI": "xli_close",
    "XLK": "xlk_close",
    "XLP": "xlp_close",
    "VNQ": "vnq_close",
    "XLU": "xlu_close",
    "XLV": "xlv_close",
    "XLY": "xly_close",
}


US_SECTOR_CONTEXT_COLUMNS: dict[str, str] = {
    "Basic Materials": "xlb_close",
    "Communication": "vox_close",
    "Communication Services": "vox_close",
    "Consumer Cyclical": "xly_close",
    "Consumer Defensive": "xlp_close",
    "Consumer Discretionary": "xly_close",
    "Consumer Staples": "xlp_close",
    "Energy": "xle_close",
    "Financial": "xlf_close",
    "Financial Services": "xlf_close",
    "Healthcare": "xlv_close",
    "Industrials": "xli_close",
    "Real Estate": "vnq_close",
    "Technology": "xlk_close",
    "Utilities": "xlu_close",
}


def _session_end(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, utc=True, errors="raise").dt.normalize()
    return dates + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def validate_us_market_context(
    frame: pd.DataFrame,
    *,
    ticker_columns: Mapping[str, str] = US_MARKET_CONTEXT_TICKERS,
    date_col: str = "date",
) -> pd.DataFrame:
    """Validate the raw, one-row-per-session ETF close panel."""

    columns = tuple(ticker_columns.values())
    missing = sorted({date_col, "source_end", *columns} - set(frame))
    if missing:
        raise ValueError(f"US market context is missing columns: {missing}")
    work = frame[[date_col, *columns, "source_end"]].copy()
    work[date_col] = pd.to_datetime(work[date_col], utc=True, errors="raise").dt.normalize()
    if work[date_col].duplicated().any():
        raise ValueError("US market context requires one row per session.")
    source_end = pd.to_datetime(work["source_end"], utc=True, errors="raise")
    if (source_end >= work[date_col] + pd.Timedelta(days=1)).any():
        raise ValueError("US market context contains a future close timestamp.")
    numeric = work.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if (~np.isfinite(numeric.to_numpy(dtype=float))).any() or (numeric <= 0).any().any():
        raise ValueError("US market context closes must be finite and positive.")
    work.loc[:, columns] = numeric
    return work.sort_values(date_col).reset_index(drop=True)


def download_us_market_context(
    pandas_module: Any,
    yfinance_module: Any,
    *,
    start: str,
    end: str | None,
    ticker_columns: Mapping[str, str] = US_MARKET_CONTEXT_TICKERS,
) -> pd.DataFrame:
    """Download adjusted ETF closes.  ``end`` follows yfinance exclusivity."""

    # Local import avoids making yfinance a runtime dependency of model code.
    from trading_system.data.download import _download_single_yahoo_close_series

    merged = None
    for ticker, column in ticker_columns.items():
        print(f"[market-context] Download {ticker} -> {column}", flush=True)
        values = _download_single_yahoo_close_series(
            pandas_module, yfinance_module, ticker, start, end,
        )
        if values.empty:
            raise RuntimeError(f"Yahoo returned no context history for {ticker}.")
        values = values.rename(columns={"value": column})
        merged = values if merged is None else merged.merge(values, on="date", how="inner")
    if merged is None or merged.empty:
        raise RuntimeError("Yahoo returned no common US market context sessions.")
    merged["date"] = pandas_module.to_datetime(merged["date"], utc=True).dt.normalize()
    merged["source_end"] = _session_end(merged["date"])
    return validate_us_market_context(merged, ticker_columns=ticker_columns)


def complete_ticker_selection(
    frame: pd.DataFrame,
    *,
    start: object,
    end: object | None = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> tuple[str, ...]:
    """Return assets with a complete calendar over the requested research era."""

    missing = sorted({date_col, ticker_col} - set(frame))
    if missing:
        raise ValueError(f"Ticker selection input is missing columns: {missing}")
    dates = pd.to_datetime(frame[date_col], utc=True, errors="raise").dt.normalize()
    start_at = pd.Timestamp(start)
    if start_at.tzinfo is None:
        start_at = start_at.tz_localize("UTC")
    else:
        start_at = start_at.tz_convert("UTC")
    mask = dates >= start_at.normalize()
    if end is not None:
        end_at = pd.Timestamp(end)
        end_at = end_at.tz_localize("UTC") if end_at.tzinfo is None else end_at.tz_convert("UTC")
        mask &= dates <= end_at.normalize()
    work = pd.DataFrame({date_col: dates[mask], ticker_col: frame.loc[mask, ticker_col].to_numpy()})
    if work.empty:
        raise ValueError("Ticker selection interval contains no observations.")
    if work.duplicated([date_col, ticker_col]).any():
        raise ValueError("Ticker selection input contains duplicate session/ticker rows.")
    sessions = work[date_col].nunique()
    counts = work.groupby(ticker_col, sort=True)[date_col].nunique()
    selected = tuple(counts.index[counts.eq(sessions)].astype(str))
    if not selected:
        raise ValueError("No ticker has a complete calendar over the requested interval.")
    return selected


def write_ticker_selection(
    path: str | Path,
    tickers: tuple[str, ...],
    *,
    start: str,
    source: str | Path,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tickers": list(tickers),
        "selection_policy": "complete common calendar from start through source maximum date",
        "start": start,
        "source": str(Path(source)),
        "count": len(tickers),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


__all__ = [
    "US_MARKET_CONTEXT_TICKERS",
    "US_SECTOR_CONTEXT_COLUMNS",
    "complete_ticker_selection",
    "download_us_market_context",
    "validate_us_market_context",
    "write_ticker_selection",
]
