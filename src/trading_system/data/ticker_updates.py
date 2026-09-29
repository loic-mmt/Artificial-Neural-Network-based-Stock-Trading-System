"""Download and incrementally refresh the fixed CAC 40 ticker price universe."""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from trading_system.data.cleaning import clean_ohlc_frame
from trading_system.data.download import extract_ticker_frame, normalize_frame
from trading_system.paths import processed_data_dir, project_root


UNIVERSE_FILE = project_root() / "configs/benchmark/cac40_diversified_10.json"
DEFAULT_OUTPUT = processed_data_dir() / "cac40_ticker_daily.parquet"
PRICE_COLUMNS = ("date", "ticker", "company", "open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits")
PARIS = ZoneInfo("Europe/Paris")


def ticker_universe(path: Path = UNIVERSE_FILE) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("tickers")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Benchmark universe has no tickers")
    tickers = tuple(row["ticker"] for row in rows)
    if len(set(tickers)) != len(tickers) or any(not ticker.endswith(".PA") for ticker in tickers):
        raise ValueError("Benchmark universe has invalid or duplicate tickers")
    return tickers


def latest_complete_date(now: datetime | None = None) -> date:
    """Skip an unfinished Paris trading day; provider may post later."""
    current = (now or datetime.now(PARIS)).astimezone(PARIS)
    day = current.date() if current.hour >= 19 else current.date() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def next_weekday_run(now: datetime, hour: int = 19, minute: int = 0) -> datetime:
    """Next Paris weekday wall-clock run, including daylight saving changes."""
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("Scheduled time must be HH:MM")
    local = now.astimezone(PARIS)
    for offset in range(8):
        day = local.date() + timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        candidate = datetime.combine(day, clock_time(hour, minute), PARIS)
        if candidate > local:
            return candidate
    raise RuntimeError("Could not find next weekday run")


def _company_names() -> dict[str, str]:
    from trading_system.data.download import CAC40_CONSTITUENTS

    return {row["ticker"]: row["company"] for row in CAC40_CONSTITUENTS}


def _validate(frame: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty or set(frame["ticker"]) != set(tickers):
        raise ValueError("Downloaded prices do not cover every configured ticker")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Downloaded prices contain duplicate ticker/date rows")
    cleaned, report = clean_ohlc_frame(frame)
    if report["rows_dropped"]:
        raise ValueError(f"Downloaded prices contain {report['rows_dropped']} invalid rows: {report['rule_violation_counts']}")
    if cleaned["date"].dt.tz is not None:
        cleaned["date"] = cleaned["date"].dt.tz_localize(None)
    return cleaned.loc[:, PRICE_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


def download_prices(
    start: date,
    end: date,
    *,
    tickers: tuple[str, ...] | None = None,
    provider: Any = None,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """Fetch daily Yahoo OHLCV/actions, with inclusive end and no macro calls."""
    if start > end:
        raise ValueError("Start date must not follow end date")
    if retries < 1 or retry_delay < 0:
        raise ValueError("Invalid retry settings")
    universe = ticker_universe()
    selected = universe if tickers is None else tuple(tickers)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(universe):
        raise ValueError("Requested ticker is outside the configured universe")
    if provider is None:
        import yfinance as provider

    company_names = _company_names()
    parts = []
    for ticker in selected:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                raw = provider.download(
                    tickers=ticker, start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(), interval="1d",
                    auto_adjust=False, actions=True, progress=False,
                    group_by="ticker", threads=False,
                )
                one = extract_ticker_frame(raw, ticker, pd)
                if one.empty:
                    raise ValueError(f"No daily prices returned for {ticker}")
                normalized = normalize_frame(one, ticker, company_names[ticker])
                normalized["date"] = pd.to_datetime(normalized["date"], utc=True).dt.tz_localize(None).dt.normalize()
                normalized = normalized.loc[normalized["date"].dt.date.between(start, end)]
                if normalized.empty:
                    raise ValueError(f"No dates in requested range for {ticker}")
                parts.append(normalized)
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(retry_delay * (attempt + 1))
        else:
            raise RuntimeError(f"Could not download {ticker} after {retries} attempts") from last_error
    downloaded = _validate(pd.concat(parts, ignore_index=True), selected)
    coverage = downloaded.groupby("ticker")["date"].agg(["min", "max"])
    if (coverage["min"] > pd.Timestamp(start + timedelta(days=14))).any():
        raise ValueError("Provider returned incomplete history near the requested start")
    if (coverage["max"] < pd.Timestamp(end - timedelta(days=14))).any():
        raise ValueError("Provider returned stale history near the requested end")
    return downloaded


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(handle)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def update_prices(
    output: Path = DEFAULT_OUTPUT,
    *,
    start: date = date(2000, 1, 1),
    end: date | None = None,
    overlap_days: int = 14,
    provider: Any = None,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> dict[str, Any]:
    """Download all ten assets; replace overlap; publish only if all pass."""
    path = Path(output)
    last_day = end or latest_complete_date()
    universe = ticker_universe()
    if overlap_days < 1:
        raise ValueError("overlap_days must be positive")
    if start > last_day:
        raise ValueError("Start date must not follow end date")
    if path.exists():
        existing = pd.read_parquet(path)
        if set(existing.columns) != set(PRICE_COLUMNS):
            raise ValueError("Existing ticker file has a different schema")
        existing = _validate(existing, universe)
        first_dates = existing.groupby("ticker")["date"].min()
        last_dates = existing.groupby("ticker")["date"].max()
        if any(first_dates > pd.Timestamp(start + timedelta(days=7))):
            fetch_start = start
        else:
            fetch_start = min((value.date() - timedelta(days=overlap_days - 1)) for value in last_dates)
            fetch_start = max(fetch_start, start)
    else:
        existing = pd.DataFrame(columns=PRICE_COLUMNS)
        fetch_start = start
    if fetch_start > last_day:
        return {"path": str(path), "rows": len(existing), "new_rows": 0, "latest_date": existing["date"].max().date().isoformat()}
    downloaded = download_prices(fetch_start, last_day, tickers=universe, provider=provider, retries=retries, retry_delay=retry_delay)
    if existing.empty:
        merged = downloaded
        new_rows = len(downloaded)
    else:
        old_keys = pd.MultiIndex.from_frame(existing[["ticker", "date"]])
        new_keys = pd.MultiIndex.from_frame(downloaded[["ticker", "date"]])
        new_rows = int((~new_keys.isin(old_keys)).sum())
        untouched = existing.loc[~old_keys.isin(new_keys)]
        merged = _validate(pd.concat([untouched, downloaded], ignore_index=True), universe)
        for ticker in universe:
            if merged.loc[merged["ticker"] == ticker, "date"].max() < existing.loc[existing["ticker"] == ticker, "date"].max():
                raise ValueError(f"Update would roll back latest date for {ticker}")
    _atomic_parquet(merged, path)
    return {"path": str(path), "rows": len(merged), "new_rows": new_rows,
            "latest_date": merged["date"].max().date().isoformat(), "tickers": len(universe)}
