"""Append-only, observation-time snapshots for macro and company context.

Yahoo company info and public macro feeds are current/revisable views. Historical
rows fetched today are never backdated: availability begins at collection time.
"""

from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading_system.data.download import (
    YAHOO_MACRO_TICKERS,
    download_credit_spread_series,
    download_rate_macro_series,
    download_yahoo_macro_series,
    normalize_sector_bucket,
)
from trading_system.data.ticker_updates import ticker_universe
from trading_system.paths import processed_data_dir


MACRO_PATH = processed_data_dir() / "cac40_macro_pit.parquet"
FUNDAMENTAL_PATH = processed_data_dir() / "cac40_fundamentals_pit.parquet"
MACRO_METRICS = tuple(YAHOO_MACRO_TICKERS.values()) + ("ust2y", "ust10y", "frt2y", "frt10y")
FUNDAMENTAL_NUMERIC = ("market_cap", "book_value", "trailing_eps", "shares_outstanding", "short_percent_float", "short_ratio")
FUNDAMENTAL_FIELDS = ("sector", "industry", "sector_bucket", *FUNDAMENTAL_NUMERIC)
MACRO_COLUMNS = ("metric", "value", "source", "observation_date", "collected_at_utc", "available_at_utc")
FUNDAMENTAL_COLUMNS = ("ticker", *FUNDAMENTAL_FIELDS, "source", "collected_at_utc", "available_at_utc")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _latest(frame: pd.DataFrame, metric: str, source: str, stamp: pd.Timestamp) -> dict[str, Any]:
    if metric not in frame:
        raise ValueError(f"Macro source lacks {metric}")
    values = frame[["date", metric]].copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce", utc=True)
    values[metric] = pd.to_numeric(values[metric], errors="coerce")
    values = values.dropna().loc[lambda data: data["date"] <= stamp].sort_values("date")
    if values.empty:
        raise ValueError(f"No observed macro value for {metric}")
    last = values.iloc[-1]
    return {"metric": metric, "value": float(last[metric]), "source": source,
            "observation_date": last["date"].date().isoformat(),
            "collected_at_utc": stamp, "available_at_utc": stamp}


def fetch_macro(*, provider: Any = None, now: datetime | None = None, lookback_days: int = 180,
                timeout: int = 30, retries: int = 2, include_credit_spread: bool = False) -> pd.DataFrame:
    """Collect latest observable values; historical source dates stay metadata."""
    if lookback_days < 120 or timeout < 1 or retries < 1:
        raise ValueError("Invalid macro fetch settings")
    if provider is None:
        import yfinance as provider

    end = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date()
    start = end - timedelta(days=lookback_days)
    start_text, end_text = start.isoformat(), end.isoformat()
    yahoo = download_yahoo_macro_series(pd, provider, start_text, (end + timedelta(days=1)).isoformat())
    rates = download_rate_macro_series(pd, start_text, end_text, timeout, retries)
    credit = None
    if include_credit_spread:
        credit = download_credit_spread_series(pd, start_text, end_text, timeout, retries)
    # Stamp after all network calls. No fetched value is claimed available before
    # the full batch completed, even if its source date is months earlier.
    stamp = pd.Timestamp(datetime.now(timezone.utc))
    rows = [_latest(yahoo, name, f"yahoo:{ticker}", stamp) for ticker, name in YAHOO_MACRO_TICKERS.items()]
    rows += [_latest(rates, name, "us_treasury" if name.startswith("ust") else "oecd", stamp)
             for name in ("ust2y", "ust10y", "frt2y", "frt10y")]
    if credit is not None:
        rows.append(_latest(credit, "credit_spread", "fred", stamp))
    return pd.DataFrame(rows, columns=MACRO_COLUMNS)


def fetch_fundamentals(*, provider: Any = None) -> pd.DataFrame:
    """Collect current company metadata; never infer historical publication dates."""
    if provider is None:
        import yfinance as provider

    rows = []
    for ticker in ticker_universe():
        instrument = provider.Ticker(ticker)
        info = instrument.get_info()
        if not isinstance(info, dict) or not info:
            raise ValueError(f"No fundamental snapshot for {ticker}")
        try:
            fast = dict(instrument.fast_info)
        except Exception:
            fast = {}
        sector = str(info.get("sector") or "unknown").strip() or "unknown"
        industry = str(info.get("industry") or "unknown").strip() or "unknown"
        row = {
            "ticker": ticker, "sector": sector, "industry": industry,
            "sector_bucket": normalize_sector_bucket(sector, industry),
            "market_cap": _finite(info.get("marketCap")) or _finite(fast.get("market_cap")),
            "book_value": _finite(info.get("bookValue")),
            "trailing_eps": _finite(info.get("trailingEps")),
            "shares_outstanding": _finite(info.get("sharesOutstanding")) or _finite(fast.get("shares")),
            "short_percent_float": _finite(info.get("shortPercentOfFloat")),
            "short_ratio": _finite(info.get("shortRatio")),
            "source": "yahoo_info",
        }
        if all(row[field] is None for field in FUNDAMENTAL_NUMERIC):
            raise ValueError(f"No numeric company data for {ticker}")
        rows.append(row)
    stamp = pd.Timestamp(datetime.now(timezone.utc))
    for row in rows:
        row["collected_at_utc"] = stamp
        row["available_at_utc"] = stamp
    return pd.DataFrame(rows, columns=FUNDAMENTAL_COLUMNS)


def asof_snapshot(frame: pd.DataFrame, as_of: datetime | pd.Timestamp, *, key: str) -> pd.DataFrame:
    """Return values actually collected by an instant, latest version per key."""
    if key not in frame or "available_at_utc" not in frame:
        raise ValueError("Point-in-time frame lacks key or availability timestamp")
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of must include timezone")
    available = pd.to_datetime(frame["available_at_utc"], utc=True, errors="coerce")
    if available.isna().any():
        raise ValueError("Point-in-time frame has invalid availability timestamps")
    eligible = frame.loc[available <= cutoff.tz_convert("UTC")].copy()
    return eligible.sort_values("available_at_utc").drop_duplicates(key, keep="last").reset_index(drop=True)


def _stage(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(descriptor)
    temporary = Path(filename)
    try:
        frame.to_parquet(temporary, index=False)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _append(existing_path: Path, new: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if tuple(new.columns) != columns or new.empty:
        raise ValueError("Invalid point-in-time snapshot schema")
    if not existing_path.exists():
        return new
    old = pd.read_parquet(existing_path)
    if tuple(old.columns) != columns:
        raise ValueError(f"Existing point-in-time file has incompatible schema: {existing_path}")
    if not old.empty and pd.to_datetime(new["available_at_utc"], utc=True).min() <= pd.to_datetime(old["available_at_utc"], utc=True).max():
        raise ValueError("Collection time must advance; old snapshots are immutable")
    return pd.concat([old, new], ignore_index=True)


def collect_context(*, macro_path: Path = MACRO_PATH, fundamental_path: Path = FUNDAMENTAL_PATH,
                    provider: Any = None, include_credit_spread: bool = False) -> dict[str, Any]:
    """Fetch both sources before writing; prior versions remain append-only."""
    macro = fetch_macro(provider=provider, include_credit_spread=include_credit_spread)
    fundamentals = fetch_fundamentals(provider=provider)
    macro_target, fundamental_target = Path(macro_path), Path(fundamental_path)
    macro_all = _append(macro_target, macro, MACRO_COLUMNS)
    fundamental_all = _append(fundamental_target, fundamentals, FUNDAMENTAL_COLUMNS)
    first: Path | None = None
    second: Path | None = None
    try:
        first = _stage(macro_all, macro_target)
        second = _stage(fundamental_all, fundamental_target)
        os.replace(first, macro_target)
        first = None
        os.replace(second, fundamental_target)
        second = None
    finally:
        for remaining in (first, second):
            if remaining is not None:
                remaining.unlink(missing_ok=True)
    return {"macro_metrics": len(macro), "fundamental_tickers": len(fundamentals),
            "available_at_utc": macro["available_at_utc"].iloc[0].isoformat(),
            "macro_path": str(macro_target), "fundamental_path": str(fundamental_target)}
