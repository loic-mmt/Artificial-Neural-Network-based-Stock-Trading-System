#!/usr/bin/env python3
"""Download non-tradable US ETF context and freeze a complete stock universe."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from trading_system.data.download import load_dependencies
from trading_system.data.us_market_context import (
    complete_ticker_selection,
    download_us_market_context,
    write_ticker_selection,
)
from trading_system.paths import processed_data_dir


DEFAULT_STOCK_DATA = processed_data_dir() / "mt5_stocks_us_daily_clean.parquet"
DEFAULT_CONTEXT_DATA = processed_data_dir() / "us_market_context_daily.parquet"
DEFAULT_SELECTION = Path("configs/benchmark/stocks_us_gnn_complete_2005.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-data", type=Path, default=DEFAULT_STOCK_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_CONTEXT_DATA)
    parser.add_argument("--ticker-selection-output", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--start", default="2005-01-03")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stocks = pd.read_parquet(args.stock_data, columns=["date", "ticker"])
    stock_dates = pd.to_datetime(stocks["date"], utc=True, errors="raise").dt.normalize()
    last_session = stock_dates.max()
    first_session = pd.Timestamp(args.start, tz="UTC")
    if first_session > last_session:
        raise ValueError("--start follows the last stock session.")
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Context output already exists: {destination}; pass --overwrite.")
    pandas_module, yfinance_module = load_dependencies()
    context = download_us_market_context(
        pandas_module,
        yfinance_module,
        start=(first_session - pd.Timedelta(days=120)).date().isoformat(),
        end=(last_session + timedelta(days=1)).date().isoformat(),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    context.to_parquet(temporary, index=False)
    temporary.replace(destination)
    selected = complete_ticker_selection(stocks, start=first_session, end=last_session)
    selection = write_ticker_selection(
        args.ticker_selection_output, selected, start=first_session.date().isoformat(),
        source=args.stock_data,
    )
    print(f"context={destination} rows={len(context)} start={context.date.min().date()} end={context.date.max().date()}")
    print(f"selection={selection} tickers={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
