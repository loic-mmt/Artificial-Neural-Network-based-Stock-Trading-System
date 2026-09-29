"""Manually download or refresh daily OHLCV for the benchmark ten tickers."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import _bootstrap  # noqa: F401

from trading_system.data.ticker_updates import DEFAULT_OUTPUT, update_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the ten CAC 40 ticker histories only")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2000, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, help="Inclusive; default is latest completed Paris day")
    parser.add_argument("--overlap-days", type=int, default=14)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    result = update_prices(args.output, start=args.start, end=args.end,
                           overlap_days=args.overlap_days, retries=args.retries)
    print(f"{result['tickers']} tickers; {result['rows']} rows; {result['new_rows']} new; through {result['latest_date']}; {result['path']}")


if __name__ == "__main__":
    main()
