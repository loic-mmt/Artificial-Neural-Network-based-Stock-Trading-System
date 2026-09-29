"""Download raw ALFRED revisions from FRED, without assuming intraday availability."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401

from trading_system.data.fred_vintages import DEFAULT_SERIES, fetch_series, save_vintages
from trading_system.paths import processed_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=processed_data_dir() / "fred_vintages_raw.parquet")
    parser.add_argument("--series", action="append", choices=tuple(DEFAULT_SERIES),
                        help="Repeat to select series; default: CPIAUCSL, UNRATE, PAYEMS, INDPRO.")
    parser.add_argument("--start", default="2000-01-01", help="Observation and real-time start date.")
    parser.add_argument("--end", help="Real-time end date, defaults to today.")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    key = os.environ.get("FRED_API_KEY")
    if not key:
        try:
            from dotenv import dotenv_values
        except ImportError:
            parser.error("Set FRED_API_KEY or install python-dotenv to read .env.")
        key = dotenv_values(Path.cwd() / ".env").get("FRED_API_KEY")
    if not key:
        parser.error("FRED_API_KEY is absent from environment and .env.")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    frames = []
    for sid in args.series or DEFAULT_SERIES:
        frame = fetch_series(key, sid, name=DEFAULT_SERIES[sid], observation_start=args.start,
                             realtime_start=args.start, realtime_end=args.end, timeout=args.timeout)
        frames.append(frame)
        print(f"{sid}: {len(frame)} vintage records", flush=True)
    print(json.dumps(save_vintages(frames, args.output), indent=2))


if __name__ == "__main__":
    main()
