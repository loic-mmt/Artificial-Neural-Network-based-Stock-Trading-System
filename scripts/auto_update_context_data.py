"""Refresh point-in-time macro and fundamental snapshots on weekdays."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import _bootstrap  # noqa: F401

from trading_system.data.pit_context import FUNDAMENTAL_PATH, MACRO_PATH, collect_context
from trading_system.data.ticker_updates import next_weekday_run


PARIS = ZoneInfo("Europe/Paris")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PIT context at 23:30 Paris time on weekdays")
    parser.add_argument("--macro-output", type=Path, default=MACRO_PATH)
    parser.add_argument("--fundamental-output", type=Path, default=FUNDAMENTAL_PATH)
    parser.add_argument("--time", default="23:30", help="Weekday Paris time, HH:MM")
    parser.add_argument("--once", action="store_true", help="Run one collection and exit; suitable for cron")
    parser.add_argument("--retry-minutes", type=int, default=30)
    parser.add_argument("--include-credit-spread", action="store_true")
    args = parser.parse_args()
    try:
        hour, minute = (int(part) for part in args.time.split(":"))
        next_weekday_run(datetime.now(PARIS), hour, minute)
    except (ValueError, TypeError):
        parser.error("--time must be HH:MM within 00:00 to 23:59")
    if args.retry_minutes < 1:
        parser.error("--retry-minutes must be positive")

    def refresh() -> None:
        result = collect_context(macro_path=args.macro_output,
                                 fundamental_path=args.fundamental_output,
                                 include_credit_spread=args.include_credit_spread)
        print(f"{datetime.now(PARIS).isoformat()} collected {result['macro_metrics']} macro values and "
              f"{result['fundamental_tickers']} company snapshots", flush=True)

    if args.once:
        refresh()
        return

    due = datetime.now(PARIS)
    print("Point-in-time context updater running; Europe/Paris weekday schedule", flush=True)
    try:
        while True:
            seconds = (due - datetime.now(PARIS)).total_seconds()
            if seconds > 0:
                time.sleep(min(seconds, 60))
                continue
            try:
                refresh()
            except Exception as exc:
                print(f"{datetime.now(PARIS).isoformat()} context update failed: {exc}; retry scheduled", flush=True)
                due = datetime.now(PARIS) + timedelta(minutes=args.retry_minutes)
            else:
                due = next_weekday_run(datetime.now(PARIS), hour, minute)
            print(f"Next attempt: {due.isoformat()}", flush=True)
    except KeyboardInterrupt:
        print("Context updater stopped", flush=True)


if __name__ == "__main__":
    main()
