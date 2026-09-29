"""Keep the ten-ticker OHLCV file refreshed on Paris market weekdays."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import _bootstrap  # noqa: F401

from trading_system.data.ticker_updates import DEFAULT_OUTPUT, next_weekday_run, update_prices


PARIS = ZoneInfo("Europe/Paris")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh ticker prices at 19:00 Europe/Paris on weekdays")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--time", default="19:00", help="Weekday Paris time, HH:MM")
    parser.add_argument("--once", action="store_true", help="Run one update and exit; suitable for cron")
    parser.add_argument("--retry-minutes", type=int, default=30)
    args = parser.parse_args()
    try:
        hour, minute = (int(part) for part in args.time.split(":"))
        next_weekday_run(datetime.now(PARIS), hour, minute)
    except (ValueError, TypeError):
        parser.error("--time must be HH:MM within 00:00 to 23:59")
    if args.retry_minutes < 1:
        parser.error("--retry-minutes must be positive")

    def refresh() -> None:
        result = update_prices(args.output)
        print(f"{datetime.now(PARIS).isoformat()} updated {result['tickers']} tickers; "
              f"{result['new_rows']} new rows; through {result['latest_date']}", flush=True)

    if args.once:
        refresh()
        return

    # Catch up once when the process starts, then keep the local weekday schedule.
    due = datetime.now(PARIS)
    print("Ticker updater running; Europe/Paris weekday schedule", flush=True)
    try:
        while True:
            seconds = (due - datetime.now(PARIS)).total_seconds()
            if seconds > 0:
                time.sleep(min(seconds, 60))
                continue
            try:
                refresh()
            except Exception as exc:
                print(f"{datetime.now(PARIS).isoformat()} update failed: {exc}; retry scheduled", flush=True)
                due = datetime.now(PARIS) + timedelta(minutes=args.retry_minutes)
            else:
                due = next_weekday_run(datetime.now(PARIS), hour, minute)
            print(f"Next attempt: {due.isoformat()}", flush=True)
    except KeyboardInterrupt:
        print("Ticker updater stopped", flush=True)


if __name__ == "__main__":
    main()
