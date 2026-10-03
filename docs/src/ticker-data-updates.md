# Ticker price updates

These jobs cover only the ten symbols in `configs/benchmark/cac40_diversified_10.json`. They fetch daily OHLCV, adjusted close, dividends, and splits from Yahoo Finance through the project's existing `yfinance` dependency. They do not download macro series, fundamental fields, sector metadata, or any other symbol.

## Manual

From the repository root:

```sh
.venv/bin/python scripts/download_ticker_data.py
```

The first run downloads history from 2000-01-01 into `data/processed/cac40_ticker_daily.parquet`. Later runs fetch a 14-calendar-day overlap plus newer rows, replacing revised values in that overlap. To request a specific inclusive range:

```sh
.venv/bin/python scripts/download_ticker_data.py --start 2026-04-01 --end 2026-09-23
```

The default end is the latest completed Paris weekday, using 19:00 local time as the cutoff. The script requires every configured ticker, validates OHLCV and duplicate keys, then replaces the Parquet atomically. A failed ticker leaves the previous file intact.

## Automatic

Start the scheduler in a persistent terminal or process manager:

```sh
.venv/bin/python scripts/auto_update_ticker_data.py
```

It updates once at startup, then each weekday at **19:00 Europe/Paris**. Failed updates retry after 30 minutes. Stop with Ctrl+C. A sleeping or stopped host cannot run this scheduler; on such a host, schedule the one-shot form with the host's cron or job service:

```sh
.venv/bin/python scripts/auto_update_ticker_data.py --once
```

The ticker file is deliberately separate from `data/processed/cac40_daily_clean.parquet`, which contains the benchmark's historical macro and fundamental columns. The GRU publisher still reads that existing clean file. The next data-integration step must join and validate new ticker rows before model refresh; these scripts do not claim that site signals are current.

Macro and company snapshot collection is described in [point-in-time-context.md](point-in-time-context.md).
