# Point-in-time macro and company snapshots

The existing `cac40_daily_clean.parquet` contains historical price rows enriched with macro data and a company metadata snapshot. Its company fields do **not** have publication timestamps and must not be treated as point-in-time historical fundamentals. The new collectors do not backfill those rows.

## Collect manually

```sh
.venv/bin/python scripts/download_context_data.py
```

Outputs:

- `data/processed/cac40_macro_pit.parquet`: CAC 40, VIX, Brent, dollar index, gold, U.S. 2y/10y, and France 2y/10y rate values. `--include-credit-spread` additionally requests the optional FRED credit series.
- `data/processed/cac40_fundamentals_pit.parquet`: sector/industry plus available Yahoo company metrics for only the ten configured tickers.

The collector records `observation_date` where the macro source provides one, `collected_at_utc`, and `available_at_utc`. A source date is **not** a release date. Every value first becomes usable at the actual collection timestamp. Company `info` is a current snapshot; it has no trustworthy historical release timestamp. Missing fields remain null. In the initial live check, Yahoo returned no short-interest values for these ten tickers.

## Collect automatically

```sh
.venv/bin/python scripts/auto_update_context_data.py
```

It collects at startup, then weekdays at **23:30 Europe/Paris**. This allows the U.S. market series to close before the scheduled snapshot. Failed runs retry after 30 minutes. The process must stay running. On a host with an external scheduler, run `scripts/auto_update_context_data.py --once` instead.

## Backtest rule

Use `trading_system.data.pit_context.asof_snapshot(frame, as_of, key=...)`, with a timezone-aware decision timestamp. It returns no value from a later collection, even when that value has an older `observation_date`. Revisions are appended as new versions, preserving the previous as-of view. Nothing in these new files is eligible for dates before the first actual collection. Do not join by fiscal period or macro observation date alone.

The current web GRU uses technical price features and still reads `cac40_daily_clean.parquet`. It does not use these new context snapshots. A future model using macro or fundamentals must join the new files as of each decision timestamp, then be trained and benchmarked again. Yahoo data and some public macro feeds can be revised; this archive proves what our collector observed and when, not the source's original publication time.
