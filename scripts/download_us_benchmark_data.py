#!/usr/bin/env python3
"""Download and clean the tracked US stock universe used by the GNN benchmark."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:  # Imported as scripts.download_us_benchmark_data in tests.
    from scripts import _bootstrap  # type: ignore  # noqa: F401
import pandas as pd

from trading_system.data.cleaning import clean_ohlc_parquet
from trading_system.data.download import (
    FredDownloadError,
    build_exogenous_daily_panel,
    build_missing_credit_spread_frame,
    download_credit_spread_series,
    download_history,
    download_rate_macro_series,
    download_yahoo_macro_series,
    download_yahoo_snapshot_metadata,
    enrich_dataset,
    load_dependencies,
    resolve_end_date_for_yfinance,
    resolve_end_date_inclusive,
    validate_date,
)
from trading_system.paths import processed_data_dir


DEFAULT_SELECTION = Path("configs/benchmark/stocks_us_gnn_complete_2005.json")
DEFAULT_RAW = processed_data_dir() / "mt5_stocks_us_daily.parquet"
DEFAULT_CLEAN = processed_data_dir() / "mt5_stocks_us_daily_clean.parquet"


def load_tickers(path: str | Path) -> tuple[str, ...]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    values = payload.get("tickers") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("Ticker selection must be a JSON list or an object with `tickers`.")
    tickers = tuple(values)
    if not tickers or len(tickers) != len(set(tickers)) or any(
        not isinstance(ticker, str) or not ticker.strip() for ticker in tickers
    ):
        raise ValueError("Ticker selection must contain unique non-empty strings.")
    return tickers


def _write_parquet(frame: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}; pass --overwrite.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker-selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--start", default="2005-01-03")
    parser.add_argument("--end", help="Inclusive end date; defaults to the latest available date.")
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--clean-output", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--include-credit-spread", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tickers = load_tickers(args.ticker_selection)
    start = validate_date(args.start)
    end_inclusive = resolve_end_date_inclusive(args.end)
    if pd.Timestamp(end_inclusive) < pd.Timestamp(start):
        raise ValueError("--end must be on or after --start.")
    end_for_yahoo = resolve_end_date_for_yfinance(args.end)
    pandas_module, yfinance_module = load_dependencies()
    cache = Path(tempfile.gettempdir()) / "trading-system-yfinance-cache"
    cache.mkdir(parents=True, exist_ok=True)
    if hasattr(yfinance_module, "set_tz_cache_location"):
        yfinance_module.set_tz_cache_location(str(cache))
    constituents = pd.DataFrame({"ticker": tickers, "company": tickers})
    prices = download_history(
        pandas_module,
        yfinance_module,
        constituents,
        start,
        end_for_yahoo,
        args.batch_size,
        dataset_label="US benchmark",
        threads=False,
        download_retries=args.retries,
    )
    missing = sorted(set(tickers) - set(prices["ticker"]))
    if missing:
        raise RuntimeError(f"Yahoo returned no price history for: {missing}")
    yahoo_macro = download_yahoo_macro_series(
        pandas_module, yfinance_module, start, end_for_yahoo,
        market_ticker="^GSPC",
    )
    rate_macro = download_rate_macro_series(
        pandas_module, start, end_inclusive, args.timeout, args.retries,
        reference_area="USA",
    )
    if args.include_credit_spread:
        try:
            credit_macro = download_credit_spread_series(
                pandas_module, start, end_inclusive, args.timeout, args.retries,
            )
        except FredDownloadError as error:
            print(f"[macro-fred] {error}; credit_spread left missing.", flush=True)
            credit_macro = build_missing_credit_spread_frame(pandas_module, prices)
    else:
        credit_macro = build_missing_credit_spread_frame(pandas_module, prices)
    metadata = download_yahoo_snapshot_metadata(
        pandas_module, yfinance_module, constituents,
    )
    exogenous = build_exogenous_daily_panel(
        pandas_module, prices, yahoo_macro, rate_macro, credit_macro,
    )
    enriched = enrich_dataset(pandas_module, prices, exogenous, metadata)
    _write_parquet(enriched, args.raw_output, overwrite=args.overwrite)
    report = clean_ohlc_parquet(
        args.raw_output,
        args.clean_output,
        report_path=args.clean_output.with_suffix(".quality.json"),
        duplicate_policy="error",
        overwrite=args.overwrite,
    )
    clean = pd.read_parquet(args.clean_output, columns=["date", "ticker"])
    present = set(clean["ticker"])
    missing_after_cleaning = sorted(set(tickers) - present)
    if missing_after_cleaning:
        raise RuntimeError(
            f"Cleaning removed the complete history of: {missing_after_cleaning}"
        )
    manifest = {
        "schema_version": 1,
        "selection": str(args.ticker_selection),
        "included_tickers": list(tickers),
        "rows": len(clean),
        "start": pd.to_datetime(clean["date"]).min().date().isoformat(),
        "end": pd.to_datetime(clean["date"]).max().date().isoformat(),
        "rows_dropped": report["rows_dropped"],
        "snapshot_sector_warning": (
            "Yahoo sector metadata is current, not point-in-time; historical sector "
            "graphs remain exploratory."
        ),
    }
    manifest_path = args.clean_output.with_suffix(".universe.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "raw_path": str(args.raw_output.resolve()),
        "clean_path": str(args.clean_output.resolve()),
        "universe_manifest": str(manifest_path.resolve()),
        "tickers": len(tickers),
        "rows": len(clean),
        "start": manifest["start"],
        "end": manifest["end"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
