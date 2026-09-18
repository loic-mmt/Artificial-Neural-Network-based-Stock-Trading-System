from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import _bootstrap

from trading_system.data.cleaning import clean_ohlc_parquet


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate OHLCV data, drop impossible rows, and write a JSON quality report. "
            "Prices are never clipped, interpolated, or otherwise fabricated."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input Parquet file.")
    parser.add_argument("--output", required=True, type=Path, help="Clean Parquet file.")
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path (default: <output stem>.quality.json).",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=1e-8,
        help="Relative tolerance for OHLC range comparisons (default: 1e-8).",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "first", "last"),
        default="error",
        help="How to handle duplicate ticker/date keys (default: error).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print counts without writing files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = clean_ohlc_parquet(
        args.input,
        args.output,
        report_path=args.report,
        relative_tolerance=args.relative_tolerance,
        duplicate_policy=args.duplicate_policy,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(
        f"rows_input={report['rows_input']} rows_output={report['rows_output']} "
        f"rows_dropped={report['rows_dropped']}"
    )
    print(f"violations={report['rule_violation_counts']}")
    if args.dry_run:
        print("dry_run=true; no files written")
    else:
        print(f"cleaned={report['output_path']}")
        print(f"quality_report={report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
