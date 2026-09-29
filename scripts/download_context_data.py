"""Manually collect macro and fundamental point-in-time snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from trading_system.data.pit_context import FUNDAMENTAL_PATH, MACRO_PATH, collect_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect current macro and company values with actual observation timestamps")
    parser.add_argument("--macro-output", type=Path, default=MACRO_PATH)
    parser.add_argument("--fundamental-output", type=Path, default=FUNDAMENTAL_PATH)
    parser.add_argument("--include-credit-spread", action="store_true", help="Also fetch optional FRED credit spread")
    args = parser.parse_args()
    result = collect_context(macro_path=args.macro_output, fundamental_path=args.fundamental_output,
                             include_credit_spread=args.include_credit_spread)
    print(f"{result['macro_metrics']} macro metrics; {result['fundamental_tickers']} tickers; "
          f"available from {result['available_at_utc']}")
    print(result["macro_path"])
    print(result["fundamental_path"])


if __name__ == "__main__":
    main()
