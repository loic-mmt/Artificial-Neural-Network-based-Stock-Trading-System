"""Download ECB real-time database vintages, without assigning release times."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from trading_system.data.ecb_rtd import DEFAULT_SERIES, fetch_csv, save_vintages
from trading_system.paths import processed_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=processed_data_dir() / "ecb_rtd_vintages_raw.parquet")
    parser.add_argument("--series", action="append", choices=tuple(DEFAULT_SERIES),
                        help="Repeat to select series; default: HICP and real GDP.")
    parser.add_argument("--input-dir", type=Path,
                        help="Use pre-downloaded <series name>.csv files instead of the API.")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    selected = tuple(args.series or DEFAULT_SERIES)
    sources = {}
    for key in selected:
        name = DEFAULT_SERIES[key]
        sources[key] = (args.input_dir / f"{name}.csv").read_bytes() if args.input_dir else fetch_csv(key, timeout=args.timeout)
    print(json.dumps(save_vintages(sources, args.output), indent=2))


if __name__ == "__main__":
    main()
