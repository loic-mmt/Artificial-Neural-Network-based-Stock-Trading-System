"""Attach user-supplied publication-dated fundamentals and free sentiment scores."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from trading_system.data.optional_sources import prepare_feature_sources
from trading_system.pipelines.feature_arguments import add_feature_source_arguments


def read(path):
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    add_feature_source_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.data.resolve():
        raise ValueError("Choose a new output path; existing data is never overwritten.")
    frame = read(args.data)
    frame, report = prepare_feature_sources(
        frame, fundamentals=args.fundamentals, sentiment=args.sentiment,
        disabled=args.no_external_features,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"saved={args.output} rows={len(frame)}")
    print(f"feature_sources={json.dumps(report, sort_keys=True)}")


if __name__ == "__main__":
    main()
