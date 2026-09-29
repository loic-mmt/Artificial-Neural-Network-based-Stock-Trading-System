"""Replay the frozen GRU Sharpe N0 folds of benchmark 04 without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from trading_system.experiments.replay_position_cv import replay_reference
from trading_system.pipelines.compare_models import load_ticker_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path,
                        default=Path("artifacts/comparisons/gru-optim/04-causal-normalization-60-atr"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/cac40_daily_clean.parquet"))
    parser.add_argument("--ticker-selection", type=Path,
                        default=Path("configs/benchmark/cac40_diversified_10.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/comparisons/gru-optim/11-execution-replay/sharpe-n0-positions.parquet"))
    parser.add_argument("--fold", type=int, action="append", choices=(0, 1, 2),
                        help="Optional smoke test: select one or more outer folds.")
    args = parser.parse_args()
    report = replay_reference(args.cv_dir, args.data, load_ticker_selection(args.ticker_selection), args.output,
                              folds=tuple(args.fold) if args.fold else None)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
