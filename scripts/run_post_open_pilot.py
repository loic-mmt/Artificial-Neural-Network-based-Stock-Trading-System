"""Compare frozen-protocol GRU Sharpe with and without today's opening gap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from trading_system.experiments.post_open_pilot import run_pilot
from trading_system.pipelines.compare_models import load_ticker_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/cac40_daily_clean.parquet"))
    parser.add_argument("--ticker-selection", type=Path,
                        default=Path("configs/benchmark/cac40_diversified_10.json"))
    parser.add_argument("--reference-cv", type=Path,
                        default=Path("artifacts/comparisons/gru-optim/04-causal-normalization-60-atr"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/comparisons/gru-optim/11-post-open-pilot"))
    parser.add_argument("--seeds", default="1,7,19")
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--draws", type=int, default=250)
    parser.add_argument("--max-epochs", type=int,
                        help="Smoke-test only; omit for the original 100-epoch training budget.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(","))
    folds = tuple(int(x) for x in args.folds.split(","))
    print(json.dumps(run_pilot(args.data, load_ticker_selection(args.ticker_selection),
                               args.reference_cv, args.output_dir, seeds=seeds,
                               folds=folds, draws=args.draws, device=args.device,
                               resume=args.resume, max_epochs=args.max_epochs), indent=2))


if __name__ == "__main__":
    main()
