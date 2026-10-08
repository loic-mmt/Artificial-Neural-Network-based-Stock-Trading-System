"""CLI for the frozen post-open GRU labels, financial loss and hybrid benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pyarrow.dataset as ds

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.label_loss_benchmark import (
    DECODERS, FAMILIES, METHODS, PROTOCOLS, LabelLossBenchmarkConfig,
    benchmark_counts, candidate_grid, run_label_loss_benchmark,
)
from trading_system.pipelines.compare_models import load_ticker_selection


def _names(value):
    return tuple(v.strip().replace("-", "_") for v in value.split(",") if v.strip())


def _integers(value):
    try:
        return tuple(int(v.strip()) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/mt5_stocks_us_daily_clean.parquet"))
    universe = parser.add_mutually_exclusive_group()
    universe.add_argument("--ticker-selection", type=Path,
        default=Path("configs/benchmark/stocks_us_gnn_complete_2005.json"))
    universe.add_argument("--tickers", nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/comparisons/us-label-loss-post-open"))
    parser.add_argument("--label-methods", type=_names, default=METHODS)
    parser.add_argument("--families", type=_names, default=FAMILIES)
    parser.add_argument("--protocols", type=_names, default=PROTOCOLS)
    parser.add_argument("--decoders", type=_names, default=DECODERS)
    parser.add_argument("--seeds", type=_integers, default=(1, 7, 19))
    parser.add_argument("--folds", type=_integers)
    parser.add_argument("--cv-folds", dest="n_folds", type=int, default=3)
    parser.add_argument("--cv-gap-bars", dest="gap_bars", type=int, default=5)
    parser.add_argument("--holdout-start", default="2023-06-22")
    parser.add_argument("--start", default="2005-01-03")
    parser.add_argument("--end", help="Optional development-only cap for a pilot; holdout remains closed.")
    parser.add_argument("--initial-train-fraction", type=float, default=.5)
    parser.add_argument("--inner-val-fraction", type=float, default=.2)
    parser.add_argument("--context-len", type=int, default=60)
    parser.add_argument("--max-features", type=int, default=32)
    parser.add_argument("--feature-groups", type=_names, default=("technical", "market", "sector"))
    parser.add_argument("--label-horizon", type=int, default=10)
    parser.add_argument("--forward-threshold", type=float, default=.002)
    parser.add_argument("--volatility-window", type=int, default=20)
    parser.add_argument("--long-threshold", type=float, default=1.)
    parser.add_argument("--short-threshold", type=float, default=1.5)
    parser.add_argument("--exit-threshold", type=float, default=.25)
    parser.add_argument("--min-holding-period", type=int, default=5)
    parser.add_argument("--epochs", dest="base_epochs", type=int, default=100, help="Base budget before multiplier.")
    parser.add_argument("--epoch-multiplier", type=int, default=3, help="Default: 300 max epochs for every family.")
    parser.add_argument("--batch-size", type=int, default=256, help="Memory block size, not optimizer update frequency.")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--hybrid-ce-weight", type=float, default=.5)
    parser.add_argument("--cost-bps", type=float, default=5.)
    parser.add_argument("--combined-pnl-weight", type=float, default=.25)
    parser.add_argument("--combined-pnl-scale", type=float, default=1e-4)
    parser.add_argument("--annualization", type=int, default=252)
    parser.add_argument("--sharpe-epsilon", type=float, default=1e-4)
    parser.add_argument("--initial-capital", type=float, default=10000.)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print grid without reading prices or training.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    values = vars(args).copy()
    for key in ("data", "ticker_selection", "tickers", "output_dir", "resume", "dry_run", "no_plots", "no_progress"):
        values.pop(key)
    try:
        config = LabelLossBenchmarkConfig(**values)
        tickers = args.tickers or load_ticker_selection(args.ticker_selection)
        if len(set(tickers)) != len(tickers):
            raise ValueError("Duplicate tickers.")
        if args.dry_run:
            print(json.dumps({"config": asdict(config), "ticker_count": len(tickers),
                "counts": benchmark_counts(config), "candidates": candidate_grid(config)}, indent=2))
            return 0
        frame = read_parquet_dataset(args.data, filter_expr=ds.field("ticker").isin(tickers))
        report = run_label_loss_benchmark(frame, tickers, args.output_dir, config=config,
            resume=args.resume, plots=not args.no_plots, progress=not args.no_progress)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"saved={args.output_dir.resolve()} fits={report['completed_fits']} paths={report['completed_evaluation_paths']} holdout_opened=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
