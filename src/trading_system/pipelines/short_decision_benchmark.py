"""CLI for independent short architectures and explicitly projected policies."""

from dataclasses import asdict
import json
from pathlib import Path

import pyarrow.dataset as ds

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.short_decision_benchmark import ShortDecisionBenchmarkConfig, benchmark_counts, run_short_decision_benchmark
from trading_system.pipelines.compare_models import load_ticker_selection
from trading_system.pipelines.label_loss_benchmark import _names
from trading_system.pipelines.trade_state_benchmark import build_parser as state_parser


def build_parser():
    parser = state_parser(); parser.description = __doc__
    parser.set_defaults(state_variants=("S0", "S3"), output_dir=Path("artifacts/comparisons/us-short-decisions"))
    parser.add_argument("--architectures", type=_names, default=("short_detector", "dual_head", "separate_grus"))
    parser.add_argument("--policies", type=_names, default=("long_short", "long_flat", "short_flat"))
    parser.add_argument("--no-controls", dest="include_controls", action="store_false")
    parser.add_argument("--long-activation-threshold", type=float, default=.5)
    parser.add_argument("--short-activation-threshold", type=float, default=.5)
    parser.add_argument("--conflict-policy", choices=("flat", "margin"), default="flat")
    parser.add_argument("--regime-window", type=int, default=200, help="Lagged per-asset SMA diagnostic; 0 disables, never filters trading.")
    return parser


def main(argv=None):
    parser = build_parser(); args = parser.parse_args(argv); values = vars(args).copy()
    for key in ("data", "ticker_selection", "tickers", "output_dir", "resume", "dry_run", "no_plots", "no_progress"): values.pop(key)
    try:
        config = ShortDecisionBenchmarkConfig(**values)
        tickers = args.tickers or load_ticker_selection(args.ticker_selection)
        if not tickers or len(set(tickers)) != len(tickers): raise ValueError("Explicit unique ticker universe required.")
        if args.dry_run:
            print(json.dumps({"config": asdict(config), "ticker_count": len(tickers), "counts": benchmark_counts(config), "candidates": config.candidates()}, indent=2)); return 0
        frame = read_parquet_dataset(args.data, filter_expr=ds.field("ticker").isin(tickers))
        report = run_short_decision_benchmark(frame, tickers, args.output_dir, config=config,
            resume=args.resume, plots=not args.no_plots, progress=not args.no_progress)
    except (ValueError, FileExistsError) as exc: parser.error(str(exc))
    print(f"saved={args.output_dir.resolve()} fits={report['completed_fits']} paths={report['completed_evaluation_paths']} holdout_opened=False")
    return 0
