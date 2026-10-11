"""Compare detached states with bounded temporal state gradients in overnight."""

from dataclasses import asdict
import json
from pathlib import Path

import pyarrow.dataset as ds

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.trade_state_benchmark import benchmark_counts, candidate_grid
from trading_system.experiments.trade_state_gradient_benchmark import (
    TradeStateGradientBenchmarkConfig, run_trade_state_gradient_benchmark,
)
from trading_system.pipelines.compare_models import load_ticker_selection
from trading_system.pipelines.trade_state_benchmark import build_parser as state_parser


def build_parser():
    parser = state_parser()
    parser.description = __doc__
    parser.set_defaults(state_variants=("S0", "S3"),
                        output_dir=Path("artifacts/comparisons/us-trade-state-gradients"))
    parser.add_argument("--tbptt-windows", type=lambda s: tuple(int(v.strip()) for v in s.split(",") if v.strip()),
                        default=(10, 40), help="Global-session truncation windows, default 10,40.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    values = vars(args).copy()
    for key in ("data", "ticker_selection", "tickers", "output_dir", "resume", "dry_run", "no_plots", "no_progress"):
        values.pop(key)
    try:
        config = TradeStateGradientBenchmarkConfig(**values)
        tickers = args.tickers or load_ticker_selection(args.ticker_selection)
        if not tickers or len(set(tickers)) != len(tickers):
            raise ValueError("Nonempty unique ticker universe required.")
        if args.dry_run:
            print(json.dumps({"config": asdict(config), "ticker_count": len(tickers),
                "counts": benchmark_counts(config), "candidates": candidate_grid(config)}, indent=2))
            return 0
        frame = read_parquet_dataset(args.data, filter_expr=ds.field("ticker").isin(tickers))
        report = run_trade_state_gradient_benchmark(frame, tickers, args.output_dir, config=config,
            resume=args.resume, plots=not args.no_plots, progress=not args.no_progress)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"saved={args.output_dir.resolve()} fits={report['completed_fits']} "
          f"paths={report['completed_evaluation_paths']} holdout_opened=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
