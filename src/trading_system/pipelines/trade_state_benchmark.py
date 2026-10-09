"""Launch the isolated S0-S3 overnight trade-state GRU benchmark."""

from dataclasses import asdict
import json
from pathlib import Path

import pyarrow.dataset as ds

from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.trade_state_benchmark import (
    TradeStateBenchmarkConfig, benchmark_counts, candidate_grid, run_trade_state_benchmark,
)
from trading_system.pipelines.compare_models import load_ticker_selection
from trading_system.pipelines.label_loss_benchmark import build_parser as base_parser


def build_parser():
    parser = base_parser()
    parser.description = __doc__
    parser.set_defaults(label_methods=("volatility_position",), families=("cross_entropy", "financial"),
        protocols=("overnight",), decoders=("continuous", "sign"),
        output_dir=Path("artifacts/comparisons/us-trade-state-overnight"))
    parser.add_argument("--state-variants", type=lambda s: tuple(v.strip().upper() for v in s.split(",") if v.strip()),
                        default=("S0", "S1", "S2", "S3"))
    parser.add_argument("--state-return-scale", type=float, default=.1)
    parser.add_argument("--state-age-scale", type=float, default=252.)
    parser.add_argument("--state-gradient-mode", choices=("detached",), default="detached")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    values = vars(args).copy()
    for key in ("data", "ticker_selection", "tickers", "output_dir", "resume", "dry_run", "no_plots", "no_progress"):
        values.pop(key)
    try:
        config = TradeStateBenchmarkConfig(**values)
        tickers = args.tickers or load_ticker_selection(args.ticker_selection)
        if not tickers or len(set(tickers)) != len(tickers):
            raise ValueError("Nonempty unique ticker universe required.")
        if args.dry_run:
            print(json.dumps({"config": asdict(config), "ticker_count": len(tickers),
                "counts": benchmark_counts(config), "candidates": candidate_grid(config)}, indent=2))
            return 0
        frame = read_parquet_dataset(args.data, filter_expr=ds.field("ticker").isin(tickers))
        report = run_trade_state_benchmark(frame, tickers, args.output_dir, config=config,
            resume=args.resume, plots=not args.no_plots, progress=not args.no_progress)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"saved={args.output_dir.resolve()} fits={report['completed_fits']} paths={report['completed_evaluation_paths']} holdout_opened=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
