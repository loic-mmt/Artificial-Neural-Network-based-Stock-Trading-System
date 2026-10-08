"""CPU-only diagnostics of signed graph/news runs, without retraining."""

import argparse
from pathlib import Path

from trading_system.analysis.learning_diagnostics import diagnose_learning


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="One run or a study parent directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New report directory outside source runs.")
    parser.add_argument("--partitions", default="inner", help="inner, outer or inner,outer; final test is forbidden.")
    parser.add_argument("--candidates", help="Optional comma-separated exact candidate names.")
    parser.add_argument("--folds", help="Optional comma-separated fold IDs.")
    parser.add_argument("--seeds", help="Optional comma-separated training seeds.")
    parser.add_argument("--flat-tolerance", type=float, default=1e-6, help="Descriptive only; does not change positions.")
    parser.add_argument("--saturation-threshold", type=float, default=.95, help="Descriptive only; no threshold fitting.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    names = lambda value: tuple(item.strip() for item in value.split(",") if item.strip()) if value is not None else None
    numbers = lambda value: tuple(int(item) for item in names(value)) if value is not None else None
    result = diagnose_learning(args.run_dir, args.output_dir, partitions=names(args.partitions),
                               candidates=names(args.candidates), folds=numbers(args.folds), seeds=numbers(args.seeds),
                               flat_tolerance=args.flat_tolerance, saturation_threshold=args.saturation_threshold)
    print(f"learning_diagnostic_saved={args.output_dir.resolve()} tasks={result['task_partitions']} "
          f"traces={result['trace_tasks']} complete={result['complete']} final_holdout_opened=False")
    return 0 if result["complete"] else 1


__all__ = ["build_parser", "main"]
