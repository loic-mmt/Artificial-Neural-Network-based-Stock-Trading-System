"""Compare frozen benchmark positions at identical gross exposure; no training."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from trading_system.experiments.exposure_comparison import compare_replay_exposure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--market-reconstruction-audit", type=Path,
                        help="Explicit derived-market-only audit; raw prices/context must still match exactly.")
    args = parser.parse_args(argv)
    return compare_replay_exposure(args.replay_dir, args.run_dir, args.output_dir,
                                   market_reconstruction_audit=args.market_reconstruction_audit)


if __name__ == "__main__":
    main()
