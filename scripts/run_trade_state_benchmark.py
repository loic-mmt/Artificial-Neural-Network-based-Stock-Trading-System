"""Entry point for the S0-S3 closed-loop overnight benchmark."""

import _bootstrap  # noqa: F401
from trading_system.pipelines.trade_state_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
