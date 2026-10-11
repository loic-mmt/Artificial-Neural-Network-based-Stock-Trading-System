"""Entry point for the matched detached/TBPTT overnight benchmark."""

import _bootstrap  # noqa: F401
from trading_system.pipelines.trade_state_gradient_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
