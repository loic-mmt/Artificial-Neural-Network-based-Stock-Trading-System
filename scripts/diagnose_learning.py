"""Inspect existing learning artifacts without training or checkpoint loading."""

import _bootstrap  # noqa: F401
from trading_system.pipelines.diagnose_learning import main


if __name__ == "__main__":
    raise SystemExit(main())
