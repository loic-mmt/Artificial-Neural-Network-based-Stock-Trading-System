"""Entry point for the post-open overnight short specialization benchmark."""

import _bootstrap  # noqa: F401
from trading_system.pipelines.short_decision_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
