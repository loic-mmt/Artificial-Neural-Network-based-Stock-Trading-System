"""Run fixed OU policy comparisons on frozen validation folds and seeds."""

import _bootstrap  # noqa: F401
from trading_system.trading.ou_validation import main

if __name__ == "__main__":
    main()
