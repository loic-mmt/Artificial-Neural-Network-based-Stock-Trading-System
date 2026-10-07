"""Prepare frozen FNSPID titles or run the separate exploratory news study."""

import _bootstrap

from trading_system.pipelines.fnspid_news_study import main


if __name__ == "__main__":
    main()
