from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_system.publishing.gru import backtest_days, read_market_data, read_universe


def test_data_is_limited_to_configured_universe() -> None:
    tickers = read_universe()
    frame = read_market_data(universe=tickers)
    assert set(frame["ticker"]) == set(tickers)
    assert len(tickers) == 10
    assert not frame.duplicated(["ticker", "date"]).any()


def test_backtest_waits_one_bar_and_charges_turnover() -> None:
    frame = pd.DataFrame({
        "ticker": ["BNP.PA"] * 4,
        "date": pd.date_range("2026-01-01", periods=4),
        "adj_close": [100.0, 110.0, 121.0, 133.1],
        "label_id": np.array([2, 2, 2, 2]),
    })
    days = backtest_days(frame, {"BNP.PA": "2026-01-01"}, cost_bps=5)
    assert [row["date"] for row in days] == ["2026-01-02", "2026-01-03", "2026-01-04"]
    assert days[0]["return_pct"] == 0.0
    assert days[1]["return_pct"] == pytest.approx(9.95)
    assert days[2]["return_pct"] == pytest.approx(10.0)
