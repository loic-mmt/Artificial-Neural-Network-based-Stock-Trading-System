from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_system.publishing.gru import _predict_latest_rows, backtest_days, read_market_data, read_universe


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


def test_daily_publisher_infers_one_latest_window_per_ticker(monkeypatch) -> None:
    monkeypatch.setattr("trading_system.publishing.gru._features", lambda frame, columns, fill: frame)
    frame = pd.DataFrame({
        "ticker": ["AAPL"] * 3 + ["MSFT"] * 3,
        "date": list(pd.date_range("2026-01-05", periods=3)) * 2,
        "company": ["Apple"] * 3 + ["Microsoft"] * 3,
        "adj_close": [10, 11, 12, 20, 21, 22],
        "foo": [1, 2, 3, 4, 5, 6],
    })
    class Scaler:
        def transform(self, windows):
            return windows
    class Model:
        def __init__(self):
            self.shapes = []
        def predict_proba(self, windows):
            self.shapes.append(windows.shape)
            return np.array([[0.1, 0.3, 0.6]])
    model = Model()
    latest = _predict_latest_rows(frame, model, Scaler(), ("foo",), 2, {})
    assert model.shapes == [(1, 2, 1), (1, 2, 1)]
    assert set(latest["ticker"]) == {"AAPL", "MSFT"}
    assert set(pd.to_datetime(latest["date"]).dt.date.astype(str)) == {"2026-01-07"}
    assert set(latest["label_id"]) == {2}
