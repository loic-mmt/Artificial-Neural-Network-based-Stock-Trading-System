import pandas as pd
import pytest

from trading_system.data.us_market_context import (
    complete_ticker_selection, validate_us_market_context,
)


def test_complete_ticker_selection_excludes_late_listings():
    dates = pd.date_range("2020-01-01", periods=4, freq="B", tz="UTC")
    frame = pd.DataFrame([
        {"date": day, "ticker": ticker}
        for ticker, ticker_dates in (("A", dates), ("B", dates[1:]))
        for day in ticker_dates
    ])
    assert complete_ticker_selection(frame, start=dates[0]) == ("A",)
    assert complete_ticker_selection(frame, start=dates[1]) == ("A", "B")


def test_market_context_requires_causal_positive_closes():
    dates = pd.date_range("2020-01-01", periods=2, tz="UTC")
    frame = pd.DataFrame({
        "date": dates,
        "spy_close": [100.0, 101.0],
        "source_end": dates + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1),
    })
    validated = validate_us_market_context(frame, ticker_columns={"SPY": "spy_close"})
    assert list(validated["spy_close"]) == [100.0, 101.0]
    future = frame.copy()
    future.loc[0, "source_end"] = dates[0] + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="future close"):
        validate_us_market_context(future, ticker_columns={"SPY": "spy_close"})
