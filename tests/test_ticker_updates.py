from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from trading_system.data.ticker_updates import latest_complete_date, next_weekday_run, ticker_universe, update_prices


class FakeYahoo:
    def __init__(self) -> None:
        self.fail: str | None = None
        self.revision = 0

    def download(self, *, tickers, start, end, **kwargs):
        if tickers == self.fail:
            raise OSError("provider unavailable")
        dates = pd.date_range(max(pd.Timestamp(start), pd.Timestamp("2026-04-20")),
                              min(pd.Timestamp(end) - pd.Timedelta(days=1), pd.Timestamp("2026-04-23")), freq="D")
        close = [100.0 + day.day - 20 + (self.revision if day.day == 22 else 0) for day in dates]
        return pd.DataFrame({
            "Open": close, "High": [value + 1 for value in close],
            "Low": [value - 1 for value in close], "Close": close,
            "Adj Close": close, "Volume": [1000.0] * len(dates),
            "Dividends": [0.0] * len(dates), "Stock Splits": [0.0] * len(dates),
        }, index=pd.Index(dates, name="Date"))


def test_incremental_download_replaces_overlap_and_preserves_file_on_failure(tmp_path: Path) -> None:
    provider = FakeYahoo()
    path = tmp_path / "prices.parquet"
    first = update_prices(path, start=date(2026, 4, 20), end=date(2026, 4, 22), provider=provider, retry_delay=0)
    assert first["tickers"] == 10 and first["new_rows"] == 30
    provider.revision = 2
    second = update_prices(path, start=date(2026, 4, 20), end=date(2026, 4, 23), provider=provider, retry_delay=0)
    assert second["new_rows"] == 10
    frame = pd.read_parquet(path)
    assert len(frame) == 40 and set(frame.ticker) == set(ticker_universe())
    assert frame.loc[frame.date.eq(pd.Timestamp("2026-04-22")), "close"].eq(104.0).all()
    original = path.read_bytes()
    provider.fail = ticker_universe()[2]
    with pytest.raises(RuntimeError, match="Could not download"):
        update_prices(path, start=date(2026, 4, 20), end=date(2026, 4, 23), provider=provider, retry_delay=0)
    assert path.read_bytes() == original


def test_ticker_scope_and_complete_market_day() -> None:
    paris = ZoneInfo("Europe/Paris")
    assert latest_complete_date(datetime(2026, 9, 24, 18, 0, tzinfo=paris)) == date(2026, 9, 23)
    assert latest_complete_date(datetime(2026, 9, 24, 19, 30, tzinfo=paris)) == date(2026, 9, 24)
    assert latest_complete_date(datetime(2026, 9, 27, 20, 0, tzinfo=paris)) == date(2026, 9, 25)
    assert next_weekday_run(datetime(2026, 9, 25, 19, 1, tzinfo=paris)).date() == date(2026, 9, 28)
    assert next_weekday_run(datetime(2026, 10, 23, 19, 1, tzinfo=paris)).utcoffset().total_seconds() == 3600
