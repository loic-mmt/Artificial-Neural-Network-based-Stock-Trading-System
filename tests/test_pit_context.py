from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from trading_system.data import pit_context as pit


class FakeInfo:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.fast_info = {"market_cap": 1_000_000_000, "shares": 1_000_000}

    def get_info(self) -> dict:
        return {"sector": "Industrials", "industry": "Machinery", "bookValue": 12.0,
                "trailingEps": 3.0, "marketCap": 1_000_000_000}


class FakeProvider:
    Ticker = FakeInfo


def test_macro_and_fundamentals_are_available_only_after_collection(monkeypatch) -> None:
    old_date = pd.Timestamp("2025-01-02")
    yahoo = pd.DataFrame({"date": [old_date], **{name: [float(index + 1)] for index, name in enumerate(pit.YAHOO_MACRO_TICKERS.values())}})
    rates = pd.DataFrame({"date": [old_date], "ust2y": [2.0], "ust10y": [3.0], "frt2y": [2.1], "frt10y": [3.1]})
    monkeypatch.setattr(pit, "download_yahoo_macro_series", lambda *args: yahoo)
    monkeypatch.setattr(pit, "download_rate_macro_series", lambda *args: rates)
    macro = pit.fetch_macro(provider=FakeProvider(), now=datetime(2026, 9, 24, tzinfo=timezone.utc), lookback_days=700)
    fundamentals = pit.fetch_fundamentals(provider=FakeProvider())
    assert set(macro.metric) == set(pit.MACRO_METRICS)
    assert set(fundamentals.ticker) == set(pit.ticker_universe())
    assert all(macro.observation_date == "2025-01-02")
    before = macro.available_at_utc.iloc[0] - timedelta(seconds=1)
    assert pit.asof_snapshot(macro, before, key="metric").empty
    assert pit.asof_snapshot(fundamentals, before, key="ticker").empty
    assert len(pit.asof_snapshot(macro, macro.available_at_utc.iloc[0], key="metric")) == len(pit.MACRO_METRICS)


def test_revisions_do_not_change_past_asof_results(tmp_path: Path) -> None:
    early = pd.Timestamp("2026-09-24T12:00:00Z")
    late = pd.Timestamp("2026-09-25T12:00:00Z")
    first = pd.DataFrame([{"metric": "ust10y", "value": 3.0, "source": "us_treasury",
                           "observation_date": "2026-09-23", "collected_at_utc": early, "available_at_utc": early}], columns=pit.MACRO_COLUMNS)
    revised = first.copy()
    revised.loc[0, "value"] = 3.2
    revised.loc[0, "collected_at_utc"] = late
    revised.loc[0, "available_at_utc"] = late
    path = tmp_path / "macro.parquet"
    first.to_parquet(path, index=False)
    combined = pit._append(path, revised, pit.MACRO_COLUMNS)
    assert pit.asof_snapshot(combined, early, key="metric").iloc[0]["value"] == 3.0
    assert pit.asof_snapshot(combined, late, key="metric").iloc[0]["value"] == 3.2
    assert pit.asof_snapshot(combined, early - timedelta(seconds=1), key="metric").empty


def test_failed_company_fetch_leaves_both_files_unchanged(tmp_path: Path, monkeypatch) -> None:
    macro_path = tmp_path / "macro.parquet"
    company_path = tmp_path / "company.parquet"
    macro_path.write_bytes(b"previous macro")
    company_path.write_bytes(b"previous company")
    monkeypatch.setattr(pit, "fetch_macro", lambda **kwargs: pd.DataFrame())

    def fail(**kwargs):
        raise OSError("company source offline")

    monkeypatch.setattr(pit, "fetch_fundamentals", fail)
    with pytest.raises(OSError, match="offline"):
        pit.collect_context(macro_path=macro_path, fundamental_path=company_path)
    assert macro_path.read_bytes() == b"previous macro"
    assert company_path.read_bytes() == b"previous company"
