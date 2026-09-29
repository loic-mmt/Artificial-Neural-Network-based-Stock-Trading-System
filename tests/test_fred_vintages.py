"""ALFRED archive keeps revisions and never publishes an intraday timestamp."""

import pandas as pd
import pytest

from trading_system.data import fred_vintages as fred


def test_fetch_series_paginates_and_preserves_revisions(monkeypatch, tmp_path):
    calls = []

    def page(key, series_id, **kwargs):
        offset = kwargs["offset"]
        calls.append(offset)
        return {"count": 2, "observations": [
            {"date": "2020-01-01", "value": ["1.0", "1.2"][offset],
             "realtime_start": ["2020-02-01", "2020-03-01"][offset],
             "realtime_end": ["2020-02-29", "9999-12-31"][offset]}]}

    monkeypatch.setattr(fred, "_page", page)
    frame = fred.fetch_series("fake-key", "CPIAUCSL", limit=1,
                              realtime_start="2020-01-01", realtime_end="2020-01-02")
    assert calls == [0, 1]
    assert frame["value"].tolist() == [1.0, 1.2]
    assert frame.loc[1, "realtime_end"] == "9999-12-31"
    assert "available_at_utc" not in frame
    output = tmp_path / "fred.parquet"
    report = fred.save_vintages([frame], output)
    assert report["point_in_time_ready"] is False
    assert len(pd.read_parquet(output)) == 2
    with pytest.raises(FileExistsError):
        fred.save_vintages([frame], output)


def test_fetch_series_rejects_truncated_page(monkeypatch):
    monkeypatch.setattr(fred, "_page", lambda *args, **kwargs: {"count": 2, "observations": []})
    with pytest.raises(ValueError, match="Incomplete"):
        fred.fetch_series("fake-key", "UNRATE", realtime_start="2020-01-01",
                          realtime_end="2020-01-02")


def test_merges_query_window_boundaries_without_inventing_releases():
    frame = pd.DataFrame({
        "observation_date": pd.to_datetime(["2020-01-01"] * 3),
        "value": [4.0, 4.0, 4.1],
        "realtime_start": pd.to_datetime(["2020-02-01", "2024-01-01", "2024-02-01"]),
        "realtime_end": ["2023-12-31", "2024-01-31", "9999-12-31"],
    })
    merged = fred._merge_window_fragments(frame)
    assert len(merged) == 2
    assert merged.iloc[0]["realtime_start"] == pd.Timestamp("2020-02-01")
    assert merged.iloc[0]["realtime_end"] == "2024-01-31"
