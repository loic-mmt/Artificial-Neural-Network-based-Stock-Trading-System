import pandas as pd

from trading_system.data import download


def test_reference_area_is_forwarded_only_to_oecd(monkeypatch):
    calls = {}

    def treasury(**kwargs):
        calls["treasury"] = kwargs
        return pd.DataFrame(
            {"date": [pd.Timestamp("2025-01-01")], "ust2y": [4.0], "ust10y": [4.5]}
        )

    def oecd(**kwargs):
        calls["oecd"] = kwargs
        return pd.DataFrame(
            {"date": [pd.Timestamp("2025-01-01")], "frt2y": [4.1], "frt10y": [4.6]}
        )

    monkeypatch.setattr(download, "download_treasury_yield_series", treasury)
    monkeypatch.setattr(download, "download_oecd_rate_series", oecd)
    result = download.download_rate_macro_series(
        pd, "2025-01-01", "2025-01-31", 30, 2, reference_area="USA"
    )

    assert "reference_area" not in calls["treasury"]
    assert calls["oecd"]["reference_area"] == "USA"
    assert set(result) == {"date", "ust2y", "ust10y", "frt2y", "frt10y"}
