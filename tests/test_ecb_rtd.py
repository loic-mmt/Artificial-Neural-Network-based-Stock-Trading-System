"""ECB RTD archive does not confuse database history with publication history."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from trading_system.data.ecb_rtd import parse_csv, save_vintages, series_url


KEY = "RTD.M.S0.N.P_C_OV.X"
CSV = ("KEY,TIME_PERIOD,OBS_VALUE,ACTION,VALID_FROM,VALID_TO,TITLE\n"
       f"{KEY},2020-01,104.2,Replace,2020-03-01T15:30:00+01:00,2020-07-01T15:30:00+02:00,HICP\n"
       f"{KEY},2020-01,104.5,Replace,2020-07-01T15:30:00+02:00,,HICP\n"
       f"{KEY},2010-01,,Delete,,2016-03-09T15:30:00+01:00,HICP\n").encode()


def test_ecb_rtd_versions_preserve_utc_and_have_no_publication_timestamp(tmp_path):
    frame = parse_csv(CSV, KEY, name="hicp", retrieved_at=datetime(2026, 9, 25, tzinfo=timezone.utc))
    assert len(frame) == 3
    assert "available_at_utc" not in frame
    assert frame.loc[0, "database_valid_from_utc"] == pd.Timestamp("2020-03-01 14:30:00Z")
    assert frame.loc[2, "action"] == "Delete"
    output = tmp_path / "rtd.parquet"
    report = save_vintages({KEY: CSV}, output, names={KEY: "hicp"})
    assert report["point_in_time_ready"] is False
    assert len(pd.read_parquet(output)) == 3
    with pytest.raises(FileExistsError):
        save_vintages({KEY: CSV}, output, names={KEY: "hicp"})


def test_ecb_rtd_refuses_mismatched_series_and_invalid_key():
    with pytest.raises(ValueError, match="does not match"):
        parse_csv(CSV, "RTD.Q.S0.S.G_GDPM_TO_C.E", name="gdp")
    with pytest.raises(ValueError, match="Invalid"):
        series_url("https://example.com")
