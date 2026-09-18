import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from trading_system.data.cleaning import clean_ohlc_frame, clean_ohlc_parquet


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02"],
            "ticker": ["AAA", "AAA"],
            "open": [10.5, 10.0],
            "high": [11.0, 10.5],
            "low": [10.0, 9.5],
            "close": [10.8, 10.2],
            # Adjusted prices need not lie inside the raw OHLC range.
            "adj_close": [5.4, 5.1],
            "volume": [100, 200],
        }
    )


def test_clean_ohlc_frame_preserves_valid_rows_and_sorts():
    cleaned, report = clean_ohlc_frame(_valid_frame())

    assert cleaned["date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert cleaned["adj_close"].tolist() == [5.1, 5.4]
    assert report["rows_input"] == 2
    assert report["rows_output"] == 2
    assert report["rows_dropped"] == 0


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("date", "not-a-date", "invalid_date"),
        ("open", np.nan, "non_finite_price"),
        ("adj_close", np.inf, "non_finite_price"),
        ("close", 0.0, "non_positive_price"),
        ("volume", np.nan, "non_finite_volume"),
        ("volume", -1, "negative_volume"),
        ("low", 12.0, "low_above_high"),
        ("open", 12.0, "open_outside_range"),
        ("close", 12.0, "close_outside_range"),
    ],
)
def test_clean_ohlc_frame_drops_invalid_rows(column, value, reason):
    frame = _valid_frame().iloc[[0]].copy()
    frame.loc[frame.index[0], column] = value

    cleaned, report = clean_ohlc_frame(frame)

    assert cleaned.empty
    assert report["rows_dropped"] == 1
    assert report["rule_violation_counts"][reason] == 1


def test_clean_ohlc_frame_applies_relative_tolerance():
    frame = _valid_frame().iloc[[0]].copy()
    frame.loc[frame.index[0], "close"] = 11.0 + 5e-8

    accepted, _ = clean_ohlc_frame(frame, relative_tolerance=1e-8)
    rejected, _ = clean_ohlc_frame(frame, relative_tolerance=1e-10)

    assert len(accepted) == 1
    assert rejected.empty


def test_clean_ohlc_frame_requires_explicit_duplicate_policy():
    frame = pd.concat([_valid_frame().iloc[[0]]] * 2, ignore_index=True)

    with pytest.raises(ValueError, match="duplicate keys"):
        clean_ohlc_frame(frame)

    cleaned, report = clean_ohlc_frame(frame, duplicate_policy="last")
    assert len(cleaned) == 1
    assert report["duplicate_rows_detected"] == 2
    assert report["duplicate_rows_dropped"] == 1


def test_clean_ohlc_parquet_writes_data_and_audit_report(tmp_path):
    source = tmp_path / "source.parquet"
    destination = tmp_path / "clean.parquet"
    frame = _valid_frame()
    frame.loc[frame.index[0], "close"] = 12.0
    frame.to_parquet(source, index=False)

    report = clean_ohlc_parquet(source, destination)
    report_path = tmp_path / "clean.quality.json"

    assert len(pd.read_parquet(destination)) == 1
    assert report_path.exists()
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["rows_dropped"] == 1
    assert saved_report["output_sha256"] == hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()
    assert report == saved_report

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        clean_ohlc_parquet(source, destination)


def test_clean_ohlc_parquet_dry_run_writes_nothing(tmp_path):
    source = tmp_path / "source.parquet"
    destination = tmp_path / "clean.parquet"
    _valid_frame().to_parquet(source, index=False)

    report = clean_ohlc_parquet(source, destination, dry_run=True)

    assert report["dry_run"] is True
    assert not destination.exists()
    assert not destination.with_suffix(".quality.json").exists()


def test_clean_ohlc_parquet_rejects_in_place_cleaning(tmp_path):
    source = tmp_path / "source.parquet"
    _valid_frame().to_parquet(source, index=False)

    with pytest.raises(ValueError, match="must differ"):
        clean_ohlc_parquet(source, source)
