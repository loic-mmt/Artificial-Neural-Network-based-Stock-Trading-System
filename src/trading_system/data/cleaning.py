from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .io import read_parquet_dataset


PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")
REQUIRED_COLUMNS = ("date", *PRICE_COLUMNS, "volume")
DuplicatePolicy = Literal["error", "first", "last"]


def _counts_by_ticker(frame: pd.DataFrame) -> Counter[str]:
    if "ticker" not in frame.columns or frame.empty:
        return Counter()
    return Counter(frame["ticker"].astype("string").fillna("<missing>").tolist())


def clean_ohlc_frame(
    frame: pd.DataFrame,
    *,
    relative_tolerance: float = 1e-8,
    duplicate_policy: DuplicatePolicy = "error",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate OHLCV rows and drop observations that cannot be trusted.

    ``adj_close`` is checked for finiteness and positivity but is deliberately not
    constrained to the raw daily high/low range. Adjustments for distributions and
    corporate actions commonly put it outside that range.
    """

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("relative_tolerance must be finite and non-negative.")
    if duplicate_policy not in {"error", "first", "last"}:
        raise ValueError("duplicate_policy must be one of: error, first, last.")

    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for column in (*PRICE_COLUMNS, "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")

    prices = work.loc[:, PRICE_COLUMNS].to_numpy(dtype=np.float64, na_value=np.nan)
    ohlc = work.loc[:, ("open", "high", "low", "close")].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    volume = work["volume"].to_numpy(dtype=np.float64, na_value=np.nan)

    prices_finite = np.isfinite(prices).all(axis=1)
    ohlc_finite = np.isfinite(ohlc).all(axis=1)
    scale = np.maximum(
        np.max(np.where(np.isfinite(ohlc), np.abs(ohlc), 0.0), axis=1),
        1.0,
    )
    tolerance = relative_tolerance * scale

    open_price, high, low, close = (ohlc[:, index] for index in range(4))
    rules: dict[str, np.ndarray] = {
        "invalid_date": work["date"].isna().to_numpy(),
        "non_finite_price": ~prices_finite,
        "non_positive_price": prices_finite & np.any(prices <= 0.0, axis=1),
        "non_finite_volume": ~np.isfinite(volume),
        "negative_volume": np.isfinite(volume) & (volume < 0.0),
        "low_above_high": ohlc_finite & (low > high + tolerance),
        "open_outside_range": ohlc_finite
        & ((open_price < low - tolerance) | (open_price > high + tolerance)),
        "close_outside_range": ohlc_finite
        & ((close < low - tolerance) | (close > high + tolerance)),
    }
    rejected = np.logical_or.reduce(tuple(rules.values()))

    primary_reason = np.full(len(work), "", dtype=object)
    for name, mask in rules.items():
        primary_reason[(primary_reason == "") & mask] = name

    invalid_rows = work.loc[rejected]
    cleaned = work.loc[~rejected].copy()
    dropped_by_ticker = _counts_by_ticker(invalid_rows)

    key_columns = ["date"]
    if "ticker" in cleaned.columns:
        key_columns.insert(0, "ticker")
    duplicate_mask = cleaned.duplicated(key_columns, keep=False)
    duplicate_rows_detected = int(duplicate_mask.sum())
    duplicate_rows_dropped = 0
    if duplicate_rows_detected:
        if duplicate_policy == "error":
            raise ValueError(
                f"Found {duplicate_rows_detected} rows with duplicate keys "
                f"{key_columns}; choose duplicate_policy='first' or 'last' explicitly."
            )
        keep = "first" if duplicate_policy == "first" else "last"
        drop_duplicates = cleaned.duplicated(key_columns, keep=keep)
        duplicate_rows_dropped = int(drop_duplicates.sum())
        dropped_by_ticker.update(_counts_by_ticker(cleaned.loc[drop_duplicates]))
        cleaned = cleaned.loc[~drop_duplicates].copy()

    sort_columns = ["date"]
    if "ticker" in cleaned.columns:
        sort_columns.insert(0, "ticker")
    cleaned = cleaned.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    date_range = {"start": None, "end": None}
    if not cleaned.empty:
        date_range = {
            "start": cleaned["date"].min().isoformat(),
            "end": cleaned["date"].max().isoformat(),
        }

    primary_counts = Counter(primary_reason[primary_reason != ""])
    report: dict[str, Any] = {
        "rows_input": int(len(frame)),
        "rows_output": int(len(cleaned)),
        "rows_dropped": int(len(frame) - len(cleaned)),
        "rows_rejected_by_validation": int(rejected.sum()),
        "rule_violation_counts": {
            name: int(mask.sum()) for name, mask in rules.items()
        },
        "primary_drop_reason_counts": {
            name: int(primary_counts.get(name, 0)) for name in rules
        },
        "duplicate_key_columns": key_columns,
        "duplicate_policy": duplicate_policy,
        "duplicate_rows_detected": duplicate_rows_detected,
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "dropped_rows_by_ticker": dict(sorted(dropped_by_ticker.items())),
        "date_range_output": date_range,
        "relative_tolerance": relative_tolerance,
    }
    return cleaned, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_ohlc_parquet(
    input_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    relative_tolerance: float = 1e-8,
    duplicate_policy: DuplicatePolicy = "error",
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Clean one Parquet file and atomically write data plus a JSON audit report."""

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    report_destination = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else destination.with_suffix(".quality.json")
    )
    if not source.is_file():
        raise FileNotFoundError(f"Parquet input file not found: {source}")
    if source == destination:
        raise ValueError("Input and output paths must differ.")
    if report_destination in {source, destination}:
        raise ValueError("The report path must differ from input and output paths.")
    if not dry_run and not overwrite:
        existing = [path for path in (destination, report_destination) if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing output: "
                + ", ".join(str(path) for path in existing)
            )

    frame = read_parquet_dataset(source)
    cleaned, report = clean_ohlc_frame(
        frame,
        relative_tolerance=relative_tolerance,
        duplicate_policy=duplicate_policy,
    )
    report.update(
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_path": str(source),
            "output_path": str(destination),
            "report_path": str(report_destination),
            "input_sha256": _sha256(source),
            "dry_run": dry_run,
        }
    )
    if dry_run:
        return report

    destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    output_tmp: Path | None = None
    report_tmp: Path | None = None
    try:
        output_fd, output_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(output_fd)
        output_tmp = Path(output_name)
        cleaned.to_parquet(output_tmp, index=False)
        report["output_sha256"] = _sha256(output_tmp)

        report_fd, report_name = tempfile.mkstemp(
            prefix=f".{report_destination.name}.",
            suffix=".tmp",
            dir=report_destination.parent,
        )
        os.close(report_fd)
        report_tmp = Path(report_name)
        report_tmp.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(output_tmp, destination)
        output_tmp = None
        os.replace(report_tmp, report_destination)
        report_tmp = None
    finally:
        for temporary in (output_tmp, report_tmp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    return report


__all__ = [
    "DuplicatePolicy",
    "PRICE_COLUMNS",
    "REQUIRED_COLUMNS",
    "clean_ohlc_frame",
    "clean_ohlc_parquet",
]
