"""Explicit, offline loading of optional historical feature sources."""

from hashlib import sha256
from pathlib import Path

import pandas as pd

from .feature_sources import (
    FUNDAMENTAL_FIELDS, SENTIMENT_FEATURES, attach_fundamentals, attach_sentiment,
)


def read_feature_source(path):
    """Read an explicitly requested source; missing/malformed files are errors."""
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
    else:
        raise ValueError("Historical feature sources must be CSV or parquet files.")
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return table, {
        "path": str(path), "sha256": digest.hexdigest(), "rows": len(table),
    }


def prepare_feature_sources(frame, *, fundamentals=None, sentiment=None,
                            disabled=False, group_col="ticker", date_col="date"):
    """Use explicit files or embedded columns; absent sources need no fallback data.

    The returned report describes input availability, not training eligibility.
    Feature selection still uses only the training partition. No network access,
    automatic file discovery, or fabricated values are involved.
    """
    if disabled and (fundamentals is not None or sentiment is not None):
        raise ValueError("--no-external-features cannot be combined with source paths.")
    work = frame.copy()
    inherited = frame.attrs.get("feature_sources", {})
    report = {"mode": "disabled" if disabled else "optional"}
    fundamental_columns = [f"fund_{field}" for field in FUNDAMENTAL_FIELDS]
    if disabled:
        work = work.drop(columns=[*fundamental_columns, "fund_available_at", *SENTIMENT_FEATURES], errors="ignore")
    for name, path, attach, columns in (
        ("fundamentals", fundamentals, attach_fundamentals, fundamental_columns),
        ("sentiment", sentiment, attach_sentiment, list(SENTIMENT_FEATURES)),
    ):
        details = {}
        if path is None and not disabled:
            previous = inherited.get(name, {})
            details = {key: previous[key] for key in (
                "path", "sha256", "rows", "publication_start", "publication_end"
            ) if key in previous}
        if path is not None:
            source, details = read_feature_source(path)
            work = attach(work, source, group_col=group_col, date_col=date_col)
            published = pd.to_datetime(source.available_at, utc=True, errors="raise")
            details.update(publication_start=published.min().isoformat(),
                           publication_end=published.max().isoformat())
        present = [column for column in columns if column in work]
        usable = pd.Series(False, index=work.index)
        if present:
            numeric = work[present].apply(pd.to_numeric, errors="coerce")
            usable = numeric.replace([float("inf"), -float("inf")], float("nan")).notna().any(axis=1)
        if name == "fundamentals":
            available = pd.to_datetime(work.get("fund_available_at", pd.Series(pd.NaT, index=work.index)), utc=True, errors="coerce")
            cutoff = pd.to_datetime(work[date_col], utc=True, errors="raise")
            age = (cutoff - available).dt.total_seconds() / 86400
            usable &= (available < cutoff) & age.between(0, 550)
        status = "disabled" if disabled else "file" if path is not None else "embedded" if present else "absent"
        report[name] = {
            "status": status, **details,
            "usable_rows": int(usable.sum()), "total_rows": len(work),
            "coverage_by_ticker": {
                str(ticker): float(value)
                for ticker, value in work.assign(_source_usable=usable).groupby(
                    group_col, sort=False
                )["_source_usable"].mean().items()
            } if group_col in work else {},
        }
        if name == "sentiment":
            report[name]["caution"] = (
                "Embedded sentiment timestamps cannot be audited from aggregates. "
                "Event feeds must be complete after their first publication; "
                "missing intervals must not be interpreted as no news."
            )
    work.attrs["feature_sources"] = report
    return work, report
