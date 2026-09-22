"""Offline bridge from audited FinBERT exports to the multimodal data contract.

The news-sentiment package owns article scoring, coverage, and aggregation.
This module only aligns its already-exported daily features to market sessions.
It never imports FinBERT or downloads articles/model weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SENTIMENT_STATISTICS = (
    "sentiment_mean",
    "sentiment_std",
    "p_positive_mean",
    "p_neutral_mean",
    "p_negative_mean",
    "positive_share",
    "negative_share",
    "confidence_mean",
    "sentiment_ewm",
    "sentiment_momentum",
    "hours_since_last_news",
)
SENTIMENT_COLUMNS = (
    "news_count",
    *(name for statistic in SENTIMENT_STATISTICS for name in (statistic, f"{statistic}_present")),
)
_EXPORT_COLUMNS = (
    "ticker",
    "decision_at",
    "news_count",
    "coverage_status",
    "last_contributing_available_at",
    *SENTIMENT_STATISTICS,
)


@dataclass(frozen=True)
class SentimentExport:
    """Ready-to-align frame, model columns, and the library's audit manifest."""

    frame: pd.DataFrame
    columns: tuple[str, ...]
    manifest: dict[str, Any]


def build_news_decision_points(
    market: pd.DataFrame,
    *,
    tickers: Sequence[str],
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Produce one midnight-UTC decision per observed session and chosen ticker."""

    if isinstance(tickers, str):
        raise TypeError("tickers must be an explicit sequence, not a string.")
    names = tuple(tickers)
    if not names or len(names) != len(set(names)) or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise ValueError("tickers must be non-empty, unique ticker names.")
    missing = sorted({date_col, ticker_col} - set(market))
    if missing:
        raise ValueError(f"Market data is missing columns: {missing}")
    selected = market.loc[market[ticker_col].isin(names), [date_col, ticker_col]].copy()
    if selected.empty:
        raise ValueError("No market rows match the selected tickers.")
    dates = pd.to_datetime(selected[date_col], utc=True, errors="coerce", format="mixed")
    if dates.isna().any():
        raise ValueError("Market sessions contain invalid dates.")
    selected["decision_at"] = dates.dt.normalize()
    selected = selected.rename(columns={ticker_col: "ticker"})
    if selected.duplicated(["ticker", "decision_at"]).any():
        raise ValueError("Market data has duplicate session/ticker rows.")
    ranks = {name: rank for rank, name in enumerate(names)}
    selected["_ticker_rank"] = selected["ticker"].map(ranks)
    return (
        selected.sort_values(["decision_at", "_ticker_rank"], kind="stable")
        .loc[:, ["ticker", "decision_at"]]
        .reset_index(drop=True)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aware_utc(values: pd.Series, *, name: str, nullable: bool) -> pd.Series:
    parsed: list[pd.Timestamp | pd.NaT] = []
    for row, value in enumerate(values):
        if pd.isna(value):
            if not nullable:
                raise ValueError(f"{name} row {row} is missing.")
            parsed.append(pd.NaT)
            continue
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} row {row} is invalid.") from exc
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError(f"{name} row {row} needs an explicit timezone.")
        parsed.append(timestamp.tz_convert("UTC"))
    return pd.Series(pd.to_datetime(parsed, utc=True), index=values.index)


def load_news_sentiment_export(path: str | Path) -> SentimentExport:
    """Verify an exported Parquet/manifest and map it to multimodal masks.

    A `covered` window with zero articles is available but has no article
    timestamp. The upstream library has already checked that its coverage
    evidence was known before the decision cutoff.
    """

    target = Path(path).expanduser().resolve(strict=True)
    if target.suffix.lower() != ".parquet":
        raise ValueError("Sentiment export must be a Parquet file.")
    manifest_path = target.with_suffix(".manifest.json")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported sentiment export schema version.")
    aggregation = manifest.get("aggregation")
    if not isinstance(aggregation, dict) or aggregation.get("include_at_cutoff") is not False:
        raise ValueError("Sentiment export must use include_at_cutoff=False.")
    if not isinstance(manifest.get("checkpoint"), str) or not manifest["checkpoint"].strip():
        raise ValueError("Sentiment export must identify its scoring checkpoint.")
    if manifest.get("parquet_sha256") != _sha256(target):
        raise ValueError("Sentiment export checksum does not match its manifest.")
    raw = pd.read_parquet(target)
    if raw.columns.has_duplicates or list(raw.columns) != manifest.get("feature_columns"):
        raise ValueError("Sentiment export columns do not match its manifest.")
    if len(raw) != manifest.get("feature_rows"):
        raise ValueError("Sentiment export row count does not match its manifest.")
    missing = sorted(set(_EXPORT_COLUMNS) - set(raw))
    if missing:
        raise ValueError(f"Sentiment export is missing columns: {missing}")
    if raw.empty:
        raise ValueError("Sentiment export has no decision points.")
    if not raw["ticker"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Sentiment export has invalid tickers.")

    decisions = _aware_utc(raw["decision_at"], name="decision_at", nullable=False)
    if not decisions.eq(decisions.dt.normalize()).all():
        raise ValueError("Sentiment decisions must be session midnight UTC.")
    if pd.DataFrame({"ticker": raw["ticker"], "decision_at": decisions}).duplicated().any():
        raise ValueError("Sentiment export has duplicate session/ticker rows.")
    contributed = _aware_utc(
        raw["last_contributing_available_at"],
        name="last_contributing_available_at", nullable=True,
    )
    if (contributed.notna() & contributed.ge(decisions)).any():
        raise ValueError("Sentiment export contains news unavailable before the decision.")

    counts = pd.to_numeric(raw["news_count"], errors="coerce")
    if counts.isna().any() or not np.isfinite(counts.to_numpy(dtype=float)).all() or (
        counts.lt(0) | counts.ne(np.floor(counts))
    ).any():
        raise ValueError("Sentiment news_count must be a non-negative integer.")
    if ((counts.gt(0) & contributed.isna()) | (counts.eq(0) & contributed.notna())).any():
        raise ValueError("Sentiment news_count and contributing timestamp disagree.")
    statuses = raw["coverage_status"]
    if not statuses.isin(("covered", "incomplete", "unknown")).all():
        raise ValueError("Sentiment export has an invalid coverage status.")
    if statuses.eq("covered").any():
        inputs = manifest.get("inputs")
        sources = aggregation.get("required_sources")
        if not isinstance(sources, list) or not sources or not (
            isinstance(inputs, dict) and isinstance(inputs.get("coverage"), dict)
        ):
            raise ValueError("Covered sentiment needs source and coverage provenance in its manifest.")

    frame = pd.DataFrame({
        "date": decisions,
        "ticker": raw["ticker"].to_numpy(),
        "source_available": statuses.eq("covered").to_numpy(dtype=bool),
        "available_at": contributed,
        "coverage_status": statuses.to_numpy(),
        "news_count": counts.to_numpy(dtype=np.float32),
    })
    for name in SENTIMENT_STATISTICS:
        values = pd.to_numeric(raw[name], errors="coerce")
        if (raw[name].notna() & values.isna()).any() or np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"Sentiment export has invalid {name} values.")
        if name != "hours_since_last_news" and (counts.eq(0) & values.notna()).any():
            raise ValueError(f"Sentiment {name} cannot be defined without news in the window.")
        if name != "sentiment_momentum" and (counts.gt(0) & values.isna()).any():
            raise ValueError(f"Sentiment {name} is missing despite news in the window.")
        frame[name] = values.fillna(0.0).to_numpy(dtype=np.float32)
        frame[f"{name}_present"] = values.notna().to_numpy(dtype=np.float32)
    return SentimentExport(frame=frame, columns=SENTIMENT_COLUMNS, manifest=manifest)


__all__ = [
    "SENTIMENT_COLUMNS",
    "SENTIMENT_STATISTICS",
    "SentimentExport",
    "build_news_decision_points",
    "load_news_sentiment_export",
]
