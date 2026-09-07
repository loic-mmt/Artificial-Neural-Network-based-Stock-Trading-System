"""Offline, point-in-time feature sources. Never fetch current fundamentals for history."""

import re
from functools import lru_cache

import numpy as np
import pandas as pd

FUNDAMENTAL_FIELDS = (
    "shares_outstanding", "equity", "assets", "debt", "cash", "revenue_ttm",
    "net_income_ttm", "ebitda_ttm", "eps_ttm", "revenue_ttm_prev_year",
    "eps_ttm_prev_year",
)
SENTIMENT_FEATURES = (
    "news_sentiment_7d", "news_count_7d", "earnings_sentiment_30d", "earnings_count_30d",
)


def _validate_source(source, required):
    if source.empty or any(column not in source for column in required):
        raise ValueError(f"Feature source requires non-empty columns {required}.")
    out = source.copy()
    if not out.ticker.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Source tickers must be non-empty strings.")
    out["available_at"] = pd.to_datetime(out.available_at, utc=True, errors="coerce")
    if out.available_at.isna().any():
        raise ValueError("Feature sources need valid actual publication timestamps (available_at).")
    return out


def attach_fundamentals(frame, source, *, group_col="ticker", date_col="date", max_age_days=550):
    """Backward as-of join; data must be public strictly before the row timestamp.

    Each publication is a full snapshot of known values, not a sparse update.
    Daily midnight dates conservatively defer same-day publications to next bar.
    """
    source = _validate_source(source, ["ticker", "available_at"])
    if source.duplicated(["ticker", "available_at"]).any():
        raise ValueError("Duplicate fundamental publication timestamps per ticker.")
    if not any(column in source for column in FUNDAMENTAL_FIELDS):
        raise ValueError("No supported fundamental fields supplied.")
    if max_age_days <= 0:
        raise ValueError("max_age_days must be positive.")
    names = {column: f"fund_{column}" for column in FUNDAMENTAL_FIELDS}
    source = source.reindex(columns=["ticker", "available_at", *FUNDAMENTAL_FIELDS])
    for column in FUNDAMENTAL_FIELDS:
        source[column] = pd.to_numeric(source[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    source = source.rename(columns={**names, "available_at": "fund_available_at"})
    work = frame.drop(columns=[*names.values(), "fund_available_at"], errors="ignore").copy()
    if group_col not in work:
        raise ValueError("Point-in-time enrichment requires ticker identifiers in price data.")
    work["_feature_row"] = np.arange(len(work))
    parts = []
    for ticker, group in work.groupby(group_col, sort=False, dropna=False):
        left = group.assign(_feature_cutoff=pd.to_datetime(group[date_col], utc=True, errors="raise")).sort_values("_feature_cutoff")
        right = source[source.ticker == ticker].drop(columns="ticker").sort_values("fund_available_at")
        merged = pd.merge_asof(left, right, left_on="_feature_cutoff", right_on="fund_available_at",
                               direction="backward", allow_exact_matches=False,
                               tolerance=pd.Timedelta(days=max_age_days))
        parts.append(merged)
    return pd.concat(parts).sort_values("_feature_row").drop(columns=["_feature_row", "_feature_cutoff"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _sentiment_analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def score_text(text, *, kind="news"):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Sentiment text must be non-empty.")
    pieces = re.split(r"(?<=[.!?])\s+", text) if kind == "earnings" else [text]
    return float(np.mean([_sentiment_analyzer().polarity_scores(piece)["compound"] for piece in pieces]))


def attach_sentiment(frame, source, *, group_col="ticker", date_col="date"):
    source = _validate_source(source, ["ticker", "available_at", "kind"])
    if not source.kind.isin(["news", "earnings"]).all():
        raise ValueError("Sentiment kind must be news or earnings.")
    if "score" not in source:
        if "text" not in source or "language" not in source or not source.language.eq("en").all():
            raise ValueError("VADER requires text and explicit language='en'; otherwise supply score in [-1, 1].")
        source["score"] = [score_text(text, kind=kind) for text, kind in zip(source.text, source.kind)]
    source["score"] = pd.to_numeric(source.score, errors="coerce")
    if source.score.isna().any() or not source.score.between(-1, 1).all():
        raise ValueError("Sentiment scores must be finite and in [-1, 1].")
    if "event_id" in source:
        if source.event_id.isna().any():
            raise ValueError("Sentiment event_id cannot be missing.")
        source = source.sort_values("available_at").drop_duplicates(["ticker", "kind", "event_id"], keep="first")
    work = frame.copy().reset_index(drop=True)
    if group_col not in work:
        raise ValueError("Sentiment enrichment requires ticker identifiers.")
    work[list(SENTIMENT_FEATURES)] = np.nan
    for ticker, group in work.groupby(group_col, sort=False, dropna=False):
        cutoff = pd.to_datetime(group[date_col], utc=True).to_numpy(dtype="datetime64[ns]")
        for kind, window in (("news", 7), ("earnings", 30)):
            records = source[(source.ticker == ticker) & (source.kind == kind)].sort_values("available_at")
            if records.empty:
                continue
            dates = records.available_at.to_numpy(dtype="datetime64[ns]")
            end = np.searchsorted(dates, cutoff, side="left")
            start = np.searchsorted(dates, cutoff - np.timedelta64(window, "D"), side="left")
            count = end - start
            cumulative = np.r_[0., np.cumsum(records.score.to_numpy())]
            mean = np.divide(cumulative[end] - cumulative[start], count, out=np.zeros(len(count)), where=count > 0)
            known = cutoff > dates[0]
            work.loc[group.index, f"{kind}_sentiment_{window}d"] = np.where(known, mean, np.nan)
            work.loc[group.index, f"{kind}_count_{window}d"] = np.where(known, count, np.nan)
    return work
