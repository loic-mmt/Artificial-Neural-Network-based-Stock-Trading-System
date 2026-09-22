"""Small offline fixtures for the optional FinBERT export integration."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from trading_system.data.multimodal import build_multimodal_dataset
from trading_system.data.news_sentiment import (
    SENTIMENT_STATISTICS,
    build_news_decision_points,
    load_news_sentiment_export,
)


def _raw_features():
    rows = pd.DataFrame({
        "ticker": ["B", "A", "A"],
        "decision_at": pd.to_datetime([
            "2020-01-02T00:00:00Z", "2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z",
        ], utc=True),
        "news_count": [0, 1, 1],
        "coverage_status": ["covered", "covered", "unknown"],
        "last_contributing_available_at": pd.to_datetime([
            None, "2020-01-01T20:00:00Z", "2020-01-02T20:00:00Z",
        ], utc=True),
    })
    for name in SENTIMENT_STATISTICS:
        rows[name] = [np.nan, 0.0, 0.5]
    return rows


def _write_export(tmp_path, raw=None, *, strict=False):
    if raw is None:
        raw = _raw_features()
    target = tmp_path / "sentiment.parquet"
    raw.to_parquet(target, index=False)
    manifest = {
        "schema_version": "1.0",
        "checkpoint": "finbert@revision",
        "aggregation": {"include_at_cutoff": strict, "required_sources": ["feed-a"]},
        "inputs": {"coverage": {"rows": 1, "sha256": "synthetic"}},
        "feature_columns": list(raw.columns),
        "feature_rows": len(raw),
        "parquet_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    return target


def test_decision_points_are_stable_and_use_only_observed_tickers():
    market = pd.DataFrame({
        "ticker": ["A", "B", "A", "C"],
        "date": ["2020-01-03", "2020-01-02", "2020-01-02", "2020-01-02"],
    })
    first = build_news_decision_points(market, tickers=["B", "A"])
    shuffled = build_news_decision_points(
        market.sample(frac=1, random_state=4), tickers=["B", "A"]
    )
    pd.testing.assert_frame_equal(first, shuffled)
    assert first["ticker"].tolist() == ["B", "A", "A"]
    assert first["decision_at"].tolist() == list(pd.to_datetime([
        "2020-01-02", "2020-01-02", "2020-01-03",
    ], utc=True))
    with pytest.raises(ValueError, match="duplicate"):
        build_news_decision_points(pd.concat([market, market.iloc[[0]]]), tickers=["A"])


def test_export_bridge_distinguishes_zero_news_neutral_and_unknown(tmp_path):
    exported = load_news_sentiment_export(_write_export(tmp_path))
    frame = exported.frame
    assert frame["source_available"].tolist() == [True, True, False]
    assert pd.isna(frame.loc[0, "available_at"])
    assert frame.loc[0, "news_count"] == 0
    assert frame.loc[0, "sentiment_mean_present"] == 0
    assert frame.loc[1, "sentiment_mean"] == 0
    assert frame.loc[1, "sentiment_mean_present"] == 1
    assert frame.loc[2, "coverage_status"] == "unknown"

    market = pd.DataFrame({
        "date": ["2020-01-02", "2020-01-02", "2020-01-03"],
        "ticker": ["A", "B", "A"],
        "signal": [0.1, 0.2, 0.3],
    })
    batch = build_multimodal_dataset(
        market,
        tickers=["B", "A"],
        context_len=1,
        temporal_columns=["signal"],
        sentiment_frame=frame,
        sentiment_columns=exported.columns,
    ).batch([0, 1])
    np.testing.assert_array_equal(batch.sentiment_mask, [[True, True], [False, False]])
    assert batch.sentiment[0, 0, exported.columns.index("news_count")] == 0
    assert batch.sentiment[0, 1, exported.columns.index("sentiment_mean_present")] == 1
    assert not batch.sentiment[1].any()


@pytest.mark.parametrize("mutation, message", [
    ("inclusive", "include_at_cutoff"),
    ("tampered", "checksum"),
    ("future_news", "unavailable"),
    ("duplicate", "duplicate"),
    ("unaware", "timezone"),
])
def test_export_bridge_rejects_unsafe_inputs(tmp_path, mutation, message):
    raw = _raw_features()
    strict = False
    if mutation == "future_news":
        raw.loc[1, "last_contributing_available_at"] = raw.loc[1, "decision_at"]
    if mutation == "duplicate":
        raw.loc[2, "decision_at"] = raw.loc[1, "decision_at"]
    if mutation == "unaware":
        raw["decision_at"] = raw["decision_at"].dt.tz_localize(None)
    if mutation == "inclusive":
        strict = True
    target = _write_export(tmp_path, raw, strict=strict)
    if mutation == "tampered":
        raw.assign(news_count=[0, 2, 1]).to_parquet(target, index=False)
    with pytest.raises(ValueError, match=message):
        load_news_sentiment_export(target)


def test_covered_empty_requires_coverage_evidence_in_multimodal_dataset(tmp_path):
    exported = load_news_sentiment_export(_write_export(tmp_path))
    market = pd.DataFrame({"date": ["2020-01-02"], "ticker": ["B"]})
    bad = exported.frame.loc[[0]].copy()
    bad["coverage_status"] = "unknown"
    with pytest.raises(ValueError, match="covered zero-news evidence"):
        build_multimodal_dataset(
            market, tickers=["B"], context_len=1,
            sentiment_frame=bad, sentiment_columns=exported.columns,
        )


def test_export_with_only_covered_zero_news_still_has_utc_timestamps(tmp_path):
    raw = _raw_features().iloc[[0]].reset_index(drop=True)
    ready = load_news_sentiment_export(_write_export(tmp_path, raw))
    assert str(ready.frame["available_at"].dtype).endswith("UTC]")
    assert ready.frame["source_available"].tolist() == [True]


def test_library_export_is_consumable_when_optional_package_is_installed(tmp_path):
    pytest.importorskip("news_sentiment")
    from scripts.export_news_sentiment import main

    market_path = tmp_path / "market.parquet"
    pd.DataFrame({"date": ["2020-01-02"], "ticker": ["A"]}).to_parquet(market_path)
    scored = pd.DataFrame({
        "news_id": ["article-1"], "ticker": ["A"],
        "published_at": pd.to_datetime(["2020-01-01T19:00:00Z"]),
        "available_at": pd.to_datetime(["2020-01-01T20:00:00Z"]),
        "source": ["feed-a"], "url": ["https://example.com/article-1"],
        "title": ["A result"], "text": ["A result"],
        "availability_kind": ["provider_first_seen"],
        "availability_reference": ["archive-1"],
        "label": ["neutral"], "confidence": [0.8],
        "p_negative": [0.1], "p_neutral": [0.8], "p_positive": [0.1],
        "sentiment_score": [0.0],
    })
    scored_path = tmp_path / "scored.parquet"
    scored.to_parquet(scored_path)
    coverage = pd.DataFrame({
        "source": ["feed-a"], "ticker": ["A"],
        "start_at": pd.to_datetime(["2019-12-31T00:00:00Z"]),
        "end_at": pd.to_datetime(["2020-01-02T00:00:00Z"]),
        "recorded_at": pd.to_datetime(["2020-01-01T21:00:00Z"]),
        "status": ["covered"], "evidence_reference": ["archive-1"],
    })
    coverage_path = tmp_path / "coverage.parquet"
    coverage.to_parquet(coverage_path)
    target = tmp_path / "library.parquet"
    main([
        "--data", str(market_path),
        "--tickers", "A",
        "--scored-news", str(scored_path),
        "--coverage", str(coverage_path),
        "--required-sources", "feed-a",
        "--checkpoint", "finbert@revision",
        "--output", str(target),
    ])
    ready = load_news_sentiment_export(target)
    assert ready.frame.loc[0, "source_available"]
    assert ready.frame.loc[0, "sentiment_mean"] == 0
