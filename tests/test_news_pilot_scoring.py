"""Offline scoring/route fixtures; no API requests or model downloads."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trading_system.data.news_pilot_scoring import (
    FinBERTCheckpoint,
    export_news_pilot,
    file_sha256,
    load_local_finbert,
    score_news_pilot,
    verify_local_weights,
)
from trading_system.data.news_sentiment import SENTIMENT_COLUMNS, load_news_sentiment_export


pytest.importorskip("news_sentiment")
CHECKPOINT = FinBERTCheckpoint("yiyanghkust/finbert-tone", "a" * 40, "b" * 64)


class StubAnalyzer:
    def __init__(self):
        self.calls = []

    def predict(self, texts):
        self.calls.append(list(texts))
        return pd.DataFrame({
            "text": texts, "label": "positive", "confidence": 0.8,
            "p_negative": 0.1, "p_neutral": 0.1, "p_positive": 0.8,
            "sentiment_score": 0.7,
        })


def _collector(tmp_path, *, available="2026-10-06T10:00:00Z", duplicated_content=False):
    target = tmp_path / "collected"
    target.mkdir(exist_ok=True)
    articles = pd.DataFrame({
        "article_id": ["article-1", "article-2"],
        "url": ["https://example.com/a", "https://example.com/b"],
        "source": ["example.com", "example.com"],
        "title": [" Profits   rise ", "Profits rise" if duplicated_content else "Rates fall"],
        "summary": ["Results beat expectations", "Results beat expectations" if duplicated_content else "A central bank update"],
        "published_at": pd.to_datetime(["2026-09-15T10:00:00Z", "2026-09-16T10:00:00Z"]),
        "collected_at": pd.to_datetime([available, available]),
        "available_at": pd.to_datetime([available, available]),
        "availability_kind": ["collector_first_seen"] * 2,
        "availability_reference": ["raw/response-1.json#/feed/0", "raw/response-1.json#/feed/1"],
        "raw_reference": ["raw/response-1.json"] * 2,
        "raw_sha256": ["c" * 64] * 2,
        "content_kind": ["title_summary"] * 2,
        "vendor_overall_sentiment_score": [-0.5, -0.3],
        "vendor_overall_sentiment_label": ["Bearish", "Somewhat-Bearish"],
    })
    associations = pd.DataFrame({
        "article_id": ["article-1", "article-1", "article-2", "article-2"],
        "scope_type": ["company", "macro", "macro", "macro"],
        "scope": ["AAPL", "economy_macro", "economy_monetary", "economy_fiscal"],
        "vendor_ticker": ["AAPL", None, None, None],
        "vendor_relevance_score": [0.9, None, None, None],
        "vendor_ticker_sentiment_score": [-0.8, None, None, None],
        "vendor_ticker_sentiment_label": ["Bearish", None, None, None],
        "scope_available_at": pd.to_datetime([available] * 4),
        "scope_raw_reference": ["raw/response-1.json"] * 4,
    })
    raw = target / "raw" / "response-1.json"
    raw.parent.mkdir(exist_ok=True)
    raw.write_text(json.dumps({"feed": [{"title": title, "summary": summary}
                                        for title, summary in zip(articles.title, articles.summary)]}))
    articles["raw_sha256"] = file_sha256(raw)
    associations["scope_raw_sha256"] = file_sha256(raw)
    articles.to_parquet(target / "articles.parquet", index=False)
    associations.to_parquet(target / "associations.parquet", index=False)
    (target / "collection.manifest.json").write_text(json.dumps({
        "schema_version": "1.0", "files": {
            name: {"sha256": file_sha256(target / name)}
            for name in ("articles.parquet", "associations.parquet")
        }, "requests": [{"raw_reference": "raw/response-1.json", "raw_sha256": file_sha256(raw),
                          "finished_at": available}],
    }), encoding="utf-8")
    return target


def _market(*dates):
    return pd.DataFrame([
        {"date": date, "ticker": ticker}
        for date in dates for ticker in ("AAPL", "JPM", "XOM", "WMT", "JNJ")
    ])


def _score(tmp_path, **kwargs):
    source = _collector(tmp_path, **kwargs)
    analyzer = StubAnalyzer()
    pilot = score_news_pilot(source, tmp_path / "scored", checkpoint=CHECKPOINT, analyzer=analyzer)
    return source, analyzer, pilot


def test_score_deduplicates_articles_and_company_macro_content(tmp_path):
    _, analyzer, pilot = _score(tmp_path, duplicated_content=True)
    assert analyzer.calls == [["Profits rise\nResults beat expectations"]]
    assert len(pilot.articles) == 2
    assert len(pilot.company) == 1 and len(pilot.macro) == 3
    assert "ticker" not in pilot.macro
    assert pilot.company["ticker"].tolist() == ["AAPL"]
    assert pilot.company.loc[0, "vendor_ticker_sentiment_score"] == -0.8
    assert pilot.company.loc[0, "vendor_overall_sentiment_score"] == -0.5
    assert pilot.company.loc[0, "sentiment_score"] == 0.7
    assert pilot.articles["availability_kind"].eq("pipeline_observed").all()
    assert pilot.articles["collector_availability_kind"].eq("collector_first_seen").all()
    assert pilot.manifest["unique_texts"] == 1
    assert pilot.articles["available_at"].eq(pd.Timestamp("2026-10-06T10:00:00Z")).all()


@pytest.mark.parametrize("column,value", [
    ("p_positive", 0.7), ("label", "negative"), ("confidence", 0.5),
    ("sentiment_score", -0.4), ("p_negative", float("nan")),
])
def test_rejects_inconsistent_predictions(tmp_path, column, value):
    source = _collector(tmp_path)
    def invalid(texts):
        frame = StubAnalyzer().predict(texts)
        frame[column] = value
        return frame
    with pytest.raises(ValueError):
        score_news_pilot(source, tmp_path / "scored", checkpoint=CHECKPOINT, analyzer=invalid)


def test_unknown_coverage_is_masked_and_macro_not_replicated(tmp_path):
    _, _, pilot = _score(tmp_path)
    paths = export_news_pilot(pilot, _market("2026-09-15", "2026-09-16"), tickers=["AAPL", "JPM", "XOM", "WMT", "JNJ"])
    ready = load_news_sentiment_export(paths["company_sentiment"])
    assert len(ready.columns) == 23
    assert ready.frame["coverage_status"].eq("unknown").all()
    assert not ready.frame["source_available"].any()
    assert ready.frame["news_count"].eq(0).all()
    company_panel = pd.read_parquet(paths["company_panel"])
    assert set(SENTIMENT_COLUMNS).issubset(company_panel)
    macro = pd.read_parquet(paths["macro_sentiment"])
    assert len(macro) == 2 and "ticker" not in macro
    assert not any(macro[name].any() for name in macro if name.endswith("__source_available"))
    manifest = json.loads(paths["macro_sentiment"].with_suffix(".manifest.json").read_text())
    assert manifest["domain"] == "macro"
    assert {group["topic"] for group in manifest["groups"]} == {"global", "economy_macro", "economy_monetary", "economy_fiscal"}
    assert all(len(group["feature_columns"]) == 23 for group in manifest["groups"])
    assert "__macro_topic__" not in paths["macro_sentiment"].with_suffix(".manifest.json").read_text()


def test_strict_cutoff_and_macro_global_dedup(tmp_path):
    _, _, pilot = _score(tmp_path, available="2026-10-07T00:00:00Z")
    paths = export_news_pilot(pilot, _market("2026-10-07", "2026-10-08"), tickers=["AAPL"] , lookback="48h")
    ready = load_news_sentiment_export(paths["company_sentiment"])
    assert ready.frame["news_count"].tolist() == [0, 1]
    assert ready.frame["source_available"].tolist() == [False, False]
    macro = pd.read_parquet(paths["macro_sentiment"])
    assert macro["macro_global__news_count"].tolist() == [0, 2]
    assert macro["macro_global__available_at"].iloc[1] < macro["date"].iloc[1]
    assert macro["macro_global__coverage_status"].eq("unknown").all()
    manifest = json.loads(paths["company_sentiment"].with_suffix(".manifest.json").read_text())
    assert manifest["aggregation"]["include_at_cutoff"] is False
    assert manifest["input_identifiers"]["domain"] == "company"


def test_resume_uses_verified_cache_without_analyzer(tmp_path):
    source, _, pilot = _score(tmp_path)
    resumed = score_news_pilot(source, pilot.output_dir, checkpoint=CHECKPOINT, resume=True)
    assert resumed.manifest["newly_scored_texts"] == 0
    assert resumed.manifest["resume_cache_status"] == "valid_cache"
    pd.testing.assert_frame_equal(resumed.articles, pilot.articles)


@pytest.mark.parametrize("mutation,status", [
    ("tamper", "artifact_checksum_mismatch"), ("checkpoint", "changed_checkpoint_or_settings"),
    ("input", "changed_inputs"), ("content", "changed_content"),
])
def test_resume_invalidates_tamper_checkpoint_input_and_content(tmp_path, mutation, status):
    source, _, pilot = _score(tmp_path)
    checkpoint = CHECKPOINT
    target = pilot.output_dir / "scored_articles.parquet"
    if mutation in ("tamper", "content"):
        frame = pd.read_parquet(target)
        frame.loc[0, "text"] = "Tampered text"
        frame.to_parquet(target, index=False)
        if mutation == "content":
            manifest_path = pilot.output_dir / "scoring.manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"][target.name]["sha256"] = file_sha256(target)
            manifest_path.write_text(json.dumps(manifest))
    elif mutation == "checkpoint":
        checkpoint = FinBERTCheckpoint(CHECKPOINT.repository, "d" * 40, CHECKPOINT.weights_sha256)
    else:
        articles = pd.read_parquet(source / "articles.parquet")
        articles.loc[0, "summary"] = "New content"
        articles.to_parquet(source / "articles.parquet", index=False)
        collection_path = source / "collection.manifest.json"
        collection = json.loads(collection_path.read_text())
        collection["files"]["articles.parquet"]["sha256"] = file_sha256(source / "articles.parquet")
        collection_path.write_text(json.dumps(collection))
    analyzer = StubAnalyzer()
    original_manifest = (pilot.output_dir / "scoring.manifest.json").read_bytes()
    with pytest.raises(ValueError, match=status):
        score_news_pilot(source, pilot.output_dir, checkpoint=checkpoint, analyzer=analyzer, resume=True)
    assert not analyzer.calls
    assert (pilot.output_dir / "scoring.manifest.json").read_bytes() == original_manifest


def test_genuine_coverage_is_explicit_and_point_in_time(tmp_path):
    _, _, pilot = _score(tmp_path, available="2026-10-06T10:00:00Z")
    coverage = pd.DataFrame({
        "source": ["audited-feed"], "ticker": ["AAPL"],
        "start_at": pd.to_datetime(["2026-10-05T00:00:00Z"]),
        "end_at": pd.to_datetime(["2026-10-07T00:00:00Z"]),
        "recorded_at": pd.to_datetime(["2026-10-06T12:00:00Z"]),
        "status": ["covered"], "evidence_reference": ["archive-proof-1"],
    })
    paths = export_news_pilot(pilot, _market("2026-10-07"), tickers=["AAPL", "JPM"], coverage=coverage, required_sources=["audited-feed"])
    ready = load_news_sentiment_export(paths["company_sentiment"])
    assert ready.frame["source_available"].tolist() == [True, False]
    assert not pd.read_parquet(paths["macro_sentiment"])["macro_global__source_available"].any()
    coverage["recorded_at"] = pd.Timestamp("2026-10-07T00:00:00Z")
    paths = export_news_pilot(pilot, _market("2026-10-07"), tickers=["AAPL"], coverage=coverage, required_sources=["audited-feed"])
    assert not load_news_sentiment_export(paths["company_sentiment"]).frame["source_available"].any()
    with pytest.raises(ValueError, match="together"):
        export_news_pilot(pilot, _market("2026-10-07"), tickers=["AAPL"], coverage=coverage)


def test_local_loader_stages_legacy_metadata_without_mutation(tmp_path, monkeypatch):
    import news_sentiment
    directory = tmp_path / "local-model"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"architectures": ["BertForSequenceClassification"]}))
    (directory / "vocab.txt").write_text("[PAD]\n[UNK]\n")
    weights = directory / "pytorch_model.bin"
    weights.write_bytes(b"offline fixture, not actual model weights")
    checkpoint = FinBERTCheckpoint(CHECKPOINT.repository, CHECKPOINT.revision, file_sha256(weights))
    original_hash = file_sha256(directory / "config.json")
    def factory(model_name, **kwargs):
        stage = Path(model_name)
        assert json.loads((stage / "config.json").read_text())["model_type"] == "bert"
        assert json.loads((stage / "tokenizer_config.json").read_text())["tokenizer_class"] == "BertTokenizer"
        assert file_sha256(stage / weights.name) == file_sha256(weights)
        return SimpleNamespace(predict=lambda texts: StubAnalyzer().predict(texts), **kwargs)
    monkeypatch.setattr(news_sentiment, "SentimentAnalyzer", factory)
    loaded = load_local_finbert(directory, checkpoint=checkpoint)
    assert loaded.device == "cpu"
    assert file_sha256(directory / "config.json") == original_hash
    assert not (directory / "tokenizer_config.json").exists()
    with pytest.raises(ValueError, match="weights do not match"):
        verify_local_weights(directory, "f" * 64)


def test_cli_date_filter_and_verified_resume(tmp_path):
    from scripts.score_news_sentiment_pilot import main
    source, _, pilot = _score(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    weights = model_dir / "pytorch_model.bin"
    weights.write_bytes(b"offline fixture")
    checkpoint = FinBERTCheckpoint(CHECKPOINT.repository, CHECKPOINT.revision, file_sha256(weights))
    settings = {
        "device": "cpu", "batch_size": 32, "max_length": 512,
        "config_sha256": file_sha256(model_dir / "config.json"),
        "metadata_sha256": {"config.json": file_sha256(model_dir / "config.json")},
        "local_metadata_staging": "copy metadata; add missing BERT model_type/tokenizer_class; hard-link verified weights or copy fallback",
    }
    output = tmp_path / "cli-scored"
    score_news_pilot(source, output, checkpoint=checkpoint, analyzer=StubAnalyzer(), inference_settings=settings)
    market_path = tmp_path / "market.parquet"
    _market("2026-09-01", "2026-09-15", "2026-10-01").to_parquet(market_path, index=False)
    main([
        "--input-dir", str(source), "--output-dir", str(output),
        "--model-dir", str(model_dir), "--model-repository", checkpoint.repository,
        "--model-revision", checkpoint.revision, "--model-weights-sha256", checkpoint.weights_sha256,
        "--resume", "--data", str(market_path), "--tickers", "AAPL,JPM,XOM,WMT,JNJ",
        "--decision-start", "2026-09-02", "--decision-end", "2026-09-30",
    ])
    ready = load_news_sentiment_export(output / "company_sentiment.parquet")
    assert len(ready.frame) == 5
    assert ready.frame["date"].eq(pd.Timestamp("2026-09-15", tz="UTC")).all()


@pytest.mark.parametrize("tamper", ["raw", "table", "timestamp", "path"])
def test_scoring_verifies_collection_evidence_before_inference(tmp_path, tamper):
    source = _collector(tmp_path)
    if tamper == "raw":
        (source / "raw" / "response-1.json").write_text("{}")
    else:
        frame = pd.read_parquet(source / "articles.parquet")
        if tamper == "timestamp":
            frame["available_at"] = pd.Timestamp("2026-10-05T10:00:00Z")
        elif tamper == "path":
            frame["raw_reference"] = "../outside.json"
        else:
            frame["title"] = "Changed title"
        frame.to_parquet(source / "articles.parquet", index=False)
        if tamper != "table":
            path = source / "collection.manifest.json"
            manifest = json.loads(path.read_text())
            manifest["files"]["articles.parquet"]["sha256"] = file_sha256(source / "articles.parquet")
            path.write_text(json.dumps(manifest))
    analyzer = StubAnalyzer()
    output = tmp_path / "failed-scoring"
    with pytest.raises(ValueError):
        score_news_pilot(source, output, checkpoint=CHECKPOINT, analyzer=analyzer)
    assert not analyzer.calls and not output.exists()


def test_scoring_never_overwrites_existing_version(tmp_path):
    source, _, pilot = _score(tmp_path)
    with pytest.raises(FileExistsError):
        score_news_pilot(source, pilot.output_dir, checkpoint=CHECKPOINT, analyzer=StubAnalyzer())


def test_collector_to_scoring_to_exports_with_real_contract(tmp_path):
    from datetime import datetime, timezone
    from trading_system.data.news_collection import HTTPResponse, PilotConfig, collect_news_sentiment

    observed = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    source = tmp_path / "collector-end-to-end"
    def transport(params, key, timeout):
        return HTTPResponse(200, json.dumps({"items": "1", "feed": [{
            "title": "Profits rise", "summary": "Results beat expectations",
            "url": "https://example.com/one", "source": "fixture-feed",
            "time_published": "20260915T100000",
            "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.9"}],
            "topics": [{"topic": "economy_macro", "relevance_score": "0.8"}],
        }]}).encode())
    result = collect_news_sentiment(
        PilotConfig(tickers=("AAPL",), topics=("economy_macro",)), source,
        api_key="offline-fixture-not-a-real-key", ledger_path=tmp_path / "quota.sqlite3",
        transport=transport, now=lambda: observed,
    )
    assert result["complete"] and result["invocation_calls"] == 2
    analyzer = StubAnalyzer()
    pilot = score_news_pilot(source, tmp_path / "end-to-end-scored", checkpoint=CHECKPOINT, analyzer=analyzer)
    assert sum(map(len, analyzer.calls)) == 1
    paths = export_news_pilot(pilot, _market("2026-09-16"), tickers=["AAPL"])
    exported = load_news_sentiment_export(paths["company_sentiment"])
    assert not exported.frame.source_available.any()
    assert exported.frame.news_count.eq(0).all()
    macro = pd.read_parquet(paths["macro_sentiment"])
    assert "ticker" not in macro and not macro.macro_global__source_available.any()


def test_later_macro_association_is_not_backdated_across_midnight(tmp_path):
    from datetime import datetime
    from trading_system.data.news_collection import HTTPResponse, PilotConfig, collect_news_sentiment

    clock = iter(datetime.fromisoformat(value.replace("Z", "+00:00")) for value in (
        "2026-10-06T23:50:00Z", "2026-10-06T23:51:00Z", "2026-10-06T23:59:00Z",
        "2026-10-07T00:01:00Z", "2026-10-07T00:02:00Z", "2026-10-07T00:03:00Z",
    ))
    def transport(params, key, timeout):
        return HTTPResponse(200, json.dumps({"feed": [{
            "title": "Profits rise", "summary": "Results beat expectations",
            "url": "https://example.com/one", "source": "fixture-feed",
            "time_published": "20260915T100000",
        }]}).encode())
    source = tmp_path / "cross-midnight"
    collect_news_sentiment(
        PilotConfig(tickers=("AAPL",), topics=("economy_macro",)), source,
        api_key="offline-fixture-not-a-real-key", ledger_path=tmp_path / "quota.sqlite3",
        transport=transport, now=lambda: next(clock),
    )
    pilot = score_news_pilot(source, tmp_path / "scored", checkpoint=CHECKPOINT, analyzer=StubAnalyzer())
    assert pilot.company.available_at.iloc[0] == pd.Timestamp("2026-10-06T23:59:00Z")
    assert pilot.macro.available_at.iloc[0] == pd.Timestamp("2026-10-07T00:02:00Z")
    paths = export_news_pilot(pilot, _market("2026-10-07", "2026-10-08"), tickers=["AAPL"])
    assert load_news_sentiment_export(paths["company_sentiment"]).frame.news_count.tolist() == [1, 0]
    assert pd.read_parquet(paths["macro_sentiment"]).macro_global__news_count.tolist() == [0, 1]
