"""Offline quota, snapshot, provenance, and transport tests. No real API calls."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.request import Request

import pandas as pd
import pytest

from trading_system.data.news_collection import (
    DEFAULT_TICKERS, DEFAULT_TOPICS, HTTPResponse, PilotConfig, QuotaExceeded,
    QuotaLedger, SafeTransportError, collect_news_sentiment, stdlib_http_transport,
)


NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
KEY = "synthetic-secret-only"


def _article(index=1, *, published="20260901T120000", url=None):
    return {"title": f"Company result {index}", "summary": "Synthetic summary, not full text.",
            "url": url or f"https://example.test/news/{index}", "source": "Synthetic wire",
            "time_published": published, "overall_sentiment_score": "0.5",
            "overall_sentiment_label": "Bullish", "ticker_sentiment": [
                {"ticker": "AAPL", "relevance_score": "0.8", "ticker_sentiment_score": "0.6",
                 "ticker_sentiment_label": "Bullish"},
                {"ticker": "MSFT", "relevance_score": "0.4", "ticker_sentiment_score": "-0.2"}],
            "topics": [{"topic": topic, "relevance_score": "0.7"} for topic in DEFAULT_TOPICS]}


def _response(feed=(), **metadata):
    return HTTPResponse(200, json.dumps({"items": str(len(feed)), "feed": list(feed), **metadata}).encode())


def _collect(tmp_path, config=None, transport=None, **kwargs):
    return collect_news_sentiment(config or PilotConfig(), tmp_path / "snapshot", api_key=KEY,
                                  ledger_path=tmp_path / "quota.sqlite3", now=lambda: NOW,
                                  transport=transport or (lambda *args: _response([_article()])), **kwargs)


def _process_reservations(arguments):
    path, timestamp = arguments
    ledger = QuotaLedger(path)
    accepted = 0
    for _ in range(12):
        try:
            ledger.reserve(KEY, now=datetime.fromisoformat(timestamp))
            accepted += 1
        except QuotaExceeded:
            pass
    return accepted


def test_quota_is_atomic_across_processes_and_persists_restarts(tmp_path):
    path = tmp_path / "shared.sqlite3"
    QuotaLedger(path)
    with ProcessPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(_process_reservations, [(str(path), NOW.isoformat())] * 4))
    assert sum(counts) == 25
    reopened = QuotaLedger(path)
    assert reopened.usage(KEY, now=NOW) == {"paris_day_calls": 25, "rolling_24h_calls": 25}
    with pytest.raises(QuotaExceeded):
        reopened.reserve(KEY, now=NOW)
    # Per-key isolation, without plaintext keys in the database.
    reopened.reserve("second-synthetic-key", now=NOW)
    assert KEY.encode() not in path.read_bytes()


def test_quota_survives_paris_midnight_and_external_baseline_not_double_counted(tmp_path):
    ledger = QuotaLedger(tmp_path / "quota.sqlite3")
    before = datetime(2026, 10, 6, 21, 59, tzinfo=timezone.utc)  # Paris 23:59.
    ledger.reserve(KEY, now=before, calls_already_used=23)
    ledger.reserve(KEY, now=before, calls_already_used=23)
    after = before + timedelta(minutes=2)
    assert ledger.usage(KEY, now=after) == {"paris_day_calls": 0, "rolling_24h_calls": 25}
    with pytest.raises(QuotaExceeded):
        QuotaLedger(ledger.path).reserve(KEY, now=after)
    ledger.reserve(KEY, now=before + timedelta(hours=24, minutes=1))


def test_ledger_interoperates_with_external_library_schema(tmp_path):
    path = tmp_path / "quota.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE calls (key_hash TEXT NOT NULL, called_at REAL NOT NULL)")
        connection.execute("INSERT INTO calls VALUES (?, ?)", (hashlib.sha256(KEY.encode()).hexdigest(), NOW.timestamp()))
    ledger = QuotaLedger(path)
    assert ledger.reserve(KEY, now=NOW)["rolling_24h_calls"] == 2
    try:
        from news_sentiment.quota import AlphaVantageQuota
    except ImportError:
        return
    external = AlphaVantageQuota(KEY, path=path)
    external._now_epoch = lambda: NOW.timestamp()
    assert external.reserve().day_used == 3
    assert ledger.usage(KEY, now=NOW)["rolling_24h_calls"] == 3


@pytest.mark.parametrize("value", [-1, 26, True])
def test_configuration_rejects_unsafe_caps(value):
    with pytest.raises(ValueError):
        PilotConfig(max_calls=value)
    with pytest.raises(ValueError):
        PilotConfig(calls_already_used=value)


def test_no_key_means_no_transport_and_no_output(tmp_path):
    with pytest.raises(ValueError, match="API key"):
        collect_news_sentiment(PilotConfig(), tmp_path / "snapshot", ledger_path=tmp_path / "quota",
                               transport=lambda *args: pytest.fail("Network must not run."))
    assert not list(tmp_path.iterdir())


def test_dry_run_has_no_key_network_or_state_writes_and_filters_are_independent(tmp_path):
    result = collect_news_sentiment(PilotConfig(window_days=15), tmp_path / "new" / "snapshot",
                                   ledger_path=tmp_path / "new" / "quota.sqlite3", dry_run=True,
                                   transport=lambda *args: pytest.fail("Dry run invoked transport."))
    assert not list(tmp_path.iterdir())
    assert result["network_calls"] == result["state_writes"] == 0
    requests = result["requests"]
    assert len(requests) == 16
    assert [item.get("tickers") for item in requests[:5]] == list(DEFAULT_TICKERS)
    assert [item.get("topics") for item in requests[5:8]] == list(DEFAULT_TOPICS)
    assert all("tickers" not in item for item in requests[5:8])
    assert all(item["sort"] == "EARLIEST" and item["limit"] == "1000" for item in requests)
    assert all("," not in item.get("tickers", item.get("topics")) for item in requests)
    assert requests[0]["time_to"] == "20260916T0000"


def test_dedup_scope_associations_vendor_scores_and_current_pit_timestamps(tmp_path):
    calls = []
    def transport(params, key, timeout):
        assert QuotaLedger(tmp_path / "quota.sqlite3").usage(key, now=NOW)["rolling_24h_calls"] == len(calls) + 1
        calls.append(params)
        return _response([_article(url="https://EXAMPLE.test/news/1?utm_source=one#section")])
    result = _collect(tmp_path, transport=transport)
    assert result["complete"] is True
    assert result["invocation_calls"] == 8
    assert result["historical_pit_coverage"] is False
    articles = pd.read_parquet(tmp_path / "snapshot" / "articles.parquet")
    associations = pd.read_parquet(tmp_path / "snapshot" / "associations.parquet")
    assert len(articles) == 1 and len(associations) == 8
    assert articles.loc[0, "url"] == "https://example.test/news/1"
    assert articles.loc[0, "available_at"] == pd.Timestamp(NOW)
    assert articles.loc[0, "collected_at"] == pd.Timestamp(NOW)
    assert articles.loc[0, "published_at"] < articles.loc[0, "available_at"]
    assert articles.loc[0, "availability_kind"] == "collector_first_seen"
    assert articles.loc[0, "content_kind"] == "title_summary"
    assert articles.loc[0, "raw_reference"] == articles.loc[0, "availability_reference"]
    assert articles.loc[0, "vendor_overall_sentiment_score"] == 0.5
    assert not any(column.startswith("finbert") for column in articles)
    assert set(associations.loc[associations.scope_type.eq("company"), "scope"]) == set(DEFAULT_TICKERS)
    assert set(associations.loc[associations.scope_type.eq("macro"), "scope"]) == set(DEFAULT_TOPICS)
    assert associations.loc[associations.scope_type.eq("macro"), "vendor_ticker"].isna().all()
    assert associations.loc[associations.scope.eq("AAPL"), "vendor_ticker_sentiment_score"].item() == 0.6
    assert "MSFT" not in set(associations.scope)
    assert all(record["recorded_at"] == NOW.isoformat().replace("+00:00", "Z") for record in result["requests"])
    assert not (tmp_path / "snapshot" / "coverage.parquet").exists()


@pytest.mark.parametrize("failure", ["http", "timeout", "Note", "Information", "bad_json"])
def test_errors_consume_quota_are_redacted_and_never_retry_implicitly(tmp_path, failure):
    attempted = []
    def transport(params, key, timeout):
        attempted.append(params)
        if failure == "http":
            return HTTPResponse(503, key.encode())
        if failure == "timeout":
            raise TimeoutError(f"https://vendor.test/?apikey={key}")
        if failure == "bad_json":
            return HTTPResponse(200, key.encode())
        return HTTPResponse(200, json.dumps({failure: f"quota denied key={key}"}).encode())
    result = _collect(tmp_path, transport=transport)
    assert len(attempted) == 1 and result["invocation_calls"] == 1
    assert result["quota_usage"]["rolling_24h_calls"] == 1
    assert result["complete"] is False
    assert result["requests"][0]["status"] == "error"
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert KEY.encode() not in path.read_bytes()


def test_resume_skips_completed_requests_validates_config_and_preserves_first_seen(tmp_path):
    first = _collect(tmp_path, config=PilotConfig(max_calls=2))
    assert first["invocation_calls"] == 2 and not first["complete"]
    second_calls = []
    def transport(params, key, timeout):
        second_calls.append(params)
        return _response([_article()])
    second = collect_news_sentiment(PilotConfig(max_calls=20), tmp_path / "snapshot", api_key=KEY,
                                   ledger_path=tmp_path / "quota.sqlite3", transport=transport,
                                   now=lambda: NOW + timedelta(minutes=3), resume=True)
    assert second["complete"] and len(second_calls) == 6
    assert second["total_http_attempts"] == 8
    assert pd.read_parquet(tmp_path / "snapshot" / "articles.parquet").available_at.item() == pd.Timestamp(NOW)
    with pytest.raises(ValueError, match="fingerprint"):
        _collect(tmp_path, config=PilotConfig(window_days=15), resume=True,
                 transport=lambda *args: pytest.fail("Mismatched resume attempted network."))
    third = _collect(tmp_path, resume=True, transport=lambda *args: pytest.fail("Completed resume attempted network."))
    assert third["invocation_calls"] == 0


def test_explicit_retry_retains_attempt_audit_and_counts_failed_reservation(tmp_path):
    _collect(tmp_path, transport=lambda *args: HTTPResponse(429, b""))
    result = _collect(tmp_path, resume=True)
    assert result["total_http_attempts"] == 9
    attempts = result["requests"][0]["attempts"]
    assert [attempt["status"] for attempt in attempts] == ["error", "complete"]
    assert attempts[0]["http_status"] == 429
    assert "http_status" not in attempts[1]


def test_resume_dry_run_skips_completed_and_does_not_touch_existing_snapshot(tmp_path):
    _collect(tmp_path, config=PilotConfig(max_calls=2))
    before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
              for path in tmp_path.rglob("*") if path.is_file()}
    result = collect_news_sentiment(PilotConfig(), tmp_path / "snapshot", dry_run=True, resume=True,
                                   transport=lambda *args: pytest.fail("Resume dry run attempted network."))
    after = {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
             for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
    assert len(result["requests"]) == 6


def test_resume_detects_corrupted_raw_and_tables_before_network(tmp_path):
    result = _collect(tmp_path, config=PilotConfig(max_calls=1))
    raw = tmp_path / "snapshot" / result["requests"][0]["raw_reference"]
    raw.write_text("{}")
    with pytest.raises(ValueError, match="raw checksum"):
        _collect(tmp_path, resume=True, transport=lambda *args: pytest.fail("Invalid raw attempted network."))


def test_per_invocation_and_hard_caps_stop_before_transport(tmp_path):
    attempted = []
    result = _collect(tmp_path, config=PilotConfig(max_calls=20, calls_already_used=24),
                      transport=lambda *args: (attempted.append(args), _response())[-1])
    assert len(attempted) == 1
    assert result["quota_usage"]["rolling_24h_calls"] == 25
    assert "hard_call_cap" in result["incomplete_reason"]
    repeated = _collect(tmp_path, config=PilotConfig(max_calls=20, calls_already_used=24), resume=True,
                        transport=lambda *args: pytest.fail("Hard cap attempted network."))
    assert repeated["invocation_calls"] == 0
    assert repeated["quota_usage"]["rolling_24h_calls"] == 25


def test_unique_article_cap_is_explicit_not_silent_truncation(tmp_path):
    result = _collect(tmp_path, config=PilotConfig(max_unique_articles=2),
                      transport=lambda *args: _response([_article(1), _article(2), _article(3)]))
    assert result["unique_articles"] == 2
    assert result["invocation_calls"] == 1
    assert result["requests"][0]["dropped_unique_articles"] == 1
    assert "unique_article_truncation" in result["incomplete_reason"]


def test_saturation_children_run_after_other_streams_and_can_resolve(tmp_path):
    params_seen = []
    def transport(params, key, timeout):
        params_seen.append(params)
        if len(params_seen) == 1:
            return _response([_article()] * 1000)
        if params["time_from"] == "20260916T0000":
            return _response([_article(2, published="20260916T120000")])
        return _response([_article()])
    result = _collect(tmp_path, transport=transport)
    assert result["complete"]
    assert result["invocation_calls"] == 10
    assert [item.get("topics") for item in params_seen[5:8]] == list(DEFAULT_TOPICS)
    assert params_seen[8]["tickers"] == params_seen[9]["tickers"] == "AAPL"
    assert result["requests"][0]["status"] == "split"
    assert len(result["requests"][0]["children"]) == 2


def test_small_page_limit_visits_all_streams_before_global_article_cap(tmp_path):
    seen = []
    def transport(params, key, timeout):
        seen.append(params)
        start = datetime.strptime(params["time_from"], "%Y%m%dT%H%M")
        return _response([_article(len(seen) * 1000 + index, published=start.strftime("%Y%m%dT%H%M%S"))
                          for index in range(100)])
    result = _collect(tmp_path, config=PilotConfig(limit=100, max_calls=20), transport=transport)
    assert len(seen) == 20
    assert [item.get("tickers") for item in seen[:5]] == list(DEFAULT_TICKERS)
    assert [item.get("topics") for item in seen[5:8]] == list(DEFAULT_TOPICS)
    assert result["unique_articles"] == 2000
    assert not result["complete"]
    assert "invocation_call_cap" in result["incomplete_reason"]
    assert all(item["limit"] == "100" for item in seen)


def test_saturated_minute_and_exhausted_split_budget_remain_incomplete(tmp_path):
    config = PilotConfig(start=datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
                         end=datetime(2026, 9, 1, 12, 1, tzinfo=timezone.utc),
                         tickers=("AAPL",), topics=())
    result = _collect(tmp_path, config=config, transport=lambda *args: _response([_article()] * 1000))
    assert "saturated_minute_precision" in result["incomplete_reason"]
    config = replace(config, end=config.end + timedelta(minutes=1), max_calls=1)
    other = tmp_path / "other"
    result = _collect(other, config=config, transport=lambda *args: _response([_article()] * 1000))
    assert not result["complete"] and "invocation_call_cap" in result["incomplete_reason"]
    assert len(result["requests"]) == 3


def test_vendor_end_boundary_overlap_is_filtered_without_losing_last_seconds(tmp_path):
    config = PilotConfig(tickers=("AAPL",), topics=())
    result = _collect(tmp_path, config=config, transport=lambda *args: _response([
        _article(1, published="20260930T235959"), _article(2, published="20261001T000000")]))
    assert result["complete"] and result["unique_articles"] == 1
    assert result["requests"][0]["boundary_filtered_articles"] == 1


def test_stable_fallback_identity_deduplicates_source_title_publication(tmp_path):
    article = _article()
    article["url"] = None
    result = _collect(tmp_path, transport=lambda *args: _response([article, article]))
    assert result["unique_articles"] == 1 and result["association_rows"] == 8


def test_standard_library_transport_does_not_leak_key_and_disables_redirects(monkeypatch):
    from trading_system.data import news_collection
    class Opener:
        def open(self, request, timeout):
            raise RuntimeError(request.full_url)
    monkeypatch.setattr(news_collection, "build_opener", lambda handler: Opener())
    with pytest.raises(SafeTransportError) as error:
        stdlib_http_transport({"function": "NEWS_SENTIMENT"}, KEY, 1)
    assert KEY not in str(error.value)
    assert error.value.__suppress_context__
    assert error.value.__context__ is None
    with pytest.raises(SafeTransportError):
        news_collection._NoRedirect().redirect_request(Request("https://example.test"), None, 302,
                                                       "redirect", {}, "https://elsewhere.test")


def test_cli_dry_run_and_local_environment_aliases(tmp_path, monkeypatch, capsys):
    from scripts.download_news_sentiment_pilot import _local_key, main
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    env_path = tmp_path / "synthetic.env"
    env_path.write_text(f'export ALPHAVANTAGE_API_KEY="{KEY}"\n')
    assert _local_key(env_path) == KEY
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "alias-key")
    assert _local_key(env_path) == "alias-key"
    output = tmp_path / "untouched"
    assert main(["--dry-run", "--output", str(output), "--ledger", str(output / "quota.sqlite3"),
                 "--env-file", str(tmp_path / "nonexistent.env")]) == 0
    assert not output.exists()
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err
    assert json.loads(captured.out)["state_writes"] == 0
