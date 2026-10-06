"""Offline RSS provenance, routing and scorer integration; no live HTTP calls."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from html import escape
import json

import pandas as pd
import pytest

from trading_system.data.news_collection import HTTPResponse
from trading_system.data.rss_news_collection import RSSFeed, RSSPilotConfig, collect_rss_news

pytest.importorskip("news_sentiment")
pytest.importorskip("feedparser")

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
APPLE = RSSFeed("apple", "https://example.test/apple", "company", "AAPL", ("Apple",), "headline_only")
FED = RSSFeed("fed", "https://example.test/fed", "macro", "economy_monetary")


def _rss(*items):
    nodes = []
    for index, item in enumerate(items):
        date = item.get("date", "Tue, 06 Oct 2026 09:00:00 GMT")
        nodes.append("<item><title>" + escape(item.get("title", "Apple profits rise")) + "</title>"
                     + "<link>" + escape(item.get("url", f"https://example.test/article/{index}")) + "</link>"
                     + "<description>" + escape(item.get("summary", "Revenue exceeds expectations")) + "</description>"
                     + ("<pubDate>" + escape(date) + "</pubDate>" if date else "")
                     + "</item>")
    return ('<?xml version="1.0"?><rss version="2.0"><channel><title>Test wire</title>'
            + "".join(nodes) + "</channel></rss>").encode()


def _collect(tmp_path, config=None, transport=None, **kwargs):
    return collect_rss_news(config or RSSPilotConfig(feeds=(APPLE, FED)), tmp_path / "snapshot",
                            transport=transport or (lambda *args: HTTPResponse(200, _rss({}))), now=lambda: NOW, **kwargs)


def test_default_plan_is_six_independent_feeds_and_dry_run_writes_nothing(tmp_path):
    config = RSSPilotConfig()
    assert [feed.scope for feed in config.feeds] == ["AAPL", "JPM", "XOM", "WMT", "JNJ", "economy_monetary"]
    assert all(feed.text_policy == "headline_only" for feed in config.feeds[:-1])
    assert config.feeds[-1].scope_type == "macro" and not config.feeds[-1].aliases
    result = collect_rss_news(config, tmp_path / "missing", dry_run=True,
                              transport=lambda *args: pytest.fail("Dry run attempted HTTP"))
    assert result["network_calls"] == result["state_writes"] == 0
    assert result["max_entries"] == 600
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kwargs", [{"max_entries_per_feed": 0}, {"max_response_bytes": True},
                                    {"timeout_seconds": float("nan")}, {"feeds": (APPLE, APPLE)}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RSSPilotConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"scope_type": "company", "aliases": ()},
                                    {"scope_type": "macro", "aliases": ("Apple",)},
                                    {"url": "https://secret:password@example.test/rss"},
                                    {"feed_id": "../bad"}, {"text_policy": "full_text"}])
def test_invalid_feed(kwargs):
    with pytest.raises(ValueError):
        replace(APPLE, **kwargs)


def test_html_clean_headline_only_and_explicit_alias_routing(tmp_path):
    payload = _rss({"title": "<b>Apple</b> profits rise<script>ignore</script>"},
                   {"title": "Pineapple profits rise"},
                   {"title": "Microsoft profits rise", "summary": "Apple appears only in boilerplate"})
    result = _collect(tmp_path, RSSPilotConfig(feeds=(APPLE,)), lambda *args: HTTPResponse(200, payload))
    assert result["complete"] and result["unique_articles"] == 1
    articles = pd.read_parquet(tmp_path / "snapshot/articles.parquet")
    associations = pd.read_parquet(tmp_path / "snapshot/associations.parquet")
    assert articles.title.item() == "Apple profits rise"
    assert articles.summary.item() == "" and articles.rss_text_policy.item() == "headline_only"
    assert articles.available_at.item() == pd.Timestamp(NOW)
    assert articles.published_at.item() < articles.available_at.item()
    assert associations.scope.tolist() == ["AAPL"]
    assert list(associations.matched_aliases.item()) == ["Apple"]
    assert associations.vendor_relevance_score.isna().all()
    assert result["requests"][0]["rejections"] == {"unmatched_company_alias": 2}
    assert not (tmp_path / "snapshot/coverage.parquet").exists()
    assert result["historical_pit_coverage"] is False


def test_macro_news_is_not_assigned_to_company_even_if_it_mentions_apple(tmp_path):
    _collect(tmp_path, RSSPilotConfig(feeds=(FED,)))
    associations = pd.read_parquet(tmp_path / "snapshot/associations.parquet")
    assert associations.scope_type.tolist() == ["macro"]
    assert associations.scope.tolist() == ["economy_monetary"]
    assert associations.routing_method.tolist() == ["declared_macro_feed"]
    assert associations.vendor_ticker.isna().all()


def test_duplicate_version_and_later_route_keep_own_observation(tmp_path):
    second = replace(APPLE, feed_id="jpm", scope="JPM", aliases=("JPMorgan",))
    feed = _rss({"title": "Apple and JPMorgan profits rise"})
    clock = iter([NOW, NOW, NOW + timedelta(days=1), NOW + timedelta(days=1)])
    collect_rss_news(RSSPilotConfig(feeds=(APPLE, second)), tmp_path / "snapshot",
                     transport=lambda *args: HTTPResponse(200, feed), now=lambda: next(clock))
    articles = pd.read_parquet(tmp_path / "snapshot/articles.parquet")
    associations = pd.read_parquet(tmp_path / "snapshot/associations.parquet")
    assert len(articles) == 1 and len(associations) == 2
    assert articles.available_at.item() == pd.Timestamp(NOW)
    assert associations.scope_available_at.tolist() == [pd.Timestamp(NOW), pd.Timestamp(NOW + timedelta(days=1))]
    from trading_system.data.news_pilot_scoring import FinBERTCheckpoint, export_news_pilot, score_news_pilot
    pilot = score_news_pilot(tmp_path / "snapshot", tmp_path / "scored",
                            checkpoint=FinBERTCheckpoint("test", "a" * 40, "b" * 64), analyzer=_analyzer)
    assert pilot.company.available_at.tolist() == associations.scope_available_at.tolist()
    outputs = export_news_pilot(pilot, pd.DataFrame({"date": ["2026-10-07"] * 2, "ticker": ["AAPL", "JPM"]}),
                                tickers=["AAPL", "JPM"])
    frame = pd.read_parquet(outputs["company_sentiment"])
    assert frame.set_index("ticker").news_count.to_dict() == {"AAPL": 1, "JPM": 0}
    assert not pd.read_parquet(outputs["company_panel"]).source_available.any()


def test_revised_text_is_not_backdated_and_missing_date_is_explicit(tmp_path):
    second = replace(APPLE, feed_id="apple-other")
    calls = iter([_rss({"title": "Apple profits rise", "date": ""}),
                  _rss({"title": "Apple profits fall", "date": ""})])
    result = _collect(tmp_path, RSSPilotConfig(feeds=(APPLE, second)), lambda *args: HTTPResponse(200, next(calls)))
    assert result["unique_articles"] == 2
    assert pd.read_parquet(tmp_path / "snapshot/articles.parquet").published_at.isna().all()


def test_future_and_invalid_publication_dates_are_rejected(tmp_path):
    result = _collect(tmp_path, RSSPilotConfig(feeds=(APPLE,)), lambda *args: HTTPResponse(200, _rss(
        {"date": "Wed, 07 Oct 2026 12:00:00 GMT"}, {"date": "not a timestamp"})))
    assert result["unique_articles"] == 0
    assert result["requests"][0]["rejections"] == {"future_publication_timestamp": 1, "invalid_publication_timestamp": 1}


@pytest.mark.parametrize("body,status", [(b"<html>error</html>", 200), (b"<rss broken", 200), (b"no", 503)])
def test_invalid_and_http_failures_save_evidence_without_retry(tmp_path, body, status):
    calls = []
    def transport(*args):
        calls.append(args)
        return HTTPResponse(status, body)
    result = _collect(tmp_path, RSSPilotConfig(feeds=(APPLE,)), transport)
    assert not result["complete"] and len(calls) == 1
    assert result["requests"][0]["status"] == "error"
    assert (tmp_path / "snapshot/raw/apple-001.xml").read_bytes() == body
    assert result["unique_articles"] == 0


def test_caps_bound_entries_and_payload(tmp_path):
    result = _collect(tmp_path, RSSPilotConfig(feeds=(APPLE,), max_entries_per_feed=1),
                      lambda *args: HTTPResponse(200, _rss({}, {})))
    assert result["unique_articles"] == 1 and not result["complete"]
    assert result["requests"][0]["truncated"]
    result = _collect(tmp_path / "too-large", RSSPilotConfig(feeds=(APPLE,), max_response_bytes=1))
    assert not result["complete"] and result["unique_articles"] == 0
    assert not (tmp_path / "too-large/snapshot/raw").exists()


def test_empty_valid_feed_is_successful_not_coverage(tmp_path):
    result = _collect(tmp_path, RSSPilotConfig(feeds=(APPLE,)), lambda *args: HTTPResponse(200, _rss()))
    assert result["complete"] and result["unique_articles"] == 0
    assert result["publication_coverage"] == "unknown"


def test_resume_skips_successes_and_preserves_failure_attempts(tmp_path):
    calls = iter([HTTPResponse(503, b""), HTTPResponse(200, _rss({}))])
    first = _collect(tmp_path, transport=lambda *args: next(calls))
    assert first["total_http_attempts"] == 2 and not first["complete"]
    second = _collect(tmp_path, resume=True)
    assert second["total_http_attempts"] == 3 and second["invocation_calls"] == 1
    assert [attempt["status"] for attempt in second["requests"][0]["attempts"]] == ["error", "complete"]
    assert "error_type" not in second["requests"][0]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (tmp_path / "snapshot").rglob("*") if path.is_file()}
    third = _collect(tmp_path, resume=True, transport=lambda *args: pytest.fail("Cached RSS must not fetch"))
    assert third["complete"] and third["invocation_calls"] == 0
    assert third["unique_articles"] == 2  # Different title/summary policies are separate text versions.
    assert all(path.read_bytes() == content and path.stat().st_mtime_ns == modified for path, (content, modified) in before.items())


@pytest.mark.parametrize("target", ["raw/apple-001.xml", "articles.parquet", "associations.parquet", "collection.state.json"])
def test_resume_refuses_corrupt_evidence_before_network(tmp_path, target):
    _collect(tmp_path)
    with (tmp_path / "snapshot" / target).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        _collect(tmp_path, resume=True, transport=lambda *args: pytest.fail("Corrupt resume must not fetch"))


def test_resume_refuses_changed_configuration_and_overwrite(tmp_path):
    _collect(tmp_path)
    with pytest.raises(FileExistsError):
        _collect(tmp_path)
    with pytest.raises(ValueError, match="fingerprint"):
        _collect(tmp_path, RSSPilotConfig(max_entries_per_feed=5), resume=True)


def test_atom_feed_uses_same_contract_and_utc_observation(tmp_path):
    body = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <title>Test wire</title><entry><title>Apple profits rise</title>
      <link href="https://example.test/atom"/><updated>2026-10-06T10:00:00+02:00</updated>
      <summary>Revenue rises</summary></entry></feed>'''
    result = _collect(tmp_path, RSSPilotConfig(feeds=(APPLE,)), lambda *args: HTTPResponse(200, body))
    assert result["complete"] and result["unique_articles"] == 1
    articles = pd.read_parquet(tmp_path / "snapshot/articles.parquet")
    assert articles.published_at.item() == pd.Timestamp("2026-10-06T08:00:00Z")
    assert articles.available_at.item() == pd.Timestamp(NOW)


def test_http_transport_reads_only_bounded_feed_body(tmp_path, monkeypatch):
    import trading_system.data.rss_news_collection as module
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, size):
            assert size == 11
            return b"x" * 11
    calls = []
    def open_feed(request, timeout):
        calls.append(request.full_url)
        return Response()
    monkeypatch.setattr(module, "urlopen", open_feed)
    with pytest.raises(ValueError, match="max_response_bytes"):
        module.rss_http_transport(APPLE.url, 10, 10)
    assert calls == [APPLE.url]


def _analyzer(texts):
    return pd.DataFrame({"text": texts, "label": "positive", "confidence": 0.8,
                         "p_negative": 0.1, "p_neutral": 0.1, "p_positive": 0.8, "sentiment_score": 0.7})


def test_scoring_cli_exports_real_contract_and_separate_technical_preview(tmp_path, monkeypatch):
    import scripts.score_news_sentiment_pilot as cli
    from trading_system.data.news_pilot_scoring import file_sha256
    _collect(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{}')
    (model / "pytorch_model.bin").write_bytes(b"synthetic-only")
    monkeypatch.setattr(cli, "load_local_finbert", lambda *args, **kwargs: _analyzer)
    cli.main(["--input-dir", str(tmp_path / "snapshot"), "--output-dir", str(tmp_path / "scored"),
              "--model-dir", str(model), "--model-repository", "test", "--model-revision", "a" * 40,
              "--model-weights-sha256", file_sha256(model / "pytorch_model.bin"),
              "--preview-next-midnight", "--tickers", "AAPL,JPM,XOM,WMT,JNJ"])
    preview = tmp_path / "scored/technical-preview"
    company = pd.read_parquet(preview / "company_panel.parquet")
    assert company.news_count.sum() == 1
    assert not company.source_available.any()
    assert company.coverage_status.eq("unknown").all()
    manifest = json.loads((preview / "company_sentiment.manifest.json").read_text())
    assert manifest["input_identifiers"]["preview_kind"] == "synthetic_cutoffs_not_a_backtest"
    macro = pd.read_parquet(preview / "macro_sentiment.parquet")
    assert "ticker" not in macro and macro.macro_global__news_count.tolist() == [0, 1]
