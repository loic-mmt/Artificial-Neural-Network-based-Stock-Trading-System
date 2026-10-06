"""Small audited RSS/Atom snapshots, compatible with the FinBERT pilot.

No API key, publisher crawling, historical availability claims or coverage
journal. HTTP bodies are bounded, saved once and parsed locally. Company routes
require explicit aliases; a separately declared macro feed is never replicated
across tickers. Observation after parsing is the earliest usable timestamp.
"""

from __future__ import annotations

from calendar import timegm
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from trading_system.data.news_collection import (
    ARTICLE_COLUMNS, ASSOCIATION_COLUMNS, HTTPResponse, UTC, _canonical_url,
    _digest, _integer, _iso, _output_lock, _utc, _write_json, utc_now,
)


@dataclass(frozen=True)
class RSSFeed:
    feed_id: str
    url: str
    scope_type: str
    scope: str
    aliases: tuple[str, ...] = ()
    text_policy: str = "title_summary"

    def __post_init__(self) -> None:
        if not isinstance(self.feed_id, str) or not self.feed_id or not all(c.isalnum() or c in "-_" for c in self.feed_id):
            raise ValueError("feed_id must be a non-empty safe identifier.")
        if not _canonical_url(self.url):
            raise ValueError("Feed URL must be HTTP(S), without credentials.")
        if self.scope_type not in ("company", "macro") or not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("Feed must declare one company or macro scope.")
        if self.text_policy not in ("headline_only", "title_summary"):
            raise ValueError("Unsupported text_policy.")
        if isinstance(self.aliases, str) or any(not isinstance(alias, str) or not alias.strip() for alias in self.aliases):
            raise ValueError("Aliases must be individual non-empty strings.")
        if self.scope_type == "company" and not self.aliases:
            raise ValueError("Company feeds require explicit aliases.")
        if self.scope_type == "macro" and self.aliases:
            raise ValueError("Macro feeds must not declare company aliases.")
        object.__setattr__(self, "aliases", tuple(self.aliases))


def default_rss_feeds() -> tuple[RSSFeed, ...]:
    companies = (
        ("AAPL", "Apple stock", ("Apple", "AAPL")),
        ("JPM", "JPMorgan stock", ("JPMorgan", "JPMorgan Chase", "JP Morgan", "JPM")),
        ("XOM", "ExxonMobil stock", ("ExxonMobil", "Exxon Mobil", "Exxon", "XOM")),
        ("WMT", "Walmart stock", ("Walmart", "Wal-Mart", "WMT")),
        ("JNJ", '"Johnson & Johnson" stock', ("Johnson & Johnson", "Johnson and Johnson", "J&J", "JNJ")),
    )
    feeds = [RSSFeed(
        f"google-{ticker.lower()}",
        "https://news.google.com/rss/search?" + urlencode({"q": query + " when:7d", "hl": "en-US", "gl": "US", "ceid": "US:en"}),
        "company", ticker, aliases, "headline_only",
    ) for ticker, query, aliases in companies]
    feeds.append(RSSFeed("fed-monetary", "https://www.federalreserve.gov/feeds/press_monetary.xml", "macro", "economy_monetary"))
    return tuple(feeds)


@dataclass(frozen=True)
class RSSPilotConfig:
    feeds: tuple[RSSFeed, ...] = ()
    max_entries_per_feed: int = 100
    max_response_bytes: int = 2 * 1024 * 1024
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        feeds = tuple(self.feeds) or default_rss_feeds()
        if not all(isinstance(feed, RSSFeed) for feed in feeds) or len({feed.feed_id for feed in feeds}) != len(feeds):
            raise ValueError("Feeds require unique feed IDs.")
        _integer(len(feeds), "feeds", 1, 25)
        _integer(self.max_entries_per_feed, "max_entries_per_feed", 1, 1000)
        _integer(self.max_response_bytes, "max_response_bytes", 1, 32 * 1024 * 1024)
        if isinstance(self.timeout_seconds, bool) or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite.")
        object.__setattr__(self, "feeds", feeds)

    def configuration(self) -> dict[str, Any]:
        return {"feeds": [asdict(feed) for feed in self.feeds],
                "max_entries_per_feed": self.max_entries_per_feed,
                "max_response_bytes": self.max_response_bytes,
                "timeout_seconds": self.timeout_seconds, "collector_version": 1,
                "availability_kind": "collector_first_seen", "association_provenance_version": 1}

    @property
    def fingerprint(self) -> str:
        return _digest(json.dumps(self.configuration(), sort_keys=True).encode())


def rss_http_transport(url: str, timeout: float, max_bytes: int) -> HTTPResponse:
    """One feed fetch, bounded body, no retries and no article-link fetches."""
    try:
        with urlopen(Request(url, headers={"User-Agent": "news-sentiment-pilot/1.0"}), timeout=timeout) as response:
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ValueError("RSS payload exceeds max_response_bytes.")
            return HTTPResponse(response.status, body)
    except HTTPError as error:
        return HTTPResponse(error.code, b"")


def _parse(body: bytes):
    try:
        import feedparser
        from news_sentiment.news import _alias_patterns, _plain_text
    except ImportError as error:
        raise ImportError("RSS pilot requires the project's optional sentiment extra (including feedparser).") from error
    parsed = feedparser.parse(body)
    # Do not accept an HTML error page, empty document or partially malformed XML.
    if parsed.get("bozo") or not parsed.get("version"):
        raise ValueError("Invalid RSS/Atom document.")
    return parsed, _alias_patterns, _plain_text


def _normalize(entry, feed, source, observed, raw_reference, raw_hash, plain_text, patterns):
    title = plain_text(entry.get("title"))
    if not title:
        raise ValueError("missing_title")
    summary = ""
    if feed.text_policy == "title_summary":
        summary = plain_text(entry.get("summary") or entry.get("description"))
        if summary == title:
            summary = ""
    # Only the text actually scored may establish a company association.
    matched = [alias for alias, pattern in zip(feed.aliases, patterns) if pattern.search(f"{title} {summary}")]
    if feed.scope_type == "company" and not matched:
        raise ValueError("unmatched_company_alias")
    # Bypass FeedParserDict's deprecated updated -> published fallback.
    parsed_date = dict.get(entry, "published_parsed") or dict.get(entry, "updated_parsed")
    published = None
    if parsed_date is not None:
        try:
            published = datetime.fromtimestamp(timegm(parsed_date), tz=UTC)
        except (ValueError, TypeError, OverflowError):
            raise ValueError("invalid_publication_timestamp") from None
        if published > observed:
            raise ValueError("future_publication_timestamp")
    elif entry.get("published") or entry.get("updated"):
        raise ValueError("invalid_publication_timestamp")
    url = _canonical_url(entry.get("link"))
    if not url:
        raise ValueError("missing_article_url")
    publisher = plain_text(entry.get("source", {}).get("title")) or source
    # A revision of an existing URL is a new text version, never a replacement
    # retroactively carrying the first version's availability timestamp.
    identity = json.dumps([url, publisher, title, summary], ensure_ascii=False)
    article_id = _digest(identity.encode())
    article = dict.fromkeys(ARTICLE_COLUMNS)
    article.update(article_id=article_id, url=url, source=publisher, title=title,
                   summary=summary, published_at=_iso(published) if published else None,
                   collected_at=_iso(observed), available_at=_iso(observed),
                   availability_kind="collector_first_seen", availability_reference=raw_reference,
                   raw_reference=raw_reference, raw_sha256=raw_hash, content_kind="title_summary",
                   rss_text_policy=feed.text_policy)
    association = dict.fromkeys(ASSOCIATION_COLUMNS)
    association.update(article_id=article_id, scope_type=feed.scope_type, scope=feed.scope,
                       scope_available_at=_iso(observed), scope_raw_reference=raw_reference,
                       scope_raw_sha256=raw_hash, routing_method="explicit_alias" if matched else "declared_macro_feed",
                       matched_aliases=matched, routing_feed_id=feed.feed_id)
    return article, association


def _save(output: Path, state: dict[str, Any], config: RSSPilotConfig, invocation_calls: int) -> dict[str, Any]:
    import pandas as pd

    files = {}
    for name, records, columns, dates in (
        ("articles", state["articles"], (*ARTICLE_COLUMNS, "rss_text_policy"), ("published_at", "collected_at", "available_at")),
        ("associations", state["associations"], (*ASSOCIATION_COLUMNS, "routing_method", "matched_aliases", "routing_feed_id"), ("scope_available_at",)),
    ):
        frame = pd.DataFrame(list(records.values()), columns=columns)
        for column in dates:
            frame[column] = pd.to_datetime(frame[column], utc=True)
        path = output / f"{name}.parquet"
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        files[path.name] = {"sha256": _digest(path.read_bytes()), "bytes": path.stat().st_size, "rows": len(frame)}
    state_path = output / "collection.state.json"
    _write_json(state_path, state)
    files[state_path.name] = {"sha256": _digest(state_path.read_bytes()), "bytes": state_path.stat().st_size}
    complete = all(record["status"] == "complete" and not record.get("truncated") for record in state["requests"])
    manifest = {"schema_version": "1.0", "provider": "RSS", "configuration": config.configuration(),
                "configuration_fingerprint": config.fingerprint, "availability_kind": "collector_first_seen",
                "historical_pit_coverage": False, "coverage_kind": "current_rss_snapshots",
                "publication_coverage": "unknown", "complete": complete,
                "completeness_meaning": "All configured bounded feed snapshots parsed, not full news-history coverage.",
                "content_kind": "title_summary", "full_text_available": False, "finbert_scored": False,
                "unique_articles": len(state["articles"]), "association_rows": len(state["associations"]),
                "invocation_calls": invocation_calls, "total_http_attempts": state["total_http_attempts"],
                "requests": state["requests"], "files": files,
                "limitations": ["Publication timestamps do not establish historical availability.",
                                "Google News is headline-only; publisher pages are not fetched.",
                                "Alias matching is lexical routing, not validated entity resolution.",
                                "No complete publication coverage or missing-news-as-zero claim."]}
    _write_json(output / "collection.manifest.json", manifest)
    return manifest


def _resume_state(output: Path, config: RSSPilotConfig) -> dict[str, Any]:
    manifest = json.loads((output / "collection.manifest.json").read_text())
    if manifest.get("configuration_fingerprint") != config.fingerprint:
        raise ValueError("RSS resume configuration fingerprint mismatch.")
    for name, artifact in manifest["files"].items():
        if name not in ("articles.parquet", "associations.parquet", "collection.state.json") or _digest((output / name).read_bytes()) != artifact["sha256"]:
            raise ValueError("RSS resume artifact checksum mismatch.")
    state = json.loads((output / "collection.state.json").read_text())
    if state["requests"] != manifest["requests"]:
        raise ValueError("RSS resume request audit mismatch.")
    for record in state["requests"]:
        for attempt in record.get("attempts", []):
            if attempt.get("raw_reference"):
                reference = Path(attempt["raw_reference"])
                if reference.is_absolute() or ".." in reference.parts or not (output / reference).resolve().is_relative_to(output.resolve()):
                    raise ValueError("RSS raw evidence path is unsafe.")
                if _digest((output / reference).read_bytes()) != attempt["raw_sha256"]:
                    raise ValueError("RSS resume raw checksum mismatch.")
    return state


def collect_rss_news(
    config: RSSPilotConfig, output_dir: str | Path, *, dry_run: bool = False,
    resume: bool = False, transport: Callable[[str, float, int], HTTPResponse] = rss_http_transport,
    now: Callable[[], datetime] = utc_now, progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fetch each configured feed once; explicit resume retries only failed feeds.

    No Alpha Vantage client, .env file or quota ledger is accessed. A complete
    resume verifies checksums and makes zero HTTP calls.
    """
    output = Path(output_dir)
    if dry_run:
        return {"configuration": config.configuration(), "configuration_fingerprint": config.fingerprint,
                "network_calls": 0, "state_writes": 0, "max_entries": len(config.feeds) * config.max_entries_per_feed}
    # Check the optional dependency before creating output or making HTTP calls.
    _parse(b'<?xml version="1.0"?><rss version="2.0"><channel><title>Dependency check</title></channel></rss>')
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("RSS output is not empty; use a new output or explicit --resume.")
    if resume and not (output / "collection.manifest.json").is_file():
        raise ValueError("RSS resume requires an existing collection manifest.")
    output.mkdir(parents=True, exist_ok=True)
    with _output_lock(output):
        state = _resume_state(output, config) if resume else {
            "articles": {}, "associations": {}, "total_http_attempts": 0,
            "requests": [{"feed_id": feed.feed_id, "scope_type": feed.scope_type, "scope": feed.scope,
                          "status": "planned", "attempts": []} for feed in config.feeds],
        }
        invocation_calls = 0
        for feed, record in zip(config.feeds, state["requests"]):
            if record["status"] == "complete":
                continue
            # Only attempts retain previous errors/raw observations. The request
            # summary describes the latest attempt, never stale retry metadata.
            for key in set(record) - {"feed_id", "scope_type", "scope", "status", "attempts"}:
                del record[key]
            attempt = {"status": "started", "started_at": _iso(_utc(now()))}
            record["attempts"].append(attempt)
            record["status"] = "started"
            state["total_http_attempts"] += 1
            invocation_calls += 1
            _save(output, state, config, invocation_calls)
            try:
                response = transport(feed.url, config.timeout_seconds, config.max_response_bytes)
                if len(response.body) > config.max_response_bytes:
                    raise ValueError("RSS payload exceeds max_response_bytes.")
                raw_reference = f"raw/{feed.feed_id}-{len(record['attempts']):03d}.xml"
                raw_path = output / raw_reference
                raw_path.parent.mkdir(exist_ok=True)
                raw_path.write_bytes(response.body)
                attempt.update(raw_reference=raw_reference, raw_sha256=_digest(response.body), http_status=response.status_code)
                if response.status_code != 200:
                    raise ValueError("RSS HTTP response was not 200.")
                parsed, alias_patterns, plain_text = _parse(response.body)
                observed = _utc(now())
                if observed < _utc(attempt["started_at"]):
                    raise ValueError("Observation clock moved backwards.")
                attempt["finished_at"] = _iso(observed)
                patterns = alias_patterns({feed.scope: feed.aliases}).get(feed.scope, []) if feed.aliases else []
                source = plain_text(parsed.feed.get("title")) or feed.url
                rejections: dict[str, int] = {}
                accepted = 0
                for entry in parsed.entries[:config.max_entries_per_feed]:
                    try:
                        article, association = _normalize(entry, feed, source, observed, raw_reference,
                                                          attempt["raw_sha256"], plain_text, patterns)
                    except ValueError as error:
                        reason = str(error)
                        rejections[reason] = rejections.get(reason, 0) + 1
                        continue
                    state["articles"].setdefault(article["article_id"], article)
                    key = json.dumps([article["article_id"], feed.scope_type, feed.scope])
                    state["associations"].setdefault(key, association)
                    accepted += 1
                attempt.update(status="complete", entries_received=len(parsed.entries), entries_accepted=accepted,
                               rejections=rejections, truncated=len(parsed.entries) > config.max_entries_per_feed)
            except Exception as error:
                attempt.update(status="error", error_type=type(error).__name__)
                # No exception payloads: upstream URLs can contain private tokens.
                attempt.setdefault("finished_at", _iso(_utc(now())))
            record.update({key: value for key, value in attempt.items() if key != "started_at"})
            _save(output, state, config, invocation_calls)
            if progress:
                progress(f"RSS {feed.feed_id}: {record['status']} accepted={record.get('entries_accepted', 0)}")
        if invocation_calls == 0:
            # Preserve the immutable collector manifest used by scoring cache
            # identities; a verification-only resume must not alter it.
            return {**json.loads((output / "collection.manifest.json").read_text()), "invocation_calls": 0}
        return _save(output, state, config, invocation_calls)
