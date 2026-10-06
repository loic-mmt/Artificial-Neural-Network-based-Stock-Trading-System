"""Bounded Alpha Vantage NEWS_SENTIMENT snapshots (not historical PIT coverage).

``collect_news_sentiment`` accepts an injected ``transport(params, key, timeout)``
returning ``HTTPResponse``. Every invocation of that transport is preceded by a
durable SQLite reservation. The shared ledger must be reused by all collectors
using the same key; ``calls_already_used`` accounts for calls made outside it.

Articles are deduplicated globally; associations describe only the stream that
actually returned them. Macro queries never have tickers and are never expanded
into company associations. All availability timestamps are observation times,
not publication times. This module neither scrapes full text nor scores FinBERT.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Callable, Mapping
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo


UTC = timezone.utc
PARIS = ZoneInfo("Europe/Paris")
HARD_CALL_CAP = 25
DEFAULT_TICKERS = ("AAPL", "JPM", "XOM", "WMT", "JNJ")
DEFAULT_TOPICS = ("economy_macro", "economy_monetary", "economy_fiscal")
ARTICLE_COLUMNS = (
    "article_id", "url", "source", "title", "summary", "published_at",
    "collected_at", "available_at", "availability_kind", "availability_reference",
    "raw_reference", "raw_sha256", "content_kind", "vendor_overall_sentiment_score",
    "vendor_overall_sentiment_label",
)
ASSOCIATION_COLUMNS = (
    "article_id", "scope_type", "scope", "vendor_ticker", "vendor_relevance_score",
    "vendor_ticker_sentiment_score", "vendor_ticker_sentiment_label",
    "scope_available_at", "scope_raw_reference", "scope_raw_sha256",
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Timestamps require an explicit timezone.")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _integer(value: int, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")


@dataclass(frozen=True)
class PilotConfig:
    """End-exclusive, minute-aligned UTC windows and independent stream scopes.

    ``max_calls`` is per invocation (default 20); the shared ledger always enforces
    25 for both the Paris calendar day and the preceding rolling 24 hours.
    ``calls_already_used`` is a conservative count of *external* calls today,
    not a count to add again on each restart. Increasing it adds only the delta.
    ``limit`` defaults to the provider maximum 1000; choosing 100 for a bounded
    20-call/2000-article pilot allows every initial stream to be visited, at the
    cost of requiring more splits and recording incomplete saturated windows.
    """

    start: datetime = datetime(2026, 9, 1, tzinfo=UTC)
    end: datetime = datetime(2026, 10, 1, tzinfo=UTC)
    tickers: tuple[str, ...] = DEFAULT_TICKERS
    topics: tuple[str, ...] = DEFAULT_TOPICS
    window_days: int = 30
    max_calls: int = 20
    calls_already_used: int = 0
    max_unique_articles: int = 2000
    limit: int = 1000
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        start, end = _utc(self.start), _utc(self.end)
        if start >= end or any(value.second or value.microsecond for value in (start, end)):
            raise ValueError("start/end must be increasing and minute-aligned.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        _integer(self.max_calls, "max_calls", 0, HARD_CALL_CAP)
        _integer(self.calls_already_used, "calls_already_used", 0, HARD_CALL_CAP)
        _integer(self.max_unique_articles, "max_unique_articles", 1, 1_000_000)
        _integer(self.window_days, "window_days", 1, 3660)
        _integer(self.limit, "limit", 1, 1000)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite.")
        for names, label in ((self.tickers, "tickers"), (self.topics, "topics")):
            if isinstance(names, str) or len(names) != len(set(names)):
                raise ValueError(f"{label} must contain unique individual scopes.")
            if any(not isinstance(name, str) or not name or any(c in name for c in ",&= \n") for name in names):
                raise ValueError(f"{label} must contain non-empty individual scopes.")
        if not self.tickers and not self.topics:
            raise ValueError("At least one stream is required.")

    def data_configuration(self) -> dict[str, Any]:
        """Stable resume fingerprint; invocation budgets may change on resume."""
        return {
            "start": _iso(self.start), "end": _iso(self.end),
            "tickers": list(self.tickers), "topics": list(self.topics),
            "window_days": self.window_days, "max_unique_articles": self.max_unique_articles,
            "function": "NEWS_SENTIMENT", "sort": "EARLIEST", "limit": self.limit,
            "end_exclusive": True, "vendor_end_inclusivity": "undocumented_boundary_overlap_assumed",
            "availability_kind": "collector_first_seen",
            "content_kind": "title_summary",
            "association_provenance_version": 1,
        }

    @property
    def fingerprint(self) -> str:
        return _digest(json.dumps(self.data_configuration(), sort_keys=True).encode())


class QuotaExceeded(RuntimeError):
    """No HTTP request was made because the durable hard cap was reached."""


class QuotaLedger:
    """Process-safe pre-request reservations, including failed HTTP attempts.

    Synthetic external-call reservations survive restarts and the Paris midnight
    boundary. A transaction serializes check-and-reserve across all processes.
    Keys exist only as SHA-256 digests in the database, never as plaintext.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            # Match news_sentiment.AlphaVantageQuota so both clients can share
            # one physical ledger instead of accidentally using two counters.
            connection.execute("CREATE TABLE IF NOT EXISTS calls (key_hash TEXT NOT NULL, called_at REAL NOT NULL)")
            connection.execute("CREATE INDEX IF NOT EXISTS calls_key_time ON calls(key_hash, called_at)")
            connection.execute("CREATE TABLE IF NOT EXISTS external_baselines (key_hash TEXT NOT NULL, paris_day TEXT NOT NULL, call_count INTEGER NOT NULL, PRIMARY KEY(key_hash, paris_day))")

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _identity(key: str) -> str:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("A non-empty Alpha Vantage API key is required.")
        return _digest(key.strip().encode())

    @staticmethod
    def _usage(connection: sqlite3.Connection, identity: str, at: datetime) -> dict[str, int]:
        day_start = datetime.combine(at.astimezone(PARIS).date(), time(), tzinfo=PARIS).timestamp()
        # Future-dated reservations also count if the system clock moves backward.
        daily, rolling = connection.execute(
            "SELECT COALESCE(SUM(called_at >= ?), 0), COALESCE(SUM(called_at >= ?), 0) FROM calls WHERE key_hash = ?",
            (day_start, at.timestamp() - 86400, identity),
        ).fetchone()
        return {"paris_day_calls": int(daily), "rolling_24h_calls": int(rolling)}

    def usage(self, key: str, *, now: datetime) -> dict[str, int]:
        with self._connection() as connection:
            return self._usage(connection, self._identity(key), _utc(now))

    def reserve(self, key: str, *, now: datetime, calls_already_used: int = 0) -> dict[str, int]:
        _integer(calls_already_used, "calls_already_used", 0, HARD_CALL_CAP)
        identity, at = self._identity(key), _utc(now)
        day = at.astimezone(PARIS).date().isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT call_count FROM external_baselines WHERE key_hash = ? AND paris_day = ?", (identity, day)).fetchone()
            baseline = previous[0] if previous else 0
            if calls_already_used > baseline:
                connection.executemany("INSERT INTO calls (key_hash, called_at) VALUES (?, ?)", [(identity, at.timestamp())] * (calls_already_used - baseline))
                connection.execute("INSERT INTO external_baselines VALUES (?, ?, ?) ON CONFLICT(key_hash, paris_day) DO UPDATE SET call_count = excluded.call_count", (identity, day, calls_already_used))
            usage = self._usage(connection, identity, at)
            if max(usage.values()) >= HARD_CALL_CAP:
                connection.commit()  # Retain the external-call evidence even on refusal.
                raise QuotaExceeded("Alpha Vantage hard cap reached; no request attempted.")
            connection.execute("INSERT INTO calls (key_hash, called_at) VALUES (?, ?)", (identity, at.timestamp()))
            connection.commit()
            return {name: count + 1 for name, count in usage.items()}


@dataclass(frozen=True)
class HTTPResponse:
    status_code: int
    body: bytes


class SafeTransportError(RuntimeError):
    """A deliberately generic transport error, without URLs or API keys."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise SafeTransportError("HTTP redirects are disabled.")


def stdlib_http_transport(params: Mapping[str, str], api_key: str, timeout: float) -> HTTPResponse:
    """Exactly one HTTPS attempt, no redirects and no automatic retry.

    Call only through the collector, which reserves quota first. All failures are
    sanitized, including HTTPError URLs that contain the query-string secret.
    """
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("A non-empty Alpha Vantage API key is required.")
    api_key = api_key.strip()
    url = "https://www.alphavantage.co/query?" + urlencode({**params, "apikey": api_key})
    try:
        opener = build_opener(_NoRedirect())
        with opener.open(Request(url, headers={"User-Agent": "news-sentiment-pilot/1.0"}), timeout=timeout) as response:
            body = response.read(32 * 1024 * 1024 + 1)
            if len(body) > 32 * 1024 * 1024:
                raise SafeTransportError("HTTP response exceeds the bounded payload size.")
            return HTTPResponse(response.status, body)
    except HTTPError as error:
        return HTTPResponse(error.code, b"")
    except Exception:
        pass
    # Raise outside the except block so even __context__ cannot retain an
    # exception carrying the secret-bearing URL.
    raise SafeTransportError("HTTP request failed; attempt consumed quota.") from None


def _request(scope_type: str, scope: str, start: datetime, end: datetime, parent_id: str | None = None) -> dict[str, Any]:
    core = {"scope_type": scope_type, "scope": scope, "start": _iso(start), "end": _iso(end)}
    return {**core, "request_id": _digest(json.dumps(core, sort_keys=True).encode())[:24],
            "parent_id": parent_id, "status": "planned", "completeness": "unknown"}


def _initial_requests(config: PilotConfig) -> list[dict[str, Any]]:
    records = []
    start = config.start
    # Window-major ordering visits all eight scopes before advancing the window.
    while start < config.end:
        end = min(config.end, start + timedelta(days=config.window_days))
        for kind, scopes in (("company", config.tickers), ("macro", config.topics)):
            records.extend(_request(kind, scope, start, end) for scope in scopes)
        start = end
    return records


def request_parameters(record: Mapping[str, Any], *, limit: int = 1000) -> dict[str, str]:
    """One ticker OR one topic: never comma-joined AND filters across scopes."""
    params = {"function": "NEWS_SENTIMENT", "sort": "EARLIEST", "limit": str(limit),
              "time_from": _utc(record["start"]).strftime("%Y%m%dT%H%M"),
              # Inclusive vendor bounds may overlap; local filtering is strictly
              # end-exclusive. Subtracting a minute could lose its final seconds.
              "time_to": _utc(record["end"]).strftime("%Y%m%dT%H%M")}
    params["tickers" if record["scope_type"] == "company" else "topics"] = record["scope"]
    return params


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, dict):
        return {_redact(str(key), secret): _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _canonical_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        parsed = urlsplit(value.strip())
        if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            return ""
        query = sorted((key, item) for key, item in parse_qsl(parsed.query, keep_blank_values=True)
                       if not key.lower().startswith("utm_") and key.lower() not in ("fbclid", "gclid"))
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", urlencode(query), ""))
    except ValueError:
        return ""


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _published(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Article publication timestamp is missing.")
    for pattern in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.strptime(value, pattern).replace(tzinfo=UTC)
        except ValueError:
            pass
    return _utc(value)


def _normalize(item: dict[str, Any], record: Mapping[str, Any], observed: datetime, raw_reference: str, raw_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    published = _published(item.get("time_published"))
    if not _utc(record["start"]) <= published < _utc(record["end"]):
        raise ValueError("Article lies outside the requested temporal window.")
    if published > observed:
        raise ValueError("Publication cannot follow the collection observation time.")
    title, summary = item.get("title", ""), item.get("summary", "")
    if not isinstance(title, str) or not title.strip() or not isinstance(summary, str):
        raise ValueError("Article must contain a title and a textual summary.")
    source = item.get("source") or item.get("source_domain") or "unknown"
    if not isinstance(source, str):
        raise ValueError("Article source must be textual.")
    url = _canonical_url(item.get("url"))
    article_id = _digest(("url:" + url if url else "fallback:" + json.dumps([source, title, _iso(published)], ensure_ascii=False)).encode())
    article = {"article_id": article_id, "url": url, "source": source, "title": title,
               "summary": summary, "published_at": _iso(published),
               "collected_at": _iso(observed), "available_at": _iso(observed),
               "availability_kind": "collector_first_seen", "availability_reference": raw_reference,
               "raw_reference": raw_reference, "raw_sha256": raw_hash, "content_kind": "title_summary",
               "vendor_overall_sentiment_score": _number(item.get("overall_sentiment_score")),
               "vendor_overall_sentiment_label": item.get("overall_sentiment_label") if isinstance(item.get("overall_sentiment_label"), str) else None}
    company = record["scope_type"] == "company"
    metadata = next((entry for entry in item.get("ticker_sentiment", []) if isinstance(entry, dict) and entry.get("ticker") == record["scope"]), {}) if company and isinstance(item.get("ticker_sentiment", []), list) else {}
    if not company:
        metadata = next((entry for entry in item.get("topics", []) if isinstance(entry, dict) and entry.get("topic") == record["scope"]), {}) if isinstance(item.get("topics", []), list) else {}
    association = {"article_id": article_id, "scope_type": record["scope_type"], "scope": record["scope"],
                   "scope_available_at": _iso(observed), "scope_raw_reference": raw_reference,
                   "scope_raw_sha256": raw_hash,
                   "vendor_ticker": record["scope"] if company else None,
                   "vendor_relevance_score": _number(metadata.get("relevance_score")),
                   "vendor_ticker_sentiment_score": _number(metadata.get("ticker_sentiment_score")) if company else None,
                   "vendor_ticker_sentiment_label": metadata.get("ticker_sentiment_label") if company and isinstance(metadata.get("ticker_sentiment_label"), str) else None}
    return article, association


@contextmanager
def _output_lock(output: Path):
    # The shared quota is protected by SQLite; this lock protects one snapshot's
    # files against simultaneous writers, and is automatically released on exit.
    import os
    lock_path = output / ".collection.lock"
    if os.name == "nt":
        import msvcrt
        with lock_path.open("a+b") as lock:
            lock.seek(0, 2)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise ValueError("This collection output is already in use.") from None
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This collection output is already in use.") from None
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _complete(record: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> bool:
    if record["status"] == "complete":
        return True
    children = record.get("children", [])
    return bool(children) and all(_complete(records[child], records) for child in children)


def _manifest(state: dict[str, Any], config: PilotConfig, observed: datetime, invocation_calls: int, quota: dict[str, int], stop_reason: str | None) -> dict[str, Any]:
    records = {record["request_id"]: record for record in state["requests"]}
    roots = [record for record in records.values() if not record["parent_id"]]
    complete = all(_complete(record, records) for record in roots)
    reasons = sorted({record.get("incomplete_reason") for record in records.values()
                      if not _complete(record, records) and record.get("incomplete_reason")})
    if stop_reason and not complete:
        reasons.append(stop_reason)
    return {"schema_version": "1.0", "recorded_at": _iso(observed),
            "configuration": config.data_configuration(), "configuration_fingerprint": config.fingerprint,
            "availability_kind": "collector_first_seen", "historical_pit_coverage": False,
            "coverage_kind": "current_collection_snapshot", "content_kind": "title_summary",
            "full_text_available": False, "finbert_scored": False,
            "complete": complete, "incomplete_reason": sorted(set(reasons)) if not complete else [],
            "unique_articles": len(state["articles"]), "association_rows": len(state["associations"]),
            "invocation_calls": invocation_calls, "total_http_attempts": state["total_http_attempts"],
            "call_policy": {"hard_cap": HARD_CALL_CAP, "per_invocation_cap": config.max_calls,
                            "calls_already_used": config.calls_already_used, "calendar_timezone": "Europe/Paris",
                            "rolling_24h_enforced": True}, "quota_usage": quota,
            "requests": state["requests"], "files": state.get("files", {}),
            "limitations": ["Historical publication timestamps do not establish historical availability.",
                            "Title and vendor summary only; full text and source lineage are unavailable.",
                            "Vendor sentiment fields are not recomputed FinBERT scores."]}


def _save(output: Path, state: dict[str, Any], config: PilotConfig, observed: datetime, invocation_calls: int, quota: dict[str, int], stop_reason: str | None) -> dict[str, Any]:
    import pandas as pd
    files = {}
    for name, columns in (("articles", ARTICLE_COLUMNS), ("associations", ASSOCIATION_COLUMNS)):
        path = output / f"{name}.parquet"
        frame = pd.DataFrame(state[name], columns=columns)
        if name == "articles":
            for column in ("published_at", "collected_at", "available_at"):
                frame[column] = pd.to_datetime(frame[column], utc=True)
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        files[path.name] = {"bytes": path.stat().st_size, "sha256": _digest(path.read_bytes()), "rows": len(frame)}
    raw_files = sorted((output / "raw").glob("*.json"))
    files["raw"] = {"files": len(raw_files), "bytes": sum(path.stat().st_size for path in raw_files)}
    state["files"] = files
    _write_json(output / "collection.state.json", state)
    manifest = _manifest(state, config, observed, invocation_calls, quota, stop_reason)
    _write_json(output / "collection.manifest.json", manifest)
    return manifest


def _resume_state(output: Path, config: PilotConfig) -> dict[str, Any]:
    try:
        state = json.loads((output / "collection.state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("Resume requires a valid existing collection state.") from None
    if not isinstance(state, dict) or state.get("schema_version") != "1.0" or state.get("configuration_fingerprint") != config.fingerprint or state.get("configuration") != config.data_configuration():
        raise ValueError("Resume configuration fingerprint does not match the existing snapshot.")
    if not all(isinstance(state.get(name), list) for name in ("articles", "associations", "requests")):
        raise ValueError("Resume snapshot state is malformed.")
    for name, entry in state.get("files", {}).items():
        if name in ("articles.parquet", "associations.parquet"):
            path = output / name
            if not path.is_file() or _digest(path.read_bytes()) != entry.get("sha256"):
                raise ValueError("Resume table checksum validation failed.")
    raw_records = [item for record in state["requests"] for item in [record, *record.get("attempts", [])]]
    for record in raw_records:
        if record.get("raw_reference"):
            relative = Path(record["raw_reference"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Resume raw reference is unsafe.")
            raw = output / relative
            if not raw.is_file() or _digest(raw.read_bytes()) != record.get("raw_sha256"):
                raise ValueError("Resume raw checksum validation failed.")
    return state


def _finish_attempt(record: dict[str, Any]) -> None:
    """Retain every attempt's sanitized audit even when a resume retries it."""
    record["attempts"][-1].update({name: value for name, value in record.items()
        if name not in ("attempts", "children", "scope", "scope_type", "start", "end", "parent_id", "request_id")})


def collect_news_sentiment(
    config: PilotConfig, output_dir: str | Path, *, api_key: str | None = None,
    ledger_path: str | Path | None = None,
    transport: Callable[[Mapping[str, str], str, float], HTTPResponse] = stdlib_http_transport,
    now: Callable[[], datetime] = utc_now, dry_run: bool = False, resume: bool = False,
) -> dict[str, Any]:
    """Collect one bounded invocation; returns the persisted manifest (or plan).

    Dry runs need no key, perform no transport calls, and create no directories,
    ledger entries, files, or locks. Resume is explicit: completed requests are
    skipped; previous failed/interrupted attempts may be attempted once again.
    There are no implicit HTTP retries. Saturated successful windows enqueue
    minute-aligned child windows behind other streams and remain incomplete
    until all children complete. Invocation budgets never bypass the hard cap.
    """
    output = Path(output_dir).expanduser().resolve()
    observed = _utc(now())
    state = _resume_state(output, config) if resume else {
        "schema_version": "1.0", "configuration": config.data_configuration(),
        "configuration_fingerprint": config.fingerprint, "articles": [], "associations": [],
        "requests": _initial_requests(config), "total_http_attempts": 0,
    }
    if dry_run:
        return {"dry_run": True, "configuration": config.data_configuration(),
                "configuration_fingerprint": config.fingerprint, "per_invocation_cap": config.max_calls,
                "hard_cap": HARD_CALL_CAP, "requests": [request_parameters(record, limit=config.limit) for record in state["requests"]
                    if record["status"] not in ("complete", "split")], "network_calls": 0, "state_writes": 0}
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("A non-empty Alpha Vantage API key is required; no request attempted.")
    api_key = api_key.strip()
    if ledger_path is None:
        raise ValueError("An explicit shared quota ledger path is required.")
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError("Output already contains files; use --resume or a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    with _output_lock(output):
        if resume:
            state = _resume_state(output, config)  # Refresh under the writer lock.
        elif (output / "collection.state.json").exists():
            raise ValueError("Output already has a snapshot; use --resume.")
        ledger = QuotaLedger(ledger_path)
        quota = ledger.usage(api_key, now=observed)
        queue = deque(record for record in state["requests"] if record["status"] not in ("complete", "split"))
        articles = {article["article_id"]: article for article in state["articles"]}
        associations = {(item["article_id"], item["scope_type"], item["scope"]): item for item in state["associations"]}
        calls, stop_reason = 0, None
        while queue:
            if calls >= config.max_calls:
                stop_reason = "invocation_call_cap"
                break
            if len(articles) >= config.max_unique_articles:
                stop_reason = "unique_article_cap"
                break
            record = queue.popleft()
            started = _utc(now())
            try:
                quota = ledger.reserve(api_key, now=started, calls_already_used=config.calls_already_used)
            except QuotaExceeded:
                quota = ledger.usage(api_key, now=started)
                stop_reason = "hard_call_cap"
                break
            calls += 1
            state["total_http_attempts"] += 1
            for name in ("finished_at", "http_status", "raw_reference", "raw_sha256", "incomplete_reason",
                         "result_count", "accepted_new_articles", "dropped_unique_articles", "invalid_articles",
                         "boundary_filtered_articles", "saturated", "vendor_items"):
                record.pop(name, None)
            if record.get("attempts") and record["attempts"][-1].get("status") == "started":
                record["attempts"][-1].update(status="interrupted", completeness="incomplete",
                    incomplete_reason="previous_attempt_interrupted", recorded_at=_iso(started))
            record.update(status="started", started_at=_iso(started), recorded_at=_iso(started),
                          completeness="unknown", attempt_count=record.get("attempt_count", 0) + 1)
            record.setdefault("attempts", []).append({"status": "started", "started_at": _iso(started),
                "recorded_at": _iso(started), "attempt_count": record["attempt_count"]})
            _write_json(output / "collection.state.json", state)
            payload = None
            error_kind = None
            try:
                response = transport(request_parameters(record, limit=config.limit), api_key, config.timeout_seconds)
                if not 200 <= response.status_code < 300:
                    error_kind = "http_error"
                    record["http_status"] = response.status_code
                else:
                    payload = _redact(json.loads(response.body, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON constant."))), api_key)
                    if not isinstance(payload, dict):
                        error_kind = "invalid_json_payload"
                    elif any(name in payload for name in ("Note", "Information", "Error Message")):
                        error_kind = "vendor_error_or_quota"
                    elif not isinstance(payload.get("feed"), list):
                        error_kind = "missing_feed"
            except Exception:
                # Never persist arbitrary exception strings: URLs can contain keys.
                error_kind = "transport_or_json_error"
            finished = _utc(now())
            record.update(finished_at=_iso(finished), recorded_at=_iso(finished))
            if payload is not None:
                relative = f"raw/{record['request_id']}.attempt-{record['attempt_count']}.json"
                raw_path = output / relative
                _write_json(raw_path, payload)
                record.update(raw_reference=relative, raw_sha256=_digest(raw_path.read_bytes()))
            if error_kind:
                record.update(status="error", completeness="incomplete", incomplete_reason=error_kind)
                _finish_attempt(record)
                stop_reason = error_kind
                _save(output, state, config, finished, calls, quota, stop_reason)
                break  # No implicit retry; explicit resume is the only retry action.
            feed = payload["feed"]
            invalid, dropped, accepted, boundary_filtered = 0, 0, 0, 0
            for item in feed:
                try:
                    if not isinstance(item, dict):
                        raise ValueError("Invalid article.")
                    published = _published(item.get("time_published"))
                    if published >= _utc(record["end"]) and published < _utc(record["end"]) + timedelta(minutes=1):
                        boundary_filtered += 1
                        continue
                    article, association = _normalize(item, record, finished, record["raw_reference"], record["raw_sha256"])
                except (ValueError, TypeError, KeyError):
                    invalid += 1
                    continue
                identity = article["article_id"]
                if identity not in articles:
                    if len(articles) >= config.max_unique_articles:
                        dropped += 1
                        continue
                    articles[identity] = article
                    accepted += 1
                associations.setdefault((identity, association["scope_type"], association["scope"]), association)
            state["articles"] = list(articles.values())
            state["associations"] = list(associations.values())
            advertised = _number(payload.get("items"))
            saturated = len(feed) >= config.limit or (advertised is not None and advertised >= config.limit)
            vendor_truncated = not saturated and advertised is not None and advertised > len(feed)
            record.update(result_count=len(feed), accepted_new_articles=accepted, dropped_unique_articles=dropped,
                          invalid_articles=invalid, boundary_filtered_articles=boundary_filtered,
                          saturated=saturated, vendor_items=advertised)
            record.pop("incomplete_reason", None)
            if dropped or invalid or vendor_truncated:
                reason = "unique_article_truncation" if dropped else "invalid_articles" if invalid else "vendor_truncation"
                record.update(status="incomplete", completeness="incomplete", incomplete_reason=reason)
                stop_reason = reason
            elif saturated:
                start, end = _utc(record["start"]), _utc(record["end"])
                minutes = int((end - start).total_seconds() // 60)
                if minutes >= 2:
                    middle = start + timedelta(minutes=minutes // 2)
                    children = [_request(record["scope_type"], record["scope"], left, right, record["request_id"])
                                for left, right in ((start, middle), (middle, end))]
                    state["requests"].extend(children)
                    queue.extend(children)
                    record.update(status="split", children=[child["request_id"] for child in children],
                                  completeness="incomplete", incomplete_reason="saturation_pending_children")
                else:
                    record.update(status="incomplete", completeness="incomplete", incomplete_reason="saturated_minute_precision")
            else:
                record.update(status="complete", completeness="complete")
            _finish_attempt(record)
            _save(output, state, config, finished, calls, quota, stop_reason)
            if stop_reason:
                break
        return _save(output, state, config, _utc(now()), calls, quota, stop_reason)


__all__ = ["PilotConfig", "QuotaLedger", "QuotaExceeded", "HTTPResponse", "SafeTransportError",
           "collect_news_sentiment", "request_parameters", "stdlib_http_transport",
           "DEFAULT_TICKERS", "DEFAULT_TOPICS", "ARTICLE_COLUMNS", "ASSOCIATION_COLUMNS"]
