"""Bounded, title-only FNSPID import with explicitly exploratory provenance.

The dataset's publication dates and stock symbols are declarations, not evidence
of historical availability or company relevance. Text revisions are retained;
their publication chronology cannot be reconstructed from a current snapshot.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
from itertools import groupby
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError

from .news_collection import _canonical_url, _output_lock
from .news_pilot_scoring import _write_json, file_sha256


PROTOCOL = "fnspid_exploratory"
SCHEMA_VERSION = "1.0"
ARTICLE_COLUMNS = (
    "news_id", "text_hash", "text", "ticker", "published_at", "collected_at",
    "source", "url", "date_raw", "association_kind", "content_kind", "source_files",
    "raw_ticker", "title_raw", "url_raw", "date_raw_values", "raw_ticker_values",
    "title_raw_values", "url_raw_values", "source_values", "version_timing_uncertain",
    "publication_timing_uncertain", "url_version_count",
)
_LIST_COLUMNS = {
    "source_files", "date_raw_values", "raw_ticker_values", "title_raw_values",
    "url_raw_values", "source_values",
}
_SCHEMA = pa.schema([
    pa.field(column, pa.list_(pa.string()) if column in _LIST_COLUMNS else
             pa.timestamp("ns", tz="UTC") if column in ("published_at", "collected_at") else
             pa.bool_() if column.endswith("_uncertain") else
             pa.int64() if column == "url_version_count" else pa.string())
    for column in ARTICLE_COLUMNS
])


class _HashingReader(io.RawIOBase):
    """Hash the exact bytes consumed by the strict, newline-aware CSV reader."""

    def __init__(self, stream: Any):
        self.stream = stream
        self.digest = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        count = self.stream.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


def _timestamp(raw: str, zone: ZoneInfo) -> pd.Timestamp:
    value = pd.Timestamp(raw)
    if pd.isna(value):
        raise ValueError("Missing publication date.")
    if value.tzinfo is None:
        value = value.tz_localize(zone, ambiguous="raise", nonexistent="raise")
    value = value.tz_convert("UTC")
    # Enforce the nanosecond representation used in the output schema.
    _ = value.value
    return value


def _iso(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _columns(header: list[str], path: Path) -> dict[str, int | None]:
    normalized = [value.strip().lower() for value in header]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Duplicate CSV columns in {path}.")
    aliases = {
        "date": ("date",), "title": ("article_title",), "ticker": ("stock_symbol",),
        "url": ("url", "article_url", "link"),
        "source": ("publisher", "source", "source_domain"),
    }
    mapped = {name: next((normalized.index(alias) for alias in names if alias in normalized), None)
              for name, names in aliases.items()}
    missing = [name for name in ("date", "title", "ticker") if mapped[name] is None]
    if missing:
        raise ValueError(f"Missing required FNSPID columns in {path}: {missing}.")
    return mapped


def _month_labels(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    first = start.tz_localize(None).to_period("M")
    last = (end - pd.Timedelta(nanoseconds=1)).tz_localize(None).to_period("M")
    return [str(month) for month in pd.period_range(first, last, freq="M")]


def _existing(destination: Path, expected: dict[str, Any], *, resume: bool) -> dict[str, Any] | None:
    report_path = destination / "import-report.json"
    articles_path = destination / "articles.parquet"
    manifest_path = destination / "articles.manifest.json"
    paths = (report_path, articles_path, manifest_path)
    if not any(path.exists() for path in paths):
        if resume:
            raise ValueError("Cannot resume: no completed FNSPID import exists.")
        return None
    if not resume:
        raise FileExistsError("FNSPID outputs already exist; use resume with unchanged inputs.")
    if not all(path.is_file() for path in paths):
        raise ValueError("Cannot resume incomplete FNSPID import artifacts.")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise ValueError("Cannot resume unreadable FNSPID manifests.") from error
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Cannot resume: FNSPID {key} changed.")
    if report.get("historical_availability_proven") is not False or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Cannot resume a changed FNSPID exploratory protocol.")
    if report.get("status") != "complete" or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Cannot resume an incomplete or incompatible FNSPID import.")
    checksum = file_sha256(articles_path)
    if report.get("output", {}).get("sha256") != checksum or manifest.get("parquet_sha256") != checksum:
        raise ValueError("Cannot resume corrupted FNSPID articles.")
    if report.get("output", {}).get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Cannot resume a changed FNSPID article manifest.")
    return {"articles_path": articles_path, "report_path": report_path, "report": report}


def _import_fnspid_unlocked(
    csv_paths: Sequence[str | Path], destination: str | Path, *,
    tickers: Sequence[str], start: str, end: str, dataset_info: str | Path,
    timezone_assumption: str = "UTC", chunksize: int = 100_000, resume: bool = False,
) -> dict[str, Any]:
    """Import declared dates in ``[start, end)``; retain only actual titles.

    Source parsing, SQLite deduplication, and Parquet writing use bounded chunks.
    Resume validates a *completed* import, its immutable sources, settings, and
    artifact checksums. Incomplete outputs are preserved and never overwritten.
    Naive dates use the explicitly recorded timezone assumption. Ambiguous or
    nonexistent local times are counted as invalid rather than guessed.
    """
    if isinstance(csv_paths, (str, Path)) or not csv_paths:
        raise ValueError("csv_paths must contain individual CSV paths.")
    paths = sorted({Path(path).resolve() for path in csv_paths}, key=str)
    if len(paths) != len(csv_paths) or any(not path.is_file() for path in paths):
        raise ValueError("CSV inputs must be distinct existing files.")
    if isinstance(tickers, str) or not tickers or any(not isinstance(t, str) or not t.strip() for t in tickers):
        raise ValueError("tickers must contain non-empty individual symbols.")
    symbols = sorted({ticker.strip().upper() for ticker in tickers})
    if any(re.fullmatch(r"[A-Z0-9][A-Z0-9.\-^=]*", ticker) is None for ticker in symbols):
        raise ValueError("tickers must contain individual declared stock symbols.")
    if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer.")
    zone = ZoneInfo(timezone_assumption)
    begin, finish = _timestamp(start, zone), _timestamp(end, zone)
    if begin >= finish:
        raise ValueError("start must precede the exclusive end.")
    metadata_path = Path(dataset_info).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    revision = metadata.get("sha") if isinstance(metadata, dict) else None
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("dataset_info requires an immutable lowercase 40-hex sha revision.")
    inputs = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in paths]
    settings = {
        "tickers": symbols, "start": _iso(begin), "end": _iso(finish), "end_exclusive": True,
        "timezone_assumption": timezone_assumption, "chunksize": chunksize,
        "content_kind": "title", "title_normalization": "collapse_whitespace_only",
        "ticker_mapping": "strip_uppercase_no_historical_alias_inference",
        "association_kind": "dataset_stock_symbol_unverified", "language_policy": "not_inferred_or_filtered",
        "publication_conflict_policy": "earliest_declared_date_retain_all_raw_values_and_flag",
    }
    expected = {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "source_revision": revision,
        "input_files": inputs, "settings": settings,
        "dataset_info": {"path": str(metadata_path), "sha256": file_sha256(metadata_path)},
    }
    output = Path(destination).resolve()
    if output in paths or output == metadata_path:
        raise ValueError("The output directory cannot overwrite an input file.")
    previous = _existing(output, expected, resume=resume)
    if previous is not None:
        return previous
    output.mkdir(parents=True, exist_ok=True)
    collected = pd.Timestamp(datetime.now(timezone.utc))
    counts: Counter[str] = Counter({name: 0 for name in (
        "input_rows", "filtered_ticker_rows", "requested_ticker_rows", "invalid_date_rows",
        "empty_title_rows", "rejected_rows", "filtered_period_rows", "missing_or_invalid_url_rows", "accepted_rows",
    )})
    naive_dates = 0
    source_schemas: dict[str, Any] = {}
    month_counts: dict[str, Counter[str]] = {ticker: Counter() for ticker in symbols}
    raw_month_counts: dict[str, Counter[str]] = {ticker: Counter() for ticker in symbols}
    ticker_counts: Counter[str] = Counter()
    coverage_bounds: dict[str, list[int]] = {}

    # The spool contains selected rows only; no complete input CSV is resident.
    with tempfile.TemporaryDirectory(prefix=".fnspid-import-", dir=output) as staging:
        connection = sqlite3.connect(Path(staging) / "selected.sqlite3")
        try:
            connection.executescript("""
                PRAGMA journal_mode = OFF;
                PRAGMA synchronous = OFF;
                PRAGMA temp_store = FILE;
                CREATE TABLE articles (
                    news_id TEXT PRIMARY KEY, text_hash TEXT NOT NULL, text TEXT NOT NULL, url TEXT NOT NULL
                );
                CREATE TABLE associations (
                    news_id TEXT NOT NULL, ticker TEXT NOT NULL, PRIMARY KEY(news_id, ticker)
                );
                CREATE TABLE occurrences (
                    news_id TEXT NOT NULL, ticker TEXT NOT NULL, published_ns INTEGER NOT NULL,
                    date_raw TEXT NOT NULL, raw_ticker TEXT NOT NULL, title_raw TEXT NOT NULL,
                    source TEXT NOT NULL, url_raw TEXT NOT NULL, source_file TEXT NOT NULL,
                    UNIQUE(news_id, ticker, published_ns, date_raw, raw_ticker, title_raw, source, url_raw, source_file)
                );
            """)
            old_limit = csv.field_size_limit()
            csv.field_size_limit(max(old_limit, 32 * 1024 * 1024))
            try:
                for path, evidence in zip(paths, inputs):
                    with path.open("rb") as raw:
                        hashing = _HashingReader(raw)
                        with io.TextIOWrapper(io.BufferedReader(hashing), encoding="utf-8-sig", newline="") as stream:
                            reader = csv.reader(stream, strict=True)
                            try:
                                header = next(reader)
                            except StopIteration as error:
                                raise ValueError(f"Empty FNSPID CSV: {path}.") from error
                            mapping = _columns(header, path)
                            source_schemas[str(path)] = {"columns": header, "mapping": mapping}
                            pending_articles, pending_associations, pending_occurrences = [], [], []
                            selected_timestamps: dict[str, pd.Timestamp | None] = {}
                            for row in reader:
                                counts["input_rows"] += 1
                                if counts["input_rows"] % chunksize == 0:
                                    _flush(connection, pending_articles, pending_associations, pending_occurrences)
                                    selected_timestamps.clear()
                                if len(row) != len(header):
                                    raise ValueError(f"Malformed CSV row in {path} at physical line {reader.line_num}: "
                                                     f"expected {len(header)} fields, received {len(row)}.")
                                raw_ticker = row[mapping["ticker"]]
                                ticker = raw_ticker.strip().upper()
                                if ticker not in month_counts:
                                    counts["filtered_ticker_rows"] += 1
                                    continue
                                counts["requested_ticker_rows"] += 1
                                raw_date = row[mapping["date"]]
                                if raw_date not in selected_timestamps:
                                    try:
                                        selected_timestamps[raw_date] = _timestamp(raw_date, zone)
                                    except (ValueError, TypeError, OverflowError, AmbiguousTimeError, NonExistentTimeError):
                                        selected_timestamps[raw_date] = None
                                published = selected_timestamps[raw_date]
                                title_raw = row[mapping["title"]]
                                text = " ".join(title_raw.split())
                                if published is None:
                                    counts["invalid_date_rows"] += 1
                                if not text:
                                    counts["empty_title_rows"] += 1
                                if published is None or not text:
                                    counts["rejected_rows"] += 1
                                    continue
                                if not begin <= published < finish:
                                    counts["filtered_period_rows"] += 1
                                    continue
                                # This is a declared timezone, never an availability proof.
                                if pd.Timestamp(raw_date).tzinfo is None:
                                    naive_dates += 1
                                url_raw = row[mapping["url"]] if mapping["url"] is not None else ""
                                url = _canonical_url(url_raw)
                                source = row[mapping["source"]].strip() if mapping["source"] is not None else ""
                                source = source or "unknown"
                                if not url:
                                    counts["missing_or_invalid_url_rows"] += 1
                                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                                identity = ["url", url, text_hash] if url else ["fallback", source, _iso(published), text_hash]
                                news_id = _digest(identity)
                                pending_articles.append((news_id, text_hash, text, url))
                                pending_associations.append((news_id, ticker))
                                pending_occurrences.append((news_id, ticker, published.value, raw_date, raw_ticker,
                                                            title_raw, source, url_raw, str(path)))
                                counts["accepted_rows"] += 1
                                raw_month_counts[ticker][published.strftime("%Y-%m")] += 1
                            _flush(connection, pending_articles, pending_associations, pending_occurrences)
                        if hashing.digest.hexdigest() != evidence["sha256"]:
                            raise ValueError(f"FNSPID input changed while importing: {path}.")
            except csv.Error as error:
                raise ValueError(f"Malformed FNSPID CSV: {error}.") from error
            finally:
                csv.field_size_limit(old_limit)

            connection.executescript("""
                CREATE INDEX occurrence_keys ON occurrences(news_id, ticker);
                CREATE TEMP TABLE url_versions AS
                    SELECT url, COUNT(*) AS version_count FROM articles WHERE url != '' GROUP BY url;
                CREATE INDEX version_urls ON url_versions(url);
                CREATE TEMP TABLE declared_dates AS
                    SELECT news_id, COUNT(DISTINCT published_ns) AS date_count FROM occurrences GROUP BY news_id;
                CREATE INDEX date_keys ON declared_dates(news_id);
            """)
            counts["unique_articles"] = connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            counts["output_associations"] = connection.execute("SELECT COUNT(*) FROM associations").fetchone()[0]
            counts["duplicate_associations"] = counts["accepted_rows"] - counts["output_associations"]
            counts["urls_with_multiple_title_versions"] = connection.execute(
                "SELECT COUNT(*) FROM url_versions WHERE version_count > 1").fetchone()[0]
            counts["articles_with_publication_conflicts"] = connection.execute(
                "SELECT COUNT(*) FROM declared_dates WHERE date_count > 1").fetchone()[0]
            counts["articles_with_multiple_tickers"] = connection.execute(
                "SELECT COUNT(*) FROM (SELECT news_id FROM associations GROUP BY news_id HAVING COUNT(*) > 1)").fetchone()[0]
            cursor = connection.execute("""
                SELECT o.*, a.text_hash, a.text, a.url, COALESCE(v.version_count, 1), d.date_count
                FROM occurrences o JOIN articles a USING(news_id)
                LEFT JOIN url_versions v ON a.url = v.url JOIN declared_dates d USING(news_id)
                ORDER BY o.news_id, o.ticker, o.published_ns, o.date_raw, o.raw_ticker,
                         o.title_raw, o.source, o.url_raw, o.source_file
            """)
            articles_path = output / "articles.parquet"
            temporary = Path(staging) / "articles.parquet"
            rows = []
            with pq.ParquetWriter(temporary, _SCHEMA, compression="snappy") as writer:
                for _, occurrences in groupby(cursor, key=lambda row: row[:2]):
                    variants = list(occurrences)
                    first = variants[0]
                    published = pd.Timestamp(first[2], tz="UTC")
                    ticker = first[1]
                    values = lambda index: sorted({row[index] for row in variants})
                    rows.append({
                        "news_id": first[0], "ticker": ticker, "text_hash": first[9], "text": first[10],
                        "url": first[11], "published_at": published, "collected_at": collected,
                        "date_raw": first[3], "raw_ticker": first[4], "title_raw": first[5],
                        "source": first[6], "url_raw": first[7], "source_files": values(8),
                        "date_raw_values": values(3), "raw_ticker_values": values(4),
                        "title_raw_values": values(5), "source_values": values(6), "url_raw_values": values(7),
                        "association_kind": "dataset_stock_symbol_unverified", "content_kind": "title",
                        "version_timing_uncertain": first[12] > 1,
                        "publication_timing_uncertain": first[13] > 1, "url_version_count": first[12],
                    })
                    month_counts[ticker][published.strftime("%Y-%m")] += 1
                    ticker_counts[ticker] += 1
                    published_ns = published.value
                    if ticker not in coverage_bounds:
                        coverage_bounds[ticker] = [published_ns, published_ns]
                    else:
                        coverage_bounds[ticker] = [min(coverage_bounds[ticker][0], published_ns),
                                                   max(coverage_bounds[ticker][1], published_ns)]
                    if len(rows) >= chunksize:
                        writer.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA))
                        rows.clear()
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA))
            checksum = file_sha256(temporary)
            os.replace(temporary, articles_path)
        finally:
            connection.close()

    months = _month_labels(begin, finish)
    report = {
        **expected, "status": "complete", "historical_availability_proven": False,
        "collected_at": _iso(collected), "counts": dict(sorted(counts.items())),
        "naive_date_rows_assumed_timezone": naive_dates, "source_schemas": source_schemas,
        "coverage": {ticker: {
            "associations": ticker_counts[ticker],
            "first": _iso(pd.Timestamp(coverage_bounds[ticker][0], tz="UTC")) if ticker in coverage_bounds else None,
            "last": _iso(pd.Timestamp(coverage_bounds[ticker][1], tz="UTC")) if ticker in coverage_bounds else None,
            "counts_by_month": {month: month_counts[ticker][month] for month in months},
            "raw_counts_by_month": {month: raw_month_counts[ticker][month] for month in months},
            "months_without_rows": [month for month in months if not month_counts[ticker][month]],
            "complete_historical_coverage_proven": False,
        } for ticker in symbols},
        "limitations": {
            "availability": "Publication plus an assumed delay is exploratory; collection occurs now.",
            "content": "Only Article_title is used; generated Lsa/Luhn/Textrank/Lexrank summaries are excluded.",
            "versions": "Current snapshot title versions have no proven historical version-availability dates.",
            "association": "Stock_symbol declares an unverified association, not relevance or historical membership.",
            "coverage": "Months without rows do not prove covered windows with no news.",
            "language": "No language inference or filter; the dataset does not establish language eligibility.",
            "ticker_mapping": "No inferred alias/change mapping; current-universe survivorship remains unaudited.",
        },
        "output": {"path": str(articles_path), "sha256": checksum, "rows": counts["output_associations"]},
    }
    manifest_path = output / "articles.manifest.json"
    _write_json(manifest_path, {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "historical_availability_proven": False, "source_revision": revision,
        "parquet_sha256": checksum, "rows": counts["output_associations"],
        "columns": list(ARTICLE_COLUMNS), "settings": settings,
    })
    report["output"]["manifest_sha256"] = file_sha256(manifest_path)
    report_path = output / "import-report.json"
    _write_json(report_path, report)
    return {"articles_path": articles_path, "report_path": report_path, "report": report}


def import_fnspid(
    csv_paths: Sequence[str | Path], destination: str | Path, *,
    tickers: Sequence[str], start: str, end: str, dataset_info: str | Path,
    timezone_assumption: str = "UTC", chunksize: int = 100_000, resume: bool = False,
) -> dict[str, Any]:
    """Import a title-only, exploratory FNSPID snapshot under an output lock.

    The interval is start-inclusive and end-exclusive. Resume accepts completed,
    checksum-validated artifacts only; source snapshots and assumptions must match.
    """
    output = Path(destination).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("The FNSPID output must be a directory, not an input file.")
    output.mkdir(parents=True, exist_ok=True)
    # Reuse the existing portable OS lock; a stale lock file is harmless after a
    # crash because the operating system releases the actual lock on process exit.
    with _output_lock(output):
        return _import_fnspid_unlocked(
            csv_paths, output, tickers=tickers, start=start, end=end, dataset_info=dataset_info,
            timezone_assumption=timezone_assumption, chunksize=chunksize, resume=resume,
        )


def _flush(connection: sqlite3.Connection, articles: list, associations: list, occurrences: list) -> None:
    connection.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?)", articles)
    connection.executemany("INSERT OR IGNORE INTO associations VALUES (?, ?)", associations)
    connection.executemany("INSERT OR IGNORE INTO occurrences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", occurrences)
    connection.commit()
    articles.clear()
    associations.clear()
    occurrences.clear()


__all__ = ["import_fnspid", "ARTICLE_COLUMNS", "PROTOCOL"]
