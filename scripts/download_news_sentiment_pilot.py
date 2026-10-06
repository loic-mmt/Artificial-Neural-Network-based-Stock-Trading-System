#!/usr/bin/env python3
"""Collect a strictly bounded current news snapshot; never historical PIT coverage."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

# Source-tree bootstrap, without _bootstrap's plotting-cache write on dry runs.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_system.data.news_collection import PilotConfig, collect_news_sentiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "external" / "news_sentiment_pilot")
    parser.add_argument("--ledger", type=Path, default=Path(os.environ.get("NEWS_SENTIMENT_QUOTA_DB", str(ROOT / ".cache" / "alpha_vantage_quota.sqlite3"))),
                        help="Shared with news_sentiment's NEWS_SENTIMENT_QUOTA_DB; reuse for every client.")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--start", default="2026-09-01T00:00:00Z")
    parser.add_argument("--end", default="2026-10-01T00:00:00Z", help="Exclusive end, minute-aligned.")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--max-calls", type=int, default=20, help="Per invocation, never greater than 25.")
    parser.add_argument("--calls-already-used", type=int, default=0, help="External calls already made today, outside this shared ledger.")
    parser.add_argument("--max-unique-articles", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=1000,
                        help="Vendor page size (1..1000); --limit 100 keeps 20-call/2000-row pilot fair across all streams.")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true", help="No key, network, or state writes.")
    parser.add_argument("--resume", action="store_true", help="Validate configuration and skip completed requests.")
    return parser


def _local_key(path: Path) -> str | None:
    """Read only the two documented names; never execute or print .env values."""
    names = ("ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY")
    for name in names:
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    if not path.is_file():
        return None
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        name, separator, value = line.partition("=")
        if separator and name.strip() in names:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            else:
                value = value.partition(" #")[0].strip()
            values[name.strip()] = value
    return next((values[name] for name in names if values.get(name, "").strip()), None)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = PilotConfig(start=datetime.fromisoformat(args.start.replace("Z", "+00:00")),
                             end=datetime.fromisoformat(args.end.replace("Z", "+00:00")),
                             window_days=args.window_days, max_calls=args.max_calls,
                             calls_already_used=args.calls_already_used,
                             max_unique_articles=args.max_unique_articles, limit=args.limit, timeout_seconds=args.timeout)
        key = None if args.dry_run else _local_key(args.env_file)
        result = collect_news_sentiment(config, args.output, api_key=key, ledger_path=args.ledger,
                                        dry_run=args.dry_run, resume=args.resume)
    except Exception:
        # No arbitrary error strings or tracebacks: an upstream URL may hold a key.
        print("Collection refused or failed. Check configuration, key presence, resume state, and local paths.", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps({name: result[name] for name in (
            "complete", "incomplete_reason", "unique_articles", "association_rows", "invocation_calls",
            "total_http_attempts", "quota_usage", "files")}, indent=2, sort_keys=True))
        if any(record.get("status") == "error" for record in result.get("requests", [])):
            print("Provider/transport error: collection stopped without retry. Inspect collection.manifest.json and redacted raw evidence.", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
