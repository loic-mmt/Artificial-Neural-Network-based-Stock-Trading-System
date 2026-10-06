#!/usr/bin/env python3
"""Download five bounded company RSS snapshots and the separate Fed macro feed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from trading_system.data.rss_news_collection import RSSPilotConfig, collect_rss_news


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/external/news-sentiment-rss-pilot")
    parser.add_argument("--max-entries-per-feed", type=int, default=100)
    parser.add_argument("--max-response-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--dry-run", action="store_true", help="No network, dependencies or output writes.")
    parser.add_argument("--resume", action="store_true", help="Verify cached snapshots; only failed feeds are fetched again.")
    args = parser.parse_args(argv)
    config = RSSPilotConfig(max_entries_per_feed=args.max_entries_per_feed,
                            max_response_bytes=args.max_response_bytes, timeout_seconds=args.timeout)
    result = collect_rss_news(config, args.output, dry_run=args.dry_run, resume=args.resume,
                             progress=lambda message: print(message, file=sys.stderr, flush=True))
    keys = ("complete", "unique_articles", "association_rows", "invocation_calls", "total_http_attempts", "files")
    print(json.dumps(result if args.dry_run else {key: result[key] for key in keys}, indent=2, sort_keys=True))
    return 0 if args.dry_run or result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
