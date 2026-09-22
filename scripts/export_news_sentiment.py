"""Export point-in-time FinBERT features from an already-scored news archive.

Run this offline on the machine holding the archive. It never fetches news or
scores articles itself, and does not affect the legacy benchmark CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from trading_system.data.news_sentiment import (
    build_news_decision_points,
    load_news_sentiment_export,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_tickers(args: argparse.Namespace) -> list[str]:
    if args.ticker_selection is None:
        return [name.strip() for name in args.tickers.split(",")]
    with args.ticker_selection.open(encoding="utf-8") as stream:
        selection = json.load(stream)
    if not isinstance(selection, dict) or not isinstance(selection.get("tickers"), list):
        raise ValueError("Ticker selection must contain a tickers list.")
    return [entry.get("ticker") if isinstance(entry, dict) else entry for entry in selection["tickers"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Daily market Parquet.")
    parser.add_argument("--scored-news", type=Path, required=True, help="Scored, provenance-dated news Parquet.")
    parser.add_argument("--coverage", type=Path, help="Optional point-in-time coverage journal Parquet.")
    parser.add_argument("--required-sources", help="Comma-separated source identifiers in the coverage journal.")
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument("--ticker-selection", type=Path, help="Benchmark JSON ticker selection.")
    ticker_group.add_argument("--tickers", help="Comma-separated explicit ticker universe.")
    parser.add_argument("--checkpoint", required=True, help="Exact FinBERT checkpoint/revision used for scoring.")
    parser.add_argument("--lookback", default="24h")
    parser.add_argument("--short-lookback", default="6h")
    parser.add_argument("--half-life", default="6h")
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet and sibling .manifest.json.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if (args.coverage is None) != (args.required_sources is None):
        raise ValueError("--coverage and --required-sources must be supplied together.")
    sources = None
    if args.required_sources is not None:
        sources = [name.strip() for name in args.required_sources.split(",")]
        if not sources or any(not name for name in sources) or len(sources) != len(set(sources)):
            raise ValueError("--required-sources must contain unique, non-empty identifiers.")

    # This optional dependency is installed only on the archive/GPU machine.
    try:
        from news_sentiment import export_sentiment_features
    except ImportError as exc:
        raise RuntimeError("Install the optional sentiment extra: pip install -e '.[sentiment]'") from exc

    selected_tickers = _selected_tickers(args)
    market = pd.read_parquet(args.data, columns=["date", "ticker"])
    points = build_news_decision_points(market, tickers=selected_tickers)
    scored_news = pd.read_parquet(
        args.scored_news, filters=[("ticker", "in", selected_tickers)]
    )
    coverage = pd.read_parquet(args.coverage) if args.coverage is not None else None
    identifiers = {
        "market_sha256": _sha256(args.data),
        "scored_news_sha256": _sha256(args.scored_news),
    }
    if args.coverage is not None:
        identifiers["coverage_sha256"] = _sha256(args.coverage)
    export_sentiment_features(
        scored_news,
        points,
        args.output,
        checkpoint=args.checkpoint,
        input_identifiers=identifiers,
        coverage=coverage,
        required_sources=sources,
        lookback=args.lookback,
        short_lookback=args.short_lookback,
        half_life=args.half_life,
        include_at_cutoff=False,
    )
    ready = load_news_sentiment_export(args.output)
    covered = int(ready.frame["source_available"].sum())
    print(f"sentiment_saved={args.output} decisions={len(ready.frame)} covered={covered}")
    if not covered:
        print("No covered decisions: sentiment stays masked until audited coverage is supplied.")


if __name__ == "__main__":
    main()
