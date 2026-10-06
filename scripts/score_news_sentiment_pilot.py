"""Score an observed news pilot with verified local FinBERT weights, offline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # noqa: F401
import pandas as pd

from trading_system.data.news_pilot_scoring import (
    FinBERTCheckpoint,
    export_news_pilot,
    file_sha256,
    load_local_finbert,
    score_news_pilot,
    verify_local_weights,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Collector articles/associations Parquet directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="Already downloaded immutable local checkpoint; no hub calls.")
    parser.add_argument("--model-repository", required=True)
    parser.add_argument("--model-revision", required=True, help="Immutable 40-hex Hugging Face commit SHA.")
    parser.add_argument("--model-weights-sha256", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data", type=Path, help="Optional market Parquet for strict session-midnight feature exports.")
    parser.add_argument("--tickers", help="Explicit comma-separated company universe; required for feature exports.")
    parser.add_argument("--preview-next-midnight", action="store_true",
                        help="Separate technical preview before/after collection, without prices, predictions or coverage claims.")
    parser.add_argument("--decision-start", help="Inclusive UTC date; filter existing market sessions before export.")
    parser.add_argument("--decision-end", help="Exclusive UTC date; filter existing market sessions before export.")
    parser.add_argument("--coverage", type=Path, help="Genuine company coverage journal, never generated from API success.")
    parser.add_argument("--required-sources")
    parser.add_argument("--macro-coverage", type=Path, help="Separate genuine macro coverage journal; topic/global/null ticker.")
    parser.add_argument("--macro-required-sources")
    parser.add_argument("--lookback", default="24h")
    parser.add_argument("--short-lookback", default="6h")
    parser.add_argument("--half-life", default="6h")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.data is not None or args.preview_next_midnight) != bool(args.tickers is not None):
        parser.error("Feature exports require --tickers and either --data or --preview-next-midnight.")
    if args.data is None and (args.decision_start is not None or args.decision_end is not None):
        parser.error("Decision-date filters require --data.")
    for coverage, sources in ((args.coverage, args.required_sources), (args.macro_coverage, args.macro_required_sources)):
        if (coverage is None) != (sources is None):
            parser.error("Coverage journals require their explicit required-source lists.")
        if coverage is not None and args.data is None:
            parser.error("Coverage journals require --data and --tickers.")
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("--batch-size and --max-length must be positive.")
    checkpoint = FinBERTCheckpoint(args.model_repository, args.model_revision, args.model_weights_sha256)
    verify_local_weights(args.model_dir, checkpoint.weights_sha256)
    settings = {
        "device": args.device, "batch_size": args.batch_size, "max_length": args.max_length,
        "config_sha256": file_sha256(args.model_dir / "config.json"),
        "local_metadata_staging": "copy metadata; add missing BERT model_type/tokenizer_class; hard-link verified weights or copy fallback",
        "metadata_sha256": {
            path.name: file_sha256(path) for path in args.model_dir.iterdir()
            if path.is_file() and path.suffix in (".json", ".txt", ".model")
        },
    }
    pilot = score_news_pilot(
        args.input_dir, args.output_dir, checkpoint=checkpoint, resume=args.resume,
        inference_settings=settings,
        progress=lambda message: print(message, flush=True),
        analyzer_factory=lambda: load_local_finbert(
            args.model_dir, checkpoint=checkpoint, device=args.device,
            batch_size=args.batch_size, max_length=args.max_length,
        ),
    )
    outputs = {}
    if args.data is not None:
        market = pd.read_parquet(args.data, columns=["date", "ticker"])
        dates = pd.to_datetime(market["date"], utc=True, format="mixed").dt.normalize()
        bounds = {}
        for name, value in (("decision_start", args.decision_start), ("decision_end", args.decision_end)):
            if value is not None:
                timestamp = pd.Timestamp(value)
                timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
                if timestamp != timestamp.normalize():
                    parser.error("Decision-date filters must be UTC midnight dates.")
                bounds[name] = timestamp
        if bounds.get("decision_start", pd.Timestamp.min.tz_localize("UTC")) > bounds.get("decision_end", pd.Timestamp.max.tz_localize("UTC")):
            parser.error("--decision-start must not follow --decision-end.")
        if "decision_start" in bounds:
            market = market.loc[dates.ge(bounds["decision_start"])]
        if "decision_end" in bounds:
            market = market.loc[dates.loc[market.index].lt(bounds["decision_end"])]
        identifiers = {"market_sha256": file_sha256(args.data)}
        identifiers.update({key: value.isoformat() for key, value in bounds.items()})
        for name, path in (("coverage_sha256", args.coverage), ("macro_coverage_sha256", args.macro_coverage)):
            if path is not None:
                identifiers[name] = file_sha256(path)
        outputs = export_news_pilot(
            pilot, market,
            tickers=[name.strip() for name in args.tickers.split(",")],
            coverage=pd.read_parquet(args.coverage) if args.coverage else None,
            required_sources=[name.strip() for name in args.required_sources.split(",")] if args.required_sources else None,
            macro_coverage=pd.read_parquet(args.macro_coverage) if args.macro_coverage else None,
            macro_required_sources=[name.strip() for name in args.macro_required_sources.split(",")] if args.macro_required_sources else None,
            input_identifiers=identifiers, lookback=args.lookback,
            short_lookback=args.short_lookback, half_life=args.half_life,
        )
    if args.preview_next_midnight:
        # Explicit synthetic decision points, not fabricated market observations.
        # Show the strict cutoff on today's and the next UTC midnight. Unknown
        # source coverage remains masked even though raw aggregated counts exist.
        cutoff = max(pilot.articles["available_at"].max(),
                     pilot.company["available_at"].max() if not pilot.company.empty else pilot.articles["available_at"].max(),
                     pilot.macro["available_at"].max() if not pilot.macro.empty else pilot.articles["available_at"].max()).normalize()
        tickers = [name.strip() for name in args.tickers.split(",")]
        preview_points = pd.DataFrame([{"date": date, "ticker": ticker}
                                       for date in (cutoff, cutoff + pd.Timedelta(days=1)) for ticker in tickers])
        preview_dir = pilot.output_dir / "technical-preview"
        preview_dir.mkdir(exist_ok=True)
        preview_outputs = export_news_pilot(
            replace(pilot, output_dir=preview_dir), preview_points, tickers=tickers,
            input_identifiers={"preview_kind": "synthetic_cutoffs_not_a_backtest", "price_data": "none"},
            lookback=args.lookback, short_lookback=args.short_lookback, half_life=args.half_life,
        )
        outputs.update({"technical_preview_" + name: path for name, path in preview_outputs.items()})
    print(json.dumps({
        "articles": len(pilot.articles), "company_associations": len(pilot.company),
        "macro_associations": len(pilot.macro), "newly_scored_texts": pilot.manifest["newly_scored_texts"],
        "resume_cache_status": pilot.manifest["resume_cache_status"],
        "outputs": {name: str(path) for name, path in outputs.items()},
        "coverage": "unknown unless genuine separately supplied coverage proves each cutoff window",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
