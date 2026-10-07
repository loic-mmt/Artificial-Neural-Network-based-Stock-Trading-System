"""Offline FNSPID title study: explicit exploratory timing, never PIT coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trading_system.artifacts.experiment import hash_dataframe
from trading_system.artifacts.multimodal_study import atomic_write_json
from trading_system.data.news_pilot_scoring import FinBERTCheckpoint, file_sha256, _write_frame


MODEL_REPOSITORY = "yiyanghkust/finbert-tone"
MODEL_REVISION = "4921590d3c0c3832c0efea24c8381ce0bda7844b"
MODEL_SHA256 = "f31c2036e91c9854bcc35141d16669dd07b9726adfe391d1011bff1de7ea4b32"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "benchmark", "smoke", "polarity", "all",
                                           "inventory", "prepare-expanded", "expanded"), default="all")
    parser.add_argument("--fnspid-dir", type=Path, default=Path("data/external/fnspid"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/mt5_stocks_us_daily_clean.parquet"))
    parser.add_argument("--tickers", default="AAPL,JPM,XOM,WMT,JNJ")
    parser.add_argument("--start", default="2020-03-09")
    parser.add_argument("--end", default="2024-01-01", help="Exclusive end of the frozen price/calendar study.")
    parser.add_argument("--delay-hours", type=float, default=24., help="Exploratory publication delay, not a PIT timestamp.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/derived/fnspid/pilot-v1"))
    parser.add_argument("--output-dir", type=Path, help="Defaults to a separate full/smoke comparison directory.")
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/finbert-tone-4921590"))
    parser.add_argument("--model-parameter-sets", type=Path, default=Path("configs/benchmark/gru_market_context.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16, help="FinBERT inference batch size (not training date batch size).")
    parser.add_argument("--shuffle-seed", type=int, default=314159, help="Fixed article-permutation seed for --stage polarity.")
    parser.add_argument("--inventory-dir", type=Path, default=Path("data/derived/fnspid/inventory-v1"))
    parser.add_argument("--expanded-prepared-dir", type=Path, default=Path("data/derived/fnspid/expanded-v1"))
    parser.add_argument("--min-observed-train-days", type=int, default=60)
    parser.add_argument("--min-train-observed-fraction", type=float, default=.1)
    parser.add_argument("--permutation-seeds", default="314159,271828,161803,57721,141421",
                        help="At least two distinct corpus permutation seeds; independent of model seeds 1,7,19.")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--score-reuse-from", type=Path, action="append",
                       help="Completed compatible FNSPID scoring directory; defaults to the pilot cache if present.")
    cache.add_argument("--no-score-cache-reuse", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Read-only plan/validation; no import, inference or training.")
    return parser


def _settings(args):
    tickers = tuple(item.strip().upper() for item in args.tickers.split(","))
    if not tickers or any(not item for item in tickers) or len(set(tickers)) != len(tickers):
        raise ValueError("Choose non-empty unique tickers.")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start.tzinfo is not None or end.tzinfo is not None or start != start.normalize() or end != end.normalize() or start >= end:
        raise ValueError("--start/--end must be ordered timezone-naive session dates; end is exclusive.")
    if not 0 <= args.delay_hours < float("inf") or args.batch_size < 1:
        raise ValueError("Use finite non-negative delay and positive inference batch size.")
    if not 0 <= args.shuffle_seed < 2**32:
        raise ValueError("--shuffle-seed must be in [0, 2**32).")
    return {"schema_version": 1, "protocol": "fnspid-exploratory", "content_kind": "title",
            "tickers": list(tickers), "start": start.isoformat(), "end_exclusive": end.isoformat(),
            "delay_hours": args.delay_hours, "lookback_hours": 24.,
            "checkpoint": FinBERTCheckpoint(MODEL_REPOSITORY, MODEL_REVISION, MODEL_SHA256).identifier,
            "historical_coverage_claim": False,
            "warmup_policy": "Consumed inside the clipped price calendar by the existing train-only preparation; no targets outside the period.",
            "price_context_policy": "Frozen selected-price frame; same embedded market proxies and derived sector/market pool for all candidates."}


def _prepared_paths(args):
    root = args.prepared_dir
    return root / "prices-matched.parquet", root / "tickers-matched.json", root / "company_daily.parquet"


def prepare(args):
    from trading_system.data.fnspid_import import import_fnspid
    from trading_system.data.fnspid_scoring import score_fnspid
    from trading_system.data.fnspid_export import export_fnspid_sentiment, load_fnspid_sentiment_export

    settings = _settings(args)
    root = args.prepared_dir
    prices, selection, daily = _prepared_paths(args)
    price_manifest = root / "prices.manifest.json"
    identity = {"settings": settings, "source_prices_sha256": file_sha256(args.data)}
    market = pd.read_parquet(args.data)
    dates = pd.to_datetime(market.date, utc=True, errors="raise")
    selected = market.loc[market.ticker.isin(settings["tickers"]) &
                          dates.ge(pd.Timestamp(settings["start"], tz="UTC")) &
                          dates.lt(pd.Timestamp(settings["end_exclusive"], tz="UTC"))].copy()
    if selected.empty or set(selected.ticker) != set(settings["tickers"]):
        raise ValueError("Frozen price period is empty or misses selected tickers.")
    selected = selected.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    if selected.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate selected price keys.")
    if price_manifest.exists():
        saved = json.loads(price_manifest.read_text(encoding="utf-8"))
        if not args.resume or saved.get("identity") != identity:
            raise ValueError("Prepared study changed or exists without --resume; choose a new --prepared-dir.")
        if saved.get("prices_sha256") != file_sha256(prices) or saved.get("selection_sha256") != file_sha256(selection):
            raise ValueError("Prepared price/universe checksum mismatch.")
    else:
        if any(path.exists() for path in (prices, selection)):
            raise FileExistsError("Unregistered preparation artifacts; choose a new --prepared-dir.")
        root.mkdir(parents=True, exist_ok=True)
        _write_frame(prices, selected, {"protocol": "fnspid-exploratory", "settings": settings})
        atomic_write_json(selection, {"tickers": settings["tickers"]})
        atomic_write_json(price_manifest, {"identity": identity, "prices_sha256": file_sha256(prices),
                                          "selection_sha256": file_sha256(selection)})
    # Include the full trailing news window before the first price decision.
    news_start = pd.Timestamp(settings["start"]) - pd.Timedelta(hours=args.delay_hours + 24.)
    csv_paths = [args.fnspid_dir / "Stock_news" / name for name in ("All_external.csv", "nasdaq_exteral_data.csv")]
    print("FNSPID import: frozen CSVs -> title/association audit", flush=True)
    imported = import_fnspid(csv_paths, root / "import", tickers=settings["tickers"],
                             start=news_start.isoformat(), end=settings["end_exclusive"],
                             dataset_info=args.fnspid_dir / "dataset-info.json",
                             resume=args.resume and any((root / "import").glob("*")))
    print("FNSPID score: unique titles -> resumable FinBERT shards", flush=True)
    scored = score_fnspid(imported["articles_path"], root / "scoring",
                          checkpoint=FinBERTCheckpoint(MODEL_REPOSITORY, MODEL_REVISION, MODEL_SHA256),
                          model_dir=args.model_dir, device=args.device, batch_size=args.batch_size,
                          resume=args.resume)
    identifiers = {"domain": "ticker", "settings": settings,
                   "source_prices_sha256": identity["source_prices_sha256"],
                   "import_report_sha256": file_sha256(imported["report_path"]),
                   "scoring_manifest_sha256": file_sha256(scored["manifest_path"])}
    if daily.exists() or daily.with_suffix(".manifest.json").exists():
        if not args.resume:
            raise FileExistsError("Daily export exists; use --resume with unchanged inputs.")
        ready = load_fnspid_sentiment_export(daily)
        parameters = {"delay_hours": args.delay_hours, "lookback_hours": 24.,
                      "short_lookback_hours": 6., "half_life_hours": 6.}
        if (ready.manifest.get("input_identifiers") != identifiers or
                ready.manifest.get("checkpoint") != settings["checkpoint"] or
                any(ready.manifest.get("aggregation", {}).get(name) != value
                    for name, value in parameters.items()) or
                ready.manifest["inputs"]["scored"]["sha256"] != file_sha256(scored["scored_path"]) or
                ready.manifest["inputs"]["market"]["sha256"] != hash_dataframe(selected)):
            raise ValueError("Daily export differs from current prepared/scored study.")
    else:
        export_fnspid_sentiment(scored["scored_path"], selected, daily,
                                checkpoint=settings["checkpoint"], input_identifiers=identifiers,
                                delay_hours=args.delay_hours)
        ready = load_fnspid_sentiment_export(daily)
    atomic_write_json(root / "preparation.manifest.json", {"settings": settings, "artifacts": {
        "prices": {"path": str(prices), "sha256": file_sha256(prices)},
        "selection": {"path": str(selection), "sha256": file_sha256(selection)},
        "daily": {"path": str(daily), "sha256": file_sha256(daily)},
        "daily_manifest": {"path": str(daily.with_suffix('.manifest.json')), "sha256": file_sha256(daily.with_suffix('.manifest.json'))},
    }})
    print(f"FNSPID ready: decisions={len(ready.frame)} observed={int(ready.frame.source_available.sum())} protocol=EXPLORATORY", flush=True)
    return ready


def benchmark_arguments(args, *, smoke=False, polarity=False):
    prices, selection, daily = _prepared_paths(args)
    target = args.output_dir or (Path("artifacts/comparisons/fnspid-news-polarity-controls") if polarity else
                                Path("artifacts/comparisons/fnspid-news-pilot-" + ("smoke" if smoke else "full")))
    candidates = ("gru,gru_activity,gru_features,gru_features_shuffled,gru_features_neutralized" if polarity else
                  "gru,gru_features" if smoke else "gru,gru_activity,gru_features,sentiment,gru_sentiment_mean")
    tokens = [
        "--data", str(prices), "--ticker-selection", str(selection), "--preset", "multi_ticker_long_short",
        "--models", "gru", "--model-parameter-sets", str(args.model_parameter_sets),
        "--losses", "combined", "--combined-weights", "0.25", "--loss-cost-bps", "5",
        "--selection-metric", "regularized_sharpe", "--context-len", "60", "--position-mode", "long_short",
        "--execution-delay", "1", "--train-ratio", "0.7", "--val-ratio", "0.15",
        "--label-method", "triple-barrier", "--label-max-holding", "10", "--label-vol-window", "20",
        "--label-volatility-estimator", "atr", "--label-profit-barrier", "0.75", "--label-stop-barrier", "0.75",
        "--label-event-filter", "cusum", "--label-cusum-threshold", "0.5", "--label-between-events", "hold",
        "--label-cost-bps", "5", "--feature-set", "expanded", "--feature-groups", "technical,market,sector",
        "--no-external-features", "--overfitting-control", "--overfitting-max-features", "32",
        "--overfitting-max-feature-correlation", "0.95", "--news-sentiment-export", str(daily),
        "--news-protocol", "fnspid-exploratory", "--sentiment-candidates",
        candidates,
        "--date-batch-size", "16", "--cv-folds", "2" if smoke else "3", "--cv-gap-bars", "5",
        "--cv-score", "regularized_sharpe", "--seeds", "42" if smoke else "1,7,19",
        "--device", args.device, "--output-dir", str(target), "--fail-fast",
    ]
    if polarity:
        tokens.extend(("--news-scored-articles", str(args.prepared_dir / "scoring" / "scored_company.parquet"),
                       "--news-shuffle-seed", str(args.shuffle_seed)))
    return tokens


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stage in ("inventory", "prepare-expanded", "expanded"):
        from trading_system.pipelines.fnspid_expanded_study import run_expanded_stage

        return run_expanded_stage(args)
    settings = _settings(args)
    if args.stage in ("prepare", "all") and not args.dry_run:
        prepare(args)
    if args.stage == "prepare" and not args.dry_run:
        return
    manifest_path = args.prepared_dir / "preparation.manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Prepare first: use --stage prepare (or --stage all), without --dry-run.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("settings") != settings:
        raise ValueError("Study settings differ from preparation; choose the matching flags or a new --prepared-dir.")
    prices, selection, daily = _prepared_paths(args)
    expected = {"prices": prices, "selection": selection, "daily": daily,
                "daily_manifest": daily.with_suffix(".manifest.json")}
    records = manifest.get("artifacts")
    if not isinstance(records, dict) or set(records) != set(expected):
        raise ValueError("Preparation manifest must register all four study artifacts.")
    for name, target in expected.items():
        record = records[name]
        if not isinstance(record, dict) or Path(record.get("path", "")).resolve() != target.resolve():
            raise ValueError(f"Preparation artifact path mismatch: {name}.")
        if file_sha256(target) != record.get("sha256"):
            raise ValueError(f"Prepared artifact checksum mismatch: {name}.")
    from trading_system.pipelines.compare_news_sentiment import main as compare
    tokens = benchmark_arguments(args, smoke=args.stage == "smoke", polarity=args.stage == "polarity")
    if args.resume and (args.output_dir or Path(tokens[tokens.index("--output-dir") + 1])).exists():
        tokens.append("--resume")
    if args.stage == "polarity":
        # The runner validates the complete TRAIN plan before creating output
        # or fitting a model. Avoid a second identical preparation here.
        return compare([*tokens, "--dry-run"] if args.dry_run else tokens)
    # Validation always precedes the first training; --dry-run never trains.
    result = compare([*tokens, "--dry-run"])
    if not args.dry_run:
        result = compare(tokens)
    return result
