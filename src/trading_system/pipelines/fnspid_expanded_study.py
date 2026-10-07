"""Inventory-first extension; no repeated CSV parsing or reference training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from trading_system.artifacts.experiment import hash_dataframe
from trading_system.artifacts.multimodal_study import atomic_write_json
from trading_system.data.news_pilot_scoring import FinBERTCheckpoint, file_sha256, _write_frame
from trading_system.pipelines import fnspid_news_study as pilot


def _selection_bounds(args):
    """Match actual earliest TRAIN warmup/gap, without reading OUTER features."""
    from trading_system.pipelines.compare_gnn_graphs import prepare_graph_run
    from trading_system.experiments.graph_ablation import _prepare
    from trading_system.data.purged_cv import expanding_calendar_folds

    own = {"--ticker-selection", "--news-sentiment-export", "--news-protocol",
           "--sentiment-candidates", "--news-scored-articles", "--news-shuffle-seed"}
    tokens = pilot.benchmark_arguments(args, polarity=True)
    forwarded, index = [], 0
    while index < len(tokens):
        flag = tokens[index]
        if flag in own:
            index += 2
        elif flag == "--data":
            forwarded.extend((flag, str(args.data)))
            index += 2
        else:
            forwarded.append(flag)
            index += 1
    forwarded.extend(("--graph-candidates", "gru", "--graph-lookback", "3"))
    inputs = prepare_graph_run(forwarded)
    config, frame = inputs["config"], inputs["frame"]
    dates = pd.to_datetime(frame[config.date_col], utc=True, errors="raise")
    frame = frame.loc[dates.ge(pd.Timestamp(args.start, tz="UTC")) &
                      dates.lt(pd.Timestamp(args.end, tz="UTC"))].copy()
    folds, _ = expanding_calendar_folds(
        frame, n_splits=inputs["n_splits"], initial_train_fraction=inputs["initial_train_fraction"],
        inner_val_fraction=inputs["inner_val_fraction"], gap_bars=inputs["gap_bars"],
        embargo_bars=inputs["embargo_bars"], date_col=config.date_col,
        final_test_fraction=1 - config.train_ratio - config.val_ratio)
    first = folds[0]
    # Labels/features are computed only on TRAIN/INNER, not OUTER or holdout.
    before_outer = pd.to_datetime(frame[config.date_col], utc=True) < pd.Timestamp(first["split"].test_start)
    prepared = _prepare(frame.loc[before_outer], replace(config, purged_split=first["split"]), inputs["ablation"])
    training_dates = pd.to_datetime(prepared.train[config.date_col], utc=True)
    return training_dates.min().isoformat(), (training_dates.max() + pd.Timedelta(days=1)).isoformat()


def inventory(args):
    from trading_system.data.fnspid_inventory import build_fnspid_inventory

    pilot._settings(args)  # Validate dates/delay before expensive preparation.
    if args.min_observed_train_days < 1 or not 0 <= args.min_train_observed_fraction <= 1:
        raise ValueError("Use positive minimum news days and a finite fraction in [0,1].")
    start, end = _selection_bounds(args)
    print(f"FNSPID eligibility: earliest TRAIN [{start}, {end}); no performance ranking", flush=True)
    return build_fnspid_inventory(
        args.data, [args.fnspid_dir / "Stock_news" / name
                    for name in ("All_external.csv", "nasdaq_exteral_data.csv")], args.inventory_dir,
        start=args.start, end=args.end, dataset_info=args.fnspid_dir / "dataset-info.json",
        delay_hours=args.delay_hours, selection_start=start, selection_end=end,
        min_observed_train_days=args.min_observed_train_days,
        min_train_observed_fraction=args.min_train_observed_fraction,
        resume=args.resume, dry_run=args.dry_run)


def _expanded_context(args):
    from trading_system.data.fnspid_inventory import load_fnspid_inventory

    frozen = load_fnspid_inventory(args.inventory_dir)
    settings = frozen["report"]["identity"]["settings"]
    rules = settings["selection"]
    if (pd.Timestamp(settings["start"]) != pd.Timestamp(args.start, tz="UTC") or
            pd.Timestamp(settings["end_exclusive"]) != pd.Timestamp(args.end, tz="UTC") or
            settings["delay_hours"] != args.delay_hours or settings["lookback_hours"] != 24. or
            rules["min_observed_train_days"] != args.min_observed_train_days or
            rules["min_train_observed_fraction"] != args.min_train_observed_fraction):
        raise ValueError("Expanded flags differ from the frozen inventory; use matching flags or a new inventory.")
    price_input = frozen["report"]["identity"]["price_input"]
    if Path(price_input["path"]).resolve() != args.data.resolve() or price_input["sha256"] != file_sha256(args.data):
        raise ValueError("Price source differs from the frozen inventory.")
    if not frozen["tickers"]:
        raise ValueError("No ticker passed frozen TRAIN availability rules; inspect inventory-report.json before changing policy.")
    ready_args = argparse.Namespace(**vars(args))
    ready_args._pilot_prepared_dir = args.prepared_dir
    ready_args.prepared_dir = args.expanded_prepared_dir
    ready_args.tickers = ",".join(frozen["tickers"])
    study_settings = pilot._settings(ready_args) | {
        "universe_policy": "frozen_train_availability_inventory",
        "inventory_manifest_sha256": frozen["manifest_sha256"],
        "selection_snapshot_sha256": frozen["selection"]["snapshot_sha256"],
        "selection_rules": rules,
    }
    return ready_args, frozen, study_settings


def _selected_articles(frozen, root, *, resume):
    """Subset the verified import once; no second pass over 29 GB of CSVs."""
    output = root / "import"
    path, report_path = output / "articles.parquet", output / "subset-report.json"
    sidecar = path.with_suffix(".manifest.json")
    identity = {"inventory_manifest_sha256": frozen["manifest_sha256"],
                "source_articles_sha256": file_sha256(frozen["articles_path"]), "tickers": frozen["tickers"]}
    targets = (path, sidecar, report_path)
    if any(item.exists() for item in targets):
        if not resume or not all(item.is_file() for item in targets):
            raise FileExistsError("Unregistered/partial expanded article subset; choose a new expanded preparation directory.")
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        if (saved.get("identity") != identity or saved.get("articles_sha256") != file_sha256(path) or
                saved.get("manifest_sha256") != file_sha256(sidecar)):
            raise ValueError("Expanded article subset identity/checksum mismatch.")
    else:
        selected = pd.read_parquet(frozen["articles_path"], filters=[("ticker", "in", frozen["tickers"])])
        if selected.empty or not selected.ticker.isin(frozen["tickers"]).all():
            raise ValueError("Inventory selected no usable article subset.")
        selected = selected.sort_values(["news_id", "ticker"], kind="stable").reset_index(drop=True)
        output.mkdir(parents=True, exist_ok=True)
        _write_frame(path, selected, {"protocol": "fnspid-exploratory", "identity": identity})
        atomic_write_json(report_path, {"identity": identity, "rows": len(selected),
                                       "articles_sha256": file_sha256(path), "manifest_sha256": file_sha256(sidecar)})
    return path, report_path


def _score_reuse_sources(args, price_manifest):
    if args.no_score_cache_reuse:
        sources = []
    elif args.score_reuse_from is not None:
        sources = args.score_reuse_from
    elif price_manifest.is_file():
        saved = json.loads(price_manifest.read_text(encoding="utf-8"))
        if "score_reuse_directories" in saved:
            sources = [Path(item) for item in saved["score_reuse_directories"]]
        else:
            sources = None
    else:
        sources = None
    if sources is None:
        pilot_root = getattr(args, "_pilot_prepared_dir", Path("data/derived/fnspid/pilot-v1"))
        cache = pilot_root / "scoring"
        sources = [cache] if (cache / "scoring.manifest.json").is_file() else []
    return sources


def prepare_expanded(args, frozen, settings):
    from trading_system.data.fnspid_scoring import score_fnspid
    from trading_system.data.fnspid_export import export_fnspid_sentiment, load_fnspid_sentiment_export

    root = args.prepared_dir
    prices, selection, daily = pilot._prepared_paths(args)
    market = pd.read_parquet(args.data)
    dates = pd.to_datetime(market.date, utc=True, errors="raise")
    selected = market.loc[market.ticker.isin(settings["tickers"]) &
                          dates.ge(pd.Timestamp(settings["start"], tz="UTC")) &
                          dates.lt(pd.Timestamp(settings["end_exclusive"], tz="UTC"))].copy()
    selected = selected.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    if selected.empty or set(selected.ticker) != set(settings["tickers"]) or selected.duplicated(["date", "ticker"]).any():
        raise ValueError("Expanded selected price keys are empty, duplicated or incomplete.")
    identity = {"settings": settings, "source_prices_sha256": file_sha256(args.data)}
    price_manifest = root / "prices.manifest.json"
    sources = _score_reuse_sources(args, price_manifest)
    directories = sorted({str(Path(path).resolve()) for path in sources})
    if price_manifest.exists():
        saved = json.loads(price_manifest.read_text(encoding="utf-8"))
        if not args.resume or saved.get("identity") != identity:
            raise ValueError("Expanded preparation differs or exists without --resume; choose a new directory.")
        if saved.get("prices_sha256") != file_sha256(prices) or saved.get("selection_sha256") != file_sha256(selection):
            raise ValueError("Expanded price/universe checksum mismatch.")
        if saved.get("score_reuse_directories", directories) != directories:
            raise ValueError("Scoring cache sources differ from the frozen expanded preparation.")
    else:
        if any(path.exists() for path in (prices, selection, prices.with_suffix(".manifest.json"))):
            raise FileExistsError("Unregistered expanded price artifacts; choose a new directory.")
        root.mkdir(parents=True, exist_ok=True)
        _write_frame(prices, selected, {"protocol": "fnspid-exploratory", "settings": settings})
        atomic_write_json(selection, frozen["selection"])
        atomic_write_json(price_manifest, {"identity": identity, "prices_sha256": file_sha256(prices),
                                           "selection_sha256": file_sha256(selection),
                                           "score_reuse_directories": directories})
    articles, subset_report = _selected_articles(frozen, root, resume=args.resume)
    print(f"FNSPID expanded: {len(settings['tickers'])} tickers; verified text-cache reuse; inference only for missing texts", flush=True)
    scored = score_fnspid(
        articles, root / "scoring", checkpoint=FinBERTCheckpoint(pilot.MODEL_REPOSITORY, pilot.MODEL_REVISION, pilot.MODEL_SHA256),
        model_dir=args.model_dir, device=args.device, batch_size=args.batch_size, resume=args.resume, reuse_from=sources)
    identifiers = {"domain": "ticker", "settings": settings,
                   "source_prices_sha256": identity["source_prices_sha256"],
                   "subset_report_sha256": file_sha256(subset_report),
                   "scoring_manifest_sha256": file_sha256(scored["manifest_path"])}
    if daily.exists() or daily.with_suffix(".manifest.json").exists():
        if not args.resume:
            raise FileExistsError("Expanded daily export exists; use matching --resume.")
        ready = load_fnspid_sentiment_export(daily)
        parameters = {"delay_hours": args.delay_hours, "lookback_hours": 24., "short_lookback_hours": 6., "half_life_hours": 6.}
        if (ready.manifest.get("input_identifiers") != identifiers or
                ready.manifest.get("checkpoint") != settings["checkpoint"] or
                any(ready.manifest.get("aggregation", {}).get(key) != value for key, value in parameters.items()) or
                ready.manifest["inputs"]["scored"]["sha256"] != file_sha256(scored["scored_path"]) or
                ready.manifest["inputs"]["market"]["sha256"] != hash_dataframe(selected)):
            raise ValueError("Expanded daily export differs from current study.")
    else:
        export_fnspid_sentiment(scored["scored_path"], selected, daily, checkpoint=settings["checkpoint"],
                                input_identifiers=identifiers, delay_hours=args.delay_hours)
        ready = load_fnspid_sentiment_export(daily)
    artifacts = {"prices": prices, "selection": selection, "daily": daily,
                 "daily_manifest": daily.with_suffix(".manifest.json")}
    atomic_write_json(root / "preparation.manifest.json", {"settings": settings, "artifacts": {
        name: {"path": str(path), "sha256": file_sha256(path)} for name, path in artifacts.items()}})
    print(f"FNSPID expanded ready: decisions={len(ready.frame)} observed={int(ready.frame.source_available.sum())}", flush=True)
    return ready


def _validate_preparation(args, settings):
    manifest_path = args.prepared_dir / "preparation.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Prepare expanded data first: --stage prepare-expanded --device cuda --resume.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prices, selection, daily = pilot._prepared_paths(args)
    expected = {"prices": prices, "selection": selection, "daily": daily, "daily_manifest": daily.with_suffix(".manifest.json")}
    if manifest.get("settings") != settings or set(manifest.get("artifacts", {})) != set(expected):
        raise ValueError("Expanded preparation settings/artifact registry mismatch.")
    for name, path in expected.items():
        record = manifest["artifacts"][name]
        if Path(record.get("path", "")).resolve() != path.resolve() or record.get("sha256") != file_sha256(path):
            raise ValueError(f"Expanded preparation artifact mismatch: {name}.")


def _permutation_seeds(args):
    try:
        seeds = tuple(int(item.strip()) for item in args.permutation_seeds.split(","))
    except ValueError as error:
        raise ValueError("--permutation-seeds requires comma-separated integers.") from error
    # Validate inexpensive seed errors before feature preparation.
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Use at least two distinct corpus permutation seeds in [0,2**32).")
    return seeds


def run_expanded_stage(args):
    if args.stage == "inventory":
        return inventory(args)
    seeds = _permutation_seeds(args) if args.stage == "expanded" or args.dry_run else None
    ready_args, frozen, settings = _expanded_context(args)
    if args.stage == "prepare-expanded" and not args.dry_run:
        return prepare_expanded(ready_args, frozen, settings)
    _validate_preparation(ready_args, settings)
    from trading_system.experiments.fnspid_permutation_study import run_fnspid_permutation_study
    from trading_system.pipelines.compare_news_sentiment import prepare_news_sentiment_run

    target = args.output_dir or Path("artifacts/comparisons/fnspid-news-expanded-permutations")
    ready_args.output_dir = target
    inputs = prepare_news_sentiment_run(pilot.benchmark_arguments(ready_args, polarity=True))
    result = run_fnspid_permutation_study(inputs, target, permutation_seeds=seeds,
                                          resume=args.resume and target.is_dir(), dry_run=args.dry_run)
    tasks = result.get("planned_tasks", result.get("metadata", {}).get("planned_tasks"))
    completed = result.get("completed_tasks", tasks if result.get("permutations") is not None else None)
    print(f"FNSPID expanded: tasks={tasks} completed={completed} "
          f"permutations={len(seeds)} final_holdout_opened=False", flush=True)
    return result
