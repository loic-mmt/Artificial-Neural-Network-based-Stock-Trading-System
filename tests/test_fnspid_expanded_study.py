"""Offline inventory/expanded CLI wiring, immutable subsets and driver forwarding."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trading_system.artifacts.experiment import hash_dataframe
from trading_system.data.news_pilot_scoring import file_sha256
from trading_system.pipelines import fnspid_expanded_study as expanded
from trading_system.pipelines import fnspid_news_study as pilot


def _args(tmp_path, *extra):
    return pilot.build_parser().parse_args([
        "--stage", "expanded", "--data", str(tmp_path / "prices.parquet"),
        "--fnspid-dir", str(tmp_path / "raw"), "--inventory-dir", str(tmp_path / "inventory"),
        "--expanded-prepared-dir", str(tmp_path / "expanded"),
        "--prepared-dir", str(tmp_path / "original-pilot"),
        "--output-dir", str(tmp_path / "comparison"), "--model-dir", str(tmp_path / "model"),
        "--start", "2020-01-02", "--end", "2020-01-06", "--device", "cpu",
        "--min-observed-train-days", "2", "--min-train-observed-fraction", ".1", *extra,
    ])


def _tokens(args, *extra):
    return ["--stage", args.stage, "--data", str(args.data), "--fnspid-dir", str(args.fnspid_dir),
            "--inventory-dir", str(args.inventory_dir), "--expanded-prepared-dir", str(args.expanded_prepared_dir),
            "--prepared-dir", str(args.prepared_dir), "--output-dir", str(args.output_dir),
            "--model-dir", str(args.model_dir), "--start", args.start, "--end", args.end,
            "--device", args.device, "--min-observed-train-days", str(args.min_observed_train_days),
            "--min-train-observed-fraction", str(args.min_train_observed_fraction), *extra]


def _frozen(tmp_path, args):
    market = pd.DataFrame([
        {"date": day, "ticker": ticker, "close": float(index + 100)}
        for day in pd.date_range("2020-01-01", periods=7, tz="UTC")
        for index, ticker in enumerate(("AAPL", "JPM", "XOM"))
    ])
    market.to_parquet(args.data, index=False)
    articles = args.inventory_dir / "import" / "articles.parquet"
    articles.parent.mkdir(parents=True)
    pd.DataFrame({"news_id": ["b", "a", "c", "d"], "ticker": ["JPM", "AAPL", "XOM", "AAPL"],
                  "text": ["JPM title", "AAPL title", "XOM title", "Second AAPL title"]}).to_parquet(articles, index=False)
    rules = {"start": "2020-01-02T00:00:00+00:00", "end_exclusive": "2020-01-04T00:00:00+00:00",
             "min_observed_train_days": args.min_observed_train_days,
             "min_train_observed_fraction": args.min_train_observed_fraction,
             "uses_performance": False}
    return {"tickers": ["AAPL", "JPM"], "articles_path": articles,
            "manifest_sha256": "f" * 64,
            "selection": {"tickers": ["AAPL", "JPM"], "snapshot_sha256": "a" * 64, "rules": rules},
            "report": {"identity": {"price_input": {"path": str(args.data.resolve()), "sha256": file_sha256(args.data)},
                                      "settings": {"start": "2020-01-02T00:00:00+00:00",
                                                   "end_exclusive": "2020-01-06T00:00:00+00:00",
                                                   "delay_hours": 24., "lookback_hours": 24., "selection": rules}}}}


def _context(tmp_path, monkeypatch, args=None):
    from trading_system.data import fnspid_inventory

    args = args or _args(tmp_path)
    frozen = _frozen(tmp_path, args)
    monkeypatch.setattr(fnspid_inventory, "load_fnspid_inventory", lambda path: deepcopy(frozen))
    ready_args, loaded, settings = expanded._expanded_context(args)
    return args, ready_args, loaded, settings, frozen


def _scoring_stubs(monkeypatch):
    from trading_system.data import fnspid_export, fnspid_import, fnspid_scoring

    scored_calls, exported_calls = [], []
    monkeypatch.setattr(fnspid_import, "import_fnspid", lambda *a, **kw: pytest.fail("Expanded data must reuse the imported Parquet, never parse raw CSVs."))

    def fake_score(source, destination, **kwargs):
        frame = pd.read_parquet(source)
        scored_calls.append({"source": source, "frame": frame, "destination": destination, **kwargs})
        destination.mkdir(parents=True, exist_ok=True)
        scored, manifest = destination / "scored_company.parquet", destination / "scoring.manifest.json"
        if not scored.exists():
            scored.write_bytes(b"frozen scored fixture")
            manifest.write_text(json.dumps({"checkpoint": kwargs["checkpoint"].identifier,
                                            "input_sha256": file_sha256(source)}), encoding="utf-8")
        return {"scored_path": scored, "manifest_path": manifest}

    def fake_export(scored, market, daily, **kwargs):
        exported_calls.append({"scored": scored, "market": market.copy(), "daily": daily, **kwargs})
        pd.DataFrame({"source_available": [True, False]}).to_parquet(daily, index=False)
        daily.with_suffix(".manifest.json").write_text(json.dumps({
            "input_identifiers": kwargs["input_identifiers"], "checkpoint": kwargs["checkpoint"],
            "aggregation": {"delay_hours": kwargs["delay_hours"], "lookback_hours": 24.,
                            "short_lookback_hours": 6., "half_life_hours": 6.},
            "inputs": {"scored": {"sha256": file_sha256(scored)}, "market": {"sha256": hash_dataframe(market)}},
        }), encoding="utf-8")

    def fake_load(path):
        return SimpleNamespace(frame=pd.read_parquet(path),
                               manifest=json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8")))

    monkeypatch.setattr(fnspid_scoring, "score_fnspid", fake_score)
    monkeypatch.setattr(fnspid_export, "export_fnspid_sentiment", fake_export)
    monkeypatch.setattr(fnspid_export, "load_fnspid_sentiment_export", fake_load)
    return scored_calls, exported_calls


def _ready_preparation(args, settings):
    args.prepared_dir.mkdir(parents=True, exist_ok=True)
    prices, selection, daily = pilot._prepared_paths(args)
    paths = {"prices": prices, "selection": selection, "daily": daily,
             "daily_manifest": daily.with_suffix(".manifest.json")}
    for name, path in paths.items():
        path.write_bytes(("fixture " + name).encode("utf-8"))
    manifest_path = args.prepared_dir / "preparation.manifest.json"
    payload = {"settings": settings, "artifacts": {
        name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()}}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path, payload


def test_inventory_cli_dispatch_forwards_readonly_dryrun_without_scoring(tmp_path, monkeypatch):
    from trading_system.data import fnspid_inventory, fnspid_scoring

    args = _args(tmp_path, "--stage", "inventory", "--device", "cuda")
    calls = []
    monkeypatch.setattr(expanded, "_selection_bounds", lambda a: ("2020-01-02T00:00:00+00:00", "2020-01-04T00:00:00+00:00"))
    monkeypatch.setattr(pilot, "prepare", lambda a: pytest.fail("Must dispatch to inventory."))
    monkeypatch.setattr(fnspid_scoring, "score_fnspid", lambda *a, **kw: pytest.fail("Inventory must not load a model."))

    def fake_inventory(*values, **kwargs):
        calls.append((values, kwargs))
        return {"status": "planned", "dry_run": kwargs["dry_run"]}

    monkeypatch.setattr(fnspid_inventory, "build_fnspid_inventory", fake_inventory)
    result = pilot.main(_tokens(args, "--dry-run", "--resume"))
    assert result == {"status": "planned", "dry_run": True}
    assert len(calls) == 1
    values, kwargs = calls[0]
    assert values[0] == args.data and values[2] == args.inventory_dir
    assert [path.name for path in values[1]] == ["All_external.csv", "nasdaq_exteral_data.csv"]
    assert kwargs["dataset_info"] == args.fnspid_dir / "dataset-info.json"
    assert kwargs["resume"] and kwargs["dry_run"]
    assert kwargs["selection_start"] == "2020-01-02T00:00:00+00:00"
    assert kwargs["selection_end"] == "2020-01-04T00:00:00+00:00"
    assert "device" not in kwargs
    assert not args.inventory_dir.exists()


def test_selection_bounds_prepares_only_before_first_outer_and_uses_actual_train_dates(tmp_path, monkeypatch):
    from trading_system.data.purged_cv import expanding_calendar_folds
    from trading_system.experiments import graph_ablation
    from trading_system.pipelines import compare_gnn_graphs
    from trading_system.pipelines.compare_losses import PRESETS

    args = _args(tmp_path, "--end", "2020-07-01")
    dates = pd.bdate_range("2020-01-01", periods=130, tz="UTC")
    frame = pd.DataFrame({"date": dates, "ticker": "AAPL", "close": 100.})
    config = replace(PRESETS["multi_ticker_long_short"], context_len=60)
    inputs = {"frame": frame, "config": config, "ablation": object(), "n_splits": 3,
              "initial_train_fraction": .5, "inner_val_fraction": .2, "gap_bars": 5, "embargo_bars": 0}
    forwarded, prepared = [], []
    monkeypatch.setattr(compare_gnn_graphs, "prepare_graph_run", lambda tokens: forwarded.append(tokens) or inputs)

    def fake_prepare(source, passed_config, ablation):
        prepared.append((source.copy(), passed_config, ablation))
        return SimpleNamespace(train=frame.iloc[21:31].copy())

    monkeypatch.setattr(graph_ablation, "_prepare", fake_prepare)
    start, end = expanded._selection_bounds(args)
    assert pd.Timestamp(start) == dates[21]
    assert pd.Timestamp(end) == dates[30] + pd.Timedelta(days=1)
    assert len(forwarded) == len(prepared) == 1
    tokens = forwarded[0]
    assert tokens[tokens.index("--data") + 1] == str(args.data)
    assert tokens[tokens.index("--context-len") + 1] == "60"
    assert tokens[tokens.index("--graph-lookback") + 1] == "3"
    assert all(flag not in tokens for flag in ("--ticker-selection", "--news-sentiment-export", "--news-protocol",
                                              "--sentiment-candidates", "--news-scored-articles", "--news-shuffle-seed"))
    clipped = frame.loc[frame.date.ge(pd.Timestamp(args.start, tz="UTC")) & frame.date.lt(pd.Timestamp(args.end, tz="UTC"))]
    folds, _ = expanding_calendar_folds(clipped, n_splits=3, initial_train_fraction=.5, inner_val_fraction=.2,
                                       final_test_fraction=1 - config.train_ratio - config.val_ratio, gap_bars=5)
    source, passed_config, ablation = prepared[0]
    assert source.date.max() < pd.Timestamp(folds[0]["split"].test_start)
    assert source.date.min() == pd.Timestamp(args.start, tz="UTC")
    assert passed_config.purged_split == folds[0]["split"]
    assert ablation is inputs["ablation"]


@pytest.mark.parametrize("options", [
    ["--min-observed-train-days", "0"], ["--min-train-observed-fraction", "nan"],
    ["--min-train-observed-fraction", "1.1"], ["--delay-hours", "nan"],
    ["--start", "2020-01-02T12:00:00"],
])
def test_inventory_invalid_policy_rejects_before_calendar_work(tmp_path, monkeypatch, options):
    args = _args(tmp_path, "--stage", "inventory", *options)
    monkeypatch.setattr(expanded, "_selection_bounds", lambda a: pytest.fail("Invalid flags must fail before calendar preparation."))
    with pytest.raises(ValueError):
        expanded.inventory(args)


def test_context_replaces_pilot_universe_directory_and_binds_inventory_snapshot(tmp_path, monkeypatch):
    args, ready, frozen, settings, _ = _context(tmp_path, monkeypatch)
    assert args.tickers == "AAPL,JPM,XOM,WMT,JNJ"
    assert ready.tickers == "AAPL,JPM"
    assert ready.prepared_dir == args.expanded_prepared_dir != args.prepared_dir
    assert ready._pilot_prepared_dir == args.prepared_dir
    assert settings["inventory_manifest_sha256"] == frozen["manifest_sha256"]
    assert settings["selection_snapshot_sha256"] == frozen["selection"]["snapshot_sha256"]
    assert settings["selection_rules"] == frozen["selection"]["rules"]
    assert settings["historical_coverage_claim"] is False


@pytest.mark.parametrize("drift", ["start", "end", "delay", "window", "minimum_days", "fraction", "price_bytes", "price_path", "empty_selection"])
def test_expanded_context_refuses_inventory_source_period_or_policy_drift(tmp_path, monkeypatch, drift):
    from trading_system.data import fnspid_inventory

    args = _args(tmp_path)
    frozen = _frozen(tmp_path, args)
    source_settings = frozen["report"]["identity"]["settings"]
    if drift == "start":
        source_settings["start"] = "2020-01-01T00:00:00+00:00"
    elif drift == "end":
        source_settings["end_exclusive"] = "2020-01-07T00:00:00+00:00"
    elif drift == "delay":
        source_settings["delay_hours"] = 0.
    elif drift == "window":
        source_settings["lookback_hours"] = 48.
    elif drift == "minimum_days":
        source_settings["selection"]["min_observed_train_days"] = 1
    elif drift == "fraction":
        source_settings["selection"]["min_train_observed_fraction"] = .9
    elif drift == "price_bytes":
        args.data.write_bytes(b"modified prices")
    elif drift == "price_path":
        alternate = tmp_path / "alternate.parquet"
        alternate.write_bytes(args.data.read_bytes())
        args.data = alternate
    elif drift == "empty_selection":
        frozen["tickers"] = []
    monkeypatch.setattr(fnspid_inventory, "load_fnspid_inventory", lambda path: frozen)
    with pytest.raises(ValueError):
        expanded._expanded_context(args)
    assert not args.expanded_prepared_dir.exists()


def test_prepare_expanded_subsets_import_once_forwards_cache_and_resumes_export(tmp_path, monkeypatch):
    cache = tmp_path / "pilot-cache"
    args = _args(tmp_path, "--score-reuse-from", str(cache))
    _, ready, frozen, settings, _ = _context(tmp_path, monkeypatch, args)
    score_calls, export_calls = _scoring_stubs(monkeypatch)
    result = expanded.prepare_expanded(ready, frozen, settings)
    assert len(score_calls) == len(export_calls) == 1
    scored = score_calls[0]
    assert scored["reuse_from"] == [cache]
    assert scored["device"] == "cpu"
    assert scored["checkpoint"].identifier == settings["checkpoint"]
    assert scored["frame"].ticker.tolist() == ["AAPL", "JPM", "AAPL"]
    assert scored["frame"].news_id.tolist() == ["a", "b", "d"]
    selected_prices = pd.read_parquet(ready.prepared_dir / "prices-matched.parquet")
    assert set(selected_prices.ticker) == {"AAPL", "JPM"}
    assert selected_prices.date.min() == pd.Timestamp("2020-01-02", tz="UTC")
    assert selected_prices.date.max() == pd.Timestamp("2020-01-05", tz="UTC")
    assert json.loads((ready.prepared_dir / "tickers-matched.json").read_text(encoding="utf-8")) == frozen["selection"]
    assert result.frame.source_available.sum() == 1
    artifacts = [ready.prepared_dir / "import" / "articles.parquet",
                 ready.prepared_dir / "company_daily.parquet"]
    hashes = [file_sha256(path) for path in artifacts]
    ready.resume = True
    expanded.prepare_expanded(ready, frozen, settings)
    assert len(score_calls) == 2 and score_calls[1]["resume"]
    assert len(export_calls) == 1
    assert [file_sha256(path) for path in artifacts] == hashes
    expanded._validate_preparation(ready, settings)
    assert not args.prepared_dir.exists()


def test_explicit_cache_optout_is_forwarded(tmp_path, monkeypatch):
    args = _args(tmp_path, "--no-score-cache-reuse")
    _, ready, frozen, settings, _ = _context(tmp_path, monkeypatch, args)
    scored, _ = _scoring_stubs(monkeypatch)
    expanded.prepare_expanded(ready, frozen, settings)
    assert scored[0]["reuse_from"] == []


def test_default_pilot_cache_is_reused_when_completed(tmp_path, monkeypatch):
    args = _args(tmp_path)
    _, ready, frozen, settings, _ = _context(tmp_path, monkeypatch, args)
    scored, _ = _scoring_stubs(monkeypatch)
    cache = args.prepared_dir / "scoring"
    cache.mkdir(parents=True)
    (cache / "scoring.manifest.json").write_text('{"state":"complete"}', encoding="utf-8")
    expanded.prepare_expanded(ready, frozen, settings)
    assert scored[0]["reuse_from"] == [cache]
    manifest = json.loads((ready.prepared_dir / "prices.manifest.json").read_text(encoding="utf-8"))
    assert manifest["score_reuse_directories"] == [str(cache.resolve())]


@pytest.mark.parametrize("initially_present", [False, True])
def test_automatic_cache_choice_stays_frozen_when_pilot_manifest_presence_changes(tmp_path, monkeypatch, initially_present):
    args = _args(tmp_path)
    _, ready, frozen, settings, _ = _context(tmp_path, monkeypatch, args)
    scores, exports = _scoring_stubs(monkeypatch)
    cache = args.prepared_dir / "scoring"
    cache.mkdir(parents=True)
    cache_manifest = cache / "scoring.manifest.json"
    if initially_present:
        cache_manifest.write_text('{"state":"complete"}', encoding="utf-8")
    expanded.prepare_expanded(ready, frozen, settings)
    chosen = [cache] if initially_present else []
    assert scores[0]["reuse_from"] == chosen
    price_manifest = ready.prepared_dir / "prices.manifest.json"
    original_manifest = file_sha256(price_manifest)
    if initially_present:
        cache_manifest.unlink()
    else:
        cache_manifest.write_text('{"state":"complete"}', encoding="utf-8")
    ready.resume = True
    # A missing chosen cache is still forwarded for the scorer's strict source
    # validation; the orchestration must not silently change its original choice.
    assert [path.resolve() for path in expanded._score_reuse_sources(ready, price_manifest)] == [path.resolve() for path in chosen]
    expanded.prepare_expanded(ready, frozen, settings)
    assert [path.resolve() for path in scores[1]["reuse_from"]] == [path.resolve() for path in chosen]
    assert len(exports) == 1
    assert file_sha256(price_manifest) == original_manifest


@pytest.mark.parametrize("change", ["different_explicit_source", "optout"])
def test_changed_cache_choice_refuses_at_price_manifest_before_article_or_scoring_work(tmp_path, monkeypatch, change):
    source = tmp_path / "pilot-cache-one"
    args = _args(tmp_path, "--score-reuse-from", str(source))
    _, ready, frozen, settings, _ = _context(tmp_path, monkeypatch, args)
    scores, exports = _scoring_stubs(monkeypatch)
    expanded.prepare_expanded(ready, frozen, settings)
    original_manifest = file_sha256(ready.prepared_dir / "prices.manifest.json")
    original_daily = file_sha256(ready.prepared_dir / "company_daily.parquet")
    ready.resume = True
    if change == "different_explicit_source":
        ready.score_reuse_from = [tmp_path / "pilot-cache-two"]
    else:
        ready.no_score_cache_reuse = True
    monkeypatch.setattr(expanded, "_selected_articles", lambda *a, **kw: pytest.fail("Cache drift must reject before subset work."))
    with pytest.raises(ValueError, match="Scoring cache sources differ"):
        expanded.prepare_expanded(ready, frozen, settings)
    assert len(scores) == len(exports) == 1
    assert file_sha256(ready.prepared_dir / "prices.manifest.json") == original_manifest
    assert file_sha256(ready.prepared_dir / "company_daily.parquet") == original_daily


def test_prepare_expanded_refuses_overwrite_before_scoring(tmp_path, monkeypatch):
    _, ready, frozen, settings, _ = _context(tmp_path, monkeypatch)
    score_calls, export_calls = _scoring_stubs(monkeypatch)
    expanded.prepare_expanded(ready, frozen, settings)
    snapshot = file_sha256(ready.prepared_dir / "company_daily.parquet")
    with pytest.raises(ValueError, match="without --resume"):
        expanded.prepare_expanded(ready, frozen, settings)
    assert len(score_calls) == len(export_calls) == 1
    assert file_sha256(ready.prepared_dir / "company_daily.parquet") == snapshot


@pytest.mark.parametrize("drift", ["settings", "prices", "selection", "subset", "subset_sidecar", "subset_identity", "daily_window"])
def test_prepare_expanded_resume_refuses_registered_drift(tmp_path, monkeypatch, drift):
    _, ready, frozen, settings, _ = _context(tmp_path, monkeypatch)
    score_calls, export_calls = _scoring_stubs(monkeypatch)
    expanded.prepare_expanded(ready, frozen, settings)
    ready.resume = True
    if drift == "settings":
        settings = settings | {"delay_hours": 0.}
    elif drift in ("prices", "selection"):
        path = ready.prepared_dir / ("prices-matched.parquet" if drift == "prices" else "tickers-matched.json")
        path.write_bytes(b"modified prepared artifact")
    elif drift in ("subset", "subset_sidecar"):
        suffix = "articles.parquet" if drift == "subset" else "articles.manifest.json"
        (ready.prepared_dir / "import" / suffix).write_bytes(b"modified subset artifact")
    elif drift == "subset_identity":
        frozen["manifest_sha256"] = "b" * 64
    elif drift == "daily_window":
        manifest = ready.prepared_dir / "company_daily.manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["aggregation"]["lookback_hours"] = 48.
        manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        expanded.prepare_expanded(ready, frozen, settings)
    assert len(export_calls) == 1
    assert len(score_calls) == (2 if drift == "daily_window" else 1)


@pytest.mark.parametrize("args_flags", [[], ["--dry-run"], ["--resume", "--dry-run"]])
def test_expanded_driver_called_once_without_double_preparation(tmp_path, monkeypatch, args_flags):
    from trading_system.experiments import fnspid_permutation_study
    from trading_system.pipelines import compare_news_sentiment

    args = _args(tmp_path)
    _, ready, frozen, settings, _ = _context(tmp_path, monkeypatch, args)
    _ready_preparation(ready, settings)
    if "--resume" in args_flags:
        args.output_dir.mkdir()
    preparations, drivers = [], []
    inputs = {"sentinel": "prepared once"}
    monkeypatch.setattr(expanded, "prepare_expanded", lambda *a: pytest.fail("Expanded run must consume prepared artifacts."))
    monkeypatch.setattr(pilot, "prepare", lambda a: pytest.fail("Expanded run must not run pilot preparation."))
    monkeypatch.setattr(compare_news_sentiment, "prepare_news_sentiment_run", lambda tokens: preparations.append(tokens) or inputs)

    def fake_driver(*values, **kwargs):
        drivers.append((values, kwargs))
        return {"planned_tasks": 81, "completed_tasks": 0 if kwargs["dry_run"] else 81}

    monkeypatch.setattr(fnspid_permutation_study, "run_fnspid_permutation_study", fake_driver)
    result = pilot.main(_tokens(args, "--permutation-seeds", "17,23,99", *args_flags))
    assert result["planned_tasks"] == 81
    assert len(preparations) == len(drivers) == 1
    values, kwargs = drivers[0]
    assert values == (inputs, args.output_dir)
    assert kwargs["permutation_seeds"] == (17, 23, 99)
    assert kwargs["dry_run"] == ("--dry-run" in args_flags)
    assert kwargs["resume"] == ("--resume" in args_flags)
    tokens = preparations[0]
    assert tokens[tokens.index("--data") + 1] == str(ready.prepared_dir / "prices-matched.parquet")
    assert tokens[tokens.index("--news-scored-articles") + 1] == str(ready.prepared_dir / "scoring" / "scored_company.parquet")
    assert "--final-test" not in tokens and "--cv-final-test" not in tokens


def test_prepare_expanded_cli_dispatches_only_preparation(tmp_path, monkeypatch):
    args = _args(tmp_path, "--stage", "prepare-expanded")
    _, ready, frozen, settings, _ = _context(tmp_path, monkeypatch, args)
    calls = []
    monkeypatch.setattr(expanded, "prepare_expanded", lambda *values: calls.append(values) or "ready")
    monkeypatch.setattr(expanded, "_validate_preparation", lambda *a: pytest.fail("Preparing must not validate/run an old comparison."))
    assert pilot.main(_tokens(args)) == "ready"
    assert len(calls) == 1
    assert calls[0][0].prepared_dir == ready.prepared_dir
    assert calls[0][1] == frozen and calls[0][2] == settings


@pytest.mark.parametrize("value", ["1", "1,1", "1,", "one,2", "-1,2", "1,4294967296"])
def test_invalid_permutation_seeds_reject_before_context_or_feature_work(tmp_path, monkeypatch, value):
    args = _args(tmp_path, "--permutation-seeds=" + value)
    monkeypatch.setattr(expanded, "_expanded_context", lambda a: pytest.fail("Invalid seeds must fail before inventory/price work."))
    with pytest.raises(ValueError, match="permutation|distinct"):
        expanded.run_expanded_stage(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("drift", ["settings", "bytes", "registry", "path"])
def test_preparation_guard_refuses_before_driver(tmp_path, monkeypatch, drift):
    args, ready, _, settings, _ = _context(tmp_path, monkeypatch)
    manifest_path, payload = _ready_preparation(ready, settings)
    if drift == "settings":
        payload["settings"] = payload["settings"] | {"delay_hours": 0.}
    elif drift == "bytes":
        (ready.prepared_dir / "company_daily.parquet").write_bytes(b"modified daily")
    elif drift == "registry":
        del payload["artifacts"]["selection"]
    elif drift == "path":
        other = tmp_path / "other-daily.parquet"
        other.write_bytes(b"fixture daily")
        payload["artifacts"]["daily"]["path"] = str(other)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        expanded.run_expanded_stage(args)
    assert not args.output_dir.exists()


def test_missing_expanded_preparation_has_clear_guard(tmp_path, monkeypatch):
    args, _, _, _, _ = _context(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="Prepare expanded"):
        expanded.run_expanded_stage(args)
    assert not args.output_dir.exists()
