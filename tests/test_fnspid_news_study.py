"""Offline orchestration checks: frozen preparation and benchmark forwarding."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trading_system.artifacts.experiment import hash_dataframe
from trading_system.data.news_pilot_scoring import file_sha256
from trading_system.pipelines import fnspid_news_study as study


def _args(tmp_path, *extra):
    return study.build_parser().parse_args([
        "--prepared-dir", str(tmp_path / "prepared"), "--fnspid-dir", str(tmp_path / "raw"),
        "--data", str(tmp_path / "prices.parquet"), "--model-dir", str(tmp_path / "model"),
        "--tickers", "AAPL,JPM", "--start", "2020-01-02", "--end", "2020-01-06",
        "--device", "cpu", *extra,
    ])


def _tokens(args, stage, *extra):
    return ["--prepared-dir", str(args.prepared_dir), "--fnspid-dir", str(args.fnspid_dir),
            "--data", str(args.data), "--model-dir", str(args.model_dir),
            "--tickers", args.tickers, "--start", args.start, "--end", args.end,
            "--device", args.device, "--stage", stage, *extra]


def _value(tokens, flag):
    return tokens[tokens.index(flag) + 1]


def _ready_manifest(args):
    args.prepared_dir.mkdir(parents=True)
    prices, selection, daily = study._prepared_paths(args)
    paths = {"prices": prices, "selection": selection, "daily": daily,
             "daily_manifest": daily.with_suffix(".manifest.json")}
    for name, path in paths.items():
        path.write_bytes(("fixture-" + name).encode())
    payload = {"settings": study._settings(args), "artifacts": {
        name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()
    }}
    manifest = args.prepared_dir / "preparation.manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload


def _stub_compare(monkeypatch):
    import trading_system.pipelines.compare_news_sentiment as comparison

    calls = []
    monkeypatch.setattr(comparison, "main", lambda tokens: calls.append(list(tokens)) or {"calls": len(calls)})
    return calls


def test_benchmark_and_smoke_freeze_45_and_4_training_tasks(tmp_path):
    args = _args(tmp_path)
    full, smoke = study.benchmark_arguments(args), study.benchmark_arguments(args, smoke=True)
    for tokens, expected in ((full, 45), (smoke, 4)):
        tasks = len(_value(tokens, "--sentiment-candidates").split(",")) * int(_value(tokens, "--cv-folds"))
        tasks *= len(_value(tokens, "--seeds").split(","))
        assert tasks == expected
        assert _value(tokens, "--news-protocol") == "fnspid-exploratory"
        assert _value(tokens, "--news-sentiment-export") == str(args.prepared_dir / "company_daily.parquet")
        assert "--final-test" not in tokens and "--cv-final-test" not in tokens
    assert _value(full, "--output-dir") != _value(smoke, "--output-dir")


@pytest.mark.parametrize("extra", [
    ["--tickers", "AAPL,AAPL"], ["--tickers", "AAPL,"],
    ["--start", "2020-01-02T12:00:00"], ["--start", "2020-01-02T00:00:00Z"],
    ["--delay-hours", "nan"], ["--delay-hours", "inf"], ["--delay-hours", "-1"],
    ["--batch-size", "0"],
])
def test_invalid_study_settings_fail_before_work(tmp_path, extra):
    with pytest.raises(ValueError):
        study._settings(_args(tmp_path, *extra))


def test_dry_run_validates_prepared_manifest_without_preparing_or_training(tmp_path, monkeypatch):
    args = _args(tmp_path)
    _ready_manifest(args)
    monkeypatch.setattr(study, "prepare", lambda _: pytest.fail("Dry-run must not import or infer."))
    calls = _stub_compare(monkeypatch)
    assert study.main(_tokens(args, "all", "--dry-run")) == {"calls": 1}
    assert len(calls) == 1 and "--dry-run" in calls[0]


def test_training_follows_a_successful_dry_run(tmp_path, monkeypatch):
    args = _args(tmp_path)
    _ready_manifest(args)
    calls = _stub_compare(monkeypatch)
    study.main(_tokens(args, "smoke"))
    assert len(calls) == 2
    assert "--dry-run" in calls[0] and "--dry-run" not in calls[1]
    assert _value(calls[1], "--sentiment-candidates") == "gru,gru_features"


@pytest.mark.parametrize("drift", ["settings", "bytes", "missing_artifact", "path"])
def test_prepared_manifest_drift_refuses_before_comparison(tmp_path, monkeypatch, drift):
    args = _args(tmp_path)
    manifest, payload = _ready_manifest(args)
    if drift == "settings":
        payload["settings"]["delay_hours"] = 0
    elif drift == "bytes":
        (args.prepared_dir / "prices-matched.parquet").write_bytes(b"different")
    elif drift == "missing_artifact":
        del payload["artifacts"]["prices"]
    elif drift == "path":
        wrong = tmp_path / "unrelated.parquet"
        wrong.write_bytes(b"fixture-prices")
        payload["artifacts"]["prices"]["path"] = str(wrong)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    calls = _stub_compare(monkeypatch)
    with pytest.raises(ValueError):
        study.main(_tokens(args, "benchmark", "--dry-run"))
    assert not calls


def test_missing_preparation_is_a_clear_error(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls = _stub_compare(monkeypatch)
    with pytest.raises(FileNotFoundError, match="Prepare first"):
        study.main(_tokens(args, "benchmark", "--dry-run"))
    assert not calls


@pytest.mark.parametrize("manifest_drift", [
    "checkpoint", "delay_hours", "lookback_hours", "short_lookback_hours", "half_life_hours",
])
def test_prepare_freezes_prices_and_refuses_resumed_checkpoint_or_window_drift(tmp_path, monkeypatch, manifest_drift):
    from trading_system.data import fnspid_export, fnspid_import, fnspid_scoring

    args = _args(tmp_path)
    market = pd.DataFrame([
        {"date": date, "ticker": ticker, "close": 100. + index}
        for index, date in enumerate(pd.date_range("2020-01-01", periods=7, tz="UTC"))
        for ticker in ("AAPL", "JPM", "XOM")
    ])
    market.to_parquet(args.data, index=False)
    imported_calls, scored_calls, exported_calls = [], [], []

    def fake_import(paths, destination, **kwargs):
        imported_calls.append({"paths": paths, "destination": destination, **kwargs})
        destination.mkdir(parents=True, exist_ok=True)
        article, report = destination / "articles.parquet", destination / "import-report.json"
        if not article.exists():
            article.write_bytes(b"frozen articles")
            report.write_text('{"protocol":"fnspid_exploratory"}', encoding="utf-8")
        return {"articles_path": article, "report_path": report}

    def fake_score(path, destination, **kwargs):
        scored_calls.append({"articles": path, **kwargs})
        destination.mkdir(parents=True, exist_ok=True)
        scored, manifest = destination / "scored_company.parquet", destination / "scoring.manifest.json"
        if not scored.exists():
            scored.write_bytes(b"frozen predictions")
            manifest.write_text('{"checkpoint":"frozen"}', encoding="utf-8")
        return {"scored_path": scored, "manifest_path": manifest}

    def fake_export(scored, market, daily, **kwargs):
        exported_calls.append({"scored": scored, "market": market.copy(), **kwargs})
        pd.DataFrame({"source_available": [True]}).to_parquet(daily, index=False)
        daily.with_suffix(".manifest.json").write_text(json.dumps({
            "checkpoint": kwargs["checkpoint"],
            "aggregation": {"delay_hours": kwargs["delay_hours"], "lookback_hours": 24.,
                            "short_lookback_hours": 6., "half_life_hours": 6.},
            "input_identifiers": kwargs["input_identifiers"],
            "inputs": {"scored": {"sha256": file_sha256(scored)}, "market": {"sha256": hash_dataframe(market)}},
        }), encoding="utf-8")

    def fake_load(daily):
        return SimpleNamespace(frame=pd.read_parquet(daily),
                               manifest=json.loads(daily.with_suffix(".manifest.json").read_text(encoding="utf-8")))

    monkeypatch.setattr(fnspid_import, "import_fnspid", fake_import)
    monkeypatch.setattr(fnspid_scoring, "score_fnspid", fake_score)
    monkeypatch.setattr(fnspid_export, "export_fnspid_sentiment", fake_export)
    monkeypatch.setattr(fnspid_export, "load_fnspid_sentiment_export", fake_load)
    study.prepare(args)
    prices = pd.read_parquet(args.prepared_dir / "prices-matched.parquet")
    assert set(prices.ticker) == {"AAPL", "JPM"}
    assert prices.date.min() == pd.Timestamp("2020-01-02", tz="UTC")
    assert prices.date.max() == pd.Timestamp("2020-01-05", tz="UTC")
    assert pd.Timestamp(imported_calls[0]["start"]) == pd.Timestamp("2019-12-31")
    assert imported_calls[0]["end"] == "2020-01-06T00:00:00"
    assert len(imported_calls[0]["paths"]) == 2
    assert scored_calls[0]["checkpoint"].identifier == study._settings(args)["checkpoint"]
    assert exported_calls[0]["checkpoint"] == scored_calls[0]["checkpoint"].identifier
    assert exported_calls[0]["delay_hours"] == 24
    snapshot = file_sha256(args.prepared_dir / "company_daily.parquet")
    args.resume = True
    study.prepare(args)
    assert len(exported_calls) == 1
    assert imported_calls[1]["resume"] and scored_calls[1]["resume"]
    assert file_sha256(args.prepared_dir / "company_daily.parquet") == snapshot
    daily_manifest = args.prepared_dir / "company_daily.manifest.json"
    original_daily_manifest = daily_manifest.read_text(encoding="utf-8")
    altered = json.loads(original_daily_manifest)
    if manifest_drift == "checkpoint":
        altered["checkpoint"] = "different-checkpoint@" + "b" * 40
    else:
        altered["aggregation"][manifest_drift] += 1
    daily_manifest.write_text(json.dumps(altered), encoding="utf-8")
    # Keep the outer artifact checksum valid so rejection must inspect the
    # checkpoint/window identity rather than merely notice changed JSON bytes.
    preparation_manifest = args.prepared_dir / "preparation.manifest.json"
    prepared = json.loads(preparation_manifest.read_text(encoding="utf-8"))
    prepared["artifacts"]["daily_manifest"]["sha256"] = file_sha256(daily_manifest)
    preparation_manifest.write_text(json.dumps(prepared), encoding="utf-8")
    with pytest.raises(ValueError, match="Daily export differs"):
        study.prepare(args)
    assert len(exported_calls) == 1
    args.delay_hours = 0
    with pytest.raises(ValueError, match="changed"):
        study.prepare(args)
    assert len(imported_calls) == 3
