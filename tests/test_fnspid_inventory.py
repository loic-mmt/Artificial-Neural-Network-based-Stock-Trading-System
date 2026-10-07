"""Offline checks for full-universe audits and causal frozen TRAIN eligibility."""

import csv
import json

import numpy as np
import pandas as pd
import pytest

from trading_system.data import fnspid_inventory as inventory
from trading_system.data.news_pilot_scoring import file_sha256


HEADER = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Lsa_summary"]


def _sources(tmp_path, names=("AAPL", "JPM", "ZERO"), *, rows=None, missing=None, invalid=None):
    calendar = pd.bdate_range("2020-01-01", periods=130, tz="UTC")
    market = pd.DataFrame([
        {"date": day, "ticker": ticker, "close": float(index + 100),
         "pnl": 999999., "Label_id": 2}
        for ticker in names for index, day in enumerate(calendar)
        if (ticker, day.strftime("%Y-%m-%d")) != missing
    ])
    if invalid:
        mask = market.ticker.eq(invalid[0]) & market.date.eq(pd.Timestamp(invalid[1], tz="UTC"))
        market.loc[mask, "close"] = np.nan
    prices = tmp_path / "prices.parquet"
    market.to_parquet(prices, index=False)
    info = tmp_path / "dataset-info.json"
    info.write_text(json.dumps({"id": "Zdong104/FNSPID", "sha": "a" * 40}), encoding="utf-8")
    source = tmp_path / "news.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows or [])
    return prices, source, info, calendar


def _news(ticker, decision, *, suffix="", url=""):
    published = pd.Timestamp(decision, tz="UTC") - pd.Timedelta(hours=36)
    return [published.isoformat(), f"Title {ticker} {decision} {suffix}", ticker,
            url, "Fixture", "DERIVED SUMMARY MUST NOT BE USED"]


def _build(tmp_path, sources, **overrides):
    prices, source, info, _ = sources
    options = {"start": "2020-01-01", "end": "2020-07-01", "dataset_info": info,
               "selection_start": "2020-01-15", "selection_end": "2020-02-01",
               "warmup_sessions": 0, "min_observed_train_days": 3,
               "min_train_observed_fraction": .1, "long_gap_days": 5, "chunksize": 3}
    options.update(overrides)
    return inventory.build_fnspid_inventory(prices, [source], tmp_path / "inventory", **options)


def test_inventory_all_price_tickers_zero_months_and_one_import(tmp_path, monkeypatch):
    day = _news("AAPL", "2020-01-15")
    rows = [day, day, *[_news("AAPL", date) for date in ("2020-01-16", "2020-01-17")],
            _news("JPM", "2020-01-03"),
            ["NaN", "Bad date", "AAPL", "", "", ""],
            ["2020-01-10", " \t ", "AAPL", "", "", ""]]
    sources = _sources(tmp_path, rows=rows)
    calls = []
    original = inventory.import_fnspid

    def wrapped(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(inventory, "import_fnspid", wrapped)
    result = _build(tmp_path, sources)
    assert len(calls) == 1
    assert calls[0]["tickers"] == ["AAPL", "JPM", "ZERO"]
    assert pd.Timestamp(calls[0]["start"]) == pd.Timestamp("2019-12-30", tz="UTC")
    assert result["tickers"] == ["AAPL"]
    assert result["report"]["universe_count"] == 3
    monthly = pd.read_parquet(result["monthly_path"])
    assert len(monthly) == 18
    january = monthly.loc[monthly.ticker.eq("AAPL") & monthly.month.eq("2020-01")].iloc[0]
    assert january.raw_news_rows == 4
    assert january.deduplicated_news_associations == january.usable_news_associations == 3
    assert january.observed_decision_days == 3
    assert monthly.loc[monthly.ticker.eq("ZERO"), "raw_news_rows"].eq(0).all()
    assert monthly.coverage_status.eq("unknown").all()
    assert not monthly.historical_coverage_proven.any()
    zero = result["report"]["tickers"]["ZERO"]
    assert zero["long_unobserved_gaps"][0]["session_days"] == len(sources[3])
    assert zero["months_without_usable_news"] == [f"2020-{month:02d}" for month in range(1, 7)]
    assert result["report"]["tickers"]["JPM"]["observed_decision_days"] == 1
    assert result["report"]["tickers"]["JPM"]["observed_train_days"] == 0
    assert result["selection"]["rules"]["uses_performance"] is False
    assert result["manifest_sha256"] == file_sha256(result["manifest_path"])


def test_timing_versions_conflicts_and_future_news_do_not_qualify_train(tmp_path):
    rows = [_news("AAPL", day) for day in ("2020-01-15", "2020-01-16", "2020-01-17")]
    rows += [_news("VERSION", "2020-01-15", suffix="first", url="https://example.com/revised"),
             _news("VERSION", "2020-01-15", suffix="updated", url="https://example.com/revised")]
    conflict = _news("CONFLICT", "2020-01-15", url="https://example.com/conflict")
    another = list(conflict)
    another[0] = "2020-01-14T12:00:00Z"
    rows += [conflict, another]
    rows += [_news("LATE", day) for day in ("2020-03-16", "2020-03-17", "2020-03-18")]
    sources = _sources(tmp_path, names=("AAPL", "VERSION", "CONFLICT", "LATE"), rows=rows)
    result = _build(tmp_path, sources)
    assert result["tickers"] == ["AAPL"]
    monthly = pd.read_parquet(result["monthly_path"])
    version = monthly.loc[monthly.ticker.eq("VERSION") & monthly.month.eq("2020-01")].iloc[0]
    assert version.deduplicated_news_associations == version.timing_rejected_associations == 2
    assert version.usable_news_associations == version.observed_decision_days == 0
    conflict = monthly.loc[monthly.ticker.eq("CONFLICT") & monthly.month.eq("2020-01")].iloc[0]
    assert conflict.timing_rejected_associations == 1
    assert result["report"]["tickers"]["LATE"]["observed_decision_days"] == 3
    assert result["report"]["tickers"]["LATE"]["observed_train_days"] == 0


def test_decision_window_has_inclusive_lower_bound_and_exclusive_cutoff(tmp_path):
    rows = [["2020-01-13T00:00:00Z", "Lower bound", "LOWER", "", "", ""],
            ["2020-01-14T00:00:00Z", "At decision cutoff", "UPPER", "", "", ""]]
    sources = _sources(tmp_path, names=("LOWER", "UPPER"), rows=rows)
    result = _build(tmp_path, sources, selection_end="2020-01-16", min_observed_train_days=1)
    assert result["tickers"] == ["LOWER"]
    assert result["report"]["tickers"]["UPPER"]["observed_train_days"] == 0
    assert result["report"]["tickers"]["UPPER"]["observed_decision_days"] == 1


def test_price_completeness_is_explicit_full_period_quality_filter(tmp_path):
    rows = [_news(ticker, day) for ticker in ("AAPL", "MISSING", "INVALID")
            for day in ("2020-01-15", "2020-01-16", "2020-01-17")]
    sources = _sources(tmp_path, names=("AAPL", "MISSING", "INVALID"), rows=rows,
                       missing=("MISSING", "2020-06-30"), invalid=("INVALID", "2020-06-29"))
    result = _build(tmp_path, sources)
    assert result["tickers"] == ["AAPL"]
    for name in ("MISSING", "INVALID"):
        assert result["report"]["tickers"][name]["observed_train_days"] == 3
        assert result["selection"]["rejected"][name] == ["incomplete_price_calendar"]
    assert result["report"]["tickers"]["INVALID"]["invalid_close_rows"] == 1
    assert "retrospective" in result["selection"]["rules"]["price_completeness_basis"]


def test_changing_prices_and_pnl_does_not_change_news_selection(tmp_path):
    rows = [_news("AAPL", day) for day in ("2020-01-15", "2020-01-16", "2020-01-17")]
    sources = _sources(tmp_path, rows=rows)
    first = _build(tmp_path, sources)
    market = pd.read_parquet(sources[0])
    market["close"] = market["close"].iloc[::-1].to_numpy() * 100
    market["pnl"] = -1234567.
    market["Label_id"] = 0
    market.to_parquet(sources[0], index=False)
    result = inventory.build_fnspid_inventory(
        sources[0], [sources[1]], tmp_path / "other-inventory", start="2020-01-01", end="2020-07-01",
        dataset_info=sources[2], selection_start="2020-01-15", selection_end="2020-02-01",
        warmup_sessions=0, min_observed_train_days=3, long_gap_days=5, chunksize=3)
    assert first["tickers"] == result["tickers"] == ["AAPL"]
    assert first["report"]["tickers"] == result["report"]["tickers"]


def test_resume_and_loader_preserve_artifacts_and_detect_drift(tmp_path):
    sources = _sources(tmp_path, rows=[_news("AAPL", day)
                       for day in ("2020-01-15", "2020-01-16", "2020-01-17")])
    original = _build(tmp_path, sources)
    checksum = file_sha256(original["manifest_path"])
    resumed = _build(tmp_path, sources, resume=True)
    assert resumed["report"] == original["report"]
    assert file_sha256(original["manifest_path"]) == checksum
    loaded = inventory.load_fnspid_inventory(tmp_path / "inventory", verify_sources=True)
    assert loaded["tickers"] == ["AAPL"]
    with pytest.raises(FileExistsError):
        _build(tmp_path, sources)
    with pytest.raises(ValueError, match="changed"):
        _build(tmp_path, sources, resume=True, delay_hours=12)
    original["monthly_path"].write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="corrupted"):
        _build(tmp_path, sources, resume=True)
    with pytest.raises(ValueError, match="corrupted"):
        inventory.load_fnspid_inventory(tmp_path / "inventory")


def test_resume_rejects_source_drift_and_keeps_original_inventory(tmp_path):
    sources = _sources(tmp_path, rows=[_news("AAPL", "2020-01-15")])
    original = _build(tmp_path, sources)
    snapshot = file_sha256(original["manifest_path"])
    with sources[1].open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(_news("AAPL", "2020-01-16"))
    with pytest.raises(ValueError, match="input_files changed"):
        _build(tmp_path, sources, resume=True)
    assert file_sha256(original["manifest_path"]) == snapshot
    with pytest.raises(ValueError, match="original source"):
        inventory.load_fnspid_inventory(tmp_path / "inventory", verify_sources=True)


def test_loader_rejects_reusable_import_corruption(tmp_path):
    result = _build(tmp_path, _sources(tmp_path))
    result["articles_path"].write_bytes(b"changed imported articles")
    with pytest.raises(ValueError, match="import artifact checksum"):
        inventory.load_fnspid_inventory(tmp_path / "inventory")


def test_dry_run_is_readonly_does_not_scan_import_or_hash_csv(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    monkeypatch.setattr(inventory, "import_fnspid", lambda *args, **kwargs: pytest.fail("must not import"))
    original_hash = inventory.file_sha256

    def selective_hash(path):
        assert path != sources[1], "dry-run must not read the large source CSV"
        return original_hash(path)

    monkeypatch.setattr(inventory, "file_sha256", selective_hash)
    result = _build(tmp_path, sources, dry_run=True)
    assert result["status"] == "planned"
    assert result["universe_count"] == 3
    assert result["selected_tickers"] is None
    assert not (tmp_path / "inventory").exists()


def test_malformed_csv_fails_without_completed_inventory(tmp_path):
    sources = _sources(tmp_path, rows=[["2020-01-01", "Title", "AAPL", "EXTRA"]])
    with pytest.raises(ValueError, match="Malformed CSV row"):
        _build(tmp_path, sources)
    assert not (tmp_path / "inventory" / "inventory.manifest.json").exists()


@pytest.mark.parametrize("options", [
    {"delay_hours": float("nan")}, {"lookback_hours": 0},
    {"min_train_observed_fraction": 1.1}, {"min_observed_train_days": 0},
    {"warmup_sessions": True}, {"selection_start": "2020-01-15T12:00:00"},
    {"selection_end": "2020-07-01"},
])
def test_invalid_settings_refuse_before_import(tmp_path, monkeypatch, options):
    sources = _sources(tmp_path)
    monkeypatch.setattr(inventory, "import_fnspid", lambda *args, **kwargs: pytest.fail("must refuse before import"))
    with pytest.raises(ValueError):
        _build(tmp_path, sources, **options)


def test_default_selection_preserves_predeclared_thresholds_and_seals_validation(tmp_path):
    sources = _sources(tmp_path)
    result = inventory.build_fnspid_inventory(
        sources[0], [sources[1]], tmp_path / "inventory", start="2020-01-01", end="2020-07-01",
        dataset_info=sources[2], warmup_sessions=20, dry_run=True)
    rules = result["settings"]["selection"]
    assert rules["min_observed_train_days"] == 60
    assert rules["min_train_observed_fraction"] == .1
    assert pd.Timestamp(rules["start"]) == sources[3][20]
    assert pd.Timestamp(rules["end_exclusive"]) < pd.Timestamp(rules["first_inner_validation_start"])
    assert rules["cv_spec"]["gap_bars"] == 5
