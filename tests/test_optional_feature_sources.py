"""Synthetic fixtures only: no historical data or network requests needed."""

from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from trading_system.data.optional_sources import prepare_feature_sources
from trading_system.experiments.config import ExperimentConfig
from trading_system.pipelines.compare_models import build_parser
from trading_system.pipelines.feature_arguments import apply_feature_sources, build_cli_features
from test_feature_expansion import prices, fundamentals


def events():
    return pd.DataFrame({
        "ticker": ["A", "A"], "kind": ["news", "news"],
        "available_at": ["2020-01-03T12:00:00Z", "2020-05-01T12:00:00Z"],
        "score": [.5, -.3], "event_id": ["synthetic-1", "synthetic-2"],
    })


def test_absent_sources_leave_prices_unchanged():
    data = prices()
    result, report = prepare_feature_sources(data)
    pd.testing.assert_frame_equal(result, data)
    assert report["fundamentals"]["status"] == "absent"
    assert report["sentiment"]["usable_rows"] == 0
    assert "feature_sources" not in data.attrs


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
@pytest.mark.parametrize("included", [("fundamentals",), ("sentiment",), ("fundamentals", "sentiment")])
def test_optional_sources_independent_causal_and_hashed(tmp_path, suffix, included):
    paths = {}
    for name, table in (("fundamentals", fundamentals()), ("sentiment", events())):
        if name in included:
            path = tmp_path / f"{name}.{suffix}"
            if suffix == "csv":
                table.to_csv(path, index=False)
            else:
                table.to_parquet(path, index=False)
            paths[name] = path
    data = pd.concat([prices(), prices(ticker="B")], ignore_index=True)
    result, report = prepare_feature_sources(data, **paths)
    for name in included:
        assert report[name]["status"] == "file"
        assert report[name]["sha256"] == sha256(paths[name].read_bytes()).hexdigest()
        assert report[name]["coverage_by_ticker"]["B"] == 0
    a = result[result.ticker == "A"]
    if "fundamentals" in included:
        assert a.loc[a.date.eq("2020-05-01"), "fund_eps_ttm"].item() == 2
        assert a.loc[a.date.eq("2020-05-02"), "fund_eps_ttm"].item() == 2.5
    if "sentiment" in included:
        assert a.loc[a.date.eq("2020-01-03"), "news_sentiment_7d"].isna().all()
        assert a.loc[a.date.eq("2020-01-04"), "news_sentiment_7d"].item() == .5


def test_disable_embedded_sources_without_changing_macro_or_input():
    data = prices().assign(fund_available_at="2019-12-31", fund_eps_ttm=2.,
                           news_sentiment_7d=.5, market_close=100.)
    result, report = prepare_feature_sources(data, disabled=True)
    assert "fund_eps_ttm" not in result and "news_sentiment_7d" not in result
    pd.testing.assert_series_equal(result.market_close, data.market_close)
    assert "fund_eps_ttm" in data
    assert report["fundamentals"]["status"] == "disabled"


def test_embedded_expired_future_and_undated_fundamentals_not_usable():
    for timestamp in (None, "2000-01-01", "2030-01-01"):
        data = prices().assign(fund_available_at=timestamp, fund_eps_ttm=2.)
        _, report = prepare_feature_sources(data)
        assert report["fundamentals"]["status"] == "embedded"
        assert report["fundamentals"]["usable_rows"] == 0


def test_explicit_missing_empty_invalid_files_fail_instead_of_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_feature_sources(prices(), fundamentals=tmp_path / "missing.csv")
    empty = tmp_path / "empty.csv"
    pd.DataFrame(columns=["ticker", "available_at", "equity"]).to_csv(empty, index=False)
    with pytest.raises(ValueError, match="non-empty"):
        prepare_feature_sources(prices(), fundamentals=empty)
    invalid = fundamentals().assign(available_at="not a timestamp")
    invalid.to_csv(empty, index=False)
    with pytest.raises(ValueError, match="publication"):
        prepare_feature_sources(prices(), fundamentals=empty)
    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_feature_sources(prices(), fundamentals=empty, disabled=True)


def test_cli_requires_expanded_and_no_external_does_not_change_default():
    args = build_parser().parse_args([])
    result, report = apply_feature_sources(prices(), args, ExperimentConfig())
    assert report is None
    args = build_parser().parse_args(["--no-external-features"])
    with pytest.raises(ValueError, match="expanded"):
        apply_feature_sources(result, args, ExperimentConfig())


def test_walkforward_shared_feature_builder_with_and_without_sources(tmp_path):
    path = tmp_path / "fundamentals.csv"
    fundamentals().to_csv(path, index=False)
    parser = build_parser()
    with_args = parser.parse_args(["--feature-set", "expanded", "--fundamentals", str(path)])
    featured, columns, _ = build_cli_features(prices(), with_args)
    assert featured.pe_ratio.notna().any()
    assert featured.attrs["feature_sources"]["fundamentals"]["status"] == "file"
    # Disable on a raw enriched input, not already computed features.
    enriched, _ = prepare_feature_sources(prices(), fundamentals=path)
    without_args = parser.parse_args(["--feature-set", "expanded", "--no-external-features"])
    baseline, baseline_columns, _ = build_cli_features(enriched, without_args)
    assert baseline.pe_ratio.isna().all()
    assert columns == baseline_columns
    assert baseline.attrs["feature_sources"]["mode"] == "disabled"


@pytest.mark.parametrize("has_sources", [False, True])
def test_training_and_artifact_work_with_and_without_sources(tmp_path, has_sources):
    from trading_system.experiments.runner import run_experiment
    from trading_system.artifacts.experiment import build_experiment_manifest
    from trading_system.models.specs import ModelSelection

    paths = {}
    if has_sources:
        path = tmp_path / "fundamentals.csv"
        fundamentals().to_csv(path, index=False)
        paths["fundamentals"] = path
    data, report = prepare_feature_sources(prices(), **paths)
    config = ExperimentConfig(
        feature_set="expanded", context_len=3, label_mode="triple_barrier",
        triple_barrier_max_holding=3, triple_barrier_volatility_window=5,
        model=ModelSelection("manual_ann", {"epochs": 1, "hidden_size": 4}),
    )
    result = run_experiment(data, config)
    assert np.isfinite(result.test_probabilities).all()
    assert ("pe_ratio" in result.bundle.feature_columns) == has_sources
    manifest = build_experiment_manifest(data, result)
    assert manifest.experiment_parameters["feature_sources"] == report
    assert result.bundle.feature_selector.state["feature_sources"] == report


def test_report_survives_parquet_roundtrip(tmp_path):
    data, report = prepare_feature_sources(prices())
    path = tmp_path / "prices.parquet"
    data.to_parquet(path, index=False)
    assert pd.read_parquet(path).attrs["feature_sources"] == report
