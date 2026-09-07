from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from trading_system.data.feature_sources import attach_fundamentals, attach_sentiment, score_text
from trading_system.features.expanded import (
    compute_expanded_features, feature_columns, ExpandedFeatureSelector,
)
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import run_validation_experiment, evaluate_experiment_test
from trading_system.experiments.walkforward import walk_forward_classifier
from trading_system.models.specs import ModelSelection
from trading_system.artifacts.experiment import build_experiment_manifest
from trading_system.pipelines.feature_arguments import apply_feature_arguments, build_cli_features


def prices(n=260, ticker="A"):
    x = np.arange(n)
    price = 100 + x * .05 + 2 * np.sin(x / 5)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n), "ticker": ticker,
        "open": price, "high": price + 1, "low": price - 1,
        "close": price, "adj_close": price, "volume": 10000 + 100 * np.cos(x),
        "sector": "Technology",
    })


def fundamentals():
    return pd.DataFrame({
        "ticker": ["A", "A"], "available_at": ["2019-12-31T12:00Z", "2020-05-01T12:00Z"],
        "shares_outstanding": [10., 12.], "equity": [200., 240.],
        "assets": [400., 450.], "debt": [100., 110.], "cash": [50., 60.],
        "revenue_ttm": [100., 120.], "net_income_ttm": [20., 25.],
        "ebitda_ttm": [50., 60.], "eps_ttm": [2., 2.5],
        "revenue_ttm_prev_year": [80., 100.], "eps_ttm_prev_year": [1.5, 2.],
    })


def test_fundamentals_publication_join_no_backfill_same_day_or_cross_ticker():
    data = pd.concat([prices(), prices(ticker="B")], ignore_index=True).sample(frac=1, random_state=1)
    source = fundamentals().iloc[1:]
    enriched = attach_fundamentals(data, source)
    known = (enriched.ticker == "A") & (enriched.date > "2020-05-01")
    assert enriched.loc[~known, "fund_eps_ttm"].isna().all()
    assert enriched.loc[known, "fund_eps_ttm"].eq(2.5).all()
    pd.testing.assert_series_equal(enriched.date, data.date.reset_index(drop=True))


def test_fundamentals_reject_duplicates_and_expire():
    with pytest.raises(ValueError, match="Duplicate"):
        attach_fundamentals(prices(), pd.concat([fundamentals()] * 2))
    expired = attach_fundamentals(prices(), fundamentals().iloc[:1], max_age_days=10)
    assert expired.loc[expired.date > "2020-01-10", "fund_equity"].isna().all()


def test_expanded_ratio_formulas_and_negative_earnings():
    enriched = attach_fundamentals(prices(), fundamentals())
    result = compute_expanded_features(enriched)
    first = result.iloc[0]
    assert first.pe_ratio == pytest.approx(50)
    assert first.pb_ratio == pytest.approx(5)
    assert first.ev_ebitda == pytest.approx(21)
    assert first.profit_margin == pytest.approx(.2)
    assert first.return_on_equity == pytest.approx(.1)
    assert first.return_on_assets == pytest.approx(.05)
    assert first.revenue_growth_yoy == pytest.approx(.25)
    negative = enriched.assign(fund_eps_ttm=-2, fund_equity=-200)
    negative = compute_expanded_features(negative)
    assert negative.pe_ratio.isna().all()
    assert negative.pb_ratio.isna().all()
    assert (negative.earnings_yield < 0).all()


def test_undated_yahoo_snapshots_are_ignored_and_features_optional():
    data = prices().assign(market_cap=1e9, book_value=50, trailing_eps=10, shares_outstanding=1e6)
    result = compute_expanded_features(data)
    assert result.pe_ratio.isna().all()
    assert result.market_cap_log.isna().all()
    assert result.vix_level.isna().all()
    assert result.sector_momentum_60.notna().any()
    assert set(feature_columns()).issubset(result.columns)


def test_fundamental_availability_checked_even_for_pre_enriched_data():
    enriched = attach_fundamentals(prices(), fundamentals())
    enriched["fund_available_at"] = "2030-01-01"
    assert compute_expanded_features(enriched).pe_ratio.isna().all()


def test_sector_surge_uses_prior_window_and_is_causal():
    data = pd.concat([prices(), prices(ticker="B")], ignore_index=True)
    original = compute_expanded_features(data)
    changed = data.copy()
    mask = changed.date >= "2020-07-01"
    changed.loc[mask, ["open", "high", "low", "close", "adj_close"]] *= 2
    perturbed = compute_expanded_features(changed)
    prior = original.date < "2020-07-01"
    pd.testing.assert_frame_equal(original.loc[prior, list(feature_columns())], perturbed.loc[prior, list(feature_columns())])
    daily = original[original.ticker == "A"]
    expected = (daily.sector_ret_1 - daily.sector_ret_1.shift().rolling(20).mean()) / daily.sector_ret_1.shift().rolling(20).std(ddof=0)
    np.testing.assert_allclose(daily.sector_surge_z_20, expected, equal_nan=True)


def test_macro_conflicts_rejected():
    data = pd.concat([prices().assign(market_close=100), prices(ticker="B").assign(market_close=200)])
    with pytest.raises(ValueError, match="Conflicting"):
        compute_expanded_features(data)


def test_real_vader_and_english_guard():
    assert score_text("Excellent strong profit growth!") > 0
    assert score_text("Terrible failure, devastating losses.") < 0
    assert -1 <= score_text("Good profit. Terrible losses.", kind="earnings") <= 1
    source = pd.DataFrame({"ticker": ["A"], "available_at": ["2020-01-01"], "kind": ["news"], "text": ["Bonjour"], "language": ["fr"]})
    with pytest.raises(ValueError, match="language"):
        attach_sentiment(prices(), source)


def test_sentiment_windows_publication_causality_and_no_cross_ticker():
    source = pd.DataFrame({
        "ticker": ["A", "A", "A"], "kind": ["news", "news", "earnings"],
        "available_at": ["2020-01-02T12:00Z", "2020-01-04T12:00Z", "2020-01-02T12:00Z"],
        "score": [1., -1., .5],
    })
    data = pd.concat([prices(40), prices(40, ticker="B")], ignore_index=True)
    result = attach_sentiment(data, source)
    assert result.loc[1, "news_count_7d"] != result.loc[1, "news_count_7d"]  # Unknown before first observed source event.
    assert result.loc[2, "news_sentiment_7d"] == 1
    assert result.loc[4, "news_sentiment_7d"] == 0
    assert result.loc[11, "news_count_7d"] == 0
    assert result.loc[4, "earnings_sentiment_30d"] == .5
    assert result.loc[result.ticker == "B", "news_sentiment_7d"].isna().all()
    changed = source.copy()
    changed.loc[1, "score"] = .8
    pd.testing.assert_frame_equal(result.iloc[:4], attach_sentiment(data, changed).iloc[:4])


def test_selector_train_only_missingness_constants_and_ablation():
    train = pd.DataFrame({"ok": [1, 2, 3, 4], "missing": [np.nan]*4, "constant": [1]*4, "sparse": [1, np.nan, np.nan, np.nan]})
    selector = ExpandedFeatureSelector(.5).fit(train, train.columns)
    assert selector.columns == ("ok",)
    assert selector.state["dropped"] == {"missing": "missing", "constant": "constant", "sparse": "missing"}
    assert feature_columns(("fundamentals",)) == tuple(c for c in feature_columns(("fundamentals",)))
    with pytest.raises(ValueError):
        feature_columns(("bogus",))


@pytest.mark.parametrize("groups", [("technical", "market", "sector"), ("fundamentals",)])
def test_static_expansion_training_frozen_selection_and_artifacts(groups, monkeypatch):
    data = attach_fundamentals(prices(), fundamentals())
    config = ExperimentConfig(
        feature_set="expanded", expanded_feature_groups=groups, context_len=3,
        label_mode="triple_barrier", triple_barrier_max_holding=3,
        triple_barrier_volatility_window=5, decision_mode="argmax",
        model=ModelSelection("manual_ann", {"epochs": 1, "hidden_size": 4}),
    )
    validation = run_validation_experiment(data, config)
    frozen = json.dumps(validation.bundle.feature_selector.state_dict(), sort_keys=True)
    def no_fit(*args, **kwargs):
        raise AssertionError("Final test refitted feature selection")
    monkeypatch.setattr(ExpandedFeatureSelector, "fit", no_fit)
    result = evaluate_experiment_test(data, validation)
    assert json.dumps(result.bundle.feature_selector.state_dict(), sort_keys=True) == frozen
    assert np.isfinite(result.test_probabilities).all()
    manifest = build_experiment_manifest(data, result)
    assert tuple(manifest.experiment_parameters["feature_selection"]["selected"]) == result.bundle.feature_columns


def test_walkforward_expanded_selection_is_logged():
    data = compute_expanded_features(prices())
    result = walk_forward_classifier(
        data, feature_columns(), expanded_min_coverage=.5, context_len=3,
        train_ratio=.6, val_ratio=.2, walkforward_step=100,
        label_mode="triple_barrier", triple_barrier_max_holding=3,
        triple_barrier_volatility_window=5,
        model_selection=ModelSelection("manual_ann", {"epochs": 1, "hidden_size": 4}),
    )
    state = result["retrain_logs"][0]["feature_selection"]
    assert state["dropped"]["pe_ratio"] == "missing"
    assert state["train_rows"] < result["retrain_logs"][0]["n_hist"]


def test_cli_default_unchanged_and_expanded_groups_supported():
    from trading_system.pipelines.compare_models import build_parser
    parser = build_parser()
    base = ExperimentConfig()
    assert apply_feature_arguments(base, parser.parse_args([])) == base
    args = parser.parse_args(["--feature-set", "expanded", "--feature-groups", "technical,sector"])
    config = apply_feature_arguments(base, args)
    assert config.feature_set == "expanded"
    assert config.expanded_feature_groups == ("technical", "sector")
    with pytest.raises(ValueError, match="expanded"):
        apply_feature_arguments(base, parser.parse_args(["--feature-groups", "sector"]))
