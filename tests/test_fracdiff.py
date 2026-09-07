import argparse
import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from trading_system.features import fracdiff as fd
from trading_system.features.fracdiff import (
    FRACDIFF_FEATURE, FracDiffConfig, FracDiffTransformer,
    fractional_difference, fractional_weights,
)
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import run_validation_experiment, evaluate_experiment_test
from trading_system.experiments.walkforward import walk_forward_classifier, walk_forward_oracle_ann
from trading_system.models.specs import ModelSelection
from trading_system.artifacts.experiment import build_experiment_manifest
from trading_system.artifacts.serialization import stable_config_hash
from trading_system.pipelines.feature_arguments import (
    add_feature_arguments, apply_feature_arguments, fracdiff_config_from_args,
)


def market_frame(rows=400, seed=3):
    price = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.012, rows)))
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=rows),
        "adj_close": price, "open": price, "close": price,
        "high": price * 1.01, "low": price * 0.99, "volume": 1e6,
        "signal": np.r_[0, np.diff(np.log(price))],
    })


def experiment(**kwargs):
    return ExperimentConfig(
        fracdiff=FracDiffConfig(order=0.5), context_len=3,
        label_mode="triple_barrier", triple_barrier_max_holding=3,
        triple_barrier_volatility_window=5, decision_mode="argmax",
        model=ModelSelection("manual_ann", {"epochs": 1, "hidden_size": 4}),
        **kwargs,
    )


def test_weights_recurrence_and_endpoints():
    np.testing.assert_array_equal(fractional_weights(0), [1])
    np.testing.assert_array_equal(fractional_weights(1), [1, -1])
    weights = fractional_weights(0.5, threshold=0.05)
    np.testing.assert_allclose(weights, [1, -0.5, -0.125, -0.0625])
    values = np.arange(10.) ** 2
    np.testing.assert_array_equal(fractional_difference(values, [1]), values)
    np.testing.assert_allclose(fractional_difference(values, [1, -1])[1:], np.diff(values))
    result = fractional_difference(values, weights)
    assert np.isnan(result[:3]).all()
    for i in range(3, len(values)):
        assert result[i] == pytest.approx(sum(weights[k] * values[i-k] for k in range(4)))


@pytest.mark.parametrize("overrides", [
    {"order": 0}, {"order": 1}, {"order": np.nan}, {"order": True},
    {"threshold": 0}, {"threshold": 1}, {"max_terms": 0},
    {"min_samples": 3}, {"adf_pvalue": 1}, {"candidates": ()},
    {"candidates": (0.5, 0.1)}, {"candidates": (0.5, 0.5)},
])
def test_config_validation(overrides):
    with pytest.raises((ValueError, TypeError)):
        FracDiffConfig(**overrides)


def test_no_silent_truncation_or_missing_value_filling():
    with pytest.raises(ValueError, match="max_terms"):
        fractional_weights(0.5, threshold=1e-12, max_terms=3)
    with pytest.raises(ValueError, match="finite"):
        fractional_difference([1, np.nan, 3], [1, -0.5])
    assert np.isnan(fractional_difference([1, 2], [1, -0.5, -0.125])).all()


def test_causality_ticker_isolation_and_state_roundtrip():
    data = pd.concat([market_frame().assign(ticker="A"), market_frame(seed=4).assign(ticker="B")])
    data = data.sample(frac=1, random_state=1).reset_index(drop=True)
    train = data[data.date < "2020-09-01"]
    transform = FracDiffTransformer(FracDiffConfig(order=0.5), group_col="ticker").fit(train)
    frozen = transform.state_dict()
    result = transform.transform(data)
    changed = data.copy()
    changed.loc[(changed.date >= "2020-09-01") | (changed.ticker == "B"), "adj_close"] *= 2
    actual = transform.transform(changed)
    mask = (data.date < "2020-09-01") & (data.ticker == "A")
    pd.testing.assert_series_equal(result.loc[mask, FRACDIFF_FEATURE], actual.loc[mask, FRACDIFF_FEATURE])
    isolated = FracDiffTransformer(FracDiffConfig(order=0.5)).fit(train[train.ticker == "A"])
    pd.testing.assert_series_equal(
        result.loc[data.ticker == "A", FRACDIFF_FEATURE].reset_index(drop=True),
        isolated.transform(data[data.ticker == "A"])[FRACDIFF_FEATURE],
    )
    assert transform.state_dict() == frozen
    restored = FracDiffTransformer.from_state_dict(json.loads(json.dumps(frozen)))
    pd.testing.assert_frame_equal(restored.transform(data), result)
    with pytest.raises(ValueError, match="ticker"):
        restored.transform(market_frame().assign(ticker="UNSEEN"))


def test_auto_selection_uses_common_train_dates_and_smallest_passing_d(monkeypatch):
    calls = []
    pvalues = iter([0.2, 0.01, 0.005])
    def diagnostics(values, original):
        calls.append(original.copy())
        return {"adf_pvalue": next(pvalues), "correlation": 0.9, "status": "tested"}
    monkeypatch.setattr(fd, "_diagnostics", diagnostics)
    data = market_frame()
    transform = FracDiffTransformer(FracDiffConfig(candidates=(0.2, 0.5, 0.8))).fit(data.iloc[:250])
    state = transform.groups["__single__"]
    assert state["order"] == 0.5
    assert state["train_end"] == str(data.date.iloc[249])
    assert len(calls) == 3
    for values in calls:
        np.testing.assert_array_equal(values, calls[0])
    assert calls[0][-1] == pytest.approx(np.log(data.adj_close.iloc[249]))
    transform.transform(data)  # Must not rerun ADF (iterator exhausted).


def test_auto_does_not_claim_stationarity_when_adf_fails(monkeypatch):
    monkeypatch.setattr(fd, "_diagnostics", lambda *args: {"adf_pvalue": 0.9, "correlation": 1, "status": "tested"})
    with pytest.raises(ValueError, match="no candidate passes"):
        FracDiffTransformer(FracDiffConfig()).fit(market_frame())
    fixed = FracDiffTransformer(FracDiffConfig(order=0.5)).fit(market_frame())
    assert fixed.groups["__single__"]["diagnostics"][0]["adf_pvalue"] == 0.9


def test_real_adf_diagnostics_and_short_constant_inputs():
    transformer = FracDiffTransformer(FracDiffConfig(order=0.5)).fit(market_frame())
    diagnostic = transformer.groups["__single__"]["diagnostics"][0]
    assert 0 <= diagnostic["adf_pvalue"] <= 1
    assert -1 <= diagnostic["correlation"] <= 1
    with pytest.raises(ValueError, match="sufficient history"):
        FracDiffTransformer(FracDiffConfig()).fit(market_frame(10))
    with pytest.raises(ValueError, match="no candidate passes"):
        FracDiffTransformer(FracDiffConfig()).fit(market_frame().assign(adj_close=100.0))


@pytest.mark.parametrize("bad", [0, -1, np.inf, np.nan])
def test_invalid_prices_rejected(bad):
    data = market_frame()
    data.loc[4, "adj_close"] = bad
    with pytest.raises(ValueError, match="positive"):
        FracDiffTransformer(FracDiffConfig(order=0.5)).fit(data)


def test_cli_configuration_and_json_roundtrip():
    parser = argparse.ArgumentParser()
    add_feature_arguments(parser)
    base = ExperimentConfig()
    assert apply_feature_arguments(base, parser.parse_args([])) == base
    auto = apply_feature_arguments(base, parser.parse_args(["--fracdiff"]))
    assert auto.fracdiff.order is None
    fixed = apply_feature_arguments(base, parser.parse_args(["--fracdiff", "--fracdiff-order", "0.4"]))
    assert fixed.fracdiff.order == 0.4
    assert stable_config_hash(asdict(base)) != stable_config_hash(asdict(fixed))
    payload = asdict(fixed)
    payload["model"] = fixed.model  # Existing ModelSelection reconstruction is separate.
    assert ExperimentConfig(**payload).fracdiff == fixed.fracdiff
    with pytest.raises(ValueError, match="require --fracdiff"):
        fracdiff_config_from_args(parser.parse_args(["--fracdiff-order", "0.4"]))


def test_auto_fit_is_unchanged_when_validation_and_test_change(monkeypatch):
    monkeypatch.setattr(fd, "_diagnostics", lambda values, original: {
        "adf_pvalue": 0.01, "correlation": float(np.corrcoef(values, original)[0, 1]),
        "status": "test_double",
    })
    data = market_frame()
    config = replace(experiment(), fracdiff=FracDiffConfig(candidates=(0.3, 0.5)))
    original = run_validation_experiment(data, config)
    changed = data.copy()
    changed.loc[280:, ["adj_close", "open", "high", "low", "close"]] *= 1.5
    perturbed = run_validation_experiment(changed, config)
    assert original.bundle.fracdiff_transformer.state_dict() == perturbed.bundle.fracdiff_transformer.state_dict()
    group = original.bundle.fracdiff_transformer.groups["__single__"]
    assert group["train_rows"] == 280
    assert group["order"] == 0.3


def test_shared_cli_is_available_in_all_general_pipelines():
    from trading_system.pipelines.compare_models import build_parser as comparison_parser
    from trading_system.pipelines.walkforward import build_parser as walkforward_parser
    from trading_system.pipelines.gridsearch_walkforward import build_parser as search_parser
    for build_parser in (comparison_parser, walkforward_parser, search_parser):
        args = build_parser().parse_args(["--fracdiff", "--fracdiff-order", "0.5"])
        assert fracdiff_config_from_args(args) == FracDiffConfig(order=0.5)


@pytest.mark.parametrize("multi", [False, True])
def test_static_training_artifacts_and_frozen_final_test(multi, monkeypatch):
    data = market_frame()
    if multi:
        data = pd.concat([data.assign(ticker="A"), market_frame(seed=4).assign(ticker="B")], ignore_index=True)
    config = experiment(universe="multi" if multi else "single")
    validation = run_validation_experiment(data, config)
    assert FRACDIFF_FEATURE in validation.bundle.feature_columns
    state = validation.bundle.fracdiff_transformer.state_dict()
    changed = data.copy()
    # Only final-test prices change; fitted preprocessing cannot change.
    cutoff = market_frame().date.iloc[340]
    changed.loc[changed.date >= cutoff, ["adj_close", "close", "open", "high", "low"]] *= 1.2
    def no_fit(*args, **kwargs):
        raise AssertionError("Final test attempted to fit FracDiff")
    monkeypatch.setattr(FracDiffTransformer, "fit", no_fit)
    result = evaluate_experiment_test(changed, validation)
    assert result.bundle.fracdiff_transformer.state_dict() == state
    assert np.isfinite(result.test_probabilities).all()
    manifest = build_experiment_manifest(changed, result)
    assert manifest.experiment_parameters["fracdiff_state"]["groups"] == state["groups"]
    from trading_system.experiments.comparison import ComparisonRun, flatten_experiment_result
    row = flatten_experiment_result(
        result, ComparisonRun("manual_ann", {}, 1), config_hash="test", duration_seconds=0.0,
    )
    assert json.loads(row["fracdiff_state"])["groups"] == state["groups"]


@pytest.mark.parametrize("legacy", [False, True])
def test_walkforward_retraining_passes_feature_and_records_state(legacy):
    data = market_frame()
    kwargs = dict(
        train_ratio=0.6, val_ratio=0.2, walkforward_step=40,
        context_len=3, label_mode="triple_barrier", triple_barrier_max_holding=3,
        triple_barrier_volatility_window=5, fracdiff_config=FracDiffConfig(order=0.5),
        decision_mode="argmax",
    )
    if legacy:
        result = walk_forward_oracle_ann(data, ["signal"], epochs=1, hidden=4, **kwargs)
    else:
        result = walk_forward_classifier(
            data, ["signal"], model_selection=ModelSelection("manual_ann", {"epochs": 1, "hidden_size": 4}), **kwargs,
        )
    assert len(result["retrain_logs"]) == 2
    for entry in result["retrain_logs"]:
        state = entry["fracdiff_state"]["groups"]["__single__"]
        assert state["order"] == 0.5
        assert pd.Timestamp(state["train_end"]) < data.date.iloc[entry["start_idx"]]
    assert result["test_metrics"]
