import argparse
import ast
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_system.artifacts.serialization import stable_config_hash
from trading_system.data.splits import chronological_train_val_test_split
from trading_system.experiments.config import ExperimentConfig
from trading_system.labels.config import LabelConfig
from trading_system.labels.registry import LabelContext, create_default_label_registry
from trading_system.labels.triple_barrier import generate_triple_barrier_labels
from trading_system.pipelines.label_arguments import add_label_arguments, apply_label_arguments
from trading_system.pipelines.walkforward import build_parser as walkforward_parser
from trading_system.pipelines.gridsearch_walkforward import build_parser as search_parser


ESTIMATORS = ("rolling_std", "atr", "bollinger")


def frame():
    prices = np.array([100, 102, 99, 105, 103, 108, 101, 104, 107, 110, 108, 113.0])
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(prices)),
        "adj_close": prices, "close": prices, "high": prices + 2, "low": prices - 2,
    })


def label(data, estimator="rolling_std", **kwargs):
    return generate_triple_barrier_labels(
        data, volatility_estimator=estimator, volatility_window=2,
        max_holding=2, cost_bps=0.0, **kwargs,
    )


def test_default_is_unchanged_return_std():
    data = frame()
    default = generate_triple_barrier_labels(
        data, volatility_window=2, max_holding=2, cost_bps=0.0,
    )
    pd.testing.assert_frame_equal(default, label(data))
    expected = data.adj_close.pct_change().rolling(2).std(ddof=0)
    np.testing.assert_allclose(default.barrier_scale, expected, equal_nan=True)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_short_and_constant_series(estimator):
    data = frame()
    data[["adj_close", "close", "high", "low"]] = 100.0
    result = label(data, estimator)
    assert (result.barrier_touch[result._label_known] == "vertical").all()
    assert (result.barrier_scale.dropna() == 0.0).all()
    assert not label(data.iloc[:2], estimator)._label_known.any()


@pytest.mark.parametrize("estimator,expected", [
    ("atr", 4.5 / 99),
    ("bollinger", 2 * 1.5 / 100.5),
])
def test_scales_have_explicit_dimensionless_formula(estimator, expected):
    result = label(frame(), estimator)
    assert result.loc[2, "barrier_scale"] == pytest.approx(expected)
    assert result.loc[2, "upper_barrier"] == pytest.approx(expected)
    assert result.loc[2, "lower_barrier"] == pytest.approx(-expected)
    assert result.barrier_scale.iloc[:2].isna().all()


def test_atr_aligns_raw_ohlc_to_adjusted_label_price():
    data = frame()
    split = data.copy()
    split.loc[:5, ["high", "low", "close"]] *= 2
    np.testing.assert_allclose(
        label(data, "atr").barrier_scale, label(split, "atr").barrier_scale,
        equal_nan=True,
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_scale_is_price_unit_invariant(estimator):
    data = frame()
    converted = data.copy()
    converted[["adj_close", "close", "high", "low"]] *= 100
    pd.testing.assert_series_equal(label(data, estimator).Label, label(converted, estimator).Label)
    np.testing.assert_allclose(
        label(data, estimator).barrier_scale, label(converted, estimator).barrier_scale,
        equal_nan=True, atol=1e-12,
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_causal_scales_and_split_boundaries(estimator):
    data = frame()
    data["_experiment_split"] = ["train"] * 6 + ["test"] * 6
    changed = data.copy()
    changed.loc[6:, ["adj_close", "close", "high", "low"]] *= 10
    config = LabelConfig.triple_barrier(
        volatility_estimator=estimator, volatility_window=2, max_holding=2,
    )
    registry = create_default_label_registry()
    original = registry.generate(data, config).frame
    perturbed = registry.generate(changed, config).frame
    columns = ["barrier_scale", "upper_barrier", "lower_barrier", "Label", "_label_known"]
    pd.testing.assert_frame_equal(original.loc[:5, columns], perturbed.loc[:5, columns])
    assert not original.loc[4:5, "_label_known"].any()
    assert original.loc[6, "barrier_scale"] == pytest.approx(label(data, estimator).loc[6, "barrier_scale"])


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_ticker_scales_are_independent(estimator):
    data = frame().assign(ticker="AAA")
    other = data.assign(ticker="BBB", high=data.high * 4)
    combined = pd.concat([data, other], ignore_index=True).sample(frac=1, random_state=2)
    result = create_default_label_registry().generate(
        combined, LabelConfig.triple_barrier(
            volatility_estimator=estimator, volatility_window=2, max_holding=2, cost_bps=0.0,
        ), LabelContext(group_col="ticker"),
    ).frame
    pd.testing.assert_series_equal(
        result[result.ticker == "AAA"].barrier_scale.reset_index(drop=True),
        label(data, estimator).barrier_scale,
    )


def test_cusum_sampling_does_not_change_with_barrier_estimator():
    expected = label(frame(), event_filter="cusum").label_event_id.notna()
    for estimator in ESTIMATORS:
        pd.testing.assert_series_equal(
            label(frame(), estimator, event_filter="cusum").label_event_id.notna(), expected,
        )


@pytest.mark.parametrize("column,value", [("high", np.nan), ("low", -1), ("close", np.inf), ("high", 1)])
def test_atr_rejects_invalid_ohlc(column, value):
    data = frame()
    data.loc[3, column] = value
    with pytest.raises(ValueError, match="ATR"):
        label(data, "atr")


def test_ohlc_only_required_for_atr():
    data = frame()[["date", "adj_close"]]
    label(data, "rolling_std")
    label(data, "bollinger")
    with pytest.raises(ValueError, match="ATR requires OHLC"):
        label(data, "atr")


def test_invalid_estimator_rejected():
    with pytest.raises(ValueError, match="volatility_estimator"):
        label(frame(), "unknown")
    with pytest.raises(ValueError, match="volatility_estimator"):
        ExperimentConfig(triple_barrier_volatility_estimator="unknown")


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_config_cli_and_hash(estimator):
    parser = argparse.ArgumentParser()
    add_label_arguments(parser)
    args = parser.parse_args(["--label-method", "triple-barrier", "--label-volatility-estimator", estimator])
    config = apply_label_arguments(ExperimentConfig(), args)
    assert config.triple_barrier_volatility_estimator == estimator
    resolved = config.resolved_label_config()
    assert resolved.parameters["volatility_estimator"] == estimator
    hashes = {stable_config_hash(asdict(LabelConfig.triple_barrier(volatility_estimator=item))) for item in ESTIMATORS}
    assert len(hashes) == 3
    for build_parser in (walkforward_parser, search_parser):
        assert build_parser().parse_args(["--label-volatility-estimator", estimator]).label_volatility_estimator == estimator


def test_comparison_rejects_estimator_for_other_label_methods():
    parser = argparse.ArgumentParser()
    add_label_arguments(parser)
    with pytest.raises(ValueError, match="Triple-barrier-only"):
        apply_label_arguments(ExperimentConfig(), parser.parse_args(["--label-volatility-estimator", "atr"]))


def test_absent_cli_option_preserves_preset_estimator():
    parser = argparse.ArgumentParser()
    add_label_arguments(parser)
    config = ExperimentConfig(
        label_mode="triple_barrier", triple_barrier_volatility_estimator="atr",
    )
    assert apply_label_arguments(config, parser.parse_args([])) == config


@pytest.mark.parametrize("estimator", [None, *ESTIMATORS])
def test_notebook_reconstructs_estimator_and_supports_old_manifests(estimator):
    # Execute only the pure reconstruction function, not notebook I/O or plots.
    path = Path(__file__).resolve().parents[1] / "notebooks/compare_model_benchmarks.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_test_labels")
    namespace = {
        "pd": pd, "LabelConfig": LabelConfig, "LabelContext": LabelContext,
        "create_default_label_registry": create_default_label_registry,
        "chronological_train_val_test_split": chronological_train_val_test_split,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    config = asdict(ExperimentConfig(
        label_mode="triple_barrier", train_ratio=0.5, val_ratio=0.25,
        triple_barrier_max_holding=1, triple_barrier_volatility_window=2,
        triple_barrier_volatility_estimator=estimator or "rolling_std",
    ))
    if estimator is None:
        config.pop("triple_barrier_volatility_estimator")
    result = namespace["build_test_labels"](frame(), config)
    expected = label(frame(), estimator or "rolling_std").set_index("date")
    np.testing.assert_allclose(result.barrier_scale, expected.loc[result.date, "barrier_scale"])
