import argparse

import numpy as np
import pandas as pd
import pytest

from trading_system.experiments.config import ExperimentConfig
from trading_system.labels.config import LabelConfig
from trading_system.labels.forward_return import (
    build_forward_return_labels,
    build_forward_return_labels_by_ticker,
)
from trading_system.labels.registry import LabelContext, create_default_label_registry
from trading_system.pipelines.label_arguments import (
    add_label_arguments,
    apply_label_arguments,
    label_config_from_args,
)


def _frame(prices: list[float], *, ticker: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(prices)),
            "adj_close": prices,
        }
    )
    if ticker is not None:
        frame["ticker"] = ticker
    return frame


def test_forward_return_preserves_historical_threshold_semantics():
    frame = _frame([100.0, 102.0, 101.0, 98.0, 100.0, 103.0])

    labeled, report = build_forward_return_labels(
        frame,
        horizon=2,
        buy_threshold=0.02,
        sell_threshold=0.01,
    )

    assert labeled["Label"].tolist() == [
        "Hold",
        "Sell",
        "Hold",
        "Buy",
        "Hold",
        "Hold",
    ]
    pd.testing.assert_series_equal(
        labeled["label_score"], labeled["fwd_ret"], check_names=False
    )
    assert report["n_known"] == 4
    assert report["n_unknown"] == 2


def test_forward_return_registry_marks_each_split_tail_unknown():
    frame = _frame([100.0, 101.0, 103.0, 106.0, 50.0, 49.0, 47.0, 44.0])
    frame["_experiment_split"] = ["train"] * 4 + ["validation"] * 4

    result = create_default_label_registry().generate(
        frame,
        LabelConfig.forward_return(horizon=2, buy_threshold=0.0, sell_threshold=0.0),
    )

    assert result.known_mask.tolist() == [True, True, False, False] * 2
    assert result.frame.loc[~result.known_mask, "fwd_ret"].isna().all()
    assert result.frame["_label_known"].tolist() == result.known_mask.tolist()
    assert result.metadata["n_known"] == 4
    assert result.metadata["n_unknown"] == 4


def test_forward_return_train_labels_cannot_see_next_split_prices():
    frame = _frame([100.0, 101.0, 102.0, 103.0, 200.0, 210.0, 220.0, 230.0])
    frame["_experiment_split"] = ["train"] * 4 + ["validation"] * 4
    changed = frame.copy()
    changed.loc[changed["_experiment_split"] == "validation", "adj_close"] *= 100.0
    registry = create_default_label_registry()
    config = LabelConfig.forward_return(horizon=2)

    original = registry.generate(frame, config).frame
    perturbed = registry.generate(changed, config).frame
    original_train = original[original["_experiment_split"] == "train"].reset_index(
        drop=True
    )
    perturbed_train = perturbed[
        perturbed["_experiment_split"] == "train"
    ].reset_index(drop=True)

    pd.testing.assert_series_equal(original_train["Label"], perturbed_train["Label"])
    np.testing.assert_allclose(
        original_train["fwd_ret"],
        perturbed_train["fwd_ret"],
        equal_nan=True,
    )


def test_forward_return_grouping_never_crosses_tickers():
    first = _frame([100.0, 102.0, 104.0, 106.0], ticker="AAA")
    second = _frame([50.0, 49.0, 47.0, 46.0], ticker="BBB")
    combined = pd.concat([first, second], ignore_index=True).sample(
        frac=1.0, random_state=11
    )

    grouped = build_forward_return_labels_by_ticker(combined, horizon=2)
    isolated = build_forward_return_labels_by_ticker(second, horizon=2)

    actual = grouped[grouped["ticker"] == "BBB"].reset_index(drop=True)
    pd.testing.assert_series_equal(actual["Label"], isolated["Label"])
    np.testing.assert_allclose(actual["fwd_ret"], isolated["fwd_ret"], equal_nan=True)


def test_forward_return_registry_fills_defaults_and_standard_contract():
    result = create_default_label_registry().generate(
        _frame([100.0, 101.0, 102.0]),
        LabelConfig(method="forward-return", parameters={"horizon": 1}),
        LabelContext(),
    )

    assert result.semantics == "action"
    assert result.class_names == ("Sell", "Hold", "Buy")
    assert result.metadata["parameters"] == LabelConfig.forward_return(
        horizon=1
    ).parameters
    assert result.known_mask.tolist() == [True, True, False]


@pytest.mark.parametrize(
    "parameters, error",
    [
        ({"horizon": 0}, ValueError),
        ({"horizon": 1.5}, TypeError),
        ({"buy_threshold": -0.1}, ValueError),
        ({"sell_threshold": np.inf}, ValueError),
        ({"buy_threshold": True}, TypeError),
    ],
)
def test_forward_return_rejects_invalid_parameters(parameters, error):
    config = LabelConfig.forward_return(**parameters)
    with pytest.raises(error):
        create_default_label_registry().generate(_frame([1.0, 2.0, 3.0]), config)


def test_shared_cli_resolves_forward_return_and_legacy_options():
    parser = argparse.ArgumentParser()
    add_label_arguments(parser)
    args = parser.parse_args(
        [
            "--label-method",
            "forward-return",
            "--forward-horizon",
            "5",
            "--forward-buy-threshold",
            "0.01",
            "--forward-sell-threshold",
            "0.02",
        ]
    )

    config = apply_label_arguments(ExperimentConfig(), args)
    label_config = label_config_from_args(args)

    assert config.label_mode == "forward_return"
    assert config.forward_horizon == 5
    assert config.forward_buy_threshold == pytest.approx(0.01)
    assert config.forward_sell_threshold == pytest.approx(0.02)
    assert label_config == LabelConfig.forward_return(
        horizon=5,
        buy_threshold=0.01,
        sell_threshold=0.02,
    )


def test_shared_cli_rejects_forward_parameter_for_breakout():
    parser = argparse.ArgumentParser()
    add_label_arguments(parser)
    args = parser.parse_args(
        ["--label-method", "breakout", "--label-horizon", "5"]
    )

    with pytest.raises(ValueError, match="Horizon arguments"):
        apply_label_arguments(ExperimentConfig(), args)


def test_experiment_config_resolves_forward_label_config():
    config = ExperimentConfig(
        label_mode="forward_return",
        forward_horizon=10,
        forward_buy_threshold=0.015,
        forward_sell_threshold=0.02,
    )

    assert config.resolved_label_config() == LabelConfig.forward_return(
        horizon=10,
        buy_threshold=0.015,
        sell_threshold=0.02,
    )
