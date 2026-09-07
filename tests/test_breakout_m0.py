import argparse

import numpy as np
import pandas as pd
import pytest

from trading_system.backtest.engine import evaluate_strategy_vs_buy_hold
from trading_system.experiments.config import ExperimentConfig
from trading_system.labels.breakout import (
    generate_breakout_labels,
    generate_breakout_labels_by_ticker,
)
from trading_system.labels.breakout_gridsearch import label_gridsearch
from trading_system.labels.config import LabelConfig
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


def test_breakout_defaults_preserve_historical_actions():
    frame = _frame([10.0, 9.0, 8.0, 7.0, 8.0, 9.0, 10.0])

    implicit = generate_breakout_labels(frame, 2)
    explicit = generate_breakout_labels(
        frame,
        2,
        buy_buffer=0.0,
        sell_buffer=0.0,
        alternating=True,
    )

    assert implicit["Label"].tolist() == [
        "Hold",
        "Hold",
        "Buy",
        "Hold",
        "Sell",
        "Hold",
        "Hold",
    ]
    pd.testing.assert_frame_equal(implicit, explicit)


def test_breakout_buffers_and_optional_alternation_change_actions():
    falling = _frame([100.0, 100.0, 99.5, 99.0, 98.5])
    buffered = generate_breakout_labels(
        falling,
        2,
        buy_buffer=0.01,
        alternating=False,
    )
    repeated = generate_breakout_labels(falling, 2, alternating=False)
    alternating = generate_breakout_labels(falling, 2, alternating=True)

    assert buffered["Label"].eq("Hold").all()
    assert repeated["Label"].tolist()[-3:] == ["Buy", "Buy", "Buy"]
    assert alternating["Label"].tolist()[-3:] == ["Buy", "Hold", "Hold"]


def test_grouped_breakout_is_invariant_to_another_ticker():
    first = _frame([10.0, 9.0, 8.0, 9.0, 10.0], ticker="AAA")
    second = _frame([20.0, 21.0, 22.0, 21.0, 20.0], ticker="BBB")
    combined = pd.concat([first, second], ignore_index=True).sample(
        frac=1.0, random_state=7
    )

    grouped = generate_breakout_labels_by_ticker(combined, 2)
    isolated = generate_breakout_labels(second, 2)

    actual = grouped[grouped["ticker"] == "BBB"].reset_index(drop=True)
    pd.testing.assert_series_equal(actual["Label"], isolated["Label"])


def test_breakout_registry_returns_standard_contract():
    frame = _frame([10.0, 9.0, 8.0, 9.0, 10.0])
    registry = create_default_label_registry()
    config = LabelConfig.breakout(window=2, buy_buffer=0.001)

    result = registry.generate(frame, config, LabelContext())

    assert registry.names() == (
        "breakout",
        "forward_return",
        "triple_barrier",
        "volatility_position",
    )
    assert result.semantics == "action"
    assert result.class_names == ("Sell", "Hold", "Buy")
    assert result.known_mask.tolist() == [True] * len(frame)
    assert result.metadata["parameters"] == config.parameters
    assert sum(result.metadata["class_counts"].values()) == len(frame)


def test_breakout_registry_fills_method_defaults():
    result = create_default_label_registry().generate(
        _frame([10.0, 9.0, 8.0]),
        LabelConfig(method="breakout", parameters={"window": 2}),
    )

    assert result.metadata["parameters"] == LabelConfig.breakout(window=2).parameters


@pytest.mark.parametrize(
    "parameters, error",
    [
        ({"window": 0}, ValueError),
        ({"window": 2, "buy_buffer": -0.1}, ValueError),
        ({"window": 2, "sell_buffer": np.inf}, ValueError),
        ({"window": 2, "alternating": "yes"}, TypeError),
    ],
)
def test_breakout_registry_rejects_invalid_parameters(parameters, error):
    config = LabelConfig.breakout(**parameters)
    with pytest.raises(error):
        create_default_label_registry().generate(_frame([1.0, 2.0, 3.0]), config)


def test_breakout_gridsearch_uses_canonical_backtest():
    frame = _frame([10.0, 9.0, 8.0, 7.0, 8.0, 9.0, 10.0])
    parameters, metrics, results = label_gridsearch(
        frame,
        fees=1.0,
        capital=1_000.0,
        windows=(2,),
        buy_buffers=(0.0,),
        sell_buffers=(0.0,),
    )
    labeled = generate_breakout_labels(frame, 2)
    expected = evaluate_strategy_vs_buy_hold(
        labeled,
        labeled["Label_id"].to_numpy(),
        initial_capital=1_000.0,
        fee_per_trade=1.0,
        position_mode="long_only",
        execution_delay=1,
    )

    assert parameters == {
        "window": 2,
        "buy_buffer": 0.0,
        "sell_buffer": 0.0,
        "alternating": True,
    }
    assert metrics["model_final_capital"] == pytest.approx(
        expected["model_final_capital"]
    )
    assert results.loc[0, "score"] == pytest.approx(expected["outperformance"])


def test_shared_cli_resolves_breakout_overrides_and_legacy_alias():
    parser = argparse.ArgumentParser()
    add_label_arguments(parser)
    args = parser.parse_args(
        [
            "--label-mode",
            "breakout",
            "--label-window",
            "12",
            "--label-buy-buffer",
            "0.01",
            "--label-sell-buffer",
            "0.02",
            "--no-label-alternating",
        ]
    )

    config = apply_label_arguments(ExperimentConfig(), args)

    assert config.label_mode == "breakout"
    assert config.label_window == 12
    assert config.breakout_buy_buffer == pytest.approx(0.01)
    assert config.breakout_sell_buffer == pytest.approx(0.02)
    assert config.breakout_alternating is False
    assert label_config_from_args(args).parameters == {
        "window": 12,
        "buy_buffer": 0.01,
        "sell_buffer": 0.02,
        "alternating": False,
    }


def test_shared_cli_rejects_breakout_parameter_for_another_method():
    parser = argparse.ArgumentParser()
    add_label_arguments(parser)
    args = parser.parse_args(
        ["--label-method", "forward-return", "--label-window", "12"]
    )

    with pytest.raises(ValueError, match="Breakout-only"):
        apply_label_arguments(ExperimentConfig(), args)
