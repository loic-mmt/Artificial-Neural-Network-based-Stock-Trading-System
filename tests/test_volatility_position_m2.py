import argparse

import numpy as np
import pandas as pd
import pytest

from trading_system.backtest.engine import run_label_backtest
from trading_system.backtest.positions import labels_to_positions
from trading_system.artifacts.experiment import build_experiment_manifest
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import run_experiment
from trading_system.experiments import search
from trading_system.labels.config import LabelConfig
from trading_system.labels.registry import LabelContext, create_default_label_registry
from trading_system.labels.volatility_position import (
    build_persistent_positions,
    generate_volatility_position_labels,
    generate_volatility_position_labels_by_ticker,
)
from trading_system.pipelines.label_arguments import (
    add_label_arguments,
    apply_label_arguments,
    label_config_from_args,
)
from trading_system.models.specs import ModelSelection


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


def test_persistent_position_policy_enters_holds_exits_and_flips():
    positions = build_persistent_positions(
        [np.nan, 2.0, 2.0, 0.0, -2.0, -2.0, 0.0, 2.0],
        long_threshold=1.0,
        short_threshold=1.0,
        exit_threshold=0.25,
        min_holding_period=2,
        cooldown=0,
        position_mode="long_short",
    )

    assert positions.tolist() == [0, 1, 1, 0, -1, -1, 0, 1]


def test_minimum_holding_period_and_cooldown_block_early_transitions():
    positions = build_persistent_positions(
        [2.0, 0.0, 0.0, -2.0, -2.0, 0.0, 0.0],
        long_threshold=1.0,
        short_threshold=1.0,
        exit_threshold=0.25,
        min_holding_period=3,
        cooldown=1,
        position_mode="long_short",
    )

    assert positions.tolist() == [1, 1, 1, -1, -1, -1, 0]


def test_long_flat_never_emits_short_positions():
    positions = build_persistent_positions(
        [-2.0, 2.0, -2.0, -2.0],
        long_threshold=1.0,
        short_threshold=1.0,
        exit_threshold=0.25,
        min_holding_period=0,
        position_mode="long_flat",
    )

    assert positions.tolist() == [0, 1, 0, 0]


def test_volatility_position_marks_warmup_and_horizon_tail_unknown():
    frame = _frame([100, 101, 100, 102, 101, 104, 103, 106, 105, 108])

    labeled = generate_volatility_position_labels(
        frame,
        horizon=2,
        volatility_window=3,
        long_threshold=0.1,
        short_threshold=0.1,
        exit_threshold=0.0,
        min_holding_period=0,
        cost_bps=0.0,
        position_mode="long_short",
    )

    assert labeled["_label_known"].tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert {
        "fwd_ret",
        "historical_volatility",
        "cost_adjusted_fwd_ret",
        "label_score",
        "target_position",
        "Label_id",
        "Label",
        "_label_known",
    }.issubset(labeled.columns)
    assert labeled["Label_id"].between(0, 2).all()


def test_registry_prevents_future_targets_from_crossing_split_boundaries():
    frame = _frame([100, 101, 99, 102, 100, 103, 50, 51, 49, 52, 50, 53])
    frame["_experiment_split"] = ["train"] * 6 + ["validation"] * 6
    changed = frame.copy()
    changed.loc[changed["_experiment_split"] == "validation", "adj_close"] *= 100
    config = LabelConfig.volatility_position(
        horizon=2,
        volatility_window=3,
        long_threshold=0.1,
        short_threshold=0.1,
        exit_threshold=0.0,
        min_holding_period=0,
        cost_bps=0.0,
        position_mode="long_short",
    )
    registry = create_default_label_registry()

    result = registry.generate(frame, config)
    perturbed = registry.generate(changed, config)
    train = result.frame[result.frame["_experiment_split"] == "train"].reset_index(
        drop=True
    )
    changed_train = perturbed.frame[
        perturbed.frame["_experiment_split"] == "train"
    ].reset_index(drop=True)
    validation = result.frame[
        result.frame["_experiment_split"] == "validation"
    ].reset_index(drop=True)

    pd.testing.assert_series_equal(train["Label"], changed_train["Label"])
    np.testing.assert_allclose(
        train["label_score"], changed_train["label_score"], equal_nan=True
    )
    assert train["_label_known"].tolist() == [False, False, False, True, False, False]
    assert validation["_label_known"].tolist() == [True, True, True, True, False, False]
    assert validation.loc[0, "target_position"] == 0


def test_grouped_volatility_position_is_ticker_independent():
    first = _frame([100, 102, 101, 104, 102, 106, 104, 108], ticker="AAA")
    second = _frame([50, 49, 51, 48, 50, 47, 49, 46], ticker="BBB")
    combined = pd.concat([first, second], ignore_index=True).sample(
        frac=1.0, random_state=9
    )
    parameters = dict(
        horizon=2,
        volatility_window=2,
        long_threshold=0.1,
        short_threshold=0.1,
        exit_threshold=0.0,
        min_holding_period=0,
        cost_bps=0.0,
        position_mode="long_short",
    )

    grouped = generate_volatility_position_labels_by_ticker(combined, **parameters)
    isolated = generate_volatility_position_labels(second, **parameters)

    actual = grouped[grouped["ticker"] == "BBB"].reset_index(drop=True)
    pd.testing.assert_series_equal(actual["Label"], isolated["Label"])
    np.testing.assert_allclose(
        actual["label_score"], isolated["label_score"], equal_nan=True
    )


def test_cost_reduces_score_magnitude_without_changing_direction():
    frame = _frame([100, 101, 100, 102, 101, 104, 103, 106])
    common = dict(
        horizon=1,
        volatility_window=2,
        long_threshold=0.1,
        short_threshold=0.1,
        exit_threshold=0.0,
        min_holding_period=0,
        position_mode="long_short",
    )
    free = generate_volatility_position_labels(frame, cost_bps=0.0, **common)
    costly = generate_volatility_position_labels(frame, cost_bps=25.0, **common)
    known = free["_label_known"] & costly["_label_known"]

    assert (
        costly.loc[known, "label_score"].abs()
        <= free.loc[known, "label_score"].abs()
    ).all()
    assert (
        np.sign(costly.loc[known, "label_score"])
        == np.sign(free.loc[known, "label_score"])
    ).all()


def test_registry_returns_target_position_contract_and_diagnostics():
    config = LabelConfig.volatility_position(
        horizon=2, volatility_window=2, min_holding_period=0
    )
    result = create_default_label_registry().generate(
        _frame([100, 101, 100, 102, 101, 104, 103, 106]),
        config,
        LabelContext(),
    )

    assert result.semantics == "target_position"
    assert result.class_names == ("Short", "Flat", "Long")
    assert result.metadata["parameters"] == config.parameters
    assert sum(result.metadata["class_counts"].values()) == result.metadata["n_known"]
    assert result.metadata["transition_count"] >= 0
    assert result.metadata["mean_regime_length"] >= 0


def test_flat_target_closes_position_while_action_hold_keeps_it_open():
    prices = np.array([100.0, 110.0, 99.0])
    labels = np.array([2, 1, 1])

    target_result = run_label_backtest(
        prices, labels, execution_delay=0, label_semantics="target_position"
    )
    action_result = run_label_backtest(
        prices, labels, execution_delay=0, label_semantics="action"
    )

    assert target_result["target_positions"].tolist() == [1.0, 0.0, 0.0]
    assert action_result["target_positions"].tolist() == [1.0, 1.0, 1.0]
    assert target_result["model_curve"][-1] == pytest.approx(11_000.0)
    assert action_result["model_curve"][-1] == pytest.approx(9_900.0)


def test_long_only_target_semantics_maps_short_to_flat():
    positions = labels_to_positions(
        np.array([0, 1, 2]),
        position_mode="long_only",
        label_semantics="target_position",
    )

    assert positions.tolist() == [0.0, 0.0, 1.0]


def test_shared_cli_resolves_volatility_position_arguments():
    parser = argparse.ArgumentParser()
    add_label_arguments(parser)
    args = parser.parse_args(
        [
            "--label-method",
            "volatility-position",
            "--label-horizon",
            "20",
            "--label-vol-window",
            "60",
            "--label-long-threshold",
            "0.5",
            "--label-short-threshold",
            "1.0",
            "--label-exit-threshold",
            "0.1",
            "--label-min-hold",
            "10",
            "--label-cooldown",
            "2",
            "--label-cost-bps",
            "10",
            "--label-position-mode",
            "long-short",
        ]
    )

    config = apply_label_arguments(ExperimentConfig(), args)

    assert config.label_mode == "volatility_position"
    assert config.volatility_horizon == 20
    assert config.volatility_window == 60
    assert config.volatility_position_mode == "long_short"
    assert label_config_from_args(args) == LabelConfig.volatility_position(
        horizon=20,
        volatility_window=60,
        long_threshold=0.5,
        short_threshold=1.0,
        exit_threshold=0.1,
        min_holding_period=10,
        cooldown=2,
        cost_bps=10.0,
        position_mode="long_short",
    )


def test_experiment_config_resolves_m2_semantics_and_backtest_mode():
    config = ExperimentConfig(label_mode="volatility_position")

    assert config.resolved_label_config() == LabelConfig.volatility_position()
    assert config.resolved_class_names() == ("Short", "Flat", "Long")
    assert config.resolved_label_semantics() == "target_position"
    assert config.resolved_backtest_position_mode() == "long_only"


def test_m2_runs_through_static_training_backtest_and_manifest():
    rows = 160
    x = np.arange(rows, dtype=float)
    close = 100.0 + 0.03 * x + 2.0 * np.sin(x / 5.0)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=rows),
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": np.full(rows, 1_000_000.0),
        }
    )
    config = ExperimentConfig(
        label_mode="volatility_position",
        volatility_horizon=3,
        volatility_window=5,
        volatility_long_threshold=0.2,
        volatility_short_threshold=0.2,
        volatility_exit_threshold=0.05,
        volatility_min_holding_period=2,
        volatility_position_mode="long_short",
        context_len=3,
        decision_mode="argmax",
        model=ModelSelection(
            "manual_ann", {"hidden_size": 4, "epochs": 1, "batch_size": 16}
        ),
        seed=7,
    )

    result = run_experiment(frame, config)
    manifest = build_experiment_manifest(frame, result)

    assert set(result.label_stats) == {"Short", "Flat", "Long"}
    assert result.test_probabilities.shape[1] == 3
    assert np.isfinite(result.backtest["model_pnl"])
    assert manifest.class_names == ("Short", "Flat", "Long")
    assert (
        manifest.experiment_parameters["label_config"]["semantics"]
        == "target_position"
    )


def test_walkforward_search_maps_label_horizon_to_m2(monkeypatch):
    calls = []

    def fake_walkforward(frame, columns, **kwargs):
        calls.append(kwargs)
        if kwargs["evaluation_split"] == "validation":
            return {
                "val_metrics": {},
                "val_backtest": {"outperformance": 1.0},
                "n_val_rows": 10,
                "n_eval_rows": 10,
                "n_missing_val_preds": 0,
                "retrain_logs": [],
            }
        return {
            "test_metrics": {},
            "benchmark_comparison": {},
            "n_test_rows": 10,
            "n_eval_rows": 10,
            "n_missing_test_preds": 0,
            "retrain_logs": [],
        }

    monkeypatch.setattr(search, "walk_forward_oracle_ann", fake_walkforward)
    trial = search.WalkForwardTrialConfig(
        20, 0.0, 0.0, 3, 10, 4, 1, 0.001, 16, "argmax", 0.0
    )

    result = search.run_walkforward_grid_search(
        _frame(list(np.linspace(100.0, 110.0, 80))),
        ["adj_close"],
        [trial],
        common_parameters={"label_mode": "volatility_position"},
    )

    assert result.loc[0, "volatility_horizon"] == 20
    assert "forward_horizon" not in result.columns
    assert [call["volatility_horizon"] for call in calls] == [20, 20]
    assert all("forward_buy_threshold" not in call for call in calls)


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"horizon": 0}, ValueError),
        ({"volatility_window": 1.5}, TypeError),
        ({"long_threshold": 0.0}, ValueError),
        ({"short_threshold": np.inf}, ValueError),
        ({"exit_threshold": 1.0, "long_threshold": 0.5}, ValueError),
        ({"min_holding_period": -1}, ValueError),
        ({"cost_bps": True}, TypeError),
        ({"position_mode": "short_only"}, ValueError),
    ],
)
def test_volatility_position_rejects_invalid_parameters(overrides, error):
    config = LabelConfig.volatility_position(**overrides)
    with pytest.raises(error):
        create_default_label_registry().generate(
            _frame([100, 101, 102, 103]), config
        )
