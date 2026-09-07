import argparse

import numpy as np
import pandas as pd
import pytest

from trading_system.artifacts.experiment import build_experiment_manifest
from trading_system.experiments import search
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import run_experiment
from trading_system.experiments.walkforward import walk_forward_classifier
from trading_system.labels.config import LabelConfig
from trading_system.labels.registry import LabelContext, create_default_label_registry
from trading_system.labels.triple_barrier import (
    first_barrier_touch,
    generate_triple_barrier_labels,
    generate_triple_barrier_labels_by_ticker,
    symmetric_cusum_events,
)
from trading_system.models.specs import ModelSelection
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


def test_first_horizontal_barrier_wins():
    assert first_barrier_touch(
        [-0.02, 0.03], upper_barrier=0.02, lower_barrier=-0.01
    ) == (-1, 1, "stop")
    assert first_barrier_touch(
        [0.01, 0.03, -0.02], upper_barrier=0.02, lower_barrier=-0.01
    ) == (1, 2, "profit")


def test_vertical_barrier_is_neutral_when_no_horizontal_barrier_touches():
    assert first_barrier_touch(
        [0.001, -0.002, 0.003],
        upper_barrier=0.01,
        lower_barrier=-0.01,
    ) == (0, 3, "vertical")


def test_symmetric_cusum_filter_is_causal_and_resets_after_events():
    events = symmetric_cusum_events(
        [np.nan, 0.003, 0.003, -0.001, -0.006],
        [np.nan, 0.01, 0.01, 0.01, 0.01],
        threshold_multiplier=0.5,
    )

    assert events.tolist() == [False, False, True, False, True]


def test_all_events_mark_warmup_and_incomplete_vertical_barriers_unknown():
    labeled = generate_triple_barrier_labels(
        _frame([100, 101, 100, 102, 101, 104, 103, 106, 105, 108]),
        max_holding=3,
        volatility_window=2,
        cost_bps=0.0,
    )

    assert labeled["_label_known"].tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert labeled["label_event_id"].notna().sum() == 8
    assert labeled.loc[labeled["_label_known"], "label_end_date"].notna().all()
    assert {
        "label_event_id",
        "label_end_date",
        "barrier_touch",
        "upper_barrier",
        "lower_barrier",
        "event_return",
        "net_event_return",
        "label_score",
    }.issubset(labeled.columns)


def test_constant_series_reaches_vertical_barrier_with_neutral_labels():
    labeled = generate_triple_barrier_labels(
        _frame([100.0] * 10),
        max_holding=2,
        volatility_window=2,
        cost_bps=0.0,
    )
    known = labeled["_label_known"]

    assert labeled.loc[known, "Label"].eq("Hold").all()
    assert labeled.loc[known, "barrier_touch"].eq("vertical").all()
    assert labeled.loc[known, "label_score"].eq(0.0).all()


def test_costs_widen_horizontal_barriers_and_reduce_score_magnitude():
    frame = _frame([100, 101, 100, 102, 101, 104, 103, 106, 105, 108])
    common = dict(max_holding=2, volatility_window=2, profit_barrier=1.0)
    free = generate_triple_barrier_labels(frame, cost_bps=0.0, **common)
    costly = generate_triple_barrier_labels(frame, cost_bps=25.0, **common)
    known_events = free["_label_known"] & costly["_label_known"]

    assert (
        costly.loc[known_events, "upper_barrier"]
        >= free.loc[known_events, "upper_barrier"]
    ).all()
    assert (
        costly.loc[known_events, "label_score"].abs()
        <= free.loc[known_events, "label_score"].abs()
    ).all()


def test_between_event_policies_are_explicit():
    frame = _frame((100 + np.sin(np.arange(80) / 4)).tolist())
    common = dict(
        max_holding=5,
        volatility_window=5,
        event_filter="cusum",
        cusum_threshold=2.0,
        cost_bps=0.0,
    )
    hold = generate_triple_barrier_labels(
        frame, between_event_policy="hold", **common
    )
    flat = generate_triple_barrier_labels(
        frame, between_event_policy="flat", **common
    )
    carry = generate_triple_barrier_labels(
        frame, between_event_policy="carry", **common
    )
    non_events = hold["label_event_id"].isna() & hold["_label_known"]

    assert non_events.any()
    assert hold.loc[non_events, "Label"].eq("Hold").all()
    assert flat.loc[non_events, "Label"].eq("Flat").all()
    assert hold["target_position"].isna().all()
    assert carry["target_position"].notna().all()
    assert set(carry["Label"].unique()) <= {"Short", "Flat", "Long"}


def test_registry_never_uses_future_prices_across_split_boundary():
    frame = _frame([100, 101, 99, 102, 100, 103, 50, 51, 49, 52, 50, 53])
    frame["_experiment_split"] = ["train"] * 6 + ["validation"] * 6
    changed = frame.copy()
    changed.loc[changed["_experiment_split"] == "validation", "adj_close"] *= 100
    config = LabelConfig.triple_barrier(
        max_holding=2, volatility_window=2, cost_bps=0.0
    )
    registry = create_default_label_registry()

    original = registry.generate(frame, config).frame
    perturbed = registry.generate(changed, config).frame
    train = original[original["_experiment_split"] == "train"].reset_index(drop=True)
    changed_train = perturbed[
        perturbed["_experiment_split"] == "train"
    ].reset_index(drop=True)
    validation = original[
        original["_experiment_split"] == "validation"
    ].reset_index(drop=True)

    pd.testing.assert_series_equal(train["Label"], changed_train["Label"])
    np.testing.assert_allclose(
        train["event_return"], changed_train["event_return"], equal_nan=True
    )
    assert train["_label_known"].tolist() == [False, False, True, True, False, False]
    assert validation["_label_known"].tolist() == [True, True, True, True, False, False]


def test_grouped_triple_barrier_is_invariant_to_another_ticker():
    first = _frame([100, 102, 101, 104, 102, 106, 104, 108], ticker="AAA")
    second = _frame([50, 49, 51, 48, 50, 47, 49, 46], ticker="BBB")
    combined = pd.concat([first, second], ignore_index=True).sample(
        frac=1.0, random_state=9
    )
    parameters = dict(
        max_holding=2, volatility_window=2, cost_bps=0.0, event_filter="all"
    )

    grouped = generate_triple_barrier_labels_by_ticker(combined, **parameters)
    isolated = generate_triple_barrier_labels(second, **parameters)
    actual = grouped[grouped["ticker"] == "BBB"].reset_index(drop=True)

    pd.testing.assert_series_equal(actual["Label"], isolated["Label"])
    pd.testing.assert_series_equal(actual["barrier_touch"], isolated["barrier_touch"])
    np.testing.assert_allclose(
        actual["event_return"], isolated["event_return"], equal_nan=True
    )


def test_registry_contract_contains_event_diagnostics():
    config = LabelConfig.triple_barrier(
        max_holding=2,
        volatility_window=2,
        event_filter="cusum",
        cusum_threshold=0.5,
        cost_bps=0.0,
    )
    result = create_default_label_registry().generate(
        _frame([100, 101, 100, 102, 101, 104, 103, 106]),
        config,
        LabelContext(),
    )

    assert result.semantics == "action"
    assert result.class_names == ("Sell", "Hold", "Buy")
    assert result.metadata["parameters"] == config.parameters
    assert sum(result.metadata["class_counts"].values()) == result.metadata["n_known"]
    assert sum(result.metadata["touch_counts"].values()) == result.metadata[
        "n_known_events"
    ]


def test_shared_cli_resolves_triple_barrier_arguments():
    parser = argparse.ArgumentParser()
    add_label_arguments(parser)
    args = parser.parse_args(
        [
            "--label-method",
            "triple-barrier",
            "--label-max-holding",
            "20",
            "--label-vol-window",
            "60",
            "--label-profit-barrier",
            "1.5",
            "--label-stop-barrier",
            "0.75",
            "--label-event-filter",
            "cusum",
            "--label-cusum-threshold",
            "1.0",
            "--label-cost-bps",
            "10",
            "--label-between-events",
            "carry",
        ]
    )

    config = apply_label_arguments(ExperimentConfig(), args)

    assert config.label_mode == "triple_barrier"
    assert config.triple_barrier_max_holding == 20
    assert label_config_from_args(args) == LabelConfig.triple_barrier(
        max_holding=20,
        volatility_window=60,
        profit_barrier=1.5,
        stop_barrier=0.75,
        event_filter="cusum",
        cusum_threshold=1.0,
        cost_bps=10.0,
        between_event_policy="carry",
    )


def test_experiment_config_resolves_m3_semantics():
    action = ExperimentConfig(label_mode="triple_barrier")
    positions = ExperimentConfig(
        label_mode="triple_barrier",
        triple_barrier_between_event_policy="flat",
    )

    assert action.resolved_label_config() == LabelConfig.triple_barrier()
    assert action.resolved_label_semantics() == "action"
    assert positions.resolved_label_semantics() == "target_position"
    assert positions.resolved_class_names() == ("Short", "Flat", "Long")


@pytest.mark.parametrize("estimator", ["rolling_std", "atr", "bollinger"])
def test_m3_runs_through_static_training_backtest_and_manifest(estimator):
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
        label_mode="triple_barrier",
        triple_barrier_max_holding=3,
        triple_barrier_volatility_window=5,
        triple_barrier_volatility_estimator=estimator,
        triple_barrier_profit_barrier=0.5,
        triple_barrier_stop_barrier=0.5,
        triple_barrier_cost_bps=0.0,
        context_len=3,
        decision_mode="argmax",
        model=ModelSelection(
            "manual_ann", {"hidden_size": 4, "epochs": 1, "batch_size": 16}
        ),
        seed=7,
    )

    result = run_experiment(frame, config)
    manifest = build_experiment_manifest(frame, result)

    assert set(result.label_stats) == {"Sell", "Hold", "Buy"}
    assert result.test_probabilities.shape[1] == 3
    assert np.isfinite(result.backtest["model_pnl"])
    assert manifest.class_names == ("Sell", "Hold", "Buy")
    assert manifest.experiment_parameters["label_config"]["method"] == "triple_barrier"
    assert manifest.experiment_parameters["label_config"]["parameters"][
        "volatility_estimator"
    ] == estimator


@pytest.mark.parametrize("estimator", ["rolling_std", "atr", "bollinger"])
def test_m3_runs_through_one_walkforward_chunk(estimator):
    frame = _frame(
        (100.0 + 0.03 * np.arange(100) + 2.0 * np.sin(np.arange(100) / 5.0)).tolist()
    )
    frame["signal"] = frame["adj_close"].pct_change(fill_method=None).fillna(0.0)
    frame["close"] = frame["adj_close"]
    frame["high"] = frame["close"] * 1.01
    frame["low"] = frame["close"] * 0.99

    result = walk_forward_classifier(
        frame,
        ["signal"],
        train_ratio=0.6,
        val_ratio=0.2,
        walkforward_step=100,
        label_mode="triple_barrier",
        triple_barrier_max_holding=3,
        triple_barrier_volatility_window=5,
        triple_barrier_volatility_estimator=estimator,
        triple_barrier_profit_barrier=0.5,
        triple_barrier_stop_barrier=0.5,
        triple_barrier_cost_bps=0.0,
        context_len=3,
        decision_mode="argmax",
        model_selection=ModelSelection(
            "manual_ann", {"hidden_size": 4, "epochs": 1, "batch_size": 16}
        ),
        seed=7,
    )

    assert len(result["retrain_logs"]) == 1
    assert result["retrain_logs"][0]["label_hist_info"]["method"] == "triple_barrier"
    assert result["label_eval_report"]["method"] == "triple_barrier"
    assert result["label_eval_report"]["parameters"]["volatility_estimator"] == estimator
    assert result["test_metrics"]


def test_walkforward_search_maps_label_horizon_to_m3(monkeypatch):
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
        common_parameters={
            "label_mode": "triple_barrier",
            "triple_barrier_volatility_estimator": "atr",
        },
    )

    assert result.loc[0, "triple_barrier_max_holding"] == 20
    assert "forward_horizon" not in result.columns
    assert [call["triple_barrier_max_holding"] for call in calls] == [20, 20]
    assert all("forward_buy_threshold" not in call for call in calls)
    assert all(call["triple_barrier_volatility_estimator"] == "atr" for call in calls)
    assert result.loc[0, "triple_barrier_volatility_estimator"] == "atr"


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"max_holding": 0}, ValueError),
        ({"volatility_window": 1.5}, TypeError),
        ({"profit_barrier": -1.0}, ValueError),
        ({"profit_barrier": 0.0, "stop_barrier": 0.0}, ValueError),
        ({"cusum_threshold": np.inf}, ValueError),
        ({"cost_bps": True}, TypeError),
        ({"event_filter": "zscore"}, ValueError),
        ({"between_event_policy": "implicit"}, ValueError),
    ],
)
def test_triple_barrier_rejects_invalid_parameters(overrides, error):
    config = LabelConfig.triple_barrier(**overrides)
    with pytest.raises(error):
        create_default_label_registry().generate(
            _frame([100, 101, 102, 103]), config
        )
