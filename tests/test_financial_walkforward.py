"""Financial-objective integration for expanding-window workflows."""

import numpy as np
import pandas as pd
import pytest

from trading_system.experiments import search
from trading_system.experiments.search import (
    WalkForwardModelTrialConfig, run_walkforward_grid_search,
)
from trading_system.experiments.walkforward import walk_forward_classifier
from trading_system.models.specs import ModelSelection
from trading_system.pipelines.gridsearch_walkforward import build_parser as grid_parser
from trading_system.pipelines.training_arguments import financial_loss_config_from_args
from trading_system.pipelines.walkforward import build_parser as walk_parser
from trading_system.training.financial_loss import FinancialLossConfig


def frame(rows=150, ticker="AAA"):
    x = np.arange(rows, dtype=float)
    price = 100 + .08 * x + 2 * np.sin(x / 5)
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=rows),
        "ticker": ticker,
        "open": price, "high": price + 1, "low": price - 1,
        "close": price, "adj_close": price,
        "volume": 1_000_000 + 1000 * np.cos(x),
        "signal": np.sin(x / 5),
    })


def run(loss=None, data=None, evaluation_split="test"):
    return walk_forward_classifier(
        frame() if data is None else data,
        ("signal",),
        train_ratio=.55,
        val_ratio=.2,
        walkforward_step=40,
        label_mode="forward_return",
        forward_horizon=2,
        context_len=3,
        model_selection=ModelSelection("manual_ann", {
            "hidden_size": 4, "epochs": 2, "batch_size": 16,
            "early_stopping_patience": 2,
        }),
        seed=17,
        evaluation_split=evaluation_split,
        financial_loss=loss,
    )


@pytest.mark.parametrize("objective", ["pnl", "sharpe"])
def test_financial_walkforward_trains_continuous_positions_and_reports_costs(objective):
    result = run(FinancialLossConfig(objective, cost_bps=7))
    assert result["loss_objective"] == objective
    assert result["test_metrics"] == {}
    assert result["n_eval_rows"] == result["n_test_rows"]
    positions = result["predictions"][result["evaluation_mask"]]
    assert np.isfinite(positions).all()
    assert (np.abs(positions) <= 1).all()
    report = result["benchmark_comparison"]
    assert report["model_pnl"] == report["net_pnl"]
    assert report["outperformance"] == pytest.approx(
        report["model_pnl"] - report["buy_hold_pnl"]
    )
    assert report["cost_return_sum"] >= 0
    assert all(log["loss_objective"] == objective for log in result["retrain_logs"])
    assert all("regularized_sharpe" in log["validation_metrics"] for log in result["retrain_logs"])


@pytest.mark.parametrize("model_name", ["manual_ann", "rnn", "lstm", "gru", "transformer"])
def test_financial_walkforward_supports_every_registered_neural_model(model_name):
    parameters = {
        "epochs": 1, "batch_size": 16, "early_stopping_patience": 1,
    }
    if model_name == "manual_ann":
        parameters["hidden_size"] = 4
    elif model_name == "transformer":
        parameters.update(d_model=4, n_heads=2, num_layers=1, dim_feedforward=8, dropout=0.0)
    else:
        parameters["hidden_size"] = 4
    result = walk_forward_classifier(
        frame(100), ("signal",), train_ratio=.55, val_ratio=.2,
        walkforward_step=30, label_mode="forward_return", context_len=3,
        model_selection=ModelSelection(model_name, parameters), device="cpu",
        seed=5, financial_loss=FinancialLossConfig("pnl"),
    )
    assert result["loss_objective"] == "pnl"
    assert result["n_eval_rows"] > 0
    assert {log["model_name"] for log in result["retrain_logs"]} == {model_name}


def test_explicit_cross_entropy_config_preserves_legacy_path_exactly():
    original = run()
    explicit = run(FinancialLossConfig("cross_entropy"))
    np.testing.assert_array_equal(original["predictions"], explicit["predictions"])
    assert original["test_metrics"] == explicit["test_metrics"]
    assert original["benchmark_comparison"] == explicit["benchmark_comparison"]


def test_financial_validation_cannot_see_final_test_prices():
    data = frame()
    poisoned = data.copy()
    poisoned.loc[poisoned.index >= 120, "adj_close"] *= 100
    config = FinancialLossConfig("pnl")
    original = run(config, data, "validation")
    changed = run(config, poisoned, "validation")
    np.testing.assert_allclose(original["predictions"], changed["predictions"], equal_nan=True)
    assert original["val_backtest"] == changed["val_backtest"]


def test_financial_walkforward_does_not_generate_or_consume_labels(monkeypatch):
    from trading_system.experiments import walkforward

    monkeypatch.setattr(
        walkforward, "_label_history",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Financial objective touched labels")
        ),
    )
    result = run(FinancialLossConfig("pnl"))
    assert result["n_scored_labels"] == result["n_eval_rows"]
    assert result["label_eval_report"]["method"] == "unused_for_financial_objective"


def test_financial_walkforward_rejects_incompatible_modes():
    config = FinancialLossConfig("pnl")
    with pytest.raises(ValueError, match="one ticker"):
        run(config, pd.concat([frame(), frame(ticker="BBB")], ignore_index=True))
    with pytest.raises(ValueError, match="Sample weighting"):
        walk_forward_classifier(
            frame(), ("signal",), context_len=3, financial_loss=config,
            label_mode="triple_barrier",
            sample_weighting={"mode": "net_return"},
        )
    with pytest.raises(ValueError, match="Oracle"):
        walk_forward_classifier(
            frame(), ("signal",), context_len=3,
            financial_loss=config, label_mode="oracle_dp",
        )


def test_financial_grid_search_selects_on_validation_then_tests_only_winner(monkeypatch):
    calls = []

    def fake_runner(*args, **kwargs):
        split = kwargs["evaluation_split"]
        calls.append((kwargs["model_selection"].parameters["hidden_size"], split))
        score = float(kwargs["model_selection"].parameters["hidden_size"])
        shared = {
            "retrain_logs": [{"split": split}], "n_eval_rows": 10,
            "loss_objective": "pnl", "financial_loss": {"objective": "pnl"},
        }
        if split == "validation":
            return {**shared, "val_metrics": {},
                    "val_backtest": {"regularized_sharpe": score},
                    "n_val_rows": 10, "n_missing_val_preds": 0}
        return {**shared, "test_metrics": {},
                "benchmark_comparison": {"regularized_sharpe": -score},
                "n_test_rows": 10, "n_missing_test_preds": 0}

    monkeypatch.setattr(search, "walk_forward_classifier", fake_runner)
    trials = [
        WalkForwardModelTrialConfig(
            model=ModelSelection("manual_ann", {"hidden_size": hidden}),
            forward_horizon=2, forward_buy_threshold=.01,
            forward_sell_threshold=.01, context_len=3, walkforward_step=10,
        )
        for hidden in (4, 8)
    ]
    result = run_walkforward_grid_search(
        frame(), ("signal",), trials, objective="regularized_sharpe",
        common_parameters={"financial_loss": FinancialLossConfig("pnl")},
    )
    assert calls == [(4, "validation"), (8, "validation"), (8, "test")]
    assert result.loc[result.selected, "model_parameters"].item() == '{"hidden_size":8}'
    assert result.attrs["final_test"]["benchmark_comparison"]["regularized_sharpe"] == -8


def test_grid_rejects_objective_loss_mismatch():
    trial = WalkForwardModelTrialConfig(
        model=ModelSelection("manual_ann", {"hidden_size": 4}),
        forward_horizon=2, forward_buy_threshold=.01,
        forward_sell_threshold=.01, context_len=3, walkforward_step=10,
    )
    with pytest.raises(ValueError, match="require pnl/sharpe"):
        run_walkforward_grid_search(
            frame(), ("signal",), [trial], objective="regularized_sharpe"
        )
    with pytest.raises(ValueError, match="financial validation"):
        run_walkforward_grid_search(
            frame(), ("signal",), [trial], objective="macro_f1",
            common_parameters={"financial_loss": FinancialLossConfig("pnl")},
        )


def test_cli_financial_loss_flags_and_validation():
    for parser in (walk_parser(), grid_parser()):
        assert financial_loss_config_from_args(parser.parse_args([])) is None
        args = parser.parse_args([
            "--loss-objective", "sharpe", "--loss-cost-bps", "8",
            "--loss-annualization", "365", "--loss-sharpe-epsilon", "0.002",
        ])
        assert financial_loss_config_from_args(args) == FinancialLossConfig(
            "sharpe", 8, 365, .002
        )
    args = grid_parser().parse_args([
        "--loss-objective", "pnl", "--objective", "regularized_sharpe"
    ])
    assert args.objective == "regularized_sharpe"
