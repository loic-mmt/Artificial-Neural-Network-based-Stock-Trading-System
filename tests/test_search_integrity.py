"""Selection must never learn from final-test targets, prices or scores."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_system.evaluation.classification import evaluate_predictions
from trading_system.experiments import runner, search, walkforward
from trading_system.experiments.config import ExperimentConfig
from trading_system.models.base import FitResult, TrainingHistory
from trading_system.models.manual_ann.manual_nn import ManualANNConfig


def market_frame(rows=180, ticker="AAA"):
    x = np.arange(rows, dtype=float)
    close = 100.0 + 0.04 * x + 3.0 * np.sin(x / 4.0)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=rows),
        "ticker": ticker,
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "adj_close": close,
        "volume": 1_000_000.0 + 10_000.0 * np.cos(x / 6.0),
        "signal": np.sin(x / 4.0),
    })


def config(**kwargs):
    return ExperimentConfig(**{
        "train_ratio": 0.6, "val_ratio": 0.2, "context_len": 3,
        "label_mode": "forward_return", "forward_horizon": 3,
        "decision_mode": "argmax",
        "manual_ann": ManualANNConfig(hidden_size=4, epochs=2, batch_size=16, seed=7),
        **kwargs,
    })


class RecordingClassifier:
    model_name = "recording"
    classes_ = np.arange(3)

    def __init__(self, label=1):
        self.label = label
        self.fits = []
        self.predictions = []

    def fit(self, X_train, y_train, *, X_val=None, y_val=None):
        self.fits.append(tuple(array.copy() for array in (X_train, y_train, X_val, y_val)))
        return FitResult(1, "recorded", TrainingHistory([1.0], [1.0]))

    def predict_proba(self, X):
        self.predictions.append(X.copy())
        return np.tile(np.eye(3, dtype=np.float32)[self.label], (len(X), 1))

    def state_dict(self):
        return {}


class SelectionOnly:
    val_metrics = {"macro_f1": 0.6}
    val_backtest = {"outperformance": 12.0}

    @property
    def test_metrics(self):
        raise AssertionError("Test metrics accessed during selection")

    @property
    def backtest(self):
        raise AssertionError("Test backtest accessed during selection")


def test_objectives_read_validation_only():
    assert search.objective_value(SelectionOnly(), "macro_f1") == 0.6
    assert search.objective_value(SelectionOnly(), "outperformance") == 12.0
    with pytest.raises(ValueError, match="validation objective"):
        search.objective_value(SelectionOnly(), "test_macro_f1")
    with pytest.raises(ValueError, match="not finite"):
        search.objective_value(SimpleNamespace(val_metrics={"macro_f1": np.nan}), "macro_f1")


@pytest.mark.parametrize("universe", ["single", "multi"])
@pytest.mark.parametrize("label_mode", ["breakout", "forward_return"])
def test_static_validation_cannot_see_test_prices(universe, label_mode):
    frame = market_frame()
    if universe == "multi":
        frame = pd.concat([frame, market_frame(ticker="BBB")], ignore_index=True)
    poisoned = frame.copy()
    cutoff = pd.Timestamp("2023-01-01") + pd.Timedelta(days=144)
    price_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    poisoned.loc[poisoned.date >= cutoff, price_cols] = np.nan
    settings = config(universe=universe, label_mode=label_mode)
    models = [RecordingClassifier(), RecordingClassifier()]
    before = runner.run_validation_experiment(frame, settings, models[0])
    after = runner.run_validation_experiment(poisoned, settings, models[1])
    assert not hasattr(before, "test_metrics")
    assert not hasattr(before, "backtest")
    assert before.aligned_val_frame.date.max() < cutoff
    assert before.val_metrics == after.val_metrics
    assert before.val_backtest == after.val_backtest
    for original, changed in zip(models[0].fits[0], models[1].fits[0]):
        np.testing.assert_array_equal(original, changed)


def test_static_forward_train_targets_never_use_validation_prices():
    frame = market_frame()
    changed = frame.copy()
    changed.loc[108:, ["open", "high", "low", "close", "adj_close"]] *= 100
    models = [RecordingClassifier(), RecordingClassifier()]
    original = runner.run_validation_experiment(frame, config(), models[0])
    perturbed = runner.run_validation_experiment(changed, config(), models[1])
    np.testing.assert_array_equal(original.y_train, perturbed.y_train)
    np.testing.assert_array_equal(models[0].fits[0][0], models[1].fits[0][0])
    assert original.val_label_mask.sum() == 36 - 3
    assert len(models[0].fits[0][3]) == original.val_label_mask.sum()


def test_static_final_test_reuses_frozen_model_scaler_and_policy(monkeypatch):
    frame = market_frame()
    model = RecordingClassifier()
    validation = runner.run_validation_experiment(frame, config(), model)

    def forbidden(*args, **kwargs):
        raise AssertionError("Final test attempted training/calibration")

    monkeypatch.setattr(model, "fit", forbidden)
    monkeypatch.setattr(runner.DecisionPolicy, "calibrate", forbidden)
    final = runner.evaluate_experiment_test(frame, validation)
    assert final.bundle is validation.bundle
    assert len(model.predictions) == 3  # train, validation, final test
    assert len(final.aligned_test_frame) == 36
    assert final.test_label_mask.sum() == 33
    assert final.test_metrics == evaluate_predictions(
        final.y_test[final.test_label_mask], final.test_predictions[final.test_label_mask]
    )
    with pytest.raises(ValueError, match="already been evaluated"):
        runner.evaluate_experiment_test(frame, final)


def test_static_search_tests_only_winner_without_refitting():
    models = []

    def factory(parameters):
        model = RecordingClassifier(parameters["label"])
        models.append(model)
        return model

    result = search.run_grid_search(
        market_frame(), config(), {"label": [0, 1, 2]}, factory,
        objective="outperformance",
    )
    winner = result.best_result.bundle.estimator
    assert sum(len(model.predictions) == 3 for model in models) == 1
    assert len(winner.predictions) == 3
    assert all(len(model.fits) == 1 for model in models)
    assert result.trials.selection_split.tolist() == ["validation"] * 3
    assert "test_metrics" not in result.trials.columns
    assert result.best_parameters["label"] == winner.label


def test_static_selection_order_failures_and_ties(monkeypatch):
    events = []

    def validate(frame, settings, model):
        events.append(("validation", model.label))
        if model.label == 0:
            raise ValueError("failed trial")
        result = SelectionOnly()
        result.model = model
        return result

    def evaluate(frame, result):
        events.append(("test", result.model.label))
        return result

    monkeypatch.setattr(search, "run_validation_experiment", validate)
    monkeypatch.setattr(search, "evaluate_experiment_test", evaluate)
    result = search.run_grid_search(
        market_frame(), config(), {"label": [0, 1, 2]},
        lambda parameters: RecordingClassifier(parameters["label"]),
    )
    assert events == [("validation", 0), ("validation", 1), ("validation", 2), ("test", 1)]
    assert result.trials.trial_id.tolist() == [2, 3, 1]
    assert result.trials.status.tolist() == ["ok", "ok", "error"]
    assert "failed trial" in result.trials.iloc[-1].error


@pytest.mark.parametrize("mode", ["oracle_all", "oracle_train_only"])
def test_static_search_rejects_oracle_selection(mode):
    with pytest.raises(ValueError, match="diagnostic only"):
        search.run_grid_search(market_frame(), config(label_mode=mode), {}, lambda _: None)


def test_static_search_rejects_reused_model():
    model = RecordingClassifier()
    result = search.run_grid_search(market_frame(), config(), {"ignored": [1, 2]}, lambda _: model)
    assert result.trials.status.tolist() == ["ok", "error"]
    assert "fresh estimator" in result.trials.iloc[-1].error
    assert len(model.fits) == 1


def test_static_search_all_failed_does_not_open_test(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No valid winner: final test must remain closed")

    monkeypatch.setattr(search, "evaluate_experiment_test", forbidden)
    with pytest.raises(RuntimeError, match="no valid trials"):
        search.run_grid_search(market_frame(), config(), {}, lambda _: None)


@pytest.mark.parametrize("universe", ["single", "multi"])
def test_oracle_train_only_remains_a_separate_diagnostic(universe, monkeypatch):
    frame = market_frame()
    if universe == "multi":
        frame = pd.concat([frame, market_frame(ticker="BBB")], ignore_index=True)
    calls = []
    original = runner.build_oracle_labels_train_only

    def record(history, **kwargs):
        assert history.date.max() < frame.date.iloc[108]
        calls.append(len(history))
        return original(history, **kwargs)

    monkeypatch.setattr(runner, "build_oracle_labels_train_only", record)
    result = runner.run_experiment(
        frame, config(universe=universe, label_mode="oracle_train_only"), RecordingClassifier()
    )
    assert result.test_metrics
    assert calls == [108] * (4 if universe == "multi" else 2)


def test_walkforward_validation_is_isolated_and_seeds_reproduce():
    frame = market_frame()
    poisoned = frame.copy()
    poisoned.loc[144:, ["adj_close", "signal"]] = np.nan
    kwargs = dict(
        train_ratio=0.6, val_ratio=0.2, context_len=3, walkforward_step=12,
        forward_horizon=3, epochs=2, hidden=4, batch_size=16, seed=19,
        evaluation_split="validation",
    )
    first = walkforward.walk_forward_oracle_ann(frame, ["signal"], **kwargs)
    second = walkforward.walk_forward_oracle_ann(poisoned, ["signal"], **kwargs)
    assert "test_metrics" not in first
    assert "benchmark_comparison" not in first
    assert first["val_start_idx"] == 108
    assert first["n_total_rows"] == 144
    assert first["n_val_rows"] == first["n_eval_rows"] == 36
    assert first["n_scored_labels"] == 33
    assert first["val_metrics"] == second["val_metrics"]
    assert first["val_backtest"] == second["val_backtest"]
    np.testing.assert_array_equal(first["predictions"], second["predictions"])
    seeds = []
    for log in first["retrain_logs"]:
        assert log["n_hist"] == log["start_idx"] < log["end_idx"] <= 144
        assert log["run_seed"] == 19
        assert log["seed"] == walkforward.derive_chunk_seed(19, log["chunk_id"])
        seeds.append(log["seed"])
    assert len(set(seeds)) == 3
    without_duration = lambda logs: [
        {key: value for key, value in log.items() if key != "duration_seconds"}
        for log in logs
    ]
    assert without_duration(first["retrain_logs"]) == without_duration(
        second["retrain_logs"]
    )


def test_walkforward_forward_purge_retains_feature_context():
    frame = market_frame(100)
    frame["signal"] = np.arange(100, dtype=float)
    labeled, _ = walkforward.build_forward_return_labels(frame, horizon=5)
    model = RecordingClassifier()
    bundle, history, _ = walkforward.fit_labeled_history(
        labeled, ["signal"], val_ratio=0.2, context_len=3,
        decision_mode="argmax", min_action_rate=0, estimator=model, forward_horizon=5,
    )
    X_train, _, X_val, _ = model.fits[0]
    assert len(X_train) == 80 - 2 - 5
    assert len(X_val) == 20 - 5
    assert len(history) == 100
    expected_first_val = bundle.scaler.transform(
        np.array([[[78], [79], [80]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(X_val[:1], expected_first_val)


@pytest.mark.parametrize("seed, chunk, error", [(-1, 1, ValueError), (1, -1, ValueError), (True, 1, TypeError), (1, 1.5, TypeError)])
def test_chunk_seed_validation(seed, chunk, error):
    with pytest.raises(error):
        walkforward.derive_chunk_seed(seed, chunk)


def test_chunk_seed_does_not_mutate_global_rng():
    before = np.random.get_state()
    assert walkforward.derive_chunk_seed(12, 1) != walkforward.derive_chunk_seed(13, 1)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def trial(hidden=4):
    return search.WalkForwardTrialConfig(3, 0.002, 0.002, 3, 12, hidden, 2, 0.001, 16, "argmax", 0.0)


@pytest.mark.parametrize("logs", [False, True])
def test_walkforward_search_tests_only_frozen_winner(monkeypatch, logs):
    events = []

    def execute(frame, columns, **kwargs):
        split, hidden = kwargs["evaluation_split"], kwargs["hidden"]
        events.append((split, hidden, kwargs["seed"]))
        if split == "validation":
            return {
                "val_metrics": {"macro_f1": 1.0 / hidden},
                "val_backtest": {"outperformance": -float(hidden)},
                "n_val_rows": 36, "n_eval_rows": 36, "n_missing_val_preds": 0,
                "retrain_logs": [],
            }
        assert hidden == 4
        return {
            "test_metrics": {"macro_f1": 0.0},
            "benchmark_comparison": {"outperformance": -999.0},
            "n_test_rows": 36, "n_eval_rows": 36, "n_missing_test_preds": 0,
            "retrain_logs": [],
        }

    monkeypatch.setattr(search, "walk_forward_oracle_ann", execute)
    result = search.run_walkforward_grid_search(
        market_frame(), ["signal"], [trial(6), trial(4)], seed=7,
        suppress_inner_logs=logs,
    )
    assert events == [("validation", 6, 7), ("validation", 4, 7), ("test", 4, 7)]
    assert result.trial_id.tolist() == [2, 1]
    assert result.selected.tolist() == [True, False]
    assert result.objective_score.tolist() == [-4.0, -6.0]
    assert result.attrs["final_test"]["benchmark_comparison"]["outperformance"] == -999.0
    assert "test_macro_f1" not in result.columns


def test_walkforward_search_all_failed_does_not_open_test(monkeypatch):
    def fail(frame, columns, **kwargs):
        assert kwargs["evaluation_split"] == "validation"
        raise ValueError("training failed")

    monkeypatch.setattr(search, "walk_forward_oracle_ann", fail)
    result = search.run_walkforward_grid_search(market_frame(), ["signal"], [trial()])
    assert result.status.tolist() == ["error"]
    assert not result.selected.any()
    assert "final_test" not in result.attrs


def test_walkforward_failed_final_test_never_tests_runner_up(monkeypatch):
    events = []

    def execute(frame, columns, **kwargs):
        events.append((kwargs["evaluation_split"], kwargs["hidden"]))
        if kwargs["evaluation_split"] == "test":
            raise RuntimeError("final evaluation failed")
        return {
            "val_metrics": {}, "val_backtest": {"outperformance": kwargs["hidden"]},
            "n_val_rows": 36, "n_eval_rows": 36, "n_missing_val_preds": 0,
            "retrain_logs": [],
        }

    monkeypatch.setattr(search, "walk_forward_oracle_ann", execute)
    with pytest.raises(RuntimeError, match="final evaluation failed"):
        search.run_walkforward_grid_search(market_frame(), ["signal"], [trial(4), trial(6)])
    assert events == [("validation", 4), ("validation", 6), ("test", 6)]


def test_walkforward_rejects_implicit_warm_start():
    model = RecordingClassifier()
    with pytest.raises(ValueError, match="fresh estimator"):
        walkforward.walk_forward_classifier(
            market_frame(), ["signal"], train_ratio=0.6, val_ratio=0.2,
            context_len=3, walkforward_step=12, model_factory=lambda _: model,
        )


@pytest.mark.parametrize("common", [{"label_mode": "oracle_dp"}, {"evaluation_split": "test"}])
def test_walkforward_search_rejects_unsafe_configuration(common):
    with pytest.raises(ValueError):
        search.run_walkforward_grid_search(market_frame(), ["signal"], [trial()], common_parameters=common)


def test_walkforward_search_real_end_to_end():
    result = search.run_walkforward_grid_search(
        market_frame(), ["signal"], [trial(4), trial(6)], seed=7,
        common_parameters={"train_ratio": 0.6, "val_ratio": 0.2},
    )
    assert result.status.tolist() == ["ok", "ok"]
    assert result.selected.sum() == 1
    assert result.n_val_rows.tolist() == [36, 36]
    assert set(result.attrs["validation_retrain_logs"]) == {"1", "2"}
    for logs in result.attrs["validation_retrain_logs"].values():
        assert [log["seed"] for log in logs] == [
            walkforward.derive_chunk_seed(7, chunk) for chunk in (1, 2, 3)
        ]
    assert result.attrs["final_test"]["n_test_rows"] == 36
    assert all(log["start_idx"] >= 144 for log in result.attrs["final_test"]["retrain_logs"])


def test_cli_persists_separate_final_test_report(monkeypatch, tmp_path):
    import json
    import sys
    from trading_system.pipelines import gridsearch_walkforward as cli

    monkeypatch.setattr(cli, "read_parquet_dataset", lambda _: market_frame())
    monkeypatch.setattr(cli, "compute_market_features", lambda frame: frame)
    monkeypatch.setattr(cli, "MARKET_FEATURE_COLUMNS", ["signal"])
    output = tmp_path / "search.json"
    monkeypatch.setattr(sys, "argv", [
        "gridsearch", "--ticker", "AAA", "--train-ratio", "0.6", "--val-ratio", "0.2",
        "--context-lengths", "3", "--epochs-grid", "2", "--hiddens", "4",
        "--max-trials", "1", "--output-json", str(output),
        "--output-csv", str(tmp_path / "search.csv"),
    ])
    cli.main()
    report = json.loads(output.read_text())
    assert report["selection_split"] == "validation"
    assert report["final_test"]["n_test_rows"] == 36
    assert "val_outperformance" in report["top"][0]
    assert "test_metrics" not in report["top"][0]
    assert report["best_parameters"]["hidden"] == 4
    assert "1" in report["validation_retrain_logs"]
