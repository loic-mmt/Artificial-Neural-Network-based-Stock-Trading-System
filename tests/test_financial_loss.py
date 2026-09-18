from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel, probabilities_to_positions
from trading_system.training.position_trainer import fit_position_model, predict_positions
from trading_system.models.factory import create_default_model_registry
from trading_system.models.specs import ModelBuildContext, ModelSelection
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.position_objectives import (
    _align_position_calendar, run_position_validation, evaluate_position_test,
    save_position_artifact, load_position_artifact,
)
from trading_system.experiments.runner import run_validation_experiment


def prices(n=220, ticker="A"):
    rng = np.random.default_rng(91)
    p = 100 * np.exp(np.cumsum(rng.normal(.001, .012, n)))
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "ticker": ticker,
                         "open": p, "high": p + 1, "low": p - 1, "close": p,
                         "adj_close": p, "volume": rng.integers(100, 10000, n).astype(float), "sector": "Technology"})


@pytest.mark.parametrize("objective", ["pnl", "sharpe"])
@pytest.mark.parametrize("delay", [1, 3])
def test_loss_gradient_matches_finite_difference_with_costs_and_multiple_assets(objective, delay):
    frame = pd.concat([prices(13), prices(13, "B").assign(adj_close=lambda f: f.adj_close * np.linspace(1, 1.03, len(f)))], ignore_index=True).sample(frac=1, random_state=7)
    panel = ReturnPanel(frame, group_col="ticker", execution_delay=delay)
    config = FinancialLossConfig(objective, cost_bps=17)
    positions = np.random.default_rng(3).uniform(-.7, .7, len(frame))
    loss, gradient = panel.loss_and_gradient(positions, config)
    numerical = np.zeros_like(gradient)
    for i in range(len(positions)):
        step = np.zeros_like(positions)
        step[i] = 1e-6
        numerical[i] = (panel.loss_and_gradient(positions + step, config)[0] - panel.loss_and_gradient(positions - step, config)[0]) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-5)
    assert np.isfinite(loss)
    assert np.all(gradient[panel.indices[:, -(delay + 1):]] == 0)


def test_execution_costs_terminal_liquidation_and_no_cross_ticker_turnover():
    a = prices(5).assign(adj_close=[100., 110., 121., 121., 121.])
    frame = pd.concat([a, a.assign(ticker="B", adj_close=100.)], ignore_index=True)
    panel = ReturnPanel(frame, group_col="ticker")
    positions = np.r_[np.ones(5), -np.ones(5)]
    net, executed, _, turnover, _ = panel.path(positions, FinancialLossConfig(cost_bps=100))
    np.testing.assert_allclose(executed, [[0, 1, 1, 1], [0, -1, -1, -1]])
    np.testing.assert_allclose(turnover, [[0, 1, 0, 1], [0, 1, 0, 1]])
    np.testing.assert_allclose(net, [0., .04, 0., -.01], atol=1e-12)


def test_sharpe_flat_finite_and_panel_calendar_errors():
    data = prices(10)
    panel = ReturnPanel(data)
    loss, gradient = panel.loss_and_gradient(np.zeros(10), FinancialLossConfig("sharpe"))
    assert loss == 0 and np.isfinite(gradient).all()
    assert panel.metrics(np.zeros(10), FinancialLossConfig())['net_sharpe'] == 0
    with pytest.raises(ValueError, match="identical asset calendars"):
        ReturnPanel(pd.concat([data, data.iloc[:-1].assign(ticker="B")]), group_col="ticker")
    with pytest.raises(ValueError, match="unique dates"):
        ReturnPanel(pd.concat([data, data.iloc[:1]]))
    with pytest.raises(ValueError, match="execution_delay"):
        ReturnPanel(data, execution_delay=0)


def test_position_workflow_aligns_multi_asset_calendar_without_imputation():
    first = prices(10)
    second = prices(10, "B").drop(index=[2, 7])
    frame = pd.concat([first, second], ignore_index=True)
    frame.attrs["source"] = "test"
    cfg = ExperimentConfig(universe="multi", group_col="ticker")

    aligned = _align_position_calendar(frame, cfg)

    assert aligned.groupby("ticker").size().tolist() == [8, 8]
    assert aligned.groupby("ticker").date.apply(tuple).nunique() == 1
    assert aligned.attrs["source"] == "test"
    assert not aligned.date.isin(first.loc[[2, 7], "date"]).any()


@pytest.mark.parametrize("fields", [{"objective": "bad"}, {"cost_bps": -1}, {"cost_bps": np.nan},
                                      {"annualization": 0}, {"sharpe_epsilon": 0}, {"sharpe_epsilon": np.inf}])
def test_loss_configuration_rejects_invalid_inputs(fields):
    with pytest.raises(ValueError):
        FinancialLossConfig(**fields)


def test_probability_decoder():
    p = np.array([[.2, .3, .5], [1., 0., 0.]])
    np.testing.assert_allclose(probabilities_to_positions(p, "long_short"), [.3, -1])
    np.testing.assert_allclose(probabilities_to_positions(p, "long_only"), [.5, 0])


def model(name, *, batch_size=7, epochs=3, dropout=False):
    params = {"epochs": epochs, "batch_size": batch_size, "early_stopping_min_delta": 0.}
    if name == "manual_ann":
        params.update(hidden_size=4, learning_rate=.05, dropout_probability=.2 if dropout else 0.)
    elif name == "transformer":
        params.update(d_model=4, n_heads=2, num_layers=1, dim_feedforward=8, dropout=.2 if dropout else 0.)
    else:
        params.update(hidden_size=4, num_layers=2 if dropout else 1, dropout=.2 if dropout else 0.)
    return create_default_model_registry().build(name, ModelBuildContext(2, 3, seed=4, device="cpu"), params)


@pytest.mark.parametrize("name", ["manual_ann", "rnn", "lstm", "gru", "transformer"])
@pytest.mark.parametrize("objective", ["pnl", "sharpe"])
def test_all_models_train_and_restore_finite_position_outputs(name, objective):
    if name != "manual_ann":
        pytest.importorskip("torch")
    data = prices(24)
    panel = ReturnPanel(data)
    X = np.random.default_rng(2).normal(size=(24, 3, 2)).astype(np.float32)
    first, second = model(name, dropout=True), model(name, dropout=True)
    loss = FinancialLossConfig(objective)
    result = fit_position_model(first, X, panel, X * .8, panel, loss, "long_short")
    fit_position_model(second, X, panel, X * .8, panel, loss, "long_short")
    assert result.best_epoch >= 1
    assert np.isfinite(result.history.train_loss).all()
    np.testing.assert_array_equal(predict_positions(first, X, "long_short"), predict_positions(second, X, "long_short"))
    val_loss, _ = panel.loss_and_gradient(predict_positions(first, X * .8, "long_short"), loss)
    assert val_loss == pytest.approx(min(result.history.val_loss), rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("name", ["manual_ann", "gru"])
def test_global_update_and_sharpe_independent_of_activation_block_size(name):
    if name != "manual_ann":
        pytest.importorskip("torch")
    X = np.random.default_rng(8).normal(size=(20, 3, 2)).astype(np.float32)
    panel = ReturnPanel(prices(20))
    first, second = model(name, batch_size=4, epochs=1), model(name, batch_size=20, epochs=1)
    loss = FinancialLossConfig("sharpe")
    for current in (first, second):
        fit_position_model(current, X, panel, X, panel, loss, "long_short")
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X), atol=1e-6, rtol=1e-5)


def config():
    return ExperimentConfig(context_len=3, label_mode="triple_barrier", triple_barrier_max_holding=3,
                            triple_barrier_volatility_window=5, device="cpu",
                            model=ModelSelection("manual_ann", {"epochs": 2, "hidden_size": 4}))


def test_cross_entropy_control_exactly_preserves_previous_training_and_backtest():
    frame = prices()
    previous = run_validation_experiment(frame, config())
    control = run_position_validation(frame, config(), FinancialLossConfig("cross_entropy"))
    assert control.legacy_validation_metrics == previous.val_backtest
    np.testing.assert_array_equal(control.bundle.scaler.mean_, previous.bundle.scaler.mean_)
    for name, weights in previous.bundle.estimator.estimator.state_dict().items():
        np.testing.assert_array_equal(weights, control.bundle.estimator.estimator.state_dict()[name])
    assert control.bundle.fit_result.history == previous.bundle.fit_result.history


@pytest.mark.parametrize("objective", ["cross_entropy", "pnl", "sharpe"])
def test_test_price_perturbation_cannot_change_fit_and_artifact_roundtrip(tmp_path, objective):
    frame = prices()
    cfg = config()
    loss = FinancialLossConfig(objective)
    baseline = run_position_validation(frame, cfg, loss)
    modified = frame.copy()
    test_start = int(len(frame) * (cfg.train_ratio + cfg.val_ratio))
    modified.loc[test_start:, ["open", "high", "low", "close", "adj_close"]] *= 10
    changed = run_position_validation(modified, cfg, loss)
    assert baseline.validation_metrics == changed.validation_metrics
    for key, weights in baseline.bundle.estimator.estimator.state_dict().items():
        np.testing.assert_array_equal(weights, changed.bundle.estimator.estimator.state_dict()[key])
    artifact = save_position_artifact(tmp_path / objective, frame, baseline)
    restored = load_position_artifact(artifact)
    assert restored.loss_config == loss
    assert restored.bundle.feature_columns == baseline.bundle.feature_columns
    original_metrics = evaluate_position_test(frame, baseline)
    assert evaluate_position_test(frame, restored) == original_metrics
    with pytest.raises(ValueError, match="already evaluated"):
        evaluate_position_test(frame, restored)


def test_selection_excludes_failed_seeds_and_separates_ce_control():
    from trading_system.pipelines.compare_losses import select_candidates
    rows = [dict(candidate=candidate, objective=objective, seed=seed, status="ok", model_name="manual_ann", val_net_return=value)
            for candidate, objective, value in (("ce", "cross_entropy", .2), ("pnl", "pnl", .5), ("sharpe", "sharpe", .4)) for seed in (1, 2)]
    rows[3]["status"] = "error"
    _, selected = select_candidates(rows, [1, 2], "net_return")
    assert selected == ["ce", "sharpe"]


def test_cli_comparison_seals_test_by_default_and_freezes_selection(tmp_path, monkeypatch):
    from trading_system.pipelines import compare_losses
    data = tmp_path / "prices.parquet"
    prices().to_parquet(data)
    def forbidden(*args, **kwargs):
        raise AssertionError("Final test must stay sealed")
    monkeypatch.setattr(compare_losses, "evaluate_position_test", forbidden)
    report = compare_losses.main([
        "--data", str(data), "--preset", "multi_ticker_long_short", "--models", "manual_ann",
        "--seeds", "1", "--context-len", "3", "--model-parameter-sets",
        '{"manual_ann":[{"epochs":1,"hidden_size":4}]}', "--output-dir", str(tmp_path / "report"), "--fail-fast",
    ])
    assert report["final_test"] == []
    assert len(report["validation"]) == 3
    assert all(row["status"] == "ok" for row in report["validation"])
    assert (tmp_path / "report" / "selection.json").exists()


def test_cli_final_test_only_opens_selected_candidates_after_persisting_selection(tmp_path, monkeypatch):
    import json
    from trading_system.pipelines import compare_losses
    data = tmp_path / "prices.parquet"
    prices().to_parquet(data)
    output = tmp_path / "final-report"
    original = compare_losses.evaluate_position_test
    evaluated = []
    def evaluate(frame, validation):
        selection = json.loads((output / "selection.json").read_text())
        assert len(selection["metadata"]["selection"]) == 2
        evaluated.append(validation.loss_config.objective)
        return original(frame, validation)
    monkeypatch.setattr(compare_losses, "evaluate_position_test", evaluate)
    report = compare_losses.main([
        "--data", str(data), "--preset", "multi_ticker_long_short", "--models", "manual_ann",
        "--seeds", "1", "--context-len", "3", "--model-parameter-sets",
        '{"manual_ann":[{"epochs":1,"hidden_size":4}]}', "--output-dir", str(output), "--fail-fast", "--final-test",
    ])
    assert len(report["final_test"]) == 2
    assert "cross_entropy" in evaluated
    assert len(set(evaluated) & {"pnl", "sharpe"}) == 1


def test_expanded_fracdiff_and_torch_artifact_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from trading_system.features.fracdiff import FracDiffConfig
    cfg = replace(config(), feature_set="expanded", fracdiff=FracDiffConfig(order=.5, threshold=.05),
                  model=ModelSelection("gru", {"epochs": 1, "hidden_size": 4}))
    data = prices()
    result = run_position_validation(data, cfg, FinancialLossConfig("sharpe"))
    artifact = save_position_artifact(tmp_path / "expanded-gru", data, result)
    restored = load_position_artifact(artifact)
    assert restored.bundle.feature_selector.columns == result.bundle.feature_selector.columns
    assert restored.bundle.fracdiff_transformer.state_dict() == result.bundle.fracdiff_transformer.state_dict()
    assert evaluate_position_test(data, restored) == evaluate_position_test(data, result)
