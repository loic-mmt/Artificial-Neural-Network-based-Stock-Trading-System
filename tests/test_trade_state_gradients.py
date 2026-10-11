"""Forward parity, temporal credit, truncation, frozen controls and resume."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from trading_system.experiments.trade_state_benchmark import benchmark_counts, candidate_grid
from trading_system.experiments.trade_state_gradient_benchmark import (
    TradeStateGradientBenchmarkConfig, run_trade_state_gradient_benchmark,
)
from trading_system.models.neural.trade_state_gru import create_trade_state_gru_classifier
from trading_system.models.neural.trainer import seed_torch_run
from trading_system.models.specs import ModelBuildContext
from trading_system.pipelines.trade_state_gradient_benchmark import main
from trading_system.training.financial_loss import FinancialLossConfig
from trading_system.training.post_open_panel import PostOpenReturnPanel
from trading_system.training.trade_state import OvernightTradeBook, TradeStateConfig
from trading_system.training.trade_state_gradients import DifferentiableOvernightTradeBook, state_history_vjp
from trading_system.training.trade_state_trainer import fit_trade_state_model, rollout_trade_state_model


def prices(n=190):
    dates = pd.bdate_range("2010-01-04", periods=n)
    parts = []
    for i, ticker in enumerate(("A", "B")):
        rng = np.random.default_rng(101 + i)
        opening = (80 + 30 * i) * np.exp(np.cumsum(rng.normal(.0005, .009, n)))
        closing = opening * np.exp(rng.normal(.0003, .012, n))
        parts.append(pd.DataFrame({"date": dates, "ticker": ticker, "open": opening,
            "high": np.maximum(opening, closing) * 1.01, "low": np.minimum(opening, closing) * .99,
            "close": closing, "adj_close": closing, "volume": rng.integers(10000, 1000000, n).astype(float),
            "sector": "Technology" if i == 0 else "Financials"}))
    return pd.concat(parts, ignore_index=True)


def case(n=8, frame=None, **parameters):
    frame = prices(n) if frame is None else frame
    panel = PostOpenReturnPanel(frame, protocol="overnight", tickers=("A", "B"))
    X = np.random.default_rng(44).normal(size=(len(frame), 3, 2)).astype(np.float32)
    model = create_trade_state_gru_classifier(ModelBuildContext(input_size=2, context_len=3, seed=1, device="cpu"),
        {"hidden_size": 4, "temporal_pooling": "attention", "epochs": 1, "batch_size": 5, **parameters})
    model.module.eval()
    return model, X, panel


def test_grid_dry_run_and_no_price_reads(monkeypatch, capsys):
    import trading_system.pipelines.trade_state_gradient_benchmark as pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not read prices or train")
    monkeypatch.setattr(pipeline, "read_parquet_dataset", forbidden)
    monkeypatch.setattr(pipeline, "run_trade_state_gradient_benchmark", forbidden)
    assert main(["--dry-run", "--tickers", "A", "B"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["counts"] == {"unique_configurations": 5, "fits": 45, "evaluation_paths": 90}
    assert len({v["candidate"] for v in data["candidates"]}) == 5
    assert [v["tbptt_steps"] for v in data["candidates"] if v["state_gradient_mode"] == "tbptt"] == [10, 40]
    assert benchmark_counts(TradeStateGradientBenchmarkConfig(families=("financial",)))["fits"] == 36


@pytest.mark.parametrize("changes", [
    {"tbptt_windows": ()}, {"tbptt_windows": (0,)}, {"tbptt_windows": (True,)},
    {"tbptt_windows": (2.5,)}, {"tbptt_windows": (10, 10)},
    {"state_variants": ("S3",)}, {"families": ("cross_entropy",)},
    {"state_gradient_mode": "tbptt"}, {"decoders": ("argmax",)},
])
def test_invalid_gradient_grid(changes):
    with pytest.raises(ValueError):
        TradeStateGradientBenchmarkConfig(**changes)


@pytest.mark.parametrize("variant", ["S0", "S1", "S2", "S3"])
def test_differentiable_book_has_identical_forward_states(variant):
    numpy_book, tensor_book = OvernightTradeBook(2), DifferentiableOvernightTradeBook(2)
    config = TradeStateConfig(variant, gradient_mode="tbptt")
    for q, returns in [([.4, -.5], [-.1, .1]), ([.8, -.7], [.04, -.03]),
                       ([-.6, 0], [.03, .02]), ([0, .5], [-.01, .04]), ([0, 0], [0, 0])]:
        np.testing.assert_allclose(tensor_book.observe(config).detach().numpy(),
                                   config.encode(numpy_book.observe()), atol=1e-7)
        actual = tensor_book.advance(torch.tensor(q, dtype=torch.float64, requires_grad=True), returns, 5)
        assert float(actual.detach()) == pytest.approx(numpy_book.advance(q, returns, 5))
    np.testing.assert_allclose(tensor_book.observe(config).detach().numpy(),
                               config.encode(numpy_book.observe()), atol=1e-7)


def test_continuous_credit_and_discrete_counters_and_boundary():
    book = DifferentiableOvernightTradeBook(1)
    q = torch.tensor([.4], dtype=torch.float64, requires_grad=True)
    book.advance(q, [-.1], 5)
    state = book.observe(TradeStateConfig("S3", gradient_mode="tbptt"))
    continuous = torch.autograd.grad(state[:, [0, 6, 7, 8, 9]].sum(), q, retain_graph=True)[0]
    assert continuous.abs().sum() > 0
    discrete = torch.autograd.grad(state[:, 1:6].sum(), q)[0]
    torch.testing.assert_close(discrete, torch.zeros_like(q))
    previous = state.detach().clone()
    book.detach()
    detached = book.observe(TradeStateConfig("S3", gradient_mode="tbptt"))
    assert not detached.requires_grad
    torch.testing.assert_close(detached, previous)


def test_vjp_matches_full_piecewise_finite_difference():
    model, X, panel = case(n=5)
    with torch.no_grad():
        model.module.head.weight.zero_()
        model.module.head.bias.copy_(torch.tensor([-.2, 0., .3]))
        model.module.head.weight[2, model.module.encoder.embedding_size] = .5
    config, loss = TradeStateConfig("S3", gradient_mode="tbptt"), FinancialLossConfig("pnl")
    cached = rollout_trade_state_model(model, X, panel, state_config=config, loss_config=loss, cache_embeddings=True)
    _, upstream = panel.loss_and_gradient(cached.positions, loss)
    _, head, diagnostics = state_history_vjp(model, panel, cached, upstream, config, loss, len(panel.dates))
    assert diagnostics["state_replay_max_position_error"] < 1e-6
    short, _, _ = state_history_vjp(model, panel, cached, upstream, config, loss, 1)
    assert np.isfinite(short).all()
    for index in [(2, 0), (2, model.module.encoder.embedding_size), (0, model.module.head.weight.shape[1]-1)]:
        parameter = model.module.head.weight
        original, epsilon = float(parameter[index].detach()), .002
        values = []
        for direction in (-1, 1):
            with torch.no_grad():
                parameter[index] = original + direction * epsilon
            rolled = rollout_trade_state_model(model, X, panel, state_config=config, loss_config=loss)
            values.append(panel.loss_and_gradient(rolled.positions, loss)[0])
        with torch.no_grad():
            parameter[index] = original
        expected = (values[1] - values[0]) / (2 * epsilon)
        assert float(head[0][index]) == pytest.approx(expected, rel=.04, abs=2e-7)


def test_longer_window_propagates_more_policy_credit():
    model, X, panel = case()
    config, loss = TradeStateConfig("S3", gradient_mode="tbptt"), FinancialLossConfig("combined")
    cached = rollout_trade_state_model(model, X, panel, state_config=config, loss_config=loss, cache_embeddings=True)
    _, upstream = panel.loss_and_gradient(cached.positions, loss)
    one, head_one, _ = state_history_vjp(model, panel, cached, upstream, config, loss, 1)
    full, head_full, stats = state_history_vjp(model, panel, cached, upstream, config, loss, len(panel.dates))
    assert not np.allclose(one, full, rtol=1e-4, atol=1e-9)
    assert not torch.allclose(head_one[0], head_full[0], rtol=1e-4, atol=1e-9)
    assert stats["tbptt_windows"] == 1


@pytest.mark.parametrize("variant,steps", [("S0", 4), ("S3", 1)])
def test_no_history_control_equals_detached_update(variant, steps):
    a, X, panel = case()
    b, _, _ = case()
    loss = FinancialLossConfig("combined", combined_pnl_weight=.25)
    options = dict(objective="financial", train_panel=panel, val_panel=panel, loss_config=loss)
    fit_trade_state_model(a, X, None, None, X, None, None, state_config=TradeStateConfig(variant), **options)
    fit_trade_state_model(b, X, None, None, X, None, None,
        state_config=TradeStateConfig(variant, gradient_mode="tbptt"), tbptt_steps=steps, **options)
    for key, value in a.module.state_dict().items():
        torch.testing.assert_close(value, b.module.state_dict()[key], atol=2e-6, rtol=2e-5)
    assert b.learning_diagnostics_["optimizer_updates"] == 1
    assert b.learning_diagnostics_["gradient_contract"] == "partial_tbptt_continuous_state_history"


def test_dropout_and_missing_signals_replay_without_policy_change():
    frame = prices(9).drop(index=[3, 17]).reset_index(drop=True)
    # Missing trailing quotation is valid; an internal missing quote is not.
    full = prices(9)
    panel = PostOpenReturnPanel(full, protocol="overnight", tickers=("A", "B"),
                               signal_frame=frame[["date", "ticker"]])
    model, _, _ = case(num_layers=2, dropout=.2)
    X = np.random.default_rng(4).normal(size=(len(frame), 3, 2)).astype(np.float32)
    config, loss = TradeStateConfig("S3", gradient_mode="tbptt"), FinancialLossConfig("combined")
    model.module.train()
    seed_torch_run(7, True, torch)
    cached = rollout_trade_state_model(model, X, panel, state_config=config, loss_config=loss, cache_embeddings=True)
    _, upstream = panel.loss_and_gradient(cached.positions, loss)
    embedding, head, stats = state_history_vjp(model, panel, cached, upstream, config, loss, 3)
    assert np.isfinite(embedding).all() and torch.isfinite(head[0]).all()
    assert stats["tbptt_windows"] == 3 and stats["tbptt_max_graph_sessions"] == 3
    assert stats["state_replay_max_position_error"] < 1e-6
    assert not upstream[panel.indices[0, -1]]


@pytest.mark.parametrize("mode,steps,family", [("detached", 2, "financial"), ("tbptt", None, "financial"),
                                              ("tbptt", 0, "financial"), ("tbptt", 2, "cross_entropy")])
def test_invalid_training_gradient_settings(mode, steps, family):
    model, X, panel = case()
    with pytest.raises(ValueError):
        fit_trade_state_model(model, X, None, None, X, None, None, objective=family,
            train_panel=panel, val_panel=panel, loss_config=FinancialLossConfig("pnl"),
            state_config=TradeStateConfig("S3", gradient_mode=mode), tbptt_steps=steps)


def test_tiny_gradient_grid_exports_comparisons_and_safe_resume(tmp_path, monkeypatch):
    config = TradeStateGradientBenchmarkConfig(context_len=3, max_features=4, n_folds=1,
        seeds=(1,), feature_groups=("technical",), base_epochs=1, epoch_multiplier=1, batch_size=32,
        hidden_size=4, gap_bars=3, volatility_window=10, device="cpu", tbptt_windows=(2, 4))
    report = run_trade_state_gradient_benchmark(prices(), ["A", "B"], tmp_path, config=config,
                                               plots=False, progress=False)
    assert report["completed_fits"] == 5 and report["completed_evaluation_paths"] == 10
    assert report["paired_gradient_comparisons"] == 4 and not report["final_holdout_opened"]
    results = pd.read_csv(tmp_path / "results.csv")
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert len(summary) == 10 and results.parameter_count.nunique() == 1
    assert set(results.state_gradient_mode) == {"detached", "tbptt"}
    paired = pd.read_csv(tmp_path / "gradient-paired.csv")
    assert len(paired) == 4 and "delta_short_pnl_contribution" in paired
    directory = tmp_path / "fold-0/seed-1/financial-S3-tbptt-4"
    diag = json.loads((directory / "learning_diagnostics.json").read_text())
    assert diag["optimizer_updates"] == 1 and diag["tbptt_steps"] == 4
    states = pd.read_parquet(directory / "overnight-sign-state-inputs.parquet")
    assert states.groupby("ticker").first().previous_position.eq(0).all()
    assert states.date.max() < pd.Timestamp(config.holdout_start, tz="UTC")
    import trading_system.experiments.trade_state_benchmark as experiment
    def forbidden(*args, **kwargs):
        raise AssertionError("resume retrained a completed fit")
    monkeypatch.setattr(experiment, "fit_trade_state_model", forbidden)
    assert run_trade_state_gradient_benchmark(prices(), ["A", "B"], tmp_path, config=config,
                                              resume=True, plots=False, progress=False) == report
    with pytest.raises(ValueError, match="Resume"):
        run_trade_state_gradient_benchmark(prices(), ["A", "B"], tmp_path,
            config=replace(config, tbptt_windows=(3, 4)), resume=True, plots=False, progress=False)
    (directory / "fit.json").write_text("{}")
    with pytest.raises(ValueError):
        run_trade_state_gradient_benchmark(prices(), ["A", "B"], tmp_path, config=config,
                                          resume=True, plots=False, progress=False)
