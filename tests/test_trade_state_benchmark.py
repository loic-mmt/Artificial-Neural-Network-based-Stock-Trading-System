"""Causality, detached training and integrity checks for the S0-S3 study."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from trading_system.experiments.trade_state_benchmark import (
    TradeStateBenchmarkConfig, benchmark_counts, run_trade_state_benchmark,
)
from trading_system.models.neural.trade_state_gru import create_trade_state_gru_classifier
from trading_system.models.specs import ModelBuildContext
from trading_system.pipelines.trade_state_benchmark import main
from trading_system.training.financial_loss import FinancialLossConfig
from trading_system.training.post_open_panel import PostOpenReturnPanel
from trading_system.training.trade_state import STATE_COLUMNS, OvernightTradeBook, TradeStateConfig
from trading_system.training.trade_state_trainer import fit_trade_state_model, rollout_trade_state_model
from trading_system.models.neural.trainer import seed_torch_run


def prices(n=190):
    calendar = pd.bdate_range("2010-01-04", periods=n)
    parts = []
    for i, ticker in enumerate(("A", "B")):
        rng = np.random.default_rng(101 + i)
        opening = (80 + 30 * i) * np.exp(np.cumsum(rng.normal(.0005, .009, n)))
        closing = opening * np.exp(rng.normal(.0003, .012, n))
        parts.append(pd.DataFrame({"date": calendar, "ticker": ticker, "open": opening,
            "high": np.maximum(opening, closing) * 1.01, "low": np.minimum(opening, closing) * .99,
            "close": closing, "adj_close": closing, "volume": rng.integers(10000, 1000000, n).astype(float),
            "sector": "Technology" if i == 0 else "Financials"}))
    return pd.concat(parts, ignore_index=True)


def model_and_panel(n=8, **parameters):
    frame = prices(n)
    panel = PostOpenReturnPanel(frame, protocol="overnight", tickers=("A", "B"))
    X = np.random.default_rng(44).normal(size=(len(frame), 3, 2)).astype(np.float32)
    model = create_trade_state_gru_classifier(ModelBuildContext(input_size=2, context_len=3, seed=1, device="cpu"),
        {"hidden_size": 4, "temporal_pooling": "attention", "epochs": 2, "batch_size": 5, **parameters})
    model.module.eval()
    return model, X, panel


def test_default_grid_and_dry_run_without_price_io(monkeypatch, capsys):
    import trading_system.pipelines.trade_state_benchmark as pipeline
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted price IO or training")
    monkeypatch.setattr(pipeline, "read_parquet_dataset", forbidden)
    monkeypatch.setattr(pipeline, "run_trade_state_benchmark", forbidden)
    assert main(["--tickers", "A", "B", "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["counts"] == {"unique_configurations": 8, "fits": 72, "evaluation_paths": 144}
    assert output["config"]["state_gradient_mode"] == "detached"
    assert benchmark_counts(TradeStateBenchmarkConfig()) == output["counts"]


@pytest.mark.parametrize("changes", [
    {"protocols": ("intraday",)}, {"families": ("hybrid",)}, {"decoders": ("argmax",)},
    {"state_variants": ()}, {"state_variants": ("S1", "S1")}, {"state_variants": ("S4",)},
    {"state_gradient_mode": "tbptt"}, {"state_age_scale": 0}, {"state_return_scale": float("nan")},
    {"label_methods": ("forward_return",)},
])
def test_invalid_study_settings_rejected(changes):
    with pytest.raises(ValueError):
        TradeStateBenchmarkConfig(**changes)


def test_entry_resize_underwater_inversion_and_flat():
    book = OvernightTradeBook(1)
    np.testing.assert_array_equal(book.observe()[0, :6], 0)
    book.advance([.4], [-.1], 0)
    first = book.observe()[0]
    assert first[0] == .4 and first[2] == 1 and first[5] == 1
    assert first[3] == pytest.approx(-.1) and first[4] == pytest.approx(.1)
    book.advance([.8], [0], 0)
    assert book.observe()[0, 2] == 2  # resizing is not a new episode
    assert book.observe()[0, 5] == 2
    assert book.observe()[0, 3] == pytest.approx(-.1)
    book.advance([-1], [-.1], 0)
    assert book.observe()[0, 2] == 1 and book.observe()[0, 5] == 0
    assert book.observe()[0, 3] == pytest.approx(.1)
    book.advance([0], [0], 5)
    np.testing.assert_array_equal(book.observe()[0, :6], 0)
    assert book.observe()[0, 8] == 1


def test_portfolio_snapshot_costs_drawdown_and_same_side_giveback():
    book = OvernightTradeBook(2)
    book.advance([1, -1], [.1, -.1], 5)
    state = book.observe()
    np.testing.assert_allclose(state[:, 6:9], [[1, 0, 0], [1, 0, 0]])
    assert book.equity == pytest.approx(1.0995)
    book.advance([1, -1], [-.05, .05], 5)
    assert book.observe()[0, 4] == pytest.approx(.055)
    np.testing.assert_allclose(book.observe()[:, 9], -.05)


@pytest.mark.parametrize("variant,width", [("S0", 0), ("S1", 3), ("S2", 6), ("S3", 10)])
def test_active_groups_fixed_shape_and_exante_scaling(variant, width):
    raw = np.array([[.5, 1, 252, .1, .1, 252, .5, .5, .5, -.1]])
    encoded = TradeStateConfig(variant).encode(raw)
    assert encoded.shape == (1, len(STATE_COLUMNS)) and encoded.dtype == np.float32
    np.testing.assert_array_equal(encoded[:, width:], 0)
    if width >= 3:
        assert encoded[0, 2] == 1
    if width >= 6:
        assert encoded[0, 3] == pytest.approx(np.tanh(1))


def test_state_head_detaches_inputs_but_learns_parameters_and_has_fixed_capacity():
    model, X, _ = model_and_panel()
    state = torch.ones((2, 10), requires_grad=True)
    logits = model.module(torch.from_numpy(X[:2]), state)
    logits.square().sum().backward()
    assert state.grad is None
    assert model.module.head.weight.grad is not None
    assert model.module.encoder.recurrent.weight_ih_l0.grad is not None
    count = model.parameter_count()
    for variant in ("S0", "S1", "S2", "S3"):
        encoded = TradeStateConfig(variant).encode(np.ones((2, 10)))
        assert model.module(torch.from_numpy(X[:2]), torch.from_numpy(encoded)).shape == (2, 3)
        assert model.parameter_count() == count
    with pytest.raises(RuntimeError, match="rollout"):
        model.predict_proba(X)


def test_causal_rollout_does_not_use_current_or_future_interval_return():
    model, X, panel = model_and_panel()
    config = FinancialLossConfig("combined", combined_pnl_weight=.25)
    baseline = rollout_trade_state_model(model, X, panel, state_config=TradeStateConfig("S3"), loss_config=config)
    changed = PostOpenReturnPanel(prices(8), protocol="overnight", tickers=("A", "B"))
    changed.returns[:, 3:] += .15
    perturbed = rollout_trade_state_model(model, X, changed, state_config=TradeStateConfig("S3"), loss_config=config)
    early = panel.indices[:, :4].ravel()
    np.testing.assert_array_equal(baseline.states[early], perturbed.states[early])
    np.testing.assert_array_equal(baseline.probabilities[early], perturbed.probabilities[early])
    assert not np.allclose(baseline.states[panel.indices[:, 4]], perturbed.states[panel.indices[:, 4]])
    np.testing.assert_allclose(baseline.net_returns, panel.path(baseline.positions, config)[0])


@pytest.mark.parametrize("variant", ["S0", "S3"])
def test_decoder_books_reset_and_probabilities_follow_own_states(variant):
    model, X, panel = model_and_panel()
    kwargs = {"state_config": TradeStateConfig(variant), "loss_config": FinancialLossConfig("pnl")}
    continuous = rollout_trade_state_model(model, X, panel, **kwargs)
    sign = rollout_trade_state_model(model, X, panel, decoder="sign", **kwargs)
    np.testing.assert_array_equal(continuous.raw_states[panel.indices[:, 0]], sign.raw_states[panel.indices[:, 0]])
    assert not np.allclose(continuous.raw_states, sign.raw_states)
    if variant == "S0":
        np.testing.assert_array_equal(continuous.probabilities, sign.probabilities)
    else:
        assert not np.allclose(continuous.probabilities, sign.probabilities)
    again = rollout_trade_state_model(model, X, panel, **kwargs)
    np.testing.assert_array_equal(again.states, continuous.states)
    np.testing.assert_array_equal(again.probabilities, continuous.probabilities)


def test_encoder_cache_equals_direct_forward_for_cached_states():
    model, X, panel = model_and_panel()
    rolled = rollout_trade_state_model(model, X, panel, state_config=TradeStateConfig("S3"),
                                     loss_config=FinancialLossConfig("pnl"))
    with torch.no_grad():
        direct = torch.softmax(model.module(torch.from_numpy(X), torch.from_numpy(rolled.states)), dim=1).numpy()
    np.testing.assert_allclose(direct, rolled.probabilities, atol=1e-7)


def test_dropout_stream_replays_with_cached_detached_states():
    model, X, panel = model_and_panel(num_layers=2, dropout=.2)
    model.module.train()
    seed_torch_run(7, True, torch)
    rolled = rollout_trade_state_model(model, X, panel, state_config=TradeStateConfig("S3"),
                                     loss_config=FinancialLossConfig("pnl"))
    seed_torch_run(7, True, torch)
    replay = np.empty_like(rolled.probabilities)
    with torch.no_grad():
        for start in range(0, len(X), model.config.batch_size):
            end = min(start + model.config.batch_size, len(X))
            replay[start:end] = torch.softmax(model.module(torch.from_numpy(X[start:end]),
                torch.from_numpy(rolled.states[start:end])), dim=1).numpy()
    np.testing.assert_allclose(replay, rolled.probabilities, atol=1e-7)


def test_terminal_liquidation_and_trailing_missing_quotes_remain_cash():
    model, _, full = model_and_panel()
    frame = prices(8)
    # B disappears before A; liquidation happens at B's last actual open.
    frame = frame.loc[~((frame.ticker == "B") & (frame.date >= frame.date.unique()[-2]))].copy()
    panel = PostOpenReturnPanel(frame, protocol="overnight", tickers=("A", "B"), calendar=full.dates)
    X = np.zeros((len(frame), 3, 2), np.float32)
    config = FinancialLossConfig("pnl")
    rolled = rollout_trade_state_model(model, X, panel, state_config=TradeStateConfig("S3"), loss_config=config)
    net, executed, _, turnover, _ = panel.path(rolled.positions, config)
    np.testing.assert_allclose(net, rolled.net_returns)
    assert not executed[1, -3:].any() and not executed[0, -1]
    assert turnover[1, -3] == pytest.approx(abs(executed[1, -4]))
    assert rolled.raw_states[panel.indices[0, -1], 0] == pytest.approx(executed[0, -2])


def test_identical_initial_weights_across_state_variants():
    first, _, _ = model_and_panel()
    for variant in ("S0", "S1", "S2", "S3"):
        TradeStateConfig(variant)
        other, _, _ = model_and_panel()
        for key, value in first.module.state_dict().items():
            torch.testing.assert_close(value, other.module.state_dict()[key], rtol=0, atol=0)


def test_missing_signal_forces_cash_without_compressing_calendar():
    model, _, full = model_and_panel()
    signals = full.signal_frame.drop(index=3).reset_index(drop=True)
    panel = PostOpenReturnPanel(prices(8), protocol="overnight", tickers=("A", "B"), signal_frame=signals)
    X = np.zeros((len(signals), 3, 2), np.float32)
    rolled = rollout_trade_state_model(model, X, panel, state_config=TradeStateConfig("S3"),
                                     loss_config=FinancialLossConfig("pnl"))
    assert panel.indices[0, 3] == -1
    assert rolled.raw_states[panel.indices[0, 4], 0] == 0
    assert len(rolled.net_returns) == 8


def test_reordering_ticker_universe_preserves_predictions():
    model, X, panel = model_and_panel()
    reversed_panel = PostOpenReturnPanel(prices(8), protocol="overnight", tickers=("B", "A"))
    kwargs = {"state_config": TradeStateConfig("S3"), "loss_config": FinancialLossConfig("pnl")}
    a = rollout_trade_state_model(model, X, panel, **kwargs)
    b = rollout_trade_state_model(model, X, reversed_panel, **kwargs)
    np.testing.assert_allclose(a.states, b.states, atol=1e-7)
    np.testing.assert_allclose(a.probabilities, b.probabilities, atol=1e-7)


@pytest.mark.parametrize("family", ["cross_entropy", "financial"])
def test_fit_has_one_update_per_epoch_and_roundtrips_checkpoint(family):
    model, X, panel = model_and_panel()
    y = np.arange(len(X), dtype=np.int64) % 3
    known = np.ones(len(X), bool)
    known[-1] = False
    y[-1] = -999
    loss = FinancialLossConfig("combined", combined_pnl_weight=.25)
    fit = fit_trade_state_model(model, X, y if family == "cross_entropy" else None,
        known if family == "cross_entropy" else None, X, y if family == "cross_entropy" else None,
        known if family == "cross_entropy" else None, objective=family, train_panel=panel, val_panel=panel,
        loss_config=loss, state_config=TradeStateConfig("S3"))
    assert model.fitted_ and fit.best_epoch in (1, 2)
    assert model.learning_diagnostics_["optimizer_updates"] == 2
    if family == "cross_entropy":
        assert sum(model.learning_diagnostics_["class_counts"]) == known.sum()
    else:
        assert model.learning_diagnostics_["class_counts"] is None
    assert np.isfinite(fit.history.train_loss).all()
    assert all(row["combined_gradient_norm_preclip"] > 0 for row in model.learning_trace_)
    replica, _, _ = model_and_panel()
    replica.load_state_dict(model.state_dict())
    kwargs = {"state_config": TradeStateConfig("S3"), "loss_config": loss}
    np.testing.assert_allclose(rollout_trade_state_model(model, X, panel, **kwargs).probabilities,
                              rollout_trade_state_model(replica, X, panel, **kwargs).probabilities)


def test_tiny_full_grid_exports_resume_and_integrity(tmp_path, monkeypatch):
    config = TradeStateBenchmarkConfig(context_len=3, max_features=4, n_folds=1, folds=(0,), seeds=(1,),
        feature_groups=("technical",), base_epochs=1, epoch_multiplier=1, batch_size=32, hidden_size=4,
        gap_bars=3, volatility_window=10, device="cpu")
    report = run_trade_state_benchmark(prices().sample(frac=1, random_state=7), ["A", "B"], tmp_path,
                                     config=config, plots=False, progress=False)
    assert report["completed_fits"] == 8 and report["completed_evaluation_paths"] == 16
    assert not report["final_holdout_opened"]
    result = pd.read_csv(tmp_path / "results.csv")
    assert result.parameter_count.nunique() == 1
    assert len(pd.read_csv(tmp_path / "classification.csv")) == 32
    assert (tmp_path / "common-exposure.csv").is_file()
    assert (tmp_path / "oppositions.csv").is_file()
    directory = tmp_path / "fold-0/seed-1/financial-S3"
    states = pd.read_parquet(directory / "overnight-sign-state-inputs.parquet")
    signals = pd.read_parquet(directory / "overnight-sign-probabilities.parquet")
    assert states[["date", "ticker"]].equals(signals[["date", "ticker"]])
    assert states.groupby("ticker").first().previous_position.eq(0).all()
    assert states.date.max() < pd.Timestamp(config.holdout_start, tz="UTC")
    import trading_system.experiments.trade_state_benchmark as experiment
    def forbidden(*args, **kwargs):
        raise AssertionError("resume retrained a completed fit")
    monkeypatch.setattr(experiment, "fit_trade_state_model", forbidden)
    resumed = run_trade_state_benchmark(prices(), ["A", "B"], tmp_path, config=config,
                                       resume=True, plots=False, progress=False)
    assert resumed == report
    with pytest.raises(ValueError, match="Resume"):
        run_trade_state_benchmark(prices(), ["A", "B"], tmp_path, config=replace(config, cost_bps=6),
                                  resume=True, plots=False, progress=False)
    (directory / "fit.json").write_text("{}")
    with pytest.raises(ValueError):
        run_trade_state_benchmark(prices(), ["A", "B"], tmp_path, config=config,
                                  resume=True, plots=False, progress=False)
