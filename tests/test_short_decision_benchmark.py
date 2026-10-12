"""Frozen policies, binary supervision, causal books and integrity-checked runs."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from trading_system.analysis.short_decisions import binary_head_metrics, causal_asset_trends, short_alert_diagnostics
from trading_system.experiments.short_decision_benchmark import ShortDecisionBenchmarkConfig, benchmark_counts, run_short_decision_benchmark
from trading_system.models.neural.short_decision_gru import create_short_decision_gru
from trading_system.models.neural.trade_state_gru import create_trade_state_gru_classifier
from trading_system.models.specs import ModelBuildContext
from trading_system.pipelines.short_decision_benchmark import main
from trading_system.training.financial_loss import FinancialLossConfig
from trading_system.training.post_open_panel import PostOpenReturnPanel
from trading_system.training.short_decision import ShortPolicyConfig, ShortDecisionRollout, binary_class_weights, binary_supervision, decode_short_scores, fit_short_decision_model, rollout_short_decisions
from trading_system.training.trade_state import TradeStateConfig
from trading_system.training.trade_state_trainer import rollout_trade_state_model


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


def case(architecture="short_detector", n=8, **parameters):
    frame = prices(n)
    panel = PostOpenReturnPanel(frame, protocol="overnight", tickers=("A", "B"))
    X = np.random.default_rng(44).normal(size=(len(frame), 3, 2)).astype(np.float32)
    context = ModelBuildContext(input_size=2, context_len=3, seed=1, device="cpu")
    values = {"hidden_size": 4, "temporal_pooling": "attention", "epochs": 1, "batch_size": 5, **parameters}
    model = create_trade_state_gru_classifier(context, values) if architecture == "reference" else create_short_decision_gru(context, values, architecture)
    model.module.eval()
    return model, X, panel


def test_grid_and_price_free_dry_run(monkeypatch, capsys):
    import trading_system.pipelines.short_decision_benchmark as pipeline
    def forbidden(*args, **kwargs): raise AssertionError("Dry run read prices or trained.")
    monkeypatch.setattr(pipeline, "read_parquet_dataset", forbidden)
    assert main(["--dry-run", "--tickers", "A", "B"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["counts"] == {"unique_configurations": 9, "fits": 81, "gru_encoders_trained": 99,
        "evaluation_paths": 486, "frozen_alert_counterfactual_paths": 324}
    assert {c["state_variant"] for c in d["candidates"]} == {"S0", "S3"}
    small = ShortDecisionBenchmarkConfig(families=("financial",), architectures=("short_detector",),
        policies=("long_flat",), include_controls=False)
    assert benchmark_counts(small)["fits"] == 9
    assert benchmark_counts(small)["frozen_alert_counterfactual_paths"] == 0


@pytest.mark.parametrize("changes", [{"architectures": ()}, {"architectures": ("gnn",)},
    {"architectures": ("dual_head", "dual_head")}, {"policies": ("unknown",)}, {"policies": ()},
    {"regime_window": 1}, {"regime_window": -1}, {"regime_window": True},
    {"long_activation_threshold": 0}, {"short_activation_threshold": float("nan")},
    {"conflict_policy": "future_best"}, {"state_gradient_mode": "tbptt"}, {"state_variants": ("S3",)},
    {"include_controls": 1}, {"families": ("hybrid",)}])
def test_invalid_recipe(changes):
    with pytest.raises((ValueError, TypeError)): ShortDecisionBenchmarkConfig(**changes)


@pytest.mark.parametrize("architecture", ["short_detector", "dual_head", "separate_grus"])
@pytest.mark.parametrize("decoder", ["continuous", "sign"])
@pytest.mark.parametrize("policy", ["long_short", "long_flat", "short_flat"])
def test_torch_numpy_decoding_and_bounds(architecture, decoder, policy):
    scores = np.array([[.8, .7], [.2, .8], [.8, .2], [.2, .3], [.5, .5]], dtype=np.float32)
    if architecture == "short_detector": scores = scores[:, 1:]
    a = decode_short_scores(scores, architecture, decoder=decoder, policy=policy)
    b = decode_short_scores(torch.tensor(scores), architecture, decoder=decoder, policy=policy)
    np.testing.assert_allclose(a, b.numpy())
    assert np.abs(a).max() <= 1
    if policy == "long_flat": assert (a >= 0).all()
    if policy == "short_flat": assert (a <= 0).all()
    if decoder == "sign": assert set(a) <= {-1, 0, 1}


def test_explicit_detector_no_alert_is_long_and_conflict_is_flat():
    np.testing.assert_array_equal(decode_short_scores(np.array([[.1], [.5], [.9]]), "short_detector", decoder="sign"), [1, 1, -1])
    scores = np.array([[.8, .7], [.2, .2], [.8, .2], [.2, .8]])
    np.testing.assert_array_equal(decode_short_scores(scores, "dual_head", decoder="sign"), [0, 0, 1, -1])
    np.testing.assert_array_equal(decode_short_scores(scores, "dual_head", decoder="sign", config=ShortPolicyConfig(conflict_policy="margin")), [1, 0, 1, -1])


def test_unknown_labels_and_train_only_binary_weights():
    y = np.array([0, 1, 2, -99, 0, 1]); known = np.array([True, True, True, False, True, True])
    targets, mask = binary_supervision(y, known, ("long", "short"))
    np.testing.assert_array_equal(targets[mask], [[0, 1], [0, 0], [1, 0], [0, 1], [0, 0]])
    weights = binary_class_weights(targets, mask)
    y[-3] = 2
    changed, changed_mask = binary_supervision(y, known, ("long", "short"))
    np.testing.assert_array_equal(weights, binary_class_weights(changed, changed_mask))
    rows = binary_head_metrics(y, known, np.full_like(targets, .5), ("long", "short"), (.5, .5))
    assert all(r["n_known"] == 5 and r["n_unknown"] == 1 for r in rows)
    assert rows[0]["positive_support"] == 1 and rows[1]["positive_support"] == 2


def test_unknown_only_diagnostics_are_unavailable_not_flat_targets():
    rows = binary_head_metrics(np.full(3, -99), np.zeros(3, bool), np.full((3, 1), .4), ("short",), (.5,))
    assert rows[0]["n_known"] == 0 and rows[0]["n_unknown"] == 3
    assert rows[0]["accuracy"] is None and rows[0]["roc_auc"] is None


def test_underwater_model_closure_is_not_forced_final_liquidation():
    raw = np.zeros((2, 10)); raw[:, 0] = -1; raw[:, 5] = [12, 30]
    rolled = ShortDecisionRollout(None, None, np.array([0., -1.]), None, raw, None)
    records = [{"position": -1., "asset_return": -.01}]
    result = short_alert_diagnostics(records, [], rolled, np.array([0., 0.]))
    assert result["underwater_short_closures"] == 2
    assert result["underwater_short_model_closures"] == 1 and result["forced_underwater_short_liquidations"] == 1
    assert result["mean_consecutive_underwater_at_model_closure"] == 12


@pytest.mark.parametrize("decoder", ["continuous", "sign"])
@pytest.mark.parametrize("variant", ["S0", "S3"])
def test_reference_forward_matches_original_exactly(decoder, variant):
    model, X, panel = case("reference")
    cfg, loss = TradeStateConfig(variant), FinancialLossConfig("combined")
    original = rollout_trade_state_model(model, X, panel, state_config=cfg, loss_config=loss, decoder=decoder)
    current = rollout_short_decisions(model, X, panel, state_config=cfg, loss_config=loss, decoder=decoder)
    np.testing.assert_array_equal(current.scores, original.probabilities)
    np.testing.assert_array_equal(current.positions, original.positions)
    np.testing.assert_array_equal(current.raw_states, original.raw_states)


def test_independent_encoders_and_shared_heads_parameter_accounting():
    counts = []
    for architecture in ("short_detector", "dual_head", "separate_grus"):
        model, X, _ = case(architecture)
        assert model.module(X=torch.tensor(X), states=torch.zeros(len(X), 10)).shape == (len(X), len(model.head_names))
        counts.append(model.parameter_count())
        if architecture == "separate_grus":
            a, b = model.module.encoders
            assert not ({id(p) for p in a.parameters()} & {id(p) for p in b.parameters()})
    assert counts[0] < counts[1] < counts[2] and counts[2] == 2 * counts[0]


@pytest.mark.parametrize("architecture", ["short_detector", "dual_head", "separate_grus"])
@pytest.mark.parametrize("objective", ["financial", "cross_entropy"])
def test_training_single_update_finite_and_serializable(architecture, objective):
    model, X, panel = case(architecture)
    y = np.arange(len(X)) % 3; known = np.ones(len(X), bool); known[-1] = False
    if objective == "financial": y = known = None
    initial = {k: v.clone() for k, v in model.module.state_dict().items()}
    result = fit_short_decision_model(model, X, y, known, X, y, known, objective=objective,
        train_panel=panel, val_panel=panel, loss_config=FinancialLossConfig("combined", combined_pnl_weight=.25), state_config=TradeStateConfig("S3"))
    assert result.best_epoch == 1 and model.learning_diagnostics_["optimizer_updates"] == 1
    assert np.isfinite(model.learning_trace_[0]["combined_gradient_norm_preclip"])
    assert any(not torch.equal(v, initial[k]) for k, v in model.module.state_dict().items())
    clone, _, _ = case(architecture); clone.load_state_dict(model.state_dict())
    for k, value in model.module.state_dict().items(): torch.testing.assert_close(value, clone.module.state_dict()[k])
    incompatible, _, _ = case("dual_head" if architecture == "short_detector" else "short_detector")
    with pytest.raises(ValueError): incompatible.load_state_dict(model.state_dict())


@pytest.mark.parametrize("architecture,objective", [("short_detector", "financial"),
    ("dual_head", "financial"), ("separate_grus", "financial"), ("dual_head", "cross_entropy")])
def test_memory_blocks_equal_one_global_manual_update(architecture, objective):
    model, X, panel = case(architecture)
    expected, _, _ = case(architecture)
    config, loss = TradeStateConfig("S3"), FinancialLossConfig("combined", combined_pnl_weight=.25)
    rolled = rollout_short_decisions(expected, X, panel, state_config=config, loss_config=loss)
    optimizer = torch.optim.AdamW(expected.module.parameters(), lr=expected.config.learning_rate,
                                 weight_decay=expected.config.weight_decay)
    logits = expected.module(torch.tensor(X), torch.tensor(rolled.states))
    y = np.arange(len(X)) % 3; known = np.ones(len(X), bool); known[-1] = False
    if objective == "financial":
        _, gradient = panel.loss_and_gradient(rolled.positions, loss)
        decode_short_scores(torch.sigmoid(logits), architecture).backward(torch.tensor(gradient, dtype=logits.dtype))
    else:
        targets, selected = binary_supervision(y, known, expected.head_names)
        weights = torch.tensor(binary_class_weights(targets, selected), dtype=logits.dtype)
        observed = torch.tensor(targets[selected])
        row_weights = torch.where(observed > 0, weights[:, 1], weights[:, 0])
        errors = torch.nn.functional.binary_cross_entropy_with_logits(logits[selected], observed, reduction="none")
        ((errors * row_weights).sum(dim=0) / row_weights.sum(dim=0)).mean().backward()
    torch.nn.utils.clip_grad_norm_(expected.module.parameters(), expected.config.gradient_clip_norm)
    optimizer.step()
    fit_short_decision_model(model, X, y, known, X, y, known, objective=objective,
        train_panel=panel, val_panel=panel, loss_config=loss, state_config=config)
    for key, value in expected.module.state_dict().items():
        torch.testing.assert_close(value, model.module.state_dict()[key], rtol=3e-5, atol=2e-6)


def test_closed_loop_projections_reset_state_and_never_use_future_quotes():
    model, X, panel = case("short_detector")
    with torch.no_grad():
        model.module.heads[0].weight.zero_(); model.module.heads[0].bias.fill_(2.)
    cfg, loss = TradeStateConfig("S3"), FinancialLossConfig("pnl")
    a = rollout_short_decisions(model, X, panel, state_config=cfg, loss_config=loss, decoder="sign")
    flat = rollout_short_decisions(model, X, panel, state_config=cfg, loss_config=loss, decoder="sign", policy="long_flat")
    assert (a.positions == -1).all() and (flat.positions == 0).all()
    assert (flat.raw_states[:, 0] == 0).all() and a.raw_states[:, 0].min() == -1
    future = prices(8); cutoff = future.date.sort_values().unique()[5]
    future.loc[future.date >= cutoff, ["open", "close", "adj_close", "high", "low"]] *= 1.3
    changed = PostOpenReturnPanel(future, protocol="overnight", tickers=("A", "B"))
    b = rollout_short_decisions(model, X, changed, state_config=cfg, loss_config=loss, decoder="sign")
    rows = prices(8).date.to_numpy() < cutoff
    np.testing.assert_array_equal(a.raw_states[rows], b.raw_states[rows])


def test_missing_signal_rows_preserve_calendar_and_force_cash():
    model, _, _ = case()
    with torch.no_grad():
        model.module.heads[0].weight.zero_(); model.module.heads[0].bias.fill_(2.)
    frame = prices(9); signals = frame.drop(index=[2, 15]).reset_index(drop=True)
    panel = PostOpenReturnPanel(frame, protocol="overnight", tickers=("A", "B"), signal_frame=signals[["date", "ticker"]])
    X = np.random.default_rng(45).normal(size=(len(signals), 3, 2)).astype(np.float32)
    rolled = rollout_short_decisions(model, X, panel, state_config=TradeStateConfig("S3"), loss_config=FinancialLossConfig("pnl"))
    row = signals.index[(signals.ticker == "A") & (signals.date == frame.date.iloc[3])][0]
    assert rolled.raw_states[row, 0] == 0 and len(rolled.net_returns) == 9


def test_regime_is_lagged_and_not_fitted_on_outer():
    frame = prices(10); a = causal_asset_trends(frame, 3)
    cutoff = frame.date.max(); frame.loc[frame.date == cutoff, "adj_close"] *= 100
    b = causal_asset_trends(frame, 3)
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("ticker").head(3).asset_trend.eq("unknown").all()
    assert causal_asset_trends(frame, 0).asset_trend.eq("unknown").all()


def test_single_candidate_disabled_regime_and_plots(tmp_path):
    cfg = ShortDecisionBenchmarkConfig(context_len=3, max_features=4, n_folds=1, seeds=(1,),
        feature_groups=("technical",), base_epochs=1, epoch_multiplier=1, batch_size=32,
        hidden_size=4, gap_bars=3, volatility_window=10, device="cpu", regime_window=0,
        families=("financial",), architectures=("short_detector",), policies=("long_flat",), include_controls=False)
    report = run_short_decision_benchmark(prices(), ["A", "B"], tmp_path, config=cfg, plots=True, progress=False)
    assert report["completed_fits"] == 1 and report["completed_evaluation_paths"] == 2
    assert report["completed_frozen_alert_counterfactual_paths"] == 0
    assert list(tmp_path.rglob("portfolio-equity.png"))
    assert list(tmp_path.rglob("learning-gradients.png"))


def test_tiny_full_grid_exports_policies_counterfactuals_and_resume(tmp_path, monkeypatch):
    config = ShortDecisionBenchmarkConfig(context_len=3, max_features=4, n_folds=1, seeds=(1,),
        feature_groups=("technical",), base_epochs=1, epoch_multiplier=1, batch_size=32,
        hidden_size=4, gap_bars=3, volatility_window=10, device="cpu", regime_window=3)
    report = run_short_decision_benchmark(prices(), ["A", "B"], tmp_path, config=config, plots=False, progress=False)
    assert report["completed_fits"] == 9 and report["completed_evaluation_paths"] == 54
    assert report["completed_frozen_alert_counterfactual_paths"] == 36
    assert not report["final_holdout_opened"] and not report["selection_performed_on_outer"]
    results = pd.read_csv(tmp_path / "results.csv")
    assert set(results.policy) == {"long_short", "long_flat", "short_flat"}
    assert len(pd.read_csv(tmp_path / "frozen-alert-counterfactuals.csv")) == 36
    assert len(pd.read_csv(tmp_path / "short-paired.csv")) == 36
    assert len(pd.read_csv(tmp_path / "common-exposure.csv")) == 54
    paired = pd.read_csv(tmp_path / "short-paired.csv")
    assert paired[paired.family.eq("cross_entropy")].reference.eq("cross_entropy-S3-reference").all()
    assert not pd.read_csv(tmp_path / "regimes.csv").empty
    directory = tmp_path / "fold-0/seed-1/financial-short_detector"
    states = pd.read_parquet(directory / "long_short/overnight-sign-state-inputs.parquet")
    assert states.groupby("ticker").first().previous_position.eq(0).all()
    for policy, sign in (("long_flat", 1), ("short_flat", -1)):
        positions = pd.read_parquet(directory / policy / "overnight-sign-positions.parquet")
        assert (positions.position * sign >= 0).all()
    assert states.date.max() < pd.Timestamp(config.holdout_start, tz="UTC")
    counter = pd.read_csv(tmp_path / "frozen-alert-counterfactuals.csv")
    assert counter.scope.eq("frozen_long_short_signals_no_state_feedback_no_refit").all()
    import trading_system.experiments.short_decision_benchmark as experiment
    def forbidden(*args, **kwargs): raise AssertionError("Resume retrained.")
    monkeypatch.setattr(experiment, "fit_short_decision_model", forbidden)
    monkeypatch.setattr(experiment, "fit_trade_state_model", forbidden)
    assert run_short_decision_benchmark(prices(), ["A", "B"], tmp_path, config=config, resume=True, plots=False, progress=False) == report
    with pytest.raises(ValueError, match="Resume"):
        run_short_decision_benchmark(prices(), ["A", "B"], tmp_path, config=replace(config, conflict_policy="margin"), resume=True, plots=False, progress=False)
    (directory / "fit.json").write_text("{}")
    with pytest.raises(ValueError): run_short_decision_benchmark(prices(), ["A", "B"], tmp_path, config=config, resume=True, plots=False, progress=False)
