"""Contract and exact-gradient checks for the isolated common-cadence trainer."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from trading_system.models.factory import create_default_model_registry
from trading_system.models.specs import ModelBuildContext
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel
from trading_system.training.label_loss_trainer import fit_label_loss_model
from trading_system.training.weights import compute_class_weights


def model(*, batch_size=4, epochs=1, dropout=0.0, **parameters):
    configuration = {
        "hidden_size": 4, "temporal_pooling": "attention", "batch_size": batch_size,
        "epochs": epochs, "gradient_clip_norm": None,
        "early_stopping_min_delta": 0.0, "early_stopping_patience": 20,
        "head_type": "mlp" if dropout else "linear", "head_dropout": dropout,
        **parameters,
    }
    return create_default_model_registry().build(
        "gru", ModelBuildContext(2, 3, seed=8, device="cpu"), configuration
    )


def data(n=11):
    values = np.random.default_rng(13).normal(size=(n, 3, 2)).astype(np.float32)
    labels = np.resize(np.array([0, 0, 0, 1, 2], dtype=np.int64), n)
    known = np.ones(n, dtype=bool)
    return values, labels, known


def panel(n=11):
    return ReturnPanel(pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n),
        "adj_close": 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0.001, 0.01, n))),
    }))


def fit(current, *, objective="cross_entropy", dataset=None, **kwargs):
    X, y, known = data() if dataset is None else dataset
    finance = objective in ("financial", "hybrid")
    return fit_label_loss_model(
        current, X, y, known, X * 0.9, y, known,
        objective=objective, train_panel=panel(len(X)) if finance else None,
        val_panel=panel(len(X)) if finance else None,
        loss_config=FinancialLossConfig("combined", combined_pnl_weight=0.25) if finance else None,
        **kwargs,
    )


@pytest.mark.parametrize("objective", ["cross_entropy", "financial", "hybrid"])
def test_one_adamw_update_per_epoch_and_diagnostics_are_json_safe(objective, monkeypatch):
    calls = []
    original = torch.optim.AdamW.step

    def step(optimizer, *args, **kwargs):
        calls.append(1)
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    current = model(epochs=3)
    records = []
    result = fit(current, objective=objective, progress_callback=records.append)
    assert len(calls) == len(result.history.train_loss) == 3
    assert len(records) == 3
    assert [record["optimizer_updates"] for record in records] == [1, 2, 3]
    assert current.learning_diagnostics_["update_cadence"] == "one_global_step_per_epoch"
    assert current.fit_result_ is result and current.fitted_
    json.dumps(current.learning_diagnostics_, allow_nan=False)
    np.testing.assert_allclose(current.predict_proba(data()[0]).sum(axis=1), 1, atol=1e-6)


@pytest.mark.parametrize("objective", ["cross_entropy", "financial", "hybrid"])
def test_full_parameter_gradient_matches_unblocked_autograd(objective):
    X, y, known = data()
    known[[1, 5]] = False
    y[~known] = -987
    available = np.ones(len(X), dtype=bool)
    available[8] = False
    reference = model(batch_size=len(X))
    current = model(batch_size=4)
    reference.module.train()
    logits = reference.module(torch.as_tensor(X))
    selected = known & available
    weights = compute_class_weights(y[selected], 3)
    ce = torch.nn.functional.cross_entropy(
        logits[torch.as_tensor(selected)], torch.as_tensor(y[selected]),
        weight=torch.as_tensor(weights), reduction="sum",
    ) / float(np.sum(weights[y[selected]], dtype=np.float64))
    q = torch.softmax(logits, dim=1) @ torch.tensor([-1.0, 0.0, 1.0])
    masked = q * torch.as_tensor(available)
    loss_config = FinancialLossConfig("combined", combined_pnl_weight=0.25)
    _, financial_gradient = panel().loss_and_gradient(masked.detach().numpy(), loss_config)
    ce_gradient = torch.autograd.grad(ce, tuple(reference.module.parameters()), retain_graph=True)
    fin_gradient = torch.autograd.grad(
        masked, tuple(reference.module.parameters()), grad_outputs=torch.as_tensor(financial_gradient, dtype=q.dtype)
    )
    ce_coefficient = 1 if objective == "cross_entropy" else 0.5 if objective == "hybrid" else 0
    fin_coefficient = 1 if objective == "financial" else 0.5 if objective == "hybrid" else 0
    expected = [ce_coefficient * c + fin_coefficient * f for c, f in zip(ce_gradient, fin_gradient)]
    fit(current, objective=objective, dataset=(X, y, known), available_train=available)
    for parameter, gradient in zip(current.module.parameters(), expected):
        np.testing.assert_allclose(parameter.grad.numpy(), gradient.numpy(), atol=2e-6, rtol=2e-5)
    expected_norm = np.sqrt(sum(float(gradient.double().square().sum()) for gradient in expected))
    assert current.learning_trace_[0]["combined_gradient_norm_preclip"] == pytest.approx(expected_norm, rel=2e-6)
    if objective == "hybrid":
        ce_norm = np.sqrt(sum(float(gradient.double().square().sum()) for gradient in ce_gradient))
        fin_norm = np.sqrt(sum(float(gradient.double().square().sum()) for gradient in fin_gradient))
        assert current.learning_trace_[0]["ce_gradient_norm"] == pytest.approx(ce_norm, rel=2e-6)
        assert current.learning_trace_[0]["financial_gradient_norm"] == pytest.approx(fin_norm, rel=2e-6)


@pytest.mark.parametrize("objective", ["cross_entropy", "financial", "hybrid"])
def test_uneven_activation_blocks_do_not_change_global_update(objective):
    first, second = model(batch_size=4), model(batch_size=11)
    fit(first, objective=objective)
    fit(second, objective=objective)
    np.testing.assert_allclose(first.predict_proba(data()[0]), second.predict_proba(data()[0]), atol=1e-6, rtol=1e-5)


def test_pure_financial_never_reads_labels_or_known_masks():
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError("A pure financial trainer attempted to consume labels.")

    X, y, known = data()
    first, second = model(epochs=2), model(epochs=2)
    fit(first, objective="financial")
    fit_label_loss_model(
        second, X, Poison(), Poison(), X * 0.9, Poison(), Poison(),
        objective="financial", train_panel=panel(), val_panel=panel(),
        loss_config=FinancialLossConfig("combined", combined_pnl_weight=0.25),
    )
    np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))
    assert second.learning_diagnostics_["class_weights"] is None
    assert second.learning_trace_[0]["train"]["ce_loss"] is None


def test_unknown_and_unavailable_labels_do_not_affect_train_class_weights():
    X, y, known = data()
    known[[0, 1]] = False
    available = np.ones(len(y), dtype=bool)
    available[2] = False
    first, second = model(), model()
    fit(first, dataset=(X, y.copy(), known), available_train=available, available_val=available)
    changed = y.copy()
    changed[~known | ~available] = -99
    fit(second, dataset=(X, changed, known), available_train=available, available_val=available)
    expected = compute_class_weights(y[known & available])
    np.testing.assert_allclose(first.learning_diagnostics_["class_weights"], expected)
    np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))


def test_class_weights_are_not_recomputed_from_validation():
    X, y, known = data()
    current = model()
    fit_label_loss_model(current, X, y, known, X, np.full(len(y), 2), known, objective="cross_entropy")
    np.testing.assert_allclose(current.learning_diagnostics_["class_weights"], compute_class_weights(y))


def test_unavailable_rows_remain_in_financial_panel_and_have_zero_positions():
    class RecordingPanel:
        rows = 11

        def __init__(self):
            self.positions = []

        def loss_and_gradient(self, positions, config):
            self.positions.append(positions.copy())
            return float(np.mean(positions)), np.full(self.rows, 1 / self.rows)

    X, y, known = data()
    available = np.ones(11, dtype=bool)
    available[[2, 7]] = False
    recorded = RecordingPanel()
    current = model()
    fit_label_loss_model(
        current, X, None, None, X, None, None, objective="financial",
        train_panel=recorded, val_panel=recorded,
        available_train=available, available_val=available,
    )
    assert len(recorded.positions) == 3
    for positions in recorded.positions:
        assert positions.shape == (11,)
        np.testing.assert_array_equal(positions[~available], 0)


def test_two_pass_dropout_is_replayed_and_gpu_blocks_are_bounded():
    current = model(batch_size=4, dropout=0.3)
    outputs = []
    sizes = []

    def hook(module, inputs, output):
        sizes.append(len(inputs[0]))
        outputs.append(output.detach().clone())

    handle = current.module.register_forward_hook(hook)
    try:
        fit(current, objective="hybrid")
    finally:
        handle.remove()
    assert max(sizes) <= 4
    assert len(outputs) == 12  # two frozen passes, then TRAIN/validation evaluation
    for first, replay in zip(outputs[:3], outputs[3:6]):
        torch.testing.assert_close(first, replay, atol=0, rtol=0)


def test_dropout_runs_are_deterministic_with_fixed_block_size():
    first, second = model(epochs=2, dropout=0.3), model(epochs=2, dropout=0.3)
    fit(first, objective="hybrid")
    fit(second, objective="hybrid")
    np.testing.assert_array_equal(first.predict_proba(data()[0]), second.predict_proba(data()[0]))


def test_inner_validation_early_stopping_restores_best_checkpoint():
    class CounterPanel:
        rows = 11

        def __init__(self):
            self.count = 0

        def loss_and_gradient(self, positions, config):
            self.count += 1
            return float(self.count), np.ones(11)

    X, y, known = data()
    current = model(epochs=8, early_stopping_patience=2)
    checkpoint = []

    def capture(record):
        if record["epoch"] == 1:
            checkpoint.extend(parameter.detach().clone() for parameter in current.module.parameters())

    result = fit_label_loss_model(
        current, X, None, None, X, None, None, objective="financial",
        train_panel=panel(), val_panel=CounterPanel(), progress_callback=capture,
    )
    assert result.stop_reason == "early_stopping" and result.best_epoch == 1
    assert current.learning_diagnostics_["optimizer_updates"] == 3
    assert not current.learning_diagnostics_["budget_insufficient"]
    for actual, expected in zip(current.module.parameters(), checkpoint):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_recent_inner_validation_progress_at_epoch_cap_marks_budget_insufficient():
    class ImprovingPanel:
        rows = 11

        def __init__(self):
            self.count = 0

        def loss_and_gradient(self, positions, config):
            self.count += 1
            return -float(self.count), np.ones(11)

    X, y, known = data()
    current = model(epochs=3)
    result = fit_label_loss_model(
        current, X, None, None, X, None, None, objective="financial",
        train_panel=panel(), val_panel=ImprovingPanel(),
    )
    assert result.stop_reason == "max_epochs" and result.best_epoch == 3
    assert current.learning_diagnostics_["budget_insufficient"]


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_hybrid_endpoints_match_pure_objectives(weight):
    first, second = model(), model()
    fit(first, objective="cross_entropy" if weight else "financial")
    fit(second, objective="hybrid", hybrid_ce_weight=weight)
    np.testing.assert_allclose(first.predict_proba(data()[0]), second.predict_proba(data()[0]), atol=1e-7)


@pytest.mark.parametrize("invalid", ["wrong", None])
def test_rejects_unknown_objective(invalid):
    with pytest.raises(ValueError, match="objective"):
        fit(model(), objective=invalid)


@pytest.mark.parametrize("weight", [-0.1, 1.1, np.nan])
def test_rejects_invalid_hybrid_weight(weight):
    with pytest.raises(ValueError, match="hybrid_ce_weight"):
        fit(model(), objective="hybrid", hybrid_ce_weight=weight)


def test_rejects_no_supervised_labels_invalid_mask_and_panel_alignment():
    X, y, known = data()
    with pytest.raises(ValueError, match="at least one known"):
        fit(model(), dataset=(X, y, np.zeros(11, dtype=bool)))
    with pytest.raises(ValueError, match="boolean vector"):
        fit(model(), available_train=np.ones(11))
    with pytest.raises(ValueError, match="aligned rows"):
        fit_label_loss_model(model(), X, None, None, X, None, None,
                             objective="financial", train_panel=panel(10), val_panel=panel())


def test_rejects_nonfinite_financial_gradient():
    class BadPanel:
        rows = 11

        def loss_and_gradient(self, positions, config):
            return 0.0, np.full(11, np.nan)

    X, y, known = data()
    with pytest.raises(FloatingPointError, match="financial loss/gradient"):
        fit_label_loss_model(model(), X, None, None, X, None, None,
                             objective="financial", train_panel=BadPanel(), val_panel=panel())
