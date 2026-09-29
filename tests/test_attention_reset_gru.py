from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from trading_system.models.neural.attention_reset_gru import (
    ExplicitGRU, ManualGRUClassifier, AttentionResetGRUClassifier,
)
from trading_system.models.neural.config import GRUConfig
from trading_system.models.neural.gru import GRUClassifier
from trading_system.models.specs import ModelBuildContext
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel
from trading_system.training.position_trainer import fit_position_model


def test_manual_matches_native_outputs_and_gradients():
    torch.manual_seed(4)
    native = nn.GRU(3, 4, batch_first=True).double()
    manual = ExplicitGRU(deepcopy(native))
    first = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    hx1 = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
    hx2 = hx1.detach().clone().requires_grad_(True)
    a, ah = native(first, hx1)
    b, bh = manual(second, hx2)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(ah, bh, rtol=1e-10, atol=1e-10)
    (a.square().sum() + ah.sum()).backward()
    (b.square().sum() + bh.sum()).backward()
    torch.testing.assert_close(first.grad, second.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(hx1.grad, hx2.grad, rtol=1e-10, atol=1e-10)
    assert sum(p.numel() for p in native.parameters()) == sum(p.numel() for p in manual.parameters())
    for name, p in native.named_parameters():
        torch.testing.assert_close(p.grad, dict(manual.named_parameters())[name].grad,
                                   rtol=1e-10, atol=1e-10)


def test_manual_classifier_preserves_pooling_head_initialization():
    context = ModelBuildContext(input_size=3, context_len=5, device="cpu", seed=7)
    config = GRUConfig(hidden_size=4, temporal_pooling="attention", device="cpu", seed=7)
    native = GRUClassifier(context, config)
    manual = ManualGRUClassifier(context, config)
    values = torch.randn(3, 5, 3)
    torch.testing.assert_close(native.module(values), manual.module(values))
    assert native.parameter_count() == manual.parameter_count()


@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_cell_causal_and_deterministic_backward(attention, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(3)
        cell = ExplicitGRU(nn.GRU(3, 4, batch_first=True), attention_reset=attention).to(device)
        values = torch.randn(2, 5, 3, device=device, requires_grad=True)
        output, _ = cell(values)
        changed = values.detach().clone()
        changed[:, 3:] += 100
        updated, _ = cell(changed)
        torch.testing.assert_close(output[:, :3], updated[:, :3])
        output.sum().backward()
        assert torch.isfinite(values.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cell.parameters())
        if attention:
            assert cell.query.weight.grad.abs().sum() > 0
            assert cell.key.weight.grad.abs().sum() > 0
            reset = cell.reset_gate(values[:, 0].detach(), torch.randn(2, 4, device=device))
            assert reset.shape == (2, 4)
            assert ((reset >= 0) & (reset <= 1)).all()
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("cls", [ManualGRUClassifier, AttentionResetGRUClassifier])
def test_cell_supports_exact_financial_trainer(cls):
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=8),
                          "adj_open_target": [100., 101., 99., 103., 102., 104., 105., 104.]})
    panel = ReturnPanel(frame, price_col="adj_open_target", execution_delay=0,
                        allow_same_session=True)
    values = np.random.default_rng(2).normal(size=(8, 5, 3)).astype(np.float32)
    context = ModelBuildContext(input_size=3, context_len=5, device="cpu")
    config = GRUConfig(hidden_size=4, epochs=1, batch_size=4, device="cpu",
                       temporal_pooling="attention")
    model = cls(context, config)
    result = fit_position_model(model, values, panel, values, panel,
                                FinancialLossConfig("sharpe"), "long_short")
    assert result.best_epoch == 1
    state = model.state_dict()
    restored = cls(context, config)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(model.predict_proba(values), restored.predict_proba(values))


def test_unsupported_stacked_cell_is_rejected():
    with pytest.raises(ValueError):
        ExplicitGRU(nn.GRU(3, 4, num_layers=2, batch_first=True))
