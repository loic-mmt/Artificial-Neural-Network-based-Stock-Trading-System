"""Cross-asset GRU modules preserve date grouping and masked causality."""

import numpy as np
import pandas as pd
import pytest
import torch

from trading_system.experiments.stock_mixer_ablation import (
    fit_grouped_position_model, group_sequences,
)
from trading_system.models.neural.config import GRUConfig
from trading_system.models.stock_mixer_gru import MIXERS, StockMixerGRU
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel


def _model(mixer):
    return StockMixerGRU(input_size=2, context_len=3, assets=3,
                         config=GRUConfig(hidden_size=4, temporal_pooling="attention",
                                          device="cpu", epochs=2, batch_size=6),
                         mixer=mixer, market_states=2)


@pytest.mark.parametrize("device", ("cpu", "mps"))
@pytest.mark.parametrize("mixer", MIXERS)
def test_dense_masked_backward_is_deterministic_and_finite(mixer, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(3)
        model = _model(mixer).to(device)
        values = torch.randn(3, 3, 3, 2, device=device, requires_grad=True)
        mask = torch.tensor([[True, True, True], [True, False, False],
                             [False, False, False]], device=device)
        output = model(values, mask)
        output.backward(torch.ones_like(output))
        assert torch.isfinite(output).all()
        assert torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad[~mask]) == 0
        assert all(p.grad is None or torch.isfinite(p.grad).all()
                   for p in model.parameters())
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("mixer", MIXERS)
def test_absent_asset_cannot_influence_present_assets(mixer):
    torch.manual_seed(3)
    model = _model(mixer).eval()
    features = torch.randn(2, 3, 3, 2)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    with torch.no_grad():
        original = model(features, mask)
        changed = features.clone()
        changed[~mask] = 1_000_000
        updated = model(changed, mask)
    torch.testing.assert_close(original, updated)
    assert torch.count_nonzero(updated[~mask]) == 0


@pytest.mark.parametrize("mixer", ("none", "attention_self"))
def test_independent_controls_do_not_read_other_assets(mixer):
    torch.manual_seed(7)
    model = _model(mixer).eval()
    features = torch.randn(1, 3, 3, 2)
    mask = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        original = model(features, mask)[0, 0]
        changed = features.clone()
        changed[0, 1] += 100
        updated = model(changed, mask)[0, 0]
    torch.testing.assert_close(original, updated)


@pytest.mark.parametrize("mixer", ("mean", "stock_mixer", "attention"))
def test_cross_asset_modes_backpropagate_to_other_asset(mixer):
    torch.manual_seed(11)
    model = _model(mixer).eval()
    values = torch.randn(1, 3, 3, 2, requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)
    model(values, mask)[0, 0, 0].backward()
    assert values.grad[0, 1].abs().sum() > 0


def test_group_sequences_preserves_original_position_order_and_masks():
    dates = pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-02"])
    frame = pd.DataFrame({"date": dates, "ticker": ["B", "A", "A"]})
    values = np.arange(18, dtype=np.float32).reshape(3, 3, 2)
    grouped = group_sequences(values, frame, ("A", "B"))
    assert grouped.values.shape == (2, 2, 3, 2)
    assert grouped.rows.tolist() == [[1, -1], [2, 0]]
    assert grouped.mask.tolist() == [[True, False], [True, True]]


def test_standalone_control_preserves_shared_gru_logits():
    torch.manual_seed(13)
    model = _model("none").eval()
    values = torch.randn(2, 3, 3, 2)
    mask = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        actual = model(values, mask)
        expected = model.head(model.encoder.encode(values.reshape(6, 3, 2)))
    torch.testing.assert_close(actual.reshape(6, 3), expected)


def test_cross_asset_context_never_reads_another_date():
    torch.manual_seed(17)
    model = _model("stock_mixer").eval()
    values = torch.randn(2, 3, 3, 2)
    mask = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        original = model(values, mask)[0]
        changed = values.clone()
        changed[1] += 1_000
        updated = model(changed, mask)[0]
    torch.testing.assert_close(original, updated)


@pytest.mark.parametrize("device", ("cpu", "mps"))
def test_short_financial_training_uses_date_grouped_windows(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")
    dates = pd.date_range("2020-01-01", periods=5)
    frame = pd.DataFrame({
        "date": list(dates) * 3, "ticker": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "adj_open_target": np.r_[np.linspace(100, 104, 5),
                                  np.linspace(90, 93, 5), np.linspace(110, 108, 5)],
    })
    values = np.random.default_rng(5).normal(size=(len(frame), 3, 2)).astype(np.float32)
    grouped = group_sequences(values, frame, ("A", "B", "C"))
    panel = ReturnPanel(frame, price_col="adj_open_target", group_col="ticker",
                        execution_delay=0, allow_same_session=True)
    config = GRUConfig(hidden_size=4, epochs=2, batch_size=6, device=device,
                       temporal_pooling="attention")
    torch.manual_seed(1)
    model = StockMixerGRU(input_size=2, context_len=3, assets=3, config=config,
                          mixer="stock_mixer", market_states=2).to(device)
    fitted = fit_grouped_position_model(
        model, grouped, panel, grouped, panel,
        FinancialLossConfig("sharpe", cost_bps=5), config,
        "long_short", torch.device(device),
    )
    assert 1 <= fitted["best_epoch"] <= 2
    assert np.isfinite(fitted["inner_loss"]).all()
