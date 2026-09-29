import numpy as np
import pytest
import torch

from trading_system.models.denoising import DenoisingConfig, continuous_indices, fit_denoiser


COLUMNS = ("lag1_rsi", "night_pct", "lag1_week", "lag1_sector_banks",
           "lag1_trend_regime", "lag1_sector_ret_1")


def test_discrete_columns_bypass_denoiser_but_sector_returns_do_not():
    assert continuous_indices(COLUMNS) == (0, 1, 5)


@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_denoiser_frozen_causal_and_preserves_passthrough(attention, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    rng = np.random.default_rng(2)
    train = rng.normal(size=(18, 4, 6)).astype(np.float32)
    inner = rng.normal(size=(8, 4, 6)).astype(np.float32)
    outer = rng.normal(size=(7, 4, 6)).astype(np.float32)
    original = outer.copy()
    config = DenoisingConfig(epochs=2, hidden_size=8, latent_size=2, batch_size=8)
    denoiser = fit_denoiser(train, inner, COLUMNS, config=config,
                            attention=attention, seed=7, device=device)
    before = {k: v.clone() for k, v in denoiser.model.state_dict().items()}
    output = denoiser.transform(outer)
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output[..., [2, 3, 4]], outer[..., [2, 3, 4]])
    np.testing.assert_array_equal(outer, original)
    assert not any(p.requires_grad for p in denoiser.model.parameters())
    assert denoiser.report["fit_rows"] == len(train)
    assert denoiser.report["selection_rows"] == len(inner)
    changed = outer.copy()
    changed[:, -1] += 100
    updated = denoiser.transform(changed)
    np.testing.assert_allclose(output[:, :-1], updated[:, :-1])
    for key, value in denoiser.model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert denoiser.checkpoint()["indices"] == (0, 1, 5)


def test_fit_is_reproducible_and_does_not_mutate_training_windows():
    values = np.random.default_rng(1).normal(size=(12, 3, 6)).astype(np.float32)
    original = values.copy()
    config = DenoisingConfig(epochs=1, hidden_size=8, batch_size=4)
    first = fit_denoiser(values, values[:4], COLUMNS, config=config, device="cpu")
    second = fit_denoiser(values, values[:4], COLUMNS, config=config, device="cpu")
    np.testing.assert_array_equal(first.transform(values), second.transform(values))
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("kwargs", [{"noise_std": float("nan")}, {"noise_std": -1},
                                   {"epochs": 0}, {"learning_rate": 0}])
def test_denoiser_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        DenoisingConfig(**kwargs)
