from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trading_system.models.factory import create_default_model_registry
from trading_system.models.neural.config import (
    CommonTrainingConfig,
    GRUConfig,
    LSTMConfig,
    RNNConfig,
    TransformerConfig,
)
from trading_system.models.neural.trainer import (
    build_tensor_loader,
    resolve_device,
    seed_torch_run,
)
from trading_system.models.neural.transformer import (
    build_causal_attention_mask,
    sinusoidal_positional_encoding,
)
from trading_system.models.specs import ModelBuildContext


def test_neural_config_validation_and_recurrent_dropout_policy():
    assert RNNConfig(num_layers=1, dropout=0.5).dropout == 0.0
    assert LSTMConfig(num_layers=2, dropout=0.5).dropout == 0.5
    assert GRUConfig(num_layers=1, dropout=0.5).dropout == 0.0
    with pytest.raises(ValueError, match="learning_rate"):
        CommonTrainingConfig(learning_rate=0)
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfig(d_model=10, n_heads=3)
    with pytest.raises(ValueError, match="pooling"):
        TransformerConfig(pooling="bad")


def test_torch_helpers_device_seed_loader_and_masks():
    assert str(resolve_device("cpu", torch)) == "cpu"
    seed_torch_run(17, True, torch)
    first = torch.rand(3)
    seed_torch_run(17, True, torch)
    torch.testing.assert_close(torch.rand(3), first, rtol=0, atol=0)
    X = np.arange(72, dtype=np.float32).reshape(6, 3, 4)
    y = np.arange(6) % 3
    loader_a = build_tensor_loader(
        X, y, batch_size=2, shuffle=True, seed=4, num_workers=0, torch_module=torch
    )
    loader_b = build_tensor_loader(
        X, y, batch_size=2, shuffle=True, seed=4, num_workers=0, torch_module=torch
    )
    assert torch.equal(
        torch.cat([labels for _, labels in loader_a]),
        torch.cat([labels for _, labels in loader_b]),
    )
    encoding = sinusoidal_positional_encoding(5, 7, torch_module=torch)
    assert encoding.shape == (1, 5, 7)
    torch.testing.assert_close(encoding[0, 0, 0::2], torch.zeros(4))
    mask = build_causal_attention_mask(4, torch_module=torch)
    assert torch.equal(mask, torch.triu(torch.ones(4, 4, dtype=torch.bool), 1))


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("rnn", {"hidden_size": 6}),
        ("lstm", {"hidden_size": 6}),
        ("gru", {"hidden_size": 6}),
        (
            "transformer",
            {
                "d_model": 8,
                "n_heads": 2,
                "num_layers": 1,
                "dim_feedforward": 12,
                "dropout": 0.0,
            },
        ),
    ],
)
def test_each_neural_model_fits_predicts_and_round_trips(name, parameters):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(24, 5, 4)).astype(np.float32)
    y = np.tile(np.arange(3), 8)
    context = ModelBuildContext(4, 5, seed=13, device="cpu")
    settings = {
        **parameters,
        "epochs": 2,
        "batch_size": 6,
        "early_stopping_patience": 2,
    }
    registry = create_default_model_registry()
    model = registry.build(name, context, settings)
    result = model.fit(X[:18], y[:18], X_val=X[18:], y_val=y[18:])
    probabilities = model.predict_proba(X[18:])
    assert result.best_epoch in (1, 2)
    assert probabilities.shape == (6, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert model.parameter_count() > 0

    restored = registry.build(name, context, settings)
    restored.load_state_dict(model.state_dict())
    np.testing.assert_array_equal(restored.predict_proba(X[18:]), probabilities)


def test_rnn_training_is_reproducible_and_context_owned():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(18, 4, 3)).astype(np.float32)
    y = np.tile(np.arange(3), 6)
    context = ModelBuildContext(3, 4, seed=21, device="cpu")
    parameters = {
        "hidden_size": 5,
        "epochs": 2,
        "batch_size": 6,
        "dropout": 0.0,
    }
    registry = create_default_model_registry()
    first = registry.build("rnn", context, parameters)
    second = registry.build("rnn", context, parameters)
    first.fit(X[:12], y[:12], X_val=X[12:], y_val=y[12:])
    second.fit(X[:12], y[:12], X_val=X[12:], y_val=y[12:])
    np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))
    with pytest.raises(ValueError, match="controlled"):
        registry.build("rnn", context, {**parameters, "seed": 22})
