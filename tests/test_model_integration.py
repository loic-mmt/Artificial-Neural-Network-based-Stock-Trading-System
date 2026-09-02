import numpy as np
import pandas as pd
import pytest

from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import run_experiment
from trading_system.experiments.walkforward import walk_forward_classifier
from trading_system.features.technical import (
    TECHNICAL_FEATURE_COLUMNS,
    compute_technical_features,
)
from trading_system.models.factory import create_default_model_registry
from trading_system.models.specs import ModelSelection


MODEL_PARAMETERS = {
    "manual_ann": {
        "hidden_size": 4,
        "epochs": 1,
        "batch_size": 32,
        "early_stopping_patience": 1,
    },
    "rnn": {
        "hidden_size": 4,
        "epochs": 1,
        "batch_size": 32,
        "early_stopping_patience": 1,
    },
    "lstm": {
        "hidden_size": 4,
        "epochs": 1,
        "batch_size": 32,
        "early_stopping_patience": 1,
    },
    "gru": {
        "hidden_size": 4,
        "epochs": 1,
        "batch_size": 32,
        "early_stopping_patience": 1,
    },
    "transformer": {
        "d_model": 4,
        "n_heads": 2,
        "num_layers": 1,
        "dim_feedforward": 8,
        "dropout": 0.0,
        "epochs": 1,
        "batch_size": 32,
        "early_stopping_patience": 1,
    },
}


def market_frame(rows=100):
    x = np.arange(rows, dtype=np.float64)
    close = 100 + 0.05 * x + 2.5 * np.sin(x / 4)
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=rows),
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000 + 1000 * np.cos(x / 7),
        }
    )


@pytest.mark.parametrize("model_name", MODEL_PARAMETERS)
def test_all_registered_models_run_through_static_pipeline(model_name):
    if model_name != "manual_ann":
        pytest.importorskip("torch")
    result = run_experiment(
        market_frame(120),
        ExperimentConfig(
            label_mode="forward_return",
            context_len=3,
            train_ratio=0.6,
            val_ratio=0.2,
            decision_mode="argmax",
            model=ModelSelection(model_name, MODEL_PARAMETERS[model_name]),
            seed=13,
            device="cpu",
        ),
    )
    assert result.bundle.model_selection.name == model_name
    assert result.test_probabilities.shape == (len(result.aligned_test_frame), 3)
    np.testing.assert_allclose(result.test_probabilities.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("model_name", MODEL_PARAMETERS)
def test_all_registered_models_run_through_one_walkforward_chunk(model_name):
    if model_name != "manual_ann":
        pytest.importorskip("torch")
    featured = compute_technical_features(market_frame(), group_col=None)
    result = walk_forward_classifier(
        featured,
        TECHNICAL_FEATURE_COLUMNS,
        train_ratio=0.6,
        val_ratio=0.2,
        walkforward_step=100,
        label_mode="forward_return",
        context_len=3,
        decision_mode="argmax",
        model_selection=ModelSelection(model_name, MODEL_PARAMETERS[model_name]),
        registry=create_default_model_registry(),
        seed=13,
        device="cpu",
    )
    assert len(result["retrain_logs"]) == 1
    log = result["retrain_logs"][0]
    assert log["model_name"] == model_name
    assert log["run_seed"] == 13
    assert log["seed_applied_by_registry"] is True
    assert result["test_metrics"]
