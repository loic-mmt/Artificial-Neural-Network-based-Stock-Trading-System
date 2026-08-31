"""Memory reductions must preserve model math, caller data and temporal order."""

from dataclasses import replace
import tracemalloc
import weakref

import numpy as np
import pandas as pd
import pytest

from trading_system.data import scaling
from trading_system.data.scaling import SequenceStandardizer, Standardizer
from trading_system.data.windows import (
    build_context_dataset,
    build_context_features,
    build_sequence_dataset,
    build_sequence_features,
)
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import _select_label_rows, run_validation_experiment
from trading_system.models.base import FitResult, TrainingHistory
from trading_system.models.manual_ann.manual_nn import (
    ManualANNClassifier, ManualANNConfig, forward_pass, softmax,
)
from trading_system.models.manual_ann.sequence_adapter import ManualANNSequenceAdapter
from trading_system.training.weights import compute_class_weights


def reference_softmax(logits):
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return (exponentials / exponentials.sum(axis=1, keepdims=True)).astype(np.float32)


def reference_fit(X, y, config, X_val=None, y_val=None, class_weights=None):
    """Pre-optimization SGD equations; intentionally allocation-heavy test oracle."""
    rng = np.random.default_rng(config.seed)
    W0 = (0.01 * rng.standard_normal((X.shape[1], config.hidden_size))).astype(np.float32)
    b0 = np.zeros((1, config.hidden_size), dtype=np.float32)
    W1 = (0.01 * rng.standard_normal((config.hidden_size, config.num_classes))).astype(np.float32)
    b1 = np.zeros((1, config.num_classes), dtype=np.float32)
    weights = compute_class_weights(y) if class_weights is None else class_weights
    train_losses, val_losses = [], []
    best_loss, best_epoch, stale = np.inf, 0, 0
    stop_reason = "max_epochs"

    def loss(values, labels):
        probabilities = reference_softmax(np.maximum(values @ W0 + b0, 0) @ W1 + b1)
        sample_weights = weights[labels]
        selected = probabilities[np.arange(len(labels)), labels]
        return float(-np.sum(sample_weights * np.log(selected + 1e-12)) / sample_weights.sum())

    for epoch in range(config.epochs):
        permutation = rng.permutation(len(X))
        for start in range(0, len(X), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            batch_X, batch_y = X[indices], y[indices]
            encoded = np.zeros((len(batch_y), config.num_classes), dtype=np.float32)
            encoded[np.arange(len(batch_y)), batch_y] = 1.0
            z1 = batch_X @ W0 + b0
            hidden = np.maximum(0.0, z1)
            mask = None
            if config.dropout_probability:
                keep = 1.0 - config.dropout_probability
                mask = (rng.random(hidden.shape) < keep).astype(np.float32) / keep
                hidden = hidden * mask
            probabilities = reference_softmax(hidden @ W1 + b1)
            sample_weights = weights[batch_y]
            gradient = ((probabilities - encoded) * sample_weights[:, None]) / float(sample_weights.sum())
            dW1 = hidden.T @ gradient
            db1 = gradient.sum(axis=0, keepdims=True)
            hidden_gradient = gradient @ W1.T
            if mask is not None:
                hidden_gradient *= mask
            hidden_gradient *= (z1 > 0.0).astype(np.float32)
            dW0 = batch_X.T @ hidden_gradient
            db0 = hidden_gradient.sum(axis=0, keepdims=True)
            W1 = W1 - config.learning_rate * dW1
            b1 = b1 - config.learning_rate * db1
            W0 = W0 - config.learning_rate * dW0
            b0 = b0 - config.learning_rate * db0
        train_losses.append(loss(X, y))
        selection_loss = train_losses[-1]
        if X_val is not None:
            selection_loss = loss(X_val, y_val)
            val_losses.append(selection_loss)
        if selection_loss < best_loss - config.early_stopping_min_delta:
            best_loss, best_epoch, stale = selection_loss, epoch + 1, 0
            best = dict(zip(("W0", "b0", "W1", "b1"), (a.copy() for a in (W0, b0, W1, b1))))
        else:
            stale += 1
        if stale >= config.early_stopping_patience:
            stop_reason = "early_stopping"
            break
    return best, FitResult(best_epoch, stop_reason, TrainingHistory(train_losses, val_losses))


@pytest.mark.parametrize("seed", [0, 3, 17, 29, 41])
@pytest.mark.parametrize("dropout", [0.0, 0.25, 0.7])
@pytest.mark.parametrize("with_validation", [False, True])
def test_optimized_ann_exactly_matches_reference_training(seed, dropout, with_validation):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(41, 7)).astype(np.float32)
    y = rng.integers(0, 3, len(X))
    X_val = rng.normal(size=(13, 7)).astype(np.float32) if with_validation else None
    y_val = rng.integers(0, 3, 13) if with_validation else None
    config = ManualANNConfig(
        hidden_size=5, epochs=7, batch_size=8, seed=seed,
        dropout_probability=dropout, early_stopping_patience=2,
    )
    # Both ordinary fitting and guaranteed early stopping must restore identical
    # best weights. This also catches accidental mutation of checkpoint buffers.
    for delta in (0.0, 1.0):
        settings = replace(config, early_stopping_min_delta=delta)
        state, expected = reference_fit(X, y, settings, X_val, y_val)
        model = ManualANNClassifier(settings)
        actual = model.fit(X, y, X_val=X_val, y_val=y_val)
        assert actual == expected
        for name, value in model.state_dict().items():
            np.testing.assert_array_equal(value, state[name])


def test_ann_retains_custom_weights_and_read_only_inputs():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(30, 8)).astype(np.float32)[:, ::2]
    y = rng.choice([0, 2], size=len(X))  # Missing Hold class.
    X.setflags(write=False)
    y.setflags(write=False)
    weights = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    settings = ManualANNConfig(hidden_size=9, epochs=5, batch_size=7, dropout_probability=0.3)
    expected_state, expected = reference_fit(X, y, settings, class_weights=weights)
    model = ManualANNClassifier(settings)
    assert model.fit(X, y, class_weights=weights) == expected
    for name, value in model.state_dict().items():
        np.testing.assert_array_equal(value, expected_state[name])
    np.testing.assert_array_equal(weights, [1, 3, 2])


def test_softmax_reuses_only_private_buffers():
    source = np.arange(600, dtype=np.float64).reshape(100, 6)[::3, ::2] - 250
    original = source.copy()
    source.setflags(write=False)
    expected = reference_softmax(source)
    actual = softmax(source)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    assert not np.shares_memory(source, actual)


@pytest.mark.parametrize("context_len, target_start", [(1, 0), (4, 0), (4, 10), (4, 30)])
@pytest.mark.parametrize("flat", [False, True])
def test_vectorized_windows_preserve_values_layout_and_ownership(context_len, target_start, flat):
    frame = pd.DataFrame({
        "a": np.arange(30, dtype=np.float32),
        "b": np.arange(100, 130, dtype=np.float32),
        "Label_id": np.arange(30) % 3,
    })
    original = frame.copy(deep=True)
    build = build_context_dataset if flat else build_sequence_dataset
    inference = build_context_features if flat else build_sequence_features
    X, y, indices = build(frame, ["a", "b"], context_len, target_start, True)
    features = inference(frame, ["a", "b"], context_len, target_start)
    expected_indices = np.arange(max(target_start, context_len - 1), len(frame))
    expected = np.empty((len(expected_indices), context_len, 2), dtype=np.float32)
    for row, index in enumerate(expected_indices):
        expected[row] = frame.iloc[index - context_len + 1 : index + 1][["a", "b"]]
    if flat:
        expected = expected.reshape(len(expected_indices), context_len * 2)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(y, frame.Label_id.iloc[expected_indices])
    np.testing.assert_array_equal(X, expected)
    np.testing.assert_array_equal(features, expected)
    assert X.flags.c_contiguous and X.flags.writeable
    if not flat and len(X):
        assert np.shares_memory(ManualANNSequenceAdapter._flatten_sequences(X), X)
    if len(X) > 1:
        second = X[1].copy()
        X[0].fill(-99)
        np.testing.assert_array_equal(X[1], second)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_blocked_sequence_statistics_match_full_precision_reference(layout, monkeypatch):
    rng = np.random.default_rng(21)
    X = rng.normal(size=(1003, 9, 5)).astype(np.float32)
    X[..., 1] *= 1e6
    X[..., 2] *= 1e-9
    X[..., 3] = np.float32(100 * np.log(1.02))
    if layout == "F":
        X = np.asfortranarray(X)
    if layout == "strided":
        X = X[::2, ::2]
    monkeypatch.setattr(scaling, "_STATISTICS_BUFFER_BYTES", 32768)
    expected_mean = X.mean(axis=(0, 1), dtype=np.float64, keepdims=True).astype(np.float32)
    std = X.std(axis=(0, 1), dtype=np.float64, keepdims=True)
    expected_scale = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    X.setflags(write=False)
    scaler = SequenceStandardizer().fit(X)
    np.testing.assert_array_equal(scaler.mean_, expected_mean)
    np.testing.assert_array_equal(scaler.scale_, expected_scale)
    np.testing.assert_array_equal(scaler.transform(X), (X - expected_mean) / expected_scale)


@pytest.mark.parametrize("sequence", [False, True])
def test_scaler_transform_leaves_input_and_state_unchanged(sequence):
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(71, 4, 3)).astype(np.float32)
    X = raw[::2] if sequence else raw.reshape(71, 12)[::2]
    scaler = (SequenceStandardizer() if sequence else Standardizer()).fit(X)
    before = X.copy()
    mean, scale = scaler.mean_.copy(), scaler.scale_.copy()
    expected = ((X - mean) / scale).astype(np.float32)
    X.setflags(write=False)
    actual = scaler.transform(X)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(X, before)
    np.testing.assert_array_equal(scaler.mean_, mean)
    np.testing.assert_array_equal(scaler.scale_, scale)
    assert not np.shares_memory(actual, X)


def peak_allocation(function):
    tracemalloc.start()
    try:
        value = function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, peak


def test_sequence_scaling_uses_bounded_workspace(monkeypatch):
    X = np.random.default_rng(8).normal(size=(4096, 8, 5)).astype(np.float32)
    monkeypatch.setattr(scaling, "_STATISTICS_BUFFER_BYTES", 32768)
    _, before = peak_allocation(lambda: X.std(axis=(0, 1), dtype=np.float64))
    _, after = peak_allocation(lambda: SequenceStandardizer().fit(X))
    assert after < before * 0.4


def test_prediction_uses_one_hidden_buffer():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2048, 16)).astype(np.float32)
    model = ManualANNClassifier(ManualANNConfig(hidden_size=128))
    model.W0_ = rng.normal(size=(16, 128)).astype(np.float32)
    model.b0_ = np.zeros((1, 128), dtype=np.float32)
    model.W1_ = rng.normal(size=(128, 3)).astype(np.float32)
    model.b1_ = np.zeros((1, 3), dtype=np.float32)
    original_state = model.state_dict()
    expected, before = peak_allocation(lambda: forward_pass(X, *model._state())[-1])
    actual, after = peak_allocation(lambda: model.predict_proba(X))
    np.testing.assert_array_equal(actual, expected)
    assert after < before * 0.65
    for name, value in model.state_dict().items():
        np.testing.assert_array_equal(value, original_state[name])


@pytest.mark.parametrize("mask", [[True] * 8, [True] * 5 + [False] * 3, [False, True] * 4, [False] * 8])
def test_label_row_selection_preserves_order(mask):
    X = np.arange(48).reshape(8, 2, 3)
    mask = np.asarray(mask)
    selected = _select_label_rows(X, mask)
    np.testing.assert_array_equal(selected, X[mask])
    if mask[0] and mask[:mask.sum()].all():
        assert np.shares_memory(selected, X)


def test_runner_releases_raw_windows_before_model_fit(monkeypatch):
    raw_references = []
    original_transform = SequenceStandardizer.transform

    def record_transform(self, X):
        raw_references.append(weakref.ref(X))
        return original_transform(self, X)

    monkeypatch.setattr(SequenceStandardizer, "transform", record_transform)

    class Model:
        model_name = "memory_probe"
        classes_ = np.arange(3)

        def fit(self, X_train, y_train, *, X_val=None, y_val=None):
            assert len(raw_references) == 2
            assert all(reference() is None for reference in raw_references)
            assert X_train.flags.c_contiguous and X_val.flags.c_contiguous
            return FitResult(1, "probe", TrainingHistory([1.0], [1.0]))

        def predict_proba(self, X):
            return np.tile(np.array([0.2, 0.6, 0.2], dtype=np.float32), (len(X), 1))

        def state_dict(self):
            return {}

    x = np.arange(180, dtype=float)
    close = 100 + x * 0.01 + 3 * np.sin(x / 4)
    frame = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=len(x)),
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "adj_close": close, "volume": 1e6 + 1e4 * np.cos(x / 6),
    })
    original_frame = frame.copy(deep=True)
    result = run_validation_experiment(
        frame, ExperimentConfig(context_len=3, label_mode="forward_return"), Model()
    )
    assert result.val_metrics and result.val_backtest
    pd.testing.assert_frame_equal(frame, original_frame)
