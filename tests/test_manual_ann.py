import copy

import numpy as np
import pytest

from trading_system.experiments.runner import align_probability_columns
from trading_system.models.adapters import SklearnClassifierAdapter
from trading_system.models.manual_ann.manual_nn import (
    ManualANNClassifier,
    ManualANNConfig,
)
from trading_system.models.manual_ann.sequence_adapter import ManualANNSequenceAdapter


def test_manual_ann_sequence_adapter_initializes_unfitted_state():
    config = ManualANNConfig(hidden_size=8, epochs=2, batch_size=2, seed=9)

    adapter = ManualANNSequenceAdapter(config)

    assert adapter.config is config
    assert isinstance(adapter.estimator, ManualANNClassifier)
    assert adapter.estimator.config is config
    np.testing.assert_array_equal(adapter.classes_, [0, 1, 2])
    assert adapter.context_len_ is None
    assert adapter.feature_count_ is None
    assert adapter.fit_result_ is None


def test_manual_ann_sequence_adapter_flattens_in_time_feature_order():
    sequences = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    flattened = ManualANNSequenceAdapter._flatten_sequences(sequences)

    assert flattened.shape == (2, 12)
    assert flattened.dtype == np.float32
    assert flattened.flags.c_contiguous
    np.testing.assert_array_equal(flattened[0], np.arange(12))
    np.testing.assert_array_equal(flattened[1], np.arange(12, 24))


@pytest.mark.parametrize(
    "invalid",
    [
        np.ones((2, 3), dtype=np.float32),
        np.empty((0, 3, 2), dtype=np.float32),
        np.asarray([[[1.0], [np.nan]]], dtype=np.float32),
    ],
)
def test_manual_ann_sequence_adapter_rejects_invalid_sequences(invalid):
    with pytest.raises(ValueError):
        ManualANNSequenceAdapter._flatten_sequences(invalid)


def test_manual_ann_sequence_adapter_requires_three_classes():
    with pytest.raises(ValueError, match="three trading classes"):
        ManualANNSequenceAdapter(ManualANNConfig(num_classes=4))


def test_manual_ann_sequence_adapter_fits_3d_train_and_validation_data():
    X = np.arange(18, dtype=np.float32).reshape(9, 2, 1)
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    adapter = ManualANNSequenceAdapter(
        ManualANNConfig(
            hidden_size=5,
            epochs=4,
            batch_size=3,
            early_stopping_patience=2,
            seed=4,
        )
    )

    result = adapter.fit(X, y, X_val=X[:6], y_val=y[:6])

    assert result is adapter.fit_result_
    assert result is adapter.estimator.fit_result_
    assert adapter.context_len_ == 2
    assert adapter.feature_count_ == 1
    assert adapter.estimator.W0_.shape == (2, 5)
    np.testing.assert_array_equal(adapter.classes_, [0, 1, 2])


def test_manual_ann_sequence_adapter_fit_accepts_no_validation_pair():
    X = np.arange(18, dtype=np.float32).reshape(9, 2, 1)
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    adapter = ManualANNSequenceAdapter(
        ManualANNConfig(epochs=2, batch_size=3, early_stopping_patience=2)
    )

    result = adapter.fit(X, y)

    assert result is adapter.fit_result_
    assert adapter.context_len_ == 2
    assert adapter.feature_count_ == 1


def test_manual_ann_sequence_adapter_fit_validates_validation_pair_and_shape():
    X = np.arange(18, dtype=np.float32).reshape(9, 2, 1)
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    adapter = ManualANNSequenceAdapter(ManualANNConfig(epochs=2))

    with pytest.raises(ValueError, match="supplied together"):
        adapter.fit(X, y, X_val=X)
    with pytest.raises(ValueError, match="dimensions must match"):
        adapter.fit(
            X,
            y,
            X_val=np.ones((3, 3, 1), dtype=np.float32),
            y_val=np.asarray([0, 1, 2]),
        )

    assert adapter.context_len_ is None
    assert adapter.feature_count_ is None
    assert adapter.fit_result_ is None


def test_manual_ann_sequence_adapter_predicts_from_3d_sequences():
    X = np.arange(18, dtype=np.float32).reshape(9, 2, 1)
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    adapter = ManualANNSequenceAdapter(
        ManualANNConfig(epochs=3, batch_size=3, early_stopping_patience=2)
    )
    adapter.fit(X, y)

    probabilities = adapter.predict_proba(X)
    expected = adapter.estimator.predict_proba(adapter._flatten_sequences(X))

    assert probabilities.shape == (9, 3)
    np.testing.assert_allclose(probabilities, expected)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_manual_ann_sequence_adapter_predict_requires_fit_and_matching_shape():
    X = np.arange(18, dtype=np.float32).reshape(9, 2, 1)
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    adapter = ManualANNSequenceAdapter(
        ManualANNConfig(epochs=2, batch_size=3, early_stopping_patience=2)
    )

    with pytest.raises(RuntimeError, match="not fitted"):
        adapter.predict_proba(X)

    adapter.fit(X, y)
    with pytest.raises(ValueError, match="dimensions must match"):
        adapter.predict_proba(np.ones((3, 3, 1), dtype=np.float32))


def test_manual_ann_sequence_adapter_state_round_trip_preserves_predictions():
    X = np.arange(36, dtype=np.float32).reshape(9, 2, 2)
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    adapter = ManualANNSequenceAdapter(
        ManualANNConfig(
            hidden_size=5,
            epochs=3,
            batch_size=3,
            early_stopping_patience=2,
            seed=8,
        )
    )
    adapter.fit(X, y)
    expected = adapter.predict_proba(X)

    state = adapter.state_dict()
    restored = ManualANNSequenceAdapter()
    restored.load_state_dict(state)

    assert restored.config == adapter.config
    assert restored.context_len_ == 2
    assert restored.feature_count_ == 2
    assert restored.fit_result_ is None
    np.testing.assert_array_equal(restored.classes_, [0, 1, 2])
    np.testing.assert_allclose(restored.predict_proba(X), expected)

    # Export and load both copy arrays. Mutating caller-owned state afterward must
    # not mutate either trained adapter.
    state["classes"][0] = 99
    state["weights"]["W0"].fill(99.0)
    np.testing.assert_allclose(adapter.predict_proba(X), expected)
    np.testing.assert_allclose(restored.predict_proba(X), expected)


def test_manual_ann_sequence_adapter_state_rejects_invalid_data_atomically():
    X = np.arange(18, dtype=np.float32).reshape(9, 2, 1)
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    fitted = ManualANNSequenceAdapter(
        ManualANNConfig(
            hidden_size=4,
            epochs=2,
            batch_size=3,
            early_stopping_patience=2,
        )
    )
    fitted.fit(X, y)
    valid_state = fitted.state_dict()
    target = ManualANNSequenceAdapter()

    invalid_state = copy.deepcopy(valid_state)
    invalid_state["weights"]["W0"] = np.zeros((99, 99), dtype=np.float32)
    with pytest.raises(ValueError, match="weight W0 has shape"):
        target.load_state_dict(invalid_state)

    assert target.context_len_ is None
    assert target.feature_count_ is None
    assert target.fit_result_ is None

    missing_state = copy.deepcopy(valid_state)
    del missing_state["classes"]
    with pytest.raises(ValueError, match="missing"):
        target.load_state_dict(missing_state)


def test_manual_ann_sequence_adapter_cannot_export_before_fit():
    with pytest.raises(RuntimeError, match="unfitted"):
        ManualANNSequenceAdapter().state_dict()


def test_manual_ann_is_deterministic_and_probabilistic():
    X = np.asarray(
        [[-2.0], [-1.0], [-0.5], [0.0], [0.5], [1.0], [2.0], [3.0], [-3.0]],
        dtype=np.float32,
    )
    y = np.asarray([0, 0, 0, 1, 1, 2, 2, 2, 0], dtype=np.int64)
    config = ManualANNConfig(
        hidden_size=6,
        epochs=20,
        batch_size=3,
        early_stopping_patience=5,
        seed=7,
    )
    first = ManualANNClassifier(config)
    second = ManualANNClassifier(config)
    first.fit(X, y, X_val=X, y_val=y)
    second.fit(X, y, X_val=X, y_val=y)
    first_probabilities = first.predict_proba(X)
    second_probabilities = second.predict_proba(X)
    np.testing.assert_allclose(first_probabilities, second_probabilities)
    np.testing.assert_allclose(first_probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_sklearn_style_adapter_aligns_missing_class_columns():
    class FakeEstimator:
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, X):
            return np.tile(np.asarray([[0.25, 0.75]]), (len(X), 1))

    adapter = SklearnClassifierAdapter(FakeEstimator())
    X = np.asarray([[0.0], [1.0]], dtype=np.float32)
    adapter.fit(X, np.asarray([0, 2]))
    aligned = align_probability_columns(adapter, adapter.predict_proba(X))
    assert aligned.shape == (2, 3)
    np.testing.assert_allclose(aligned[:, 1], 0.0)
    np.testing.assert_allclose(aligned.sum(axis=1), 1.0)
