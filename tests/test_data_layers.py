import numpy as np
import pandas as pd
import pytest

from trading_system.data.scaling import SequenceStandardizer, Standardizer
from trading_system.data.splits import chronological_train_val_test_split
from trading_system.data.windows import (
    build_context_dataset_with_history,
    build_sequence_dataset,
    build_sequence_dataset_with_history,
    build_sequence_features,
)


def test_sequence_dataset_preserves_time_and_feature_axes():
    frame = pd.DataFrame(
        {
            "feature_a": np.arange(5, dtype=np.float32),
            "feature_b": np.arange(10, 15, dtype=np.float32),
            "Label_id": [0, 1, 2, 1, 0],
        }
    )

    X, y, indices = build_sequence_dataset(
        frame,
        ["feature_a", "feature_b"],
        context_len=3,
        return_indices=True,
    )

    assert X.shape == (3, 3, 2)
    assert X.dtype == np.float32
    np.testing.assert_array_equal(X[0], frame.iloc[:3][["feature_a", "feature_b"]])
    np.testing.assert_array_equal(indices, [2, 3, 4])
    np.testing.assert_array_equal(y, frame.iloc[indices]["Label_id"])


def test_sequence_dataset_target_start_and_empty_shapes():
    frame = pd.DataFrame(
        {
            "feature": np.arange(5, dtype=np.float32),
            "Label_id": [0, 1, 2, 1, 0],
        }
    )

    X, y, indices = build_sequence_dataset(
        frame,
        ["feature"],
        context_len=3,
        target_start=4,
        return_indices=True,
    )
    assert X.shape == (1, 3, 1)
    np.testing.assert_array_equal(X[0, :, 0], [2, 3, 4])
    np.testing.assert_array_equal(y, [0])
    np.testing.assert_array_equal(indices, [4])

    empty_X, empty_y, empty_indices = build_sequence_dataset(
        frame.iloc[:2],
        ["feature"],
        context_len=3,
        return_indices=True,
    )
    assert empty_X.shape == (0, 3, 1)
    assert empty_y.shape == (0,)
    assert empty_indices.shape == (0,)


def test_sequence_dataset_rejects_non_integer_labels():
    frame = pd.DataFrame({"feature": [1.0, 2.0], "Label_id": [0.0, 1.5]})

    with pytest.raises(ValueError, match="integer class identifiers"):
        build_sequence_dataset(frame, ["feature"], context_len=2)


def test_sequence_features_matches_labeled_builder_without_requiring_labels():
    features = pd.DataFrame(
        {
            "feature_a": np.arange(5, dtype=np.float32),
            "feature_b": np.arange(10, 15, dtype=np.float32),
        }
    )
    labeled = features.assign(Label_id=[0, 1, 2, 1, 0])

    inference_X, inference_indices = build_sequence_features(
        features,
        ["feature_a", "feature_b"],
        context_len=3,
        return_indices=True,
    )
    labeled_X, _, labeled_indices = build_sequence_dataset(
        labeled,
        ["feature_a", "feature_b"],
        context_len=3,
        return_indices=True,
    )

    assert inference_X.shape == (3, 3, 2)
    np.testing.assert_array_equal(inference_X, labeled_X)
    np.testing.assert_array_equal(inference_indices, labeled_indices)


def test_sequence_features_respects_target_start_and_empty_contract():
    frame = pd.DataFrame({"feature": np.arange(5, dtype=np.float32)})

    X, indices = build_sequence_features(
        frame,
        ["feature"],
        context_len=3,
        target_start=4,
        return_indices=True,
    )
    assert X.shape == (1, 3, 1)
    np.testing.assert_array_equal(X[0, :, 0], [2, 3, 4])
    np.testing.assert_array_equal(indices, [4])

    empty_X, empty_indices = build_sequence_features(
        frame.iloc[:2],
        ["feature"],
        context_len=3,
        return_indices=True,
    )
    assert empty_X.shape == (0, 3, 1)
    assert empty_indices.shape == (0,)


def test_sequence_dataset_with_unlabeled_history_aligns_target_rows():
    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "feature": [0.0, 1.0, 2.0],
        }
    )
    target = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-04", periods=2),
            "feature": [3.0, 4.0],
            "Label_id": [1, 2],
        }
    )

    X, y, aligned = build_sequence_dataset_with_history(
        target,
        ["feature"],
        context_len=3,
        history_frame=history,
        group_col=None,
        return_aligned_rows=True,
    )

    np.testing.assert_array_equal(X[:, :, 0], [[1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(y, [1, 2])
    assert aligned["feature"].tolist() == [3.0, 4.0]


def test_grouped_split_preserves_each_ticker_chronology():
    dates = pd.date_range("2024-01-01", periods=10)
    frame = pd.concat(
        [
            pd.DataFrame({"ticker": ticker, "date": dates, "value": np.arange(10)})
            for ticker in ("AAA", "BBB")
        ],
        ignore_index=True,
    )
    train, val, test = chronological_train_val_test_split(
        frame,
        train_ratio=0.6,
        val_ratio=0.2,
        group_col="ticker",
    )
    assert train.groupby("ticker").size().to_dict() == {"AAA": 6, "BBB": 6}
    assert val.groupby("ticker").size().to_dict() == {"AAA": 2, "BBB": 2}
    assert test.groupby("ticker").size().to_dict() == {"AAA": 2, "BBB": 2}
    assert all(
        group["date"].is_monotonic_increasing for _, group in train.groupby("ticker")
    )


def test_grouped_context_windows_never_cross_tickers():
    dates = pd.date_range("2024-01-01", periods=5)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "feature": values,
                    "Label_id": 1,
                }
            )
            for ticker, values in (("AAA", np.arange(5)), ("BBB", np.arange(100, 105)))
        ],
        ignore_index=True,
    )
    windows, labels, aligned = build_context_dataset_with_history(
        frame,
        ["feature"],
        3,
        group_col="ticker",
        return_aligned_rows=True,
    )
    assert len(windows) == len(labels) == len(aligned) == 6
    assert all(np.ptp(window) == 2 for window in windows)
    assert not any(window.min() < 50 < window.max() for window in windows)


def test_standardizer_uses_fitted_training_statistics_only():
    train = np.asarray([[0.0], [2.0]], dtype=np.float32)
    test = np.asarray([[100.0]], dtype=np.float32)
    scaler = Standardizer()
    scaler.fit(train)
    transformed = scaler.transform(test)
    assert scaler.mean_[0, 0] == 1.0
    assert scaler.scale_[0, 0] == 1.0
    assert transformed[0, 0] == 99.0


def test_sequence_standardizer_fits_one_statistic_per_feature():
    sequences = np.asarray(
        [
            [[1.0, 10.0], [3.0, 14.0]],
            [[5.0, 18.0], [7.0, 22.0]],
        ],
        dtype=np.float32,
    )

    scaler = SequenceStandardizer().fit(sequences)

    assert scaler.mean_.shape == (1, 1, 2)
    assert scaler.scale_.shape == (1, 1, 2)
    assert scaler.mean_.dtype == np.float32
    assert scaler.scale_.dtype == np.float32
    np.testing.assert_allclose(scaler.mean_[0, 0], [4.0, 16.0])
    np.testing.assert_allclose(scaler.scale_[0, 0], [np.sqrt(5), np.sqrt(20)])


def test_sequence_standardizer_uses_unit_scale_for_constant_features():
    sequences = np.asarray(
        [[[1.0, 5.0], [3.0, 5.0]], [[5.0, 5.0], [7.0, 5.0]]],
        dtype=np.float32,
    )

    scaler = SequenceStandardizer().fit(sequences)

    assert scaler.scale_[0, 0, 1] == 1.0


def test_sequence_standardizer_transform_broadcasts_fitted_feature_stats():
    train = np.asarray(
        [[[1.0, 10.0], [3.0, 14.0]], [[5.0, 18.0], [7.0, 22.0]]],
        dtype=np.float32,
    )
    inference = np.asarray(
        [[[9.0, 26.0], [11.0, 30.0], [13.0, 34.0]]],
        dtype=np.float64,
    )
    original_inference = inference.copy()
    scaler = SequenceStandardizer().fit(train)
    original_mean = scaler.mean_.copy()
    original_scale = scaler.scale_.copy()

    transformed = scaler.transform(inference)
    expected = (inference.astype(np.float32) - original_mean) / original_scale

    assert transformed.shape == (1, 3, 2)
    assert transformed.dtype == np.float32
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_array_equal(inference, original_inference)
    np.testing.assert_array_equal(scaler.mean_, original_mean)
    np.testing.assert_array_equal(scaler.scale_, original_scale)


def test_sequence_standardizer_transform_normalizes_training_features():
    train = np.asarray(
        [[[1.0, 10.0], [3.0, 14.0]], [[5.0, 18.0], [7.0, 22.0]]],
        dtype=np.float32,
    )
    scaler = SequenceStandardizer().fit(train)

    transformed = scaler.transform(train)

    np.testing.assert_allclose(transformed.mean(axis=(0, 1)), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(transformed.std(axis=(0, 1)), [1.0, 1.0], atol=1e-6)


def test_sequence_standardizer_does_not_amplify_float32_constant_rounding():
    constant = np.float32(100.0 * np.log(1.02))
    train = np.full((79, 5, 1), constant, dtype=np.float32)

    scaler = SequenceStandardizer().fit(train)
    transformed = scaler.transform(train)

    np.testing.assert_array_equal(scaler.scale_, np.ones((1, 1, 1), dtype=np.float32))
    np.testing.assert_array_equal(transformed, np.zeros_like(train))


def test_sequence_standardizer_fit_transform_matches_composed_operations():
    sequences = np.asarray(
        [[[1.0, 10.0], [3.0, 14.0]], [[5.0, 18.0], [7.0, 22.0]]],
        dtype=np.float64,
    )
    composed_scaler = SequenceStandardizer()
    direct_scaler = SequenceStandardizer()

    expected = composed_scaler.fit(sequences).transform(sequences)
    transformed = direct_scaler.fit_transform(sequences)

    assert transformed.dtype == np.float32
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_array_equal(direct_scaler.mean_, composed_scaler.mean_)
    np.testing.assert_array_equal(direct_scaler.scale_, composed_scaler.scale_)


def test_sequence_standardizer_transform_validates_fit_and_features():
    with pytest.raises(RuntimeError, match="fitted before transform"):
        SequenceStandardizer().transform(np.ones((1, 2, 1), dtype=np.float32))

    scaler = SequenceStandardizer().fit(np.asarray([[[1.0], [3.0]]], dtype=np.float32))
    with pytest.raises(ValueError, match="feature count"):
        scaler.transform(np.ones((1, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="finite sequence values"):
        scaler.transform(np.asarray([[[np.nan]]], dtype=np.float32))


@pytest.mark.parametrize(
    "invalid",
    [
        np.ones((2, 3), dtype=np.float32),
        np.empty((0, 2, 1), dtype=np.float32),
        np.empty((1, 0, 1), dtype=np.float32),
        np.empty((1, 1, 0), dtype=np.float32),
        np.asarray([[[np.nan]]], dtype=np.float32),
    ],
)
def test_sequence_standardizer_fit_rejects_invalid_sequences(invalid):
    with pytest.raises(ValueError):
        SequenceStandardizer().fit(invalid)


def test_sequence_standardizer_failed_refit_preserves_previous_state():
    scaler = SequenceStandardizer().fit(np.asarray([[[1.0], [3.0]]], dtype=np.float32))
    original_mean = scaler.mean_.copy()
    original_scale = scaler.scale_.copy()

    with pytest.raises(ValueError):
        scaler.fit(np.asarray([[[np.inf]]], dtype=np.float32))

    np.testing.assert_array_equal(scaler.mean_, original_mean)
    np.testing.assert_array_equal(scaler.scale_, original_scale)


def test_sequence_standardizer_state_round_trip_preserves_transform():
    train = np.asarray(
        [[[1.0, 10.0], [3.0, 14.0]], [[5.0, 18.0], [7.0, 22.0]]],
        dtype=np.float32,
    )
    inference = np.asarray([[[9.0, 26.0], [11.0, 30.0]]], dtype=np.float32)
    scaler = SequenceStandardizer().fit(train)
    expected = scaler.transform(inference)

    state = scaler.state_dict()
    restored = SequenceStandardizer.from_state_dict(state)

    np.testing.assert_allclose(restored.transform(inference), expected)
    assert restored.mean_.dtype == np.float32
    assert restored.scale_.dtype == np.float32

    state["mean"].fill(999.0)
    state["scale"].fill(999.0)
    np.testing.assert_allclose(scaler.transform(inference), expected)
    np.testing.assert_allclose(restored.transform(inference), expected)


def test_sequence_standardizer_state_requires_fitted_valid_data():
    with pytest.raises(RuntimeError, match="unfitted"):
        SequenceStandardizer().state_dict()
    with pytest.raises(TypeError, match="mapping"):
        SequenceStandardizer.from_state_dict([])

    mean = np.zeros((1, 1, 2), dtype=np.float32)
    scale = np.ones((1, 1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="missing"):
        SequenceStandardizer.from_state_dict({"mean": mean})
    with pytest.raises(ValueError, match="matching"):
        SequenceStandardizer.from_state_dict(
            {"mean": mean, "scale": np.ones((1, 1, 1), dtype=np.float32)}
        )
    with pytest.raises(ValueError, match="positive"):
        SequenceStandardizer.from_state_dict(
            {"mean": mean, "scale": np.zeros_like(scale)}
        )
    with pytest.raises(ValueError, match="finite"):
        SequenceStandardizer.from_state_dict(
            {"mean": np.full_like(mean, np.nan), "scale": scale}
        )


def test_sequence_standardizer_transform_rejects_corrupted_live_state():
    scaler = SequenceStandardizer().fit(np.asarray([[[1.0], [3.0]]], dtype=np.float32))
    scaler.scale_[0, 0, 0] = 0.0

    with pytest.raises(RuntimeError, match="invalid fitted state"):
        scaler.transform(np.ones((1, 2, 1), dtype=np.float32))
