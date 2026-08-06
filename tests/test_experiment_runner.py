import numpy as np
import pandas as pd
import pytest

from trading_system.data.scaling import SequenceStandardizer
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import run_experiment
from trading_system.experiments.walkforward import walk_forward_oracle_ann
from trading_system.features.technical import (
    TECHNICAL_FEATURE_COLUMNS,
    compute_technical_features,
)
from trading_system.models.base import FitResult, TrainingHistory
from trading_system.models.manual_ann.manual_nn import ManualANNConfig
from trading_system.models.manual_ann.sequence_adapter import ManualANNSequenceAdapter


class RecordingSequenceClassifier:
    """Small protocol implementation proving runner inputs stay three-dimensional."""

    model_name = "recording_sequence"

    def __init__(self):
        self.classes_ = np.arange(3, dtype=np.int64)
        self.train_shape: tuple[int, ...] | None = None
        self.val_shape: tuple[int, ...] | None = None

    def fit(self, X_train, y_train, *, X_val=None, y_val=None):
        assert X_train.ndim == 3
        assert X_val is not None and X_val.ndim == 3
        assert y_val is not None
        self.train_shape = X_train.shape
        self.val_shape = X_val.shape
        np.testing.assert_allclose(X_train.mean(axis=(0, 1)), 0.0, atol=1e-5)
        return FitResult(
            best_epoch=1,
            stop_reason="recorded",
            history=TrainingHistory(train_loss=[1.0], val_loss=[1.0]),
        )

    def predict_proba(self, X):
        assert X.ndim == 3
        row = np.asarray([[0.2, 0.6, 0.2]], dtype=np.float32)
        return np.tile(row, (len(X), 1))

    def state_dict(self):
        return {}


def test_static_experiment_runs_end_to_end_with_shared_layers():
    rows = 180
    x = np.arange(rows, dtype=np.float64)
    close = 100.0 + 0.05 * x + 4.0 * np.sin(x / 5.0)
    frame = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": pd.date_range("2023-01-01", periods=rows),
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000.0 + 10_000.0 * np.cos(x / 7.0),
        }
    )
    config = ExperimentConfig(
        universe="single",
        ticker="AAA",
        feature_set="technical",
        label_mode="breakout",
        position_mode="long_only",
        label_window=5,
        train_ratio=0.60,
        val_ratio=0.20,
        context_len=5,
        manual_ann=ManualANNConfig(
            hidden_size=8,
            epochs=8,
            batch_size=16,
            early_stopping_patience=3,
            seed=11,
        ),
    )
    model = RecordingSequenceClassifier()
    result = run_experiment(frame, config, model=model)
    assert result.bundle.estimator is model
    assert isinstance(result.bundle.scaler, SequenceStandardizer)
    assert model.train_shape is not None
    assert model.train_shape[1:] == (
        config.context_len,
        len(TECHNICAL_FEATURE_COLUMNS),
    )
    assert model.val_shape is not None
    assert model.val_shape[1:] == (
        config.context_len,
        len(TECHNICAL_FEATURE_COLUMNS),
    )
    assert result.test_probabilities.shape[1] == 3
    assert len(result.test_predictions) == len(result.aligned_test_frame)
    assert result.backtest["initial_capital"] == config.initial_capital

    bundle_probabilities = result.bundle.predict_proba(
        np.ones(
            (2, config.context_len, len(TECHNICAL_FEATURE_COLUMNS)),
            dtype=np.float32,
        )
    )
    assert bundle_probabilities.shape == (2, 3)
    np.testing.assert_allclose(bundle_probabilities.sum(axis=1), 1.0)

    with pytest.raises(ValueError, match="bundle dimensions"):
        result.bundle.predict_proba(
            np.ones(
                (1, config.context_len + 1, len(TECHNICAL_FEATURE_COLUMNS)),
                dtype=np.float32,
            )
        )


def test_multi_ticker_experiment_uses_same_runner():
    rows = 120
    parts = []
    for offset, ticker in enumerate(("AAA", "BBB")):
        x = np.arange(rows, dtype=np.float64)
        close = 100.0 + 5.0 * offset + 0.04 * x + 3.0 * np.sin(x / (4.0 + offset))
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": pd.date_range("2023-01-01", periods=rows),
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000_000.0 + 10_000.0 * np.cos(x / 6.0),
                }
            )
        )
    config = ExperimentConfig(
        universe="multi",
        feature_set="technical",
        label_mode="breakout",
        position_mode="long_short",
        label_window=5,
        train_ratio=0.60,
        val_ratio=0.20,
        context_len=5,
        manual_ann=ManualANNConfig(
            hidden_size=6,
            epochs=3,
            batch_size=16,
            early_stopping_patience=2,
        ),
    )
    result = run_experiment(pd.concat(parts, ignore_index=True), config)
    assert isinstance(result.bundle.scaler, SequenceStandardizer)
    assert isinstance(result.bundle.estimator, ManualANNSequenceAdapter)
    assert result.bundle.estimator.context_len_ == config.context_len
    assert result.bundle.estimator.feature_count_ == len(TECHNICAL_FEATURE_COLUMNS)
    assert result.aligned_test_frame.groupby("ticker").size().to_dict() == {
        "AAA": 24,
        "BBB": 24,
    }
    assert result.backtest["initial_capital"] == 10_000.0


def test_walkforward_reuses_manual_model_and_shared_evaluation():
    rows = 110
    x = np.arange(rows, dtype=np.float64)
    close = 100.0 + 0.04 * x + 3.0 * np.sin(x / 4.0)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows),
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000.0 + 10_000.0 * np.cos(x / 6.0),
        }
    )
    featured = compute_technical_features(frame, group_col=None)
    result = walk_forward_oracle_ann(
        featured,
        TECHNICAL_FEATURE_COLUMNS,
        train_ratio=0.70,
        val_ratio=0.15,
        walkforward_step=20,
        label_mode="forward_return",
        decision_mode="argmax",
        position_mode="long_only",
        context_len=5,
        epochs=2,
        hidden=6,
        batch_size=16,
        early_stopping_patience=2,
    )
    assert result["n_eval_rows"] == result["n_test_rows"]
    assert result["retrain_logs"]
