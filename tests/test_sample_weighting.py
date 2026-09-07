from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from trading_system.training.sample_weighting import (
    SampleWeightConfig, SampleWeightTransformer, event_uniqueness, prepare_sample_weights,
)
from trading_system.training.weights import validate_sample_weight
from trading_system.models.manual_ann.manual_nn import ManualANNClassifier, ManualANNConfig
from trading_system.models.manual_ann.sequence_adapter import ManualANNSequenceAdapter
from trading_system.models.specs import ModelSelection
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import run_validation_experiment, evaluate_experiment_test
from trading_system.experiments.walkforward import walk_forward_classifier
from trading_system.artifacts.experiment import build_experiment_manifest


def events():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=4),
        "label_end_date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05", None]),
        "label_event_id": pd.array([0, 1, 2, None], dtype="Int64"),
        "net_event_return": [-0.01, 0.02, 0.04, np.nan],
        "label_score": [-1.0, 3.0, 2.0, np.nan], "_label_known": True,
    })


def prices():
    x = np.arange(180)
    close = 100 + x * 0.03 + 2 * np.sin(x / 5)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=len(x)),
        "adj_close": close, "close": close, "open": close,
        "high": close * 1.01, "low": close * 0.99, "volume": 1e6,
        "signal": np.sin(x / 5),
    })


@pytest.mark.parametrize("mode", ["net_return", "volatility", "uniqueness"])
def test_train_normalization_and_validation_freeze(mode):
    frame = events()
    transformer = SampleWeightTransformer(SampleWeightConfig(mode=mode, clip_quantile=1))
    weights = transformer.fit_transform(frame)
    assert weights.mean() == pytest.approx(1)
    assert (weights > 0).all()
    before = json.dumps(transformer.state, sort_keys=True)
    changed = frame.copy()
    changed.net_event_return *= 1e6
    changed.label_score *= 1e6
    actual = transformer.transform(changed)
    assert np.isfinite(actual).all()
    assert json.dumps(transformer.state, sort_keys=True) == before
    assert actual[-1] == weights[-1]  # Non-events keep the neutral base weight.


def test_return_and_volatility_ordering_are_distinct():
    raw = SampleWeightTransformer(SampleWeightConfig()).fit_transform(events())
    scaled = SampleWeightTransformer(SampleWeightConfig(mode="volatility")).fit_transform(events())
    assert raw[0] < raw[1] < raw[2]
    assert scaled[0] < scaled[2] < scaled[1]


def test_uniqueness_overlap_and_ticker_independence():
    frame = events().iloc[:3].copy()
    np.testing.assert_allclose(event_uniqueness(frame), [0.75, 0.5, 0.75])
    combined = pd.concat([frame.assign(ticker="A"), frame.assign(ticker="B")], ignore_index=True)
    np.testing.assert_allclose(event_uniqueness(combined, group_col="ticker"), [0.75, 0.5, 0.75] * 2)
    identical = pd.concat([frame.iloc[:1]] * 2, ignore_index=True)
    np.testing.assert_allclose(event_uniqueness(identical), [0.5, 0.5])
    np.testing.assert_allclose(event_uniqueness(frame.iloc[:1]), [1])


def test_zero_magnitudes_fall_back_explicitly_and_sparse_signals_survive():
    frame = events().assign(net_event_return=0)
    transformer = SampleWeightTransformer(SampleWeightConfig())
    np.testing.assert_array_equal(transformer.fit_transform(frame), np.ones(4))
    assert transformer.state["fallback"] == "uniform_zero_magnitude"
    frame.loc[2, "net_event_return"] = 10
    weights = transformer.fit_transform(frame)
    assert weights[2] > weights[0]
    assert transformer.state["fallback"] is None


@pytest.mark.parametrize("kwargs", [{"mode": "bad"}, {"min_weight": 0}, {"max_weight": 0.5}, {"clip_quantile": 0}, {"clip_quantile": np.nan}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SampleWeightConfig(**kwargs)


@pytest.mark.parametrize("weights", [[1], [[1, 1]], [0, 0], [1, -1], [1, np.nan], [1, np.inf]])
def test_model_weight_validation(weights):
    with pytest.raises(ValueError):
        validate_sample_weight(weights, 2)


def test_missing_diagnostics_and_unknown_labels_are_not_silently_weighted():
    transformer = SampleWeightTransformer(SampleWeightConfig())
    with pytest.raises(ValueError, match="diagnostics"):
        transformer.fit_transform(events().drop(columns="net_event_return"))
    with pytest.raises(ValueError, match="observed"):
        transformer.fit_transform(events().assign(_label_known=False))
    with pytest.raises(ValueError, match="observed training event"):
        transformer.fit_transform(events().iloc[3:])
    with pytest.raises(ValueError, match="triple_barrier"):
        ExperimentConfig(sample_weighting=SampleWeightConfig())


def test_ann_joint_weighted_loss_formula():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    labels = np.array([0, 1])
    class_weights = np.array([2., 3., 1.])
    sample = np.array([1., 4.])
    expected = -(2 * np.log(0.7) + 12 * np.log(0.6)) / 14
    assert ManualANNClassifier._weighted_cross_entropy(probabilities, labels, class_weights, sample) == pytest.approx(expected)


def test_ann_unit_weights_preserve_training_and_importance_changes_it():
    X = np.random.default_rng(3).normal(size=(12, 4)).astype(np.float32)
    y = np.arange(12) % 3
    config = ManualANNConfig(epochs=3, hidden_size=4, batch_size=12, seed=4)
    base, same, weighted = [ManualANNClassifier(config) for _ in range(3)]
    base.fit(X, y)
    same.fit(X, y, sample_weight=np.ones(12))
    weighted.fit(X, y, sample_weight=np.where(y == 2, 10., 0.1))
    np.testing.assert_array_equal(base.predict_proba(X), same.predict_proba(X))
    assert not np.allclose(base.predict_proba(X), weighted.predict_proba(X))


def test_sequence_adapter_preserves_sample_order_and_zero_batches():
    X = np.random.default_rng(4).normal(size=(6, 2, 3)).astype(np.float32)
    y = np.arange(6) % 3
    weights = np.array([0, 0, 0, 1, 4, 2], dtype=np.float32)
    config = ManualANNConfig(epochs=1, hidden_size=4, batch_size=1)
    adapter = ManualANNSequenceAdapter(config)
    result = adapter.fit(X, y, X_val=X, y_val=y, sample_weight=weights, sample_weight_val=weights)
    assert np.isfinite(result.history.val_loss).all()
    with pytest.raises(ValueError, match="validation"):
        adapter.fit(X, y, sample_weight_val=weights)


@pytest.mark.parametrize("mode", ["net_return", "volatility", "uniqueness"])
def test_static_alignment_frozen_test_and_manifest(mode, monkeypatch):
    data = pd.concat([prices().assign(ticker="A"), prices().assign(ticker="B")], ignore_index=True)
    captured = []
    aligned_labels = []
    import trading_system.experiments.runner as runner
    prepare = runner.prepare_sample_weights
    def record_alignment(train, val, *args, **kwargs):
        aligned_labels.append((train.Label_id.to_numpy(), val.Label_id.to_numpy()))
        return prepare(train, val, *args, **kwargs)
    monkeypatch.setattr(runner, "prepare_sample_weights", record_alignment)
    original = ManualANNSequenceAdapter.fit
    def recording(self, X_train, y_train, **kwargs):
        assert len(kwargs["sample_weight"]) == len(y_train) == len(X_train)
        assert len(kwargs["sample_weight_val"]) == len(kwargs["y_val"])
        np.testing.assert_array_equal(y_train, aligned_labels[-1][0])
        np.testing.assert_array_equal(kwargs["y_val"], aligned_labels[-1][1])
        captured.append(kwargs["sample_weight"].copy())
        return original(self, X_train, y_train, **kwargs)
    monkeypatch.setattr(ManualANNSequenceAdapter, "fit", recording)
    config = ExperimentConfig(
        universe="multi", label_mode="triple_barrier", triple_barrier_max_holding=3,
        triple_barrier_volatility_window=5, context_len=3, decision_mode="argmax",
        sample_weighting=SampleWeightConfig(mode=mode),
        model=ModelSelection("manual_ann", {"epochs": 1, "hidden_size": 4}),
    )
    validation = run_validation_experiment(data, config)
    assert captured[0].mean() == pytest.approx(1)
    assert "net_event_return" not in validation.bundle.feature_columns
    def no_fit(*args, **kwargs):
        raise AssertionError("Test attempted to fit sample weights")
    monkeypatch.setattr(SampleWeightTransformer, "fit_transform", no_fit)
    result = evaluate_experiment_test(data, validation)
    manifest = build_experiment_manifest(data, result)
    assert manifest.experiment_parameters["sample_weight_state"]["config"]["mode"] == mode
    assert len(captured) == 1


def test_walkforward_weighted_training():
    result = walk_forward_classifier(
        prices(), ["signal"], train_ratio=0.6, val_ratio=0.2, walkforward_step=100,
        context_len=3, label_mode="triple_barrier", triple_barrier_max_holding=3,
        triple_barrier_volatility_window=5, sample_weighting=SampleWeightConfig(mode="uniqueness"),
        model_selection=ModelSelection("manual_ann", {"epochs": 1, "hidden_size": 4}),
    )
    assert result["retrain_logs"][0]["sample_weight_state"]["train"]["mean"] == pytest.approx(1)


def test_cli_and_config_roundtrip():
    from trading_system.pipelines.compare_models import build_parser
    from trading_system.pipelines.training_arguments import sample_weight_config_from_args
    args = build_parser().parse_args(["--sample-weighting", "net_return", "--sample-weight-max", "5"])
    config = sample_weight_config_from_args(args)
    assert config.max_weight == 5
    experiment = ExperimentConfig(label_mode="triple_barrier", sample_weighting=asdict(config))
    assert experiment.sample_weighting == config


def test_event_ids_stay_aligned_after_temporal_splitting():
    from trading_system.labels.config import LabelConfig
    from trading_system.labels.registry import create_default_label_registry
    frame = prices()
    frame["_experiment_split"] = ["train"] * 100 + ["val"] * 40 + ["test"] * 40
    result = create_default_label_registry().generate(frame, LabelConfig.triple_barrier(
        max_holding=3, volatility_window=5, event_filter="cusum", cusum_threshold=3.0,
    )).frame
    np.testing.assert_array_equal(result.label_event_id.notna(), result.barrier_touch != "not_event")


def test_weighting_rejects_mixed_splits():
    frame = events().assign(_experiment_split=["train", "train", "val", "val"])
    with pytest.raises(ValueError, match="temporal split"):
        SampleWeightTransformer(SampleWeightConfig()).fit_transform(frame)
