from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_system.artifacts.serialization import (
    ArtifactManifest,
    load_model_artifact,
    save_model_artifact,
    stable_config_hash,
    to_jsonable,
    validate_artifact_compatibility,
)
from trading_system.artifacts.experiment import (
    build_experiment_manifest,
    hash_dataframe,
    save_experiment_artifact,
)
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import run_experiment
from trading_system.models.specs import ModelSelection
from trading_system.reporting.warnings import (
    SURVIVOR_BIAS_WARNING,
    current_universe_warning,
)


def make_manifest(**changes):
    values = {
        "format_version": 1,
        "model_name": "manual_ann",
        "model_parameters": {"hidden_size": 8},
        "experiment_parameters": {"train_ratio": 0.7},
        "feature_columns": ("return_1d", "volatility"),
        "context_len": 20,
        "decision_parameters": {"mode": "argmax"},
        "runtime_metadata": {"python": "3.10"},
    }
    values.update(changes)
    config = {
        key: values[key]
        for key in (
            "model_name",
            "model_parameters",
            "experiment_parameters",
            "decision_parameters",
        )
    }
    values.setdefault("config_hash", stable_config_hash(config))
    return ArtifactManifest(**values)


def test_json_conversion_and_hash_are_canonical():
    class Choice(Enum):
        VALUE = "value"

    @dataclass
    class Config:
        path: Path
        choice: Choice
        count: np.int64

    converted = to_jsonable(Config(Path("data/input"), Choice.VALUE, np.int64(3)))
    assert converted == {"path": "data/input", "choice": "value", "count": 3}
    assert stable_config_hash({"b": 2, "a": 1}) == stable_config_hash(
        {"a": 1, "b": 2}
    )
    with pytest.raises(ValueError, match="NaN"):
        to_jsonable(float("nan"))
    with pytest.raises(TypeError, match="NumPy arrays"):
        to_jsonable(np.arange(2))


def test_manifest_validates_hash_schema_and_defensive_metadata():
    parameters = {"hidden_size": 8}
    manifest = make_manifest(model_parameters=parameters)
    parameters["hidden_size"] = 999
    assert manifest.model_parameters == {"hidden_size": 8}
    assert manifest.model_name == "manual_ann"
    with pytest.raises(ValueError, match="config_hash"):
        make_manifest(config_hash="0" * 64)
    with pytest.raises(ValueError, match="unique"):
        make_manifest(feature_columns=("x", "x"))
    with pytest.raises(ValueError, match="class_names"):
        make_manifest(class_names=("Hold", "Sell", "Buy"))


def test_artifact_round_trip_overwrite_and_compatibility(tmp_path):
    destination = tmp_path / "model"
    manifest = make_manifest()
    model_state = {
        "weights": {
            "W0": np.arange(12, dtype=np.float32).reshape(4, 3),
            "b0": np.zeros((1, 3), dtype=np.float32),
        },
        "shape": (4, 3),
    }
    scaler_state = {
        "mean": np.array([[[1.0, 2.0]]], dtype=np.float32),
        "scale": np.array([[[0.5, 1.5]]], dtype=np.float32),
    }
    saved = save_model_artifact(
        destination,
        manifest=manifest,
        model_state=model_state,
        scaler_state=scaler_state,
        training_history={"train_loss": [1.0, 0.5]},
        metrics={"macro_f1": 0.4},
    )
    assert saved == destination.resolve()
    with pytest.raises(FileExistsError):
        save_model_artifact(
            destination,
            manifest=manifest,
            model_state=model_state,
            scaler_state=scaler_state,
            training_history={},
            metrics={},
        )
    loaded, loaded_model, loaded_scaler, diagnostics = load_model_artifact(saved)
    assert loaded == manifest
    assert loaded_model["shape"] == (4, 3)
    np.testing.assert_array_equal(loaded_model["weights"]["W0"], model_state["weights"]["W0"])
    np.testing.assert_array_equal(loaded_scaler["scale"], scaler_state["scale"])
    assert diagnostics["metrics"] == {"macro_f1": 0.4}
    validate_artifact_compatibility(
        loaded,
        model_name="MANUAL_ANN",
        feature_columns=("return_1d", "volatility"),
        context_len=20,
    )
    with pytest.raises(ValueError, match="context_len"):
        validate_artifact_compatibility(
            loaded,
            model_name="manual_ann",
            feature_columns=("return_1d", "volatility"),
            context_len=10,
        )

    save_model_artifact(
        destination,
        manifest=manifest,
        model_state={"revision": np.array([2])},
        scaler_state=scaler_state,
        training_history={},
        metrics={},
        overwrite=True,
    )
    _, overwritten, _, _ = load_model_artifact(destination)
    np.testing.assert_array_equal(overwritten["revision"], [2])


def test_artifact_rejects_checksum_tampering_and_object_arrays(tmp_path):
    manifest = make_manifest()
    destination = tmp_path / "model"
    with pytest.raises(TypeError, match="Object arrays"):
        save_model_artifact(
            destination,
            manifest=manifest,
            model_state={"unsafe": np.array([object()], dtype=object)},
            scaler_state={},
            training_history={},
            metrics={},
        )
    assert not destination.exists()

    save_model_artifact(
        destination,
        manifest=manifest,
        model_state={"weights": np.arange(3)},
        scaler_state={},
        training_history={},
        metrics={},
    )
    (destination / "metrics.json").write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model_artifact(destination)


def test_experiment_artifact_contains_rerun_manifest_and_diagnostics(tmp_path):
    rows = 150
    x = np.arange(rows, dtype=float)
    close = 100 + 0.02 * x + np.sin(x / 4)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=rows),
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": np.full(rows, 1e6),
        }
    )
    config = ExperimentConfig(
        label_mode="forward_return",
        context_len=3,
        decision_mode="argmax",
        model=ModelSelection(
            "manual_ann", {"hidden_size": 4, "epochs": 2, "batch_size": 16}
        ),
        seed=7,
    )
    result = run_experiment(frame, config)
    manifest = build_experiment_manifest(
        frame, result, dataset_path="data/processed/cac40_daily.parquet"
    )
    assert manifest.experiment_parameters["dataset"]["sha256"] == hash_dataframe(frame)
    assert (
        manifest.experiment_parameters["dataset"]["survivor_bias_warning"]
        == SURVIVOR_BIAS_WARNING
    )
    destination = save_experiment_artifact(
        tmp_path / "experiment",
        frame,
        result,
        dataset_path="data/processed/cac40_daily.parquet",
        advanced_diagnostics={"overfit_verdict": "caution"},
    )
    loaded, model_state, scaler_state, diagnostics = load_model_artifact(destination)
    assert loaded.config_hash == manifest.config_hash
    assert set(model_state) >= {"model_name", "weights"}
    assert set(scaler_state) == {"mean", "scale"}
    assert diagnostics["metrics"]["advanced"] == {"overfit_verdict": "caution"}


def test_survivor_bias_warning_only_targets_current_universe_paths():
    assert current_universe_warning("data/processed/market_universe_daily.parquet")
    assert current_universe_warning("data/processed/cac40_daily.parquet")
    assert current_universe_warning("data/private_ticker.parquet") is None
