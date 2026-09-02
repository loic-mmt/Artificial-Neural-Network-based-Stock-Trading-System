from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields

import numpy as np

from trading_system.models.base import FitResult
from trading_system.models.specs import ModelBuildContext

from .manual_nn import ManualANNClassifier, ManualANNConfig


class ManualANNSequenceAdapter:
    """Adapt canonical 3D sequences to the existing flat NumPy ANN."""

    model_name = "manual_ann"
    estimator: ManualANNClassifier
    classes_: np.ndarray

    def __init__(self, config: ManualANNConfig | None = None):
        if config is not None and not isinstance(config, ManualANNConfig):
            raise TypeError("config must be a ManualANNConfig or None.")

        resolved_config = config or ManualANNConfig()
        if resolved_config.num_classes != 3:
            raise ValueError(
                "ManualANNSequenceAdapter requires the three trading classes."
            )

        # The adapter owns the flat ANN. It does not duplicate training weights or
        # mathematical operations; it only translates the input representation.
        self.config = resolved_config
        self.estimator = ManualANNClassifier(resolved_config)

        # Copy the array so callers cannot accidentally mutate the estimator's
        # canonical Sell/Hold/Buy class order through the adapter attribute.
        self.classes_ = self.estimator.classes_.copy()

        # Sequence dimensions are unknown until fit receives `(N, T, F)`. Future
        # prediction calls will compare their `T` and `F` against these values.
        self.context_len_: int | None = None
        self.feature_count_: int | None = None
        self.fit_result_: FitResult | None = None

    @staticmethod
    def _flatten_sequences(X: np.ndarray) -> np.ndarray:
        # np.asarray accepts NumPy-compatible inputs and guarantees the dtype
        # required by ManualANNClassifier.
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 3 or any(dimension == 0 for dimension in values.shape):
            raise ValueError("X must be a non-empty 3D array shaped (N, T, F).")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite sequence values.")

        n_samples, context_len, feature_count = values.shape

        # C-order flattening produces, for each sample:
        # timestep 0 features, timestep 1 features, ..., timestep T-1 features.
        # Only `(T, F)` are merged; the sample axis and temporal order stay intact.
        flattened = values.reshape(
            n_samples,
            context_len * feature_count,
            order="C",
        )

        # A contiguous float32 result is predictable for NumPy matrix operations,
        # even when the input was a transposed or otherwise non-contiguous view.
        return np.ascontiguousarray(flattened, dtype=np.float32)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        # Validation features and labels form one optional pair. Supplying only
        # one would make validation loss meaningless or misaligned.
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together.")

        # _flatten_sequences owns 3D, finite-value and dtype validation. Keep
        # original shape only to record/check temporal dimensions.
        train_sequences = np.asarray(X_train, dtype=np.float32)
        flattened_train = self._flatten_sequences(train_sequences)
        context_len = int(train_sequences.shape[1])
        feature_count = int(train_sequences.shape[2])

        flattened_val: np.ndarray | None = None
        if X_val is not None:
            validation_sequences = np.asarray(X_val, dtype=np.float32)
            flattened_val = self._flatten_sequences(validation_sequences)

            # Number of validation samples may differ. Time length and feature
            # count must match train, otherwise ANN input meaning/width changes.
            if validation_sequences.shape[1:] != (context_len, feature_count):
                raise ValueError(
                    "Validation dimensions must match train dimensions (T, F)."
                )

        # Adapter changes representation only. ANN remains sole owner of label
        # validation, weights, optimization, early stopping, and best-state restore.
        fit_result = self.estimator.fit(
            X_train=flattened_train,
            y_train=y_train,
            X_val=flattened_val,
            y_val=y_val,
        )

        # Commit adapter state only after successful training. Failed fit leaves
        # sequence dimensions unfitted instead of publishing partial metadata.
        self.context_len_ = context_len
        self.feature_count_ = feature_count
        self.classes_ = self.estimator.classes_.copy()
        self.fit_result_ = fit_result
        return fit_result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.context_len_ is None or self.feature_count_ is None:
            raise RuntimeError("ManualANNSequenceAdapter is not fitted.")

        sequences = np.asarray(X, dtype=np.float32)
        flattened = self._flatten_sequences(sequences)
        expected_shape = (self.context_len_, self.feature_count_)
        if sequences.shape[1:] != expected_shape:
            raise ValueError(
                "Prediction dimensions must match fitted dimensions (T, F)."
            )

        # Adapter owns shape conversion only. Scaling and decision thresholds stay
        # outside; flat ANN remains sole owner of probability computation.
        probabilities = self.estimator.predict_proba(flattened)
        expected_probability_shape = (len(sequences), len(self.classes_))
        if probabilities.shape != expected_probability_shape:
            raise RuntimeError("ANN returned an invalid probability shape.")
        if not np.isfinite(probabilities).all():
            raise RuntimeError("ANN returned non-finite probabilities.")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise RuntimeError("ANN returned probabilities outside [0, 1].")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError("ANN returned probabilities that do not sum to one.")
        return probabilities

    def state_dict(self) -> dict[str, object]:
        if self.context_len_ is None or self.feature_count_ is None:
            raise RuntimeError("Cannot export an unfitted sequence adapter.")

        # ManualANNClassifier.state_dict returns defensive weight copies. Config is
        # a plain mapping so artifact code can serialize metadata independently.
        return {
            "format_version": 1,
            "model_name": self.model_name,
            "config": asdict(self.config),
            "context_len": self.context_len_,
            "feature_count": self.feature_count_,
            "classes": self.classes_.copy(),
            "weights": self.estimator.state_dict(),
        }

    def parameter_count(self) -> int:
        return int(sum(weight.size for weight in self.estimator._state()))

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping.")

        required_keys = {
            "format_version",
            "model_name",
            "config",
            "context_len",
            "feature_count",
            "classes",
            "weights",
        }
        state_keys = set(state)
        missing = sorted(required_keys - state_keys)
        unexpected = sorted(state_keys - required_keys)
        if missing or unexpected:
            raise ValueError(
                f"Invalid adapter state keys; missing={missing}, "
                f"unexpected={unexpected}."
            )

        format_version = state["format_version"]
        if (
            isinstance(format_version, (bool, np.bool_))
            or not isinstance(format_version, (int, np.integer))
            or int(format_version) != 1
        ):
            raise ValueError("Unsupported adapter state format_version.")
        if state["model_name"] != self.model_name:
            raise ValueError("Adapter state model_name does not match manual_ann.")

        config_state = state["config"]
        if not isinstance(config_state, Mapping):
            raise TypeError("Adapter config state must be a mapping.")
        expected_config_keys = {field.name for field in fields(ManualANNConfig)}
        config_keys = set(config_state)
        if config_keys != expected_config_keys:
            raise ValueError("Adapter config state has invalid keys.")
        try:
            loaded_config = ManualANNConfig(**dict(config_state))
        except (TypeError, ValueError) as error:
            raise ValueError("Adapter config state is invalid.") from error
        if loaded_config.num_classes != 3:
            raise ValueError("Adapter state must use three trading classes.")

        def positive_integer(value: object, field_name: str) -> int:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{field_name} must be an integer.")
            result = int(value)
            if result <= 0:
                raise ValueError(f"{field_name} must be positive.")
            return result

        context_len = positive_integer(state["context_len"], "context_len")
        feature_count = positive_integer(state["feature_count"], "feature_count")

        raw_classes = np.asarray(state["classes"])
        if not np.issubdtype(raw_classes.dtype, np.integer):
            raise ValueError("Adapter classes must be integer IDs.")
        classes = raw_classes.astype(np.int64, copy=True)
        expected_classes = np.arange(loaded_config.num_classes, dtype=np.int64)
        if classes.shape != expected_classes.shape or not np.array_equal(
            classes, expected_classes
        ):
            raise ValueError("Adapter classes must equal [0, 1, 2].")

        weights_state = state["weights"]
        if not isinstance(weights_state, Mapping):
            raise TypeError("Adapter weights state must be a mapping.")
        expected_weight_keys = {"W0", "b0", "W1", "b1"}
        if set(weights_state) != expected_weight_keys:
            raise ValueError("Adapter weights state has invalid keys.")

        input_size = context_len * feature_count
        expected_shapes = {
            "W0": (input_size, loaded_config.hidden_size),
            "b0": (1, loaded_config.hidden_size),
            "W1": (loaded_config.hidden_size, loaded_config.num_classes),
            "b1": (1, loaded_config.num_classes),
        }
        validated_weights: dict[str, np.ndarray] = {}
        for name, expected_shape in expected_shapes.items():
            try:
                weight = np.asarray(weights_state[name], dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Adapter weight {name} is not numeric.") from error
            if weight.shape != expected_shape:
                raise ValueError(
                    f"Adapter weight {name} has shape {weight.shape}; "
                    f"expected {expected_shape}."
                )
            if not np.isfinite(weight).all():
                raise ValueError(f"Adapter weight {name} contains non-finite values.")
            validated_weights[name] = weight.copy()

        # Build complete replacement first. No adapter field changes before all
        # metadata and weights pass validation, preventing partial corrupt state.
        loaded_estimator = ManualANNClassifier(loaded_config)
        loaded_estimator.W0_ = validated_weights["W0"]
        loaded_estimator.b0_ = validated_weights["b0"]
        loaded_estimator.W1_ = validated_weights["W1"]
        loaded_estimator.b1_ = validated_weights["b1"]

        self.config = loaded_config
        self.estimator = loaded_estimator
        self.context_len_ = context_len
        self.feature_count_ = feature_count
        self.classes_ = classes
        # Training history belongs to experiment artifact metadata, not model
        # weights. Loaded adapter is inference-ready but has no reconstructed fit.
        self.fit_result_ = None


def create_manual_ann_sequence_classifier(
    context: ModelBuildContext,
    parameters: Mapping[str, object],
) -> ManualANNSequenceAdapter:
    if not isinstance(context, ModelBuildContext):
        raise TypeError("context must be a ModelBuildContext.")
    allowed = {item.name for item in fields(ManualANNConfig)}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"Unknown manual_ann parameters: {unknown}")
    values = dict(parameters)
    for name, required in (("seed", context.seed), ("num_classes", context.num_classes)):
        if name in values and values[name] != required:
            raise ValueError(f"{name} is controlled by ModelBuildContext.")
        values[name] = required
    return ManualANNSequenceAdapter(ManualANNConfig(**values))


__all__ = [
    "ManualANNSequenceAdapter",
    "create_manual_ann_sequence_classifier",
]
