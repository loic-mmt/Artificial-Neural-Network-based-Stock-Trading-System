from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from trading_system.labels.schema import N_CLASSES
from trading_system.models.base import FitResult, TrainingHistory
from trading_system.training.weights import compute_class_weights


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, values)


def relu_derivative(values: np.ndarray) -> np.ndarray:
    return (values > 0.0).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - values.max(axis=1, keepdims=True)
    np.exp(shifted, out=shifted)
    shifted /= shifted.sum(axis=1, keepdims=True)
    return shifted


def one_hot(labels: np.ndarray, num_classes: int = N_CLASSES) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or (y < 0).any() or (y >= num_classes).any():
        raise ValueError("labels must be a 1D array inside the configured class range.")
    encoded = np.zeros((len(y), num_classes), dtype=np.float32)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def dropout_mask(
    shape: tuple[int, ...],
    probability: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if not 0.0 <= probability < 1.0:
        raise ValueError("dropout probability must be in [0, 1).")
    generator = rng or np.random.default_rng()
    keep_probability = 1.0 - probability
    return (generator.random(shape) < keep_probability).astype(
        np.float32
    ) / keep_probability


def forward_pass(
    X: np.ndarray,
    W0: np.ndarray,
    b0: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z1 = np.asarray(X, dtype=np.float32) @ W0 + b0
    a1 = relu(z1)
    logits = a1 @ W1 + b1
    return z1, a1, logits, softmax(logits)


@dataclass(frozen=True)
class ManualANNConfig:
    hidden_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 500
    batch_size: int = 32
    dropout_probability: float = 0.0
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4
    seed: int = 1
    num_classes: int = N_CLASSES

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("hidden_size, epochs, and batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1).")
        if self.early_stopping_patience <= 0 or self.early_stopping_min_delta < 0:
            raise ValueError("Invalid early-stopping configuration.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must exceed one.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")


class ManualANNClassifier:
    """One-hidden-layer neural classifier implemented only with NumPy."""

    def __init__(self, config: ManualANNConfig | None = None):
        self.config = config or ManualANNConfig()
        self.classes_ = np.arange(self.config.num_classes, dtype=np.int64)
        self.W0_: np.ndarray | None = None
        self.b0_: np.ndarray | None = None
        self.W1_: np.ndarray | None = None
        self.b1_: np.ndarray | None = None
        self.fit_result_: FitResult | None = None

    @staticmethod
    def _validate_X(
        X: np.ndarray, *, expected_features: int | None = None
    ) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("X must be a non-empty finite 2D array.")
        if expected_features is not None and values.shape[1] != expected_features:
            raise ValueError("X feature count does not match fitted model.")
        return values

    def _validate_y(self, y: np.ndarray, expected_rows: int) -> np.ndarray:
        labels = np.asarray(y, dtype=np.int64)
        if labels.ndim != 1 or len(labels) != expected_rows:
            raise ValueError("y must be a 1D array aligned with X.")
        if (labels < 0).any() or (labels >= self.config.num_classes).any():
            raise ValueError("y contains labels outside configured class range.")
        return labels

    @staticmethod
    def _weighted_cross_entropy(
        probabilities: np.ndarray,
        labels: np.ndarray,
        class_weights: np.ndarray,
    ) -> float:
        sample_weights = class_weights[labels]
        selected = probabilities[np.arange(len(labels)), labels]
        return float(
            -np.sum(sample_weights * np.log(selected + 1e-12)) / sample_weights.sum()
        )

    def _state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if any(weight is None for weight in (self.W0_, self.b0_, self.W1_, self.b1_)):
            raise RuntimeError("ManualANNClassifier is not fitted.")
        assert self.W0_ is not None and self.b0_ is not None
        assert self.W1_ is not None and self.b1_ is not None
        return self.W0_, self.b0_, self.W1_, self.b1_

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        class_weights: np.ndarray | None = None,
    ) -> FitResult:
        started = perf_counter()
        X = self._validate_X(X_train)
        y = self._validate_y(y_train, len(X))
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together.")
        validation_X = (
            None
            if X_val is None
            else self._validate_X(X_val, expected_features=X.shape[1])
        )
        validation_y = (
            None if y_val is None else self._validate_y(y_val, len(validation_X))
        )
        weights = (
            compute_class_weights(y, self.config.num_classes)
            if class_weights is None
            else np.asarray(class_weights, dtype=np.float32)
        )
        if weights.shape != (self.config.num_classes,) or (weights <= 0).any():
            raise ValueError("class_weights must contain one positive value per class.")

        rng = np.random.default_rng(self.config.seed)
        input_size = X.shape[1]
        self.W0_ = (
            0.01 * rng.standard_normal((input_size, self.config.hidden_size))
        ).astype(np.float32)
        self.b0_ = np.zeros((1, self.config.hidden_size), dtype=np.float32)
        self.W1_ = (
            0.01
            * rng.standard_normal((self.config.hidden_size, self.config.num_classes))
        ).astype(np.float32)
        self.b1_ = np.zeros((1, self.config.num_classes), dtype=np.float32)

        history = TrainingHistory()
        best_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        best_loss = np.inf
        best_epoch = 0
        epochs_without_improvement = 0
        stop_reason = "max_epochs"

        for epoch in range(self.config.epochs):
            permutation = rng.permutation(len(X))

            for start in range(0, len(X), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = permutation[start:end]

                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                W0, b0, W1, b1 = self._state()
                z1 = batch_X @ W0 + b0
                active = z1 > 0.0
                hidden = z1
                np.maximum(hidden, 0.0, out=hidden)
                mask = None
                if self.config.dropout_probability > 0:
                    mask = dropout_mask(
                        hidden.shape, self.config.dropout_probability, rng
                    )
                    hidden *= mask
                probabilities = softmax(hidden @ W1 + b1)
                sample_weights = weights[batch_y]
                weight_sum = float(sample_weights.sum())
                # Probabilities are no longer needed once backprop starts. Reuse
                # their buffer rather than allocating one-hot targets/gradients.
                output_gradient = probabilities
                output_gradient[np.arange(len(batch_y)), batch_y] -= 1.0
                output_gradient *= sample_weights[:, None]
                output_gradient /= weight_sum
                dW1 = hidden.T @ output_gradient
                db1 = output_gradient.sum(axis=0, keepdims=True)
                hidden_gradient = output_gradient @ W1.T
                if mask is not None:
                    hidden_gradient *= mask
                hidden_gradient *= active
                dW0 = batch_X.T @ hidden_gradient
                db0 = hidden_gradient.sum(axis=0, keepdims=True)
                # All gradients were computed with the old weights. Updating in
                # place now preserves SGD math and the copied best checkpoint.
                for parameter, gradient in (
                    (W1, dW1), (b1, db1), (W0, dW0), (b0, db0)
                ):
                    gradient *= self.config.learning_rate
                    parameter -= gradient

            train_loss = self._weighted_cross_entropy(
                self._predict_validated(X), y, weights
            )
            history.train_loss.append(train_loss)
            if validation_X is not None and validation_y is not None:
                selection_loss = self._weighted_cross_entropy(
                    self._predict_validated(validation_X),
                    validation_y,
                    weights,
                )
                history.val_loss.append(selection_loss)
            else:
                selection_loss = train_loss

            if selection_loss < best_loss - self.config.early_stopping_min_delta:
                best_loss = selection_loss
                best_epoch = epoch + 1
                best_state = tuple(weight.copy() for weight in self._state())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.config.early_stopping_patience:
                stop_reason = "early_stopping"
                break

        if best_state is None:
            raise RuntimeError("Training failed to record model weights.")
        self.W0_, self.b0_, self.W1_, self.b1_ = best_state
        self.fit_result_ = FitResult(
            best_epoch=best_epoch,
            stop_reason=stop_reason,
            history=history,
            training_duration_seconds=perf_counter() - started,
            parameter_count=int(sum(weight.size for weight in best_state)),
            seed=self.config.seed,
            device="cpu",
        )
        return self.fit_result_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        W0 = self._state()[0]
        values = self._validate_X(X, expected_features=W0.shape[0])
        return self._predict_validated(values)

    def _predict_validated(self, values: np.ndarray) -> np.ndarray:
        """Inference needs one hidden buffer, unlike the public training trace."""

        W0, b0, W1, b1 = self._state()
        hidden = values @ W0
        hidden += b0
        np.maximum(hidden, 0.0, out=hidden)
        logits = hidden @ W1
        logits += b1
        del hidden
        return softmax(logits)

    def state_dict(self) -> dict[str, np.ndarray]:
        W0, b0, W1, b1 = self._state()
        return {"W0": W0.copy(), "b0": b0.copy(), "W1": W1.copy(), "b1": b1.copy()}


__all__ = [
    "ManualANNClassifier",
    "ManualANNConfig",
    "dropout_mask",
    "forward_pass",
    "one_hot",
    "relu",
    "relu_derivative",
    "softmax",
]
