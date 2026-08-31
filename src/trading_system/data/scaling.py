from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

_STATISTICS_BUFFER_BYTES = 8 * 1024**2


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("Standardizer.fit expects a non-empty 2D array.")
        self.mean_ = values.mean(axis=0, keepdims=True).astype(np.float32)
        scale = values.std(axis=0, keepdims=True).astype(np.float32)
        self.scale_ = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fitted before transform.")
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.mean_.shape[1]:
            raise ValueError(
                "Input feature dimension does not match fitted Standardizer."
            )
        transformed = np.subtract(values, self.mean_)
        np.divide(transformed, self.scale_, out=transformed)
        return transformed.astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class SequenceStandardizer:
    """Train-only feature scaler for arrays shaped ``(samples, time, features)``."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    @staticmethod
    def _validated_state_arrays(
        mean: object,
        scale: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and normalize persisted scaler arrays without mutating them."""

        try:
            mean_values = np.asarray(mean, dtype=np.float32)
            scale_values = np.asarray(scale, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("SequenceStandardizer state must be numeric.") from error

        if (
            mean_values.ndim != 3
            or scale_values.ndim != 3
            or mean_values.shape != scale_values.shape
            or mean_values.shape[:2] != (1, 1)
            or mean_values.shape[2] == 0
        ):
            raise ValueError(
                "SequenceStandardizer state must have matching (1, 1, F) shapes."
            )
        if not np.isfinite(mean_values).all() or not np.isfinite(scale_values).all():
            raise ValueError("SequenceStandardizer state must be finite.")
        if (scale_values <= 0).any():
            raise ValueError("SequenceStandardizer scales must be positive.")
        return mean_values, scale_values

    def fit(self, X: np.ndarray) -> "SequenceStandardizer":
        # Canonical sequence layout: samples, timesteps, features. Every axis must
        # be non-empty; otherwise per-feature statistics have no meaning.
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 3 or any(dimension == 0 for dimension in values.shape):
            raise ValueError("X must be a non-empty 3D array shaped (N, T, F).")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite sequence values.")

        # Accumulate statistics in float64. A float32 reduction can invent a tiny
        # variance for a constant feature; dividing by it would turn harmless
        # rounding noise into normalized values near one. Persisted state remains
        # float32, matching model inputs.
        mean_64 = values.mean(
            axis=(0, 1),
            dtype=np.float64,
            keepdims=True,
        )

        # np.std(dtype=float64) allocates a float64 deviation for every window
        # element. Bound this workspace while retaining the exact window weights,
        # float64 arithmetic and population-variance definition (ddof=0).
        rows_per_block = max(
            1, _STATISTICS_BUFFER_BYTES // (values.shape[1] * values.shape[2] * 8)
        )
        workspace = np.empty(
            (min(len(values), rows_per_block), *values.shape[1:]), dtype=np.float64
        )
        variance = np.zeros_like(mean_64)
        for start in range(0, len(values), rows_per_block):
            block = values[start : start + rows_per_block]
            deviations = workspace[: len(block)]
            np.subtract(block, mean_64, out=deviations)
            np.square(deviations, out=deviations)
            variance += deviations.sum(axis=(0, 1), keepdims=True)
        variance /= values.shape[0] * values.shape[1]
        scale_64 = np.sqrt(variance, out=variance)

        if not np.isfinite(mean_64).all() or not np.isfinite(scale_64).all():
            raise ValueError("Sequence statistics must be finite.")

        # Constant or near-constant features need neutral divisor 1. Their centered
        # values become zero without division-by-zero or numerical amplification.
        mean = mean_64.astype(np.float32)
        safe_scale = np.where(scale_64 < 1e-8, 1.0, scale_64).astype(np.float32)

        # Commit fitted state only after all validation succeeds. Copies prevent
        # accidental aliasing with temporary calculation arrays.
        self.mean_ = mean.copy()
        self.scale_ = safe_scale.copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("SequenceStandardizer must be fitted before transform.")

        # Public state can be loaded or manually changed. Validate it before
        # broadcasting so corrupted metadata fails with a clear runtime error.
        try:
            mean, scale = self._validated_state_arrays(self.mean_, self.scale_)
        except ValueError as error:
            raise RuntimeError(
                "SequenceStandardizer has invalid fitted state."
            ) from error

        # N and T may differ from training during validation, test, or walk-forward
        # inference. Only feature count F must remain identical.
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 3 or any(dimension == 0 for dimension in values.shape):
            raise ValueError("X must be a non-empty 3D array shaped (N, T, F).")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite sequence values.")
        if values.shape[2] != mean.shape[2]:
            raise ValueError(
                "Input feature count does not match fitted SequenceStandardizer."
            )

        # `(1,1,F)` state broadcasts over samples and timesteps. Calculation
        # creates new array; fitted statistics and caller input remain unchanged.
        transformed = np.subtract(values, mean)
        np.divide(transformed, scale, out=transformed)
        if not np.isfinite(transformed).all():
            raise ValueError("Transformed sequences contain non-finite values.")
        return transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Use public methods as one canonical path. This prevents fit_transform
        # from developing different validation or normalization rules.
        return self.fit(X).transform(X)

    def state_dict(self) -> dict[str, np.ndarray]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Cannot export an unfitted SequenceStandardizer.")
        try:
            mean, scale = self._validated_state_arrays(self.mean_, self.scale_)
        except ValueError as error:
            raise RuntimeError("Cannot export invalid fitted scaler state.") from error

        # Caller owns returned arrays. Defensive copies prevent artifact code from
        # changing live preprocessing state while saving or manipulating metadata.
        return {"mean": mean.copy(), "scale": scale.copy()}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> "SequenceStandardizer":
        if not isinstance(state, Mapping):
            raise TypeError("SequenceStandardizer state must be a mapping.")

        required_keys = {"mean", "scale"}
        state_keys = set(state)
        missing = sorted(required_keys - state_keys)
        unexpected = sorted(state_keys - required_keys)
        if missing or unexpected:
            raise ValueError(
                f"Invalid scaler state keys; missing={missing}, "
                f"unexpected={unexpected}."
            )

        mean, scale = cls._validated_state_arrays(state["mean"], state["scale"])

        # New fitted instance receives private float32 copies. Later mutation of
        # artifact arrays cannot affect inference results.
        return cls(mean_=mean.copy(), scale_=scale.copy())


def standardize_features(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility helper for legacy tuple-based callers."""

    scaler = Standardizer()
    if mean is None or std is None:
        transformed = scaler.fit_transform(X)
    else:
        scaler.mean_ = np.asarray(mean, dtype=np.float32)
        raw_scale = np.asarray(std, dtype=np.float32)
        scaler.scale_ = np.where(raw_scale < 1e-8, 1.0, raw_scale).astype(np.float32)
        transformed = scaler.transform(X)
    assert scaler.mean_ is not None and scaler.scale_ is not None
    return transformed, scaler.mean_, scaler.scale_


__all__ = ["SequenceStandardizer", "Standardizer", "standardize_features"]
