from __future__ import annotations

import numpy as np


def validate_sample_weight(weights, rows: int, *, name="sample_weight"):
    if weights is None:
        return None
    values = np.asarray(weights, dtype=np.float32)
    if values.shape != (rows,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite 1D array aligned with samples.")
    if (values < 0).any() or not np.any(values > 0):
        raise ValueError(f"{name} must be non-negative with positive total weight.")
    return values


def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Class weights require a non-empty 1D label array.")
    if num_classes <= 0 or (labels < 0).any() or (labels >= num_classes).any():
        raise ValueError("Labels fall outside configured class range.")
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    return (len(labels) / (num_classes * counts)).astype(np.float32)


__all__ = ["compute_class_weights"]
