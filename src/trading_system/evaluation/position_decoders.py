"""Turn Sell/Hold/Buy probabilities into bounded trading positions."""

from __future__ import annotations

import math

import numpy as np


CLASS_POSITIONS = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
DECODER_NAMES = ("continuous", "sign", "argmax", "deadband", "confidence")


def _validated_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("Probabilities must have shape (N, 3) in Sell/Hold/Buy order.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Probabilities must be finite and non-negative.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probability rows must sum to one.")
    return values


def _validated_threshold(threshold: float | None) -> float:
    if threshold is None or isinstance(threshold, bool):
        raise ValueError("This decoder requires a numeric threshold.")
    value = float(threshold)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Decoder threshold must be finite and in [0, 1].")
    return value


def decode_positions(
    probabilities: np.ndarray,
    decoder: str,
    *,
    threshold: float | None = None,
) -> np.ndarray:
    """Decode Sell/Hold/Buy probabilities without changing their row order.

    ``continuous`` returns ``P(Buy) - P(Sell)``. ``sign`` takes its sign.
    ``argmax`` maps the most probable class to -1/0/+1. ``deadband`` makes
    small continuous signals flat. ``confidence`` uses argmax but stays flat
    unless the top probability exceeds the runner-up by ``threshold``.
    """

    values = _validated_probabilities(probabilities)
    if decoder not in DECODER_NAMES:
        raise ValueError(f"Unknown decoder {decoder!r}; choose from {DECODER_NAMES}.")
    continuous = values[:, 2] - values[:, 0]
    if decoder == "continuous":
        return continuous
    if decoder == "sign":
        return np.sign(continuous)
    classes = values.argmax(axis=1)
    positions = CLASS_POSITIONS[classes]
    if decoder == "argmax":
        return positions
    cutoff = _validated_threshold(threshold)
    if decoder == "deadband":
        return np.where(np.abs(continuous) >= cutoff, np.sign(continuous), 0.0)
    ordered = np.sort(values, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    return np.where(margin >= cutoff, positions, 0.0)

