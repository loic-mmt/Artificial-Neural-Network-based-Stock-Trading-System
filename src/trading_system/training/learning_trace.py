"""Read-only observations of financial training passes, without model forwards.

This module consumes positions/losses already produced by the optimizer loop.
It does not compute financial objectives, update tensors or sample randomness.
"""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral, Real

import numpy as np


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _finite(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return float(value)


def position_summary(positions):
    """Summarize raw signals, not delayed/executed positions or financial metrics."""
    values = np.asarray(positions)
    if (values.size == 0 or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()):
        raise ValueError("positions must contain nonempty finite real values.")
    values = values.astype(np.float64, copy=False).reshape(-1)
    return {"count": int(values.size), "mean_position": float(values.mean()),
            "mean_abs_position": float(np.abs(values).mean()),
            "std_position": float(values.std()), "min_position": float(values.min()),
            "max_position": float(values.max()), "long_fraction": float((values > 0).mean()),
            "short_fraction": float((values < 0).mean()), "flat_fraction": float((values == 0).mean())}


def gradient_norm_l2(parameters):
    """Observe the global pre-clipping L2 norm without changing any gradient.

    Tensor methods are duck-typed so importing this helper does not require
    torch. Double-precision detached reductions neither mutate nor backpropagate.
    Parameters with no gradient contribute zero, as in gradient clipping.
    """
    norms = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        detached = gradient.detach()
        if detached.is_sparse:
            detached = detached.coalesce().values()
        # Reduce on CPU: MPS does not support float64 device tensors. This is
        # opt-in observation/synchronization only, never an optimizer input.
        norm = float(detached.cpu().double().norm(2).item())
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite learning-diagnostic gradient norm.")
        norms.append(norm)
    return math.hypot(*norms)


class LearningTrace:
    """A JSON-safe epoch trace; callers keep all stopping/checkpoint decisions."""

    def __init__(self):
        self._epochs = []

    @staticmethod
    def phase(loss, positions):
        return {"loss": _finite(loss, "loss"), "positions": position_summary(positions)}

    def record(self, epoch, *, train, validation, gradient_norm_pre_clip, improved, stale, best_epoch):
        epoch = _integer(epoch, "epoch", minimum=1)
        if epoch != len(self._epochs) + 1:
            raise ValueError("Trace epochs must be consecutive, starting at 1.")
        if not isinstance(improved, bool):
            raise ValueError("improved must be boolean.")
        stale = _integer(stale, "stale")
        best_epoch = _integer(best_epoch, "best_epoch", minimum=1)
        if best_epoch > epoch or improved and (best_epoch != epoch or stale != 0):
            raise ValueError("Trace checkpoint/staleness identity is inconsistent.")
        phases = {}
        for name, phase in (("train", train), ("validation", validation)):
            if not isinstance(phase, dict) or set(phase) != {"loss", "positions"}:
                raise ValueError(f"{name} must be a phase observation.")
            observed_loss = _finite(phase["loss"], f"{name}.loss")
            summary = phase["positions"]
            if not isinstance(summary, dict) or set(summary) != {
                "count", "mean_position", "mean_abs_position", "std_position", "min_position",
                "max_position", "long_fraction", "short_fraction", "flat_fraction",
            }:
                raise ValueError(f"{name}.positions must be a position summary.")
            observed_summary = {"count": _integer(summary["count"], f"{name}.count", minimum=1)}
            for key, value in summary.items():
                if key != "count":
                    observed_summary[key] = _finite(value, f"{name}.{key}")
            for key in ("mean_abs_position", "std_position", "long_fraction", "short_fraction", "flat_fraction"):
                if summary[key] < 0 or key.endswith("_fraction") and summary[key] > 1:
                    raise ValueError(f"{name}.{key} is outside its valid range.")
            if not math.isclose(sum(summary[key] for key in ("long_fraction", "short_fraction", "flat_fraction")), 1):
                raise ValueError(f"{name} position fractions must sum to 1.")
            phases[name] = {"loss": observed_loss, "positions": observed_summary}
        self._epochs.append({"epoch": epoch, **phases,
            "gradient_norm_pre_clip": _finite(gradient_norm_pre_clip, "gradient_norm_pre_clip", minimum=0),
            "improved": improved, "stale": stale, "best_epoch": best_epoch})

    def finish(self, *, stop_reason, best_epoch):
        if not self._epochs:
            raise ValueError("A learning trace requires at least one observed epoch.")
        if not isinstance(stop_reason, str) or stop_reason not in {"early_stopping", "max_epochs"}:
            raise ValueError("Unsupported learning-trace stop reason.")
        best_epoch = _integer(best_epoch, "best_epoch", minimum=1)
        if best_epoch != self._epochs[-1]["best_epoch"]:
            raise ValueError("Final best_epoch must match the last epoch observation.")
        return {"schema_version": 1,
                "phase_semantics": {
                    "train": "pre_optimizer_update_train_mode; dropout active when configured",
                    "validation": "post_optimizer_update_eval_mode; no dropout",
                    "loss": "existing ReturnPanel financial objective, not classification loss",
                    "positions": "raw signals including unavailable FLAT fallback, before execution delay/costs; flattened ticker-date rows",
                    "gradient_norm_pre_clip": "global parameter-gradient L2 norm before clipping",
                    "checkpoint": "original validation-loss early-stopping rule; restored best_epoch",
                },
                "epochs": deepcopy(self._epochs), "stop_reason": stop_reason,
                "best_epoch": best_epoch}


__all__ = ["LearningTrace", "position_summary", "gradient_norm_l2"]
