"""Training-only event importance weights; never used as inference features."""

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SampleWeightConfig:
    mode: str = "net_return"
    clip_quantile: float = 0.99
    min_weight: float = 0.1
    max_weight: float = 10.0

    def __post_init__(self):
        if self.mode not in ("net_return", "volatility", "uniqueness"):
            raise ValueError("Sample weighting mode must be net_return, volatility, or uniqueness.")
        for name in ("clip_quantile", "min_weight", "max_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and numeric.")
        if not 0 < self.clip_quantile <= 1:
            raise ValueError("clip_quantile must be in (0, 1].")
        if not 0 < self.min_weight <= 1 <= self.max_weight:
            raise ValueError("Require 0 < min_weight <= 1 <= max_weight.")


def event_uniqueness(frame, *, date_col="date", group_col=None):
    """Calendar-duration average inverse concurrency on (entry, exit].

    Sweep endpoints independently per ticker. Input must contain only observed
    events from one split; never mix train and validation/test event intervals.
    """
    result = np.ones(len(frame), dtype=float)
    work = frame.reset_index(drop=True)
    groups = work.groupby(group_col, sort=False, dropna=False) if group_col else [(None, work)]
    for _, group in groups:
        starts = pd.to_datetime(group[date_col], errors="coerce")
        ends = pd.to_datetime(group["label_end_date"], errors="coerce")
        if starts.isna().any() or ends.isna().any() or (ends <= starts).any():
            raise ValueError("Uniqueness requires valid event intervals with end > start.")
        # Relative seconds avoid arithmetic overflow and preserve sub-day intervals.
        origin = starts.min()
        start = (starts - origin).dt.total_seconds().to_numpy()
        end = (ends - origin).dt.total_seconds().to_numpy()
        points = np.unique(np.r_[start, end])
        left, right = np.searchsorted(points, start), np.searchsorted(points, end)
        delta = np.zeros(len(points))
        np.add.at(delta, left, 1)
        np.add.at(delta, right, -1)
        concurrency = np.cumsum(delta)[:-1]
        inverse = np.divide(1.0, concurrency, out=np.zeros_like(concurrency), where=concurrency > 0)
        integral = np.r_[0, np.cumsum(np.diff(points) * inverse)]
        result[group.index] = (integral[right] - integral[left]) / (end - start)
    return result


class SampleWeightTransformer:
    def __init__(self, config, *, date_col="date", group_col=None):
        self.config = config
        self.date_col = date_col
        self.group_col = group_col
        self.state = None

    def _raw(self, frame):
        column = "label_score" if self.config.mode == "volatility" else "net_event_return"
        required = ["label_event_id", column]
        if self.group_col:
            required.append(self.group_col)
        if any(name not in frame for name in required):
            raise ValueError("Sample weighting requires triple-barrier event diagnostics.")
        if "_label_known" in frame and (frame["_label_known"].isna().any() or not frame["_label_known"].all()):
            raise ValueError("Sample weights require observed, aligned labels only.")
        if "_experiment_split" in frame and frame["_experiment_split"].nunique(dropna=False) > 1:
            raise ValueError("Compute sample weights separately for each temporal split.")
        events = frame.label_event_id.notna().to_numpy()
        values = np.abs(pd.to_numeric(frame.loc[events, column], errors="coerce").to_numpy(float))
        if not np.isfinite(values).all():
            raise ValueError("Observed event magnitudes must be finite.")
        if self.config.mode == "uniqueness":
            values *= event_uniqueness(frame.loc[events], date_col=self.date_col, group_col=self.group_col)
        return events, values

    def _bounded(self, events, values):
        output = np.ones(len(events), dtype=float)
        if self.state["scale"] > 0:
            output[events] = np.clip(values, 0, self.state["clip_value"]) / self.state["scale"]
        return np.clip(output, self.config.min_weight, self.config.max_weight)

    def fit_transform(self, train):
        events, values = self._raw(train)
        if not len(values):
            raise ValueError("Sample weighting needs at least one observed training event.")
        positive = values[values > 0]
        cap = float(np.quantile(positive, self.config.clip_quantile)) if len(positive) else 0.0
        # Scale before averaging to avoid overflow from summing finite outliers.
        scale = cap * float((np.minimum(values, cap) / cap).mean()) if cap > 0 else 0.0
        self.state = {
            "config": asdict(self.config), "clip_value": cap, "scale": scale,
            "normalizer": 1.0, "train_events": int(events.sum()),
            "fallback": "uniform_zero_magnitude" if scale == 0 else None,
        }
        weights = self._bounded(events, values)
        self.state["normalizer"] = float(weights.mean())
        return (weights / self.state["normalizer"]).astype(np.float32)

    def transform(self, frame):
        if self.state is None:
            raise ValueError("Fit sample-weight normalization on training data first.")
        events, values = self._raw(frame)
        return (self._bounded(events, values) / self.state["normalizer"]).astype(np.float32)


def prepare_sample_weights(train, val, config, *, date_col="date", group_col=None):
    if config is None:
        return {}, None
    if train.empty or val.empty:
        raise ValueError("Sample weighting requires observed training and validation samples.")
    transformer = SampleWeightTransformer(config, date_col=date_col, group_col=group_col)
    train_weights = transformer.fit_transform(train)
    val_weights = transformer.transform(val)
    state = dict(transformer.state)
    for name, weights in (("train", train_weights), ("validation", val_weights)):
        state[name] = {
            "count": len(weights), "mean": float(weights.mean()),
            "min": float(weights.min()), "max": float(weights.max()),
        }
    return {"sample_weight": train_weights, "sample_weight_val": val_weights}, state
