"""Calendar folds and closed-interval purging, independent of model training."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedSplit:
    validation_start: str
    test_start: str
    gap_bars: int = 0
    embargo_bars: int = 0

    def __post_init__(self):
        for name in ("validation_start", "test_start"):
            value = pd.to_datetime(getattr(self, name), utc=True, errors="raise")
            if pd.isna(value):
                raise ValueError(f"{name} must be a valid timestamp.")
            object.__setattr__(self, name, value.isoformat())
        if pd.Timestamp(self.validation_start) >= pd.Timestamp(self.test_start):
            raise ValueError("validation_start must precede test_start.")
        for name in ("gap_bars", "embargo_bars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

    def split(self, frame, date_col="date"):
        dates = pd.to_datetime(frame[date_col], utc=True, errors="raise")
        if dates.isna().any():
            raise ValueError("Split dates cannot be missing.")
        val, test = pd.Timestamp(self.validation_start), pd.Timestamp(self.test_start)
        return tuple(frame.loc[mask].copy() for mask in (dates < val, (dates >= val) & (dates < test), dates >= test))


def purge_intervals(starts, ends, train_indices, validation_indices, *, embargo_bars=0):
    """Exclude train intervals intersecting any validation interval, across assets.

    Intervals are closed: a training event ending exactly at validation entry is
    excluded. Embargo counts unique dates after the latest validation event end,
    not ticker rows. NaT ends are unobserved and never eligible for training.
    """
    if isinstance(embargo_bars, bool) or not isinstance(embargo_bars, int) or embargo_bars < 0:
        raise ValueError("embargo_bars must be a non-negative integer.")
    starts = pd.DatetimeIndex(pd.to_datetime(starts, utc=True, errors="raise")).as_unit("ns")
    ends = pd.DatetimeIndex(pd.to_datetime(ends, utc=True, errors="raise")).as_unit("ns")
    if len(starts) != len(ends) or starts.isna().any():
        raise ValueError("Interval starts and ends must align; starts cannot be missing.")
    if ((ends < starts) & ~ends.isna()).any():
        raise ValueError("Interval ends cannot precede starts.")
    def indices(values):
        values = np.asarray(values)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Interval indices must be a 1D integer array.")
        if (values < 0).any() or (values >= len(starts)).any() or len(np.unique(values)) != len(values):
            raise ValueError("Interval indices must be unique and in range.")
        return values
    train, validation = indices(train_indices), indices(validation_indices)
    if not len(validation) or ends[validation].isna().any():
        raise ValueError("Validation requires observed interval ends.")
    if np.intersect1d(train, validation).size:
        raise ValueError("Training and validation indices must be disjoint.")
    # Merge validation intervals once; binary search avoids a quadratic N x M mask.
    merged = []
    for start, end in sorted(zip(starts[validation], ends[validation])):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    left = pd.DatetimeIndex([item[0] for item in merged]).asi8
    right = pd.DatetimeIndex([item[1] for item in merged]).asi8
    candidate = np.searchsorted(left, ends[train].asi8, side="right") - 1
    overlap = (candidate >= 0) & (right[np.maximum(candidate, 0)] >= starts[train].asi8)
    unknown = ends[train].isna()
    embargo_dates = starts.unique().sort_values()
    embargo_dates = embargo_dates[embargo_dates > ends[validation].max()][:embargo_bars]
    embargo = starts[train].isin(embargo_dates)
    keep = ~(unknown | overlap | embargo)
    return train[keep], {"candidates": len(train), "kept": int(keep.sum()),
                         "unknown": int(unknown.sum()), "overlap": int((overlap & ~unknown).sum()),
                         "embargo": int((embargo & ~overlap & ~unknown).sum())}


def expanding_calendar_folds(frame, *, n_splits=3, initial_train_fraction=.5,
                             inner_val_fraction=.2, final_test_fraction=.15,
                             gap_bars=0, embargo_bars=0, date_col="date"):
    """Reserve final dates first; split development into expanding outer folds."""
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2.")
    for value in (initial_train_fraction, inner_val_fraction, final_test_fraction):
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError("CV fractions must be finite and between zero and one.")
    dates = pd.DatetimeIndex(pd.to_datetime(frame[date_col], utc=True, errors="raise")).unique().sort_values()
    if dates.isna().any():
        raise ValueError("CV dates cannot be missing.")
    dev_end = int(len(dates) * (1 - final_test_fraction))
    initial = int(dev_end * initial_train_fraction)
    if dev_end >= len(dates) or initial < 4 or dev_end - initial < 2 * n_splits:
        raise ValueError("Insufficient dates for requested CV folds and final holdout.")
    blocks = np.array_split(np.arange(initial, dev_end), n_splits)
    folds = []
    for fold_id, block in enumerate(blocks):
        outer_start = int(block[0])
        inner_start = int(outer_start * (1 - inner_val_fraction))
        split = PurgedSplit(dates[inner_start].isoformat(), dates[outer_start].isoformat(), gap_bars, embargo_bars)
        if inner_start <= gap_bars + 1 or outer_start - inner_start <= gap_bars + 1:
            raise ValueError("Gap leaves insufficient inner training or validation dates.")
        folds.append({"fold": fold_id, "split": split, "end": dates[int(block[-1])].isoformat()})
    final_split = PurgedSplit(dates[int(dev_end * (1 - inner_val_fraction))].isoformat(),
                              dates[dev_end].isoformat(), gap_bars, embargo_bars)
    return folds, final_split
