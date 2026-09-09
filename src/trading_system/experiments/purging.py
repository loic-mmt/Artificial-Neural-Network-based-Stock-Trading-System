"""Apply event-end purging while retaining historical rows as causal context."""

import numpy as np
import pandas as pd

from trading_system.data.purged_cv import purge_intervals
from trading_system.labels.registry import LabelContext, create_default_label_registry


def purge_labeled_splits(source, labeled, config):
    split = config.purged_split
    keys = [config.date_col]
    grouped = config.universe == "multi"
    if grouped:
        keys.insert(0, config.group_col)
    work = labeled.copy()
    if work.duplicated(keys).any():
        raise ValueError("Purged splits require unique dates per asset.")
    # M3 labels are allowed to inspect the *available* inner validation history
    # to discover actual first-touch ends. All overlapping targets are then
    # excluded before any fitted preprocessing or supervised selection.
    if config.label_mode == "triple_barrier":
        full = create_default_label_registry().generate(
            source, config.resolved_label_config(),
            LabelContext(price_col=config.price_col, date_col=config.date_col,
                         group_col=config.group_col if grouped else None, split_col=None),
        ).frame
        full = full.set_index(keys).reindex(work.set_index(keys).index).reset_index()
        train = work["_experiment_split"].eq("train").to_numpy()
        for column in full.columns:
            if column not in keys and column != "_experiment_split":
                work.loc[train, column] = full.loc[train, column].to_numpy()
    dates = pd.to_datetime(work[config.date_col], utc=True)
    ends = dates.copy()
    groups = work.groupby(config.group_col, sort=False).groups.values() if grouped else [work.index]
    for index in groups:
        ordered = work.loc[index].sort_values(config.date_col)
        idx = ordered.index
        current = dates.loc[idx]
        if config.label_mode == "triple_barrier":
            event_ends = pd.to_datetime(ordered.label_end_date, utc=True)
            if config.triple_barrier_between_event_policy == "carry":
                event_ends = event_ends.groupby(ordered["_experiment_split"], sort=False).ffill()
                ends.loc[idx] = event_ends.where(event_ends > current, current)
            else:
                ends.loc[idx] = event_ends.where(ordered.label_event_id.notna(), current)
        elif config.label_mode in ("forward_return", "volatility_position"):
            horizon = config.forward_horizon if config.label_mode == "forward_return" else config.volatility_horizon
            ends.loc[idx] = current.shift(-horizon)
        elif config.label_mode != "breakout":
            raise ValueError("Purged CV supports breakout, forward_return, volatility_position and triple_barrier.")
    work["_information_end"] = ends
    work["_cv_gap"] = False
    reports = {}
    for name, boundary in (("train", split.validation_start), ("val", split.test_start)):
        mask = work["_experiment_split"].eq(name)
        known = mask & work["_label_known"].astype(bool)
        indices = np.flatnonzero(known.to_numpy())
        # A closed boundary interval conservatively excludes equality. The
        # embargo cannot remove past-only training rows in an expanding fold.
        starts_array = pd.DatetimeIndex([*dates, pd.Timestamp(boundary)])
        ends_array = pd.DatetimeIndex([*ends, pd.Timestamp(boundary)])
        keep, report = purge_intervals(starts_array, ends_array, indices,
                                      np.array([len(work)]), embargo_bars=split.embargo_bars)
        work.loc[mask, "_label_known"] = False
        work.loc[keep, "_label_known"] = True
        gap_dates = dates.loc[mask].drop_duplicates().sort_values().tail(split.gap_bars) if split.gap_bars else []
        gap = mask & dates.isin(gap_dates)
        gap_overlap = mask & ends.ge(gap_dates.min()) if len(gap_dates) else gap
        report["gap"] = int(((gap | gap_overlap) & work["_label_known"].astype(bool)).sum())
        work.loc[gap | gap_overlap, "_label_known"] = False
        work.loc[gap, "_cv_gap"] = True
        report["kept_after_gap"] = int(work.loc[mask, "_label_known"].sum())
        reports[name] = report
    work["_fit_eligible"] = work["_label_known"].astype(bool)
    work.attrs["purging"] = reports
    return work
