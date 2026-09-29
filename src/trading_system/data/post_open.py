"""Build a daily post-open snapshot without using the unfinished session bar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.features.expanded import compute_expanded_features, feature_columns


def build_post_open_frame(frame: pd.DataFrame, *, groups: tuple[str, ...],
                          date_col: str = "date", group_col: str = "ticker") -> tuple[pd.DataFrame, tuple[str, ...]]:
    """At session J retain completed J-1 features plus J open and overnight gap.

    The J open is deliberately not supplied as an absolute price input: the
    causal gap relative to J-1 close carries its new information without scale.
    Adjusted open is an evaluation target only, never a model input.
    """
    required = {date_col, group_col, "open", "high", "low", "close", "adj_close"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Post-open frame lacks columns: {sorted(missing)}")
    work = frame.sort_values([group_col, date_col]).copy()
    if work.duplicated([group_col, date_col]).any():
        raise ValueError("Post-open frame requires unique ticker/session rows.")
    columns = feature_columns(groups)
    completed = compute_expanded_features(work, group_col=group_col, date_col=date_col)
    completed = completed.sort_values([group_col, date_col]).reset_index(drop=True)
    prior = completed.groupby(group_col, sort=False)
    lagged = prior[list(columns)].shift(1)
    lagged.columns = [f"lag1_{name}" for name in columns]
    raw_close_previous = prior["close"].shift(1)
    raw_open = pd.to_numeric(completed["open"], errors="coerce")
    gap = raw_open / raw_close_previous - 1.0
    if "stock_splits" in completed:
        split = pd.to_numeric(completed["stock_splits"], errors="coerce").fillna(0.0)
        gap = gap.mask(split.ne(0))
    adjustment = pd.to_numeric(completed["adj_close"], errors="coerce") / pd.to_numeric(
        completed["close"], errors="coerce")
    result = pd.concat([
        completed[[date_col, group_col, "open", "high", "low", "close", "adj_close"]],
        pd.DataFrame({"adj_open_target": raw_open * adjustment,
                      "night_pct": gap.replace([np.inf, -np.inf], np.nan),
                      "completed_bar_date": prior[date_col].shift(1).to_numpy()}),
        lagged,
    ], axis=1).copy()
    result = result.dropna(subset=["completed_bar_date", "adj_open_target"])
    if (result["adj_open_target"] <= 0).any():
        raise ValueError("Adjusted open target must be positive.")
    return result.reset_index(drop=True), tuple(lagged.columns)
