"""Explicit post-open targets for the label/loss benchmark.

These contracts deliberately do not change the historical label implementations.
The horizon counts global sessions: H=10 means open J to close J+9. Historical
volatility stops at close J-1, while labels are retrospective supervision only.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .schema import POSITION_ID_TO_NAME
from .volatility_position import build_persistent_positions


def adjusted_open_prices(frame: pd.DataFrame) -> np.ndarray:
    """Return coherent adjusted opens without filling absent quotations.

    ``adj_open_target`` is an explicit execution/target price, never an input
    feature. Otherwise use the same-day adjustment ratio adj_close/raw close.
    Raw open is sufficient when the dataset has no adjusted-close column.
    """
    if "adj_open_target" in frame:
        return pd.to_numeric(frame["adj_open_target"], errors="coerce").to_numpy(dtype=float)
    opening = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    if "adj_close" not in frame:
        return opening
    raw_close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    adjusted_close = pd.to_numeric(frame["adj_close"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return opening * adjusted_close / raw_close


def _session_bound(value):
    if value is None:
        return None
    return pd.to_datetime(value, utc=True).normalize()


def build_post_open_labels(
    frame: pd.DataFrame,
    method: str,
    *,
    horizon: int = 10,
    volatility_window: int = 20,
    forward_threshold: float = 0.002,
    long_threshold: float = 1.0,
    short_threshold: float = 1.5,
    exit_threshold: float = 0.25,
    min_holding_period: int = 5,
    cost_bps: float = 5.0,
    partition_start=None,
    partition_end=None,
) -> pd.DataFrame:
    """Label rows in original order, with explicit Short/Flat/Long targets.

    Rows outside the supplied inclusive partition, missing prices, incomplete
    horizons, and unknown volatility warmup are not supervised. The neutral
    class ID on an unknown row is only a placeholder; ``_label_known`` is the
    authoritative distinction. Volatility state starts flat at partition_start.
    Missing global sessions are not compressed into a shorter ticker calendar.
    """
    method = method.replace("-", "_")
    if method not in {"intraday_return", "forward_return", "volatility_position"}:
        raise ValueError("method must be intraday_return, forward_return, or volatility_position.")
    for name, value in (("horizon", horizon), ("volatility_window", volatility_window)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name, value in (("forward_threshold", forward_threshold), ("cost_bps", cost_bps)):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if cost_bps >= 10000:
        raise ValueError("cost_bps must be less than 10000.")
    missing = [column for column in ("date", "ticker", "open", "close") if column not in frame]
    if missing or frame.empty:
        raise ValueError(f"A nonempty frame with date/ticker/open/close is required; missing={missing}.")
    out = frame.copy()
    dates = pd.to_datetime(out["date"], utc=True, errors="raise").dt.normalize()
    if dates.isna().any() or out["ticker"].isna().any():
        raise ValueError("Session dates and tickers cannot be missing.")
    keys = pd.DataFrame({"date": dates.to_numpy(), "ticker": out["ticker"].to_numpy()})
    if keys.duplicated().any():
        raise ValueError("Post-open labels require unique session dates per ticker.")
    start, end = _session_bound(partition_start), _session_bound(partition_end)
    if start is not None and end is not None and start > end:
        raise ValueError("partition_start cannot follow partition_end.")
    calendar = pd.DatetimeIndex(dates.unique()).sort_values()
    size = len(out)
    scores = np.full(size, np.nan)
    forward_returns = np.full(size, np.nan)
    volatilities = np.full(size, np.nan)
    known_values = np.zeros(size, dtype=bool)
    target_values = np.zeros(size, dtype=np.int8)
    label_end = pd.Series(pd.NaT, index=range(size), dtype="datetime64[ns, UTC]")
    work = out.reset_index(drop=True).copy()
    work["date"] = dates.to_numpy()
    work["_post_open_row"] = np.arange(size)
    for _, part in work.groupby("ticker", sort=False):
        aligned = part.set_index("date").reindex(calendar)
        row_map = aligned["_post_open_row"].to_numpy()
        present = np.isfinite(row_map)
        opening = pd.to_numeric(aligned["open"], errors="coerce")
        raw_close = pd.to_numeric(aligned["close"], errors="coerce")
        in_partition = np.ones(len(calendar), dtype=bool)
        if start is not None:
            in_partition &= calendar >= start
        if end is not None:
            in_partition &= calendar <= end
        if method == "intraday_return":
            with np.errstate(divide="ignore", invalid="ignore"):
                future = raw_close / opening - 1.0
            known = (present & in_partition & np.isfinite(future.to_numpy())
                     & (opening.to_numpy() > 0) & (raw_close.to_numpy() > 0))
            score = future.to_numpy()
            position = np.sign(np.where(known, score, 0)).astype(np.int8)
            dependencies = pd.Series(calendar, index=calendar)
            historical = np.full(len(calendar), np.nan)
        else:
            adjusted_open = adjusted_open_prices(aligned)
            close = pd.to_numeric(aligned.get("adj_close", raw_close), errors="coerce")
            terminal = close.shift(-(horizon - 1))
            dependencies = pd.Series(calendar, index=calendar).shift(-(horizon - 1))
            with np.errstate(divide="ignore", invalid="ignore"):
                future = terminal / adjusted_open - 1.0
            # An absent quotation inside H sessions invalidates the horizon;
            # neither endpoint arithmetic nor pct_change may bridge a hole.
            valid_bar = present & np.isfinite(close.to_numpy()) & (close.to_numpy() > 0)
            complete = (pd.Series(valid_bar, index=calendar).rolling(horizon, min_periods=horizon)
                        .sum().shift(-(horizon - 1)).eq(horizon).to_numpy())
            known = (present & in_partition & complete & np.isfinite(future.to_numpy())
                     & np.isfinite(adjusted_open) & (adjusted_open > 0) & (terminal.to_numpy() > 0))
            if end is not None:
                known &= (dependencies <= end).to_numpy()
            historical = (close.pct_change(fill_method=None)
                          .rolling(volatility_window, min_periods=volatility_window)
                          .std(ddof=0).shift(1).to_numpy())
            score = future.to_numpy()
            if method == "forward_return":
                position = np.zeros(len(calendar), dtype=np.int8)
                position[known & (score > forward_threshold)] = 1
                position[known & (score < -forward_threshold)] = -1
            else:
                known &= np.isfinite(historical)
                net_magnitude = np.maximum(np.abs(score) - cost_bps / 10000, 0)
                denominator = np.maximum(historical * math.sqrt(horizon), np.finfo(float).eps)
                score = np.sign(score) * net_magnitude / denominator
                score[~known] = np.nan
                # Before partition_start all known flags are false: the reused
                # state machine starts flat rather than inheriting another split.
                position = build_persistent_positions(
                    score, known_mask=known, long_threshold=long_threshold,
                    short_threshold=short_threshold, exit_threshold=exit_threshold,
                    min_holding_period=min_holding_period, position_mode="long_short",
                )
        if "_label_known" in aligned:
            existing = aligned["_label_known"].fillna(False).to_numpy(dtype=bool)
            if method == "volatility_position" and not np.all(existing[present]):
                # Rebuild persistence after supervision exclusions, rather than
                # letting a hidden excluded label choose the next visible state.
                known &= existing
                score[~known] = np.nan
                position = build_persistent_positions(
                    score, known_mask=known, long_threshold=long_threshold,
                    short_threshold=short_threshold, exit_threshold=exit_threshold,
                    min_holding_period=min_holding_period, position_mode="long_short",
                )
            else:
                known &= existing
        rows = row_map[present].astype(int)
        scores[rows] = np.where(known, score, np.nan)[present]
        forward_returns[rows] = np.where(known, future.to_numpy(), np.nan)[present]
        volatilities[rows] = historical[present]
        known_values[rows] = known[present]
        target_values[rows] = np.where(known, position, 0)[present]
        label_end.iloc[rows] = dependencies.to_numpy()[present]
    out["date"] = dates.to_numpy()
    out["fwd_ret"] = forward_returns
    out["label_score"] = scores
    out["historical_volatility"] = volatilities
    out["_label_known"] = known_values
    out["target_position"] = target_values
    out["Label_id"] = target_values.astype(np.int64) + 1
    out["Label"] = out["Label_id"].map(POSITION_ID_TO_NAME)
    out["label_end_date"] = label_end.to_numpy()
    out.attrs["post_open_label_contract"] = {
        "method": method, "semantics": "target_position", "neutral_policy": "flat",
        "horizon_sessions": 1 if method == "intraday_return" else int(horizon),
        "horizon_convention": "J_through_J_plus_H_minus_1_inclusive",
        "volatility_information_cutoff": "close_J_minus_1",
        "partition_start": None if start is None else start.isoformat(),
        "partition_end": None if end is None else end.isoformat(),
    }
    return out
