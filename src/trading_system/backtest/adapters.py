from __future__ import annotations

import numpy as np
import pandas as pd

from .positions import labels_to_positions, target_positions_to_actions


def coerce_utc_datetime_index(
    frame: pd.DataFrame, date_col: str = "date"
) -> pd.DataFrame:
    if isinstance(frame.index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True, errors="coerce"))
    elif date_col in frame.columns:
        index = pd.DatetimeIndex(
            pd.to_datetime(frame[date_col], utc=True, errors="coerce")
        )
    else:
        raise ValueError(f"Cannot build datetime index: missing {date_col}.")
    if index.isna().any() or index.has_duplicates:
        raise ValueError("Backtest timestamps must be valid and unique.")
    out = frame.copy()
    out.index = index
    return out.sort_index()


def prepare_advanced_backtest_inputs(
    test_frame: pd.DataFrame,
    predicted_labels: np.ndarray,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(test_frame) != len(predicted_labels):
        raise ValueError("Prediction count does not match test rows.")
    market = coerce_utc_datetime_index(test_frame, date_col=date_col)
    if "close" not in market.columns and "adj_close" in market.columns:
        market["close"] = market["adj_close"]
    if "close" not in market.columns:
        raise ValueError("Backtest frame requires close or adj_close.")
    close = pd.to_numeric(market["close"], errors="coerce")
    if close.isna().any():
        raise ValueError("close contains invalid values.")
    market["close"] = close.astype(float)
    market["open"] = (
        pd.to_numeric(market["open"], errors="coerce").astype(float)
        if "open" in market.columns
        else market["close"]
    )
    market["high"] = (
        pd.to_numeric(market["high"], errors="coerce").astype(float)
        if "high" in market.columns
        else market[["open", "close"]].max(axis=1)
    )
    market["low"] = (
        pd.to_numeric(market["low"], errors="coerce").astype(float)
        if "low" in market.columns
        else market[["open", "close"]].min(axis=1)
    )
    market["volume"] = (
        pd.to_numeric(market["volume"], errors="coerce").fillna(0.0).astype(float)
        if "volume" in market.columns
        else 0.0
    )
    target = labels_to_positions(predicted_labels, position_mode="long_short").astype(
        np.int8
    )
    labels = pd.DataFrame(
        {
            "target_position": target,
            "action": target_positions_to_actions(target),
            "model_label_id": np.asarray(predicted_labels, dtype=np.int64),
        },
        index=market.index,
    )
    return market, labels


__all__ = ["coerce_utc_datetime_index", "prepare_advanced_backtest_inputs"]
