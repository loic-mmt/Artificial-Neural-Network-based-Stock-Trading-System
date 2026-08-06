from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .schema import LABEL_NAME_TO_ID


def enforce_alternating_signals(labels: Sequence[str]) -> list[str]:
    """Replace repeated consecutive Buy or Sell actions with Hold."""

    filtered: list[str] = []
    last_action: str | None = None
    for label in labels:
        if label == "Hold":
            filtered.append(label)
        elif label in ("Buy", "Sell"):
            if last_action is None or label != last_action:
                filtered.append(label)
                last_action = label
            else:
                filtered.append("Hold")
        else:
            raise ValueError(f"Unknown label: {label}")
    return filtered


def generate_breakout_labels(
    frame: pd.DataFrame,
    window: int,
    *,
    price_col: str = "adj_close",
    date_col: str = "date",
) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive.")
    missing = [
        column for column in (date_col, price_col) if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing breakout columns: {missing}")
    out = frame.sort_values(date_col).copy()
    prices = pd.to_numeric(out[price_col], errors="coerce")
    if prices.isna().any():
        raise ValueError(f"{price_col} contains invalid values.")
    previous_min = prices.shift(1).rolling(window).min()
    previous_max = prices.shift(1).rolling(window).max()
    raw = np.where(
        prices <= previous_min,
        "Buy",
        np.where(prices >= previous_max, "Sell", "Hold"),
    )
    raw_labels = pd.Series(raw, index=out.index, dtype="object")
    raw_labels.loc[previous_min.isna() | previous_max.isna()] = "Hold"
    out["Label"] = enforce_alternating_signals(raw_labels.tolist())
    out["Label_id"] = out["Label"].map(LABEL_NAME_TO_ID).astype(np.int64)
    return out


def generate_breakout_labels_by_ticker(
    frame: pd.DataFrame,
    window: int,
    *,
    price_col: str = "adj_close",
    group_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    if group_col not in frame.columns:
        return generate_breakout_labels(
            frame, window, price_col=price_col, date_col=date_col
        )
    parts = [
        generate_breakout_labels(group, window, price_col=price_col, date_col=date_col)
        for _, group in frame.groupby(group_col, sort=False, dropna=False)
    ]
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values([group_col, date_col])
        .reset_index(drop=True)
    )


def label_statistics(frame: pd.DataFrame) -> dict[str, int]:
    if "Label" not in frame.columns:
        raise ValueError("Missing Label column.")
    counts = frame["Label"].value_counts()
    return {name: int(counts.get(name, 0)) for name in ("Buy", "Hold", "Sell")}


# Compatibility names used by existing scripts and notebooks.
add_labels = generate_breakout_labels


def labelling(frame: pd.DataFrame, window: int, price_col: str = "adj_close"):
    labeled = generate_breakout_labels(frame, window, price_col=price_col)
    return labeled, label_statistics(labeled)


def labelling_all(frame: pd.DataFrame, window: int, price_col: str = "adj_close"):
    return generate_breakout_labels_by_ticker(frame, window, price_col=price_col)


__all__ = [
    "add_labels",
    "enforce_alternating_signals",
    "generate_breakout_labels",
    "generate_breakout_labels_by_ticker",
    "label_statistics",
    "labelling",
    "labelling_all",
]
