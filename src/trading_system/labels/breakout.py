from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import pandas as pd

from .schema import LABEL_NAME_TO_ID


def _validate_breakout_parameters(
    window: int,
    buy_buffer: float,
    sell_buffer: float,
    alternating: bool,
) -> tuple[int, float, float, bool]:
    if isinstance(window, (bool, np.bool_)) or not isinstance(
        window, (int, np.integer)
    ):
        raise TypeError("window must be an integer.")
    if window <= 0:
        raise ValueError("window must be positive.")
    for name, value in (("buy_buffer", buy_buffer), ("sell_buffer", sell_buffer)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be numeric.")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1).")
    if not isinstance(alternating, (bool, np.bool_)):
        raise TypeError("alternating must be a boolean.")
    return int(window), float(buy_buffer), float(sell_buffer), bool(alternating)


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
    buy_buffer: float = 0.0,
    sell_buffer: float = 0.0,
    alternating: bool = True,
) -> pd.DataFrame:
    window, buy_buffer, sell_buffer, alternating = _validate_breakout_parameters(
        window,
        buy_buffer,
        sell_buffer,
        alternating,
    )
    missing = [
        column for column in (date_col, price_col) if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing breakout columns: {missing}")
    out = frame.sort_values(date_col).copy()
    prices = pd.to_numeric(out[price_col], errors="coerce")
    if prices.isna().any() or not np.isfinite(prices.to_numpy()).all():
        raise ValueError(f"{price_col} contains invalid values.")
    previous_min = prices.shift(1).rolling(window).min()
    previous_max = prices.shift(1).rolling(window).max()
    raw = np.where(
        prices <= previous_min * (1.0 - buy_buffer),
        "Buy",
        np.where(prices >= previous_max * (1.0 + sell_buffer), "Sell", "Hold"),
    )
    raw_labels = pd.Series(raw, index=out.index, dtype="object")
    raw_labels.loc[previous_min.isna() | previous_max.isna()] = "Hold"
    out["Label"] = (
        enforce_alternating_signals(raw_labels.tolist())
        if alternating
        else raw_labels.tolist()
    )
    out["Label_id"] = out["Label"].map(LABEL_NAME_TO_ID).astype(np.int64)
    return out


def generate_breakout_labels_by_ticker(
    frame: pd.DataFrame,
    window: int,
    *,
    price_col: str = "adj_close",
    group_col: str = "ticker",
    date_col: str = "date",
    buy_buffer: float = 0.0,
    sell_buffer: float = 0.0,
    alternating: bool = True,
) -> pd.DataFrame:
    if group_col not in frame.columns:
        return generate_breakout_labels(
            frame,
            window,
            price_col=price_col,
            date_col=date_col,
            buy_buffer=buy_buffer,
            sell_buffer=sell_buffer,
            alternating=alternating,
        )
    parts = [
        generate_breakout_labels(
            group,
            window,
            price_col=price_col,
            date_col=date_col,
            buy_buffer=buy_buffer,
            sell_buffer=sell_buffer,
            alternating=alternating,
        )
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


def build_breakout_label_result(frame, config, context):
    """Registry adapter for the M0 breakout baseline."""

    from .config import LabelConfig
    from .registry import LabelContext, LabelResult

    if not isinstance(config, LabelConfig) or config.method != "breakout":
        raise ValueError("Breakout builder requires method='breakout'.")
    if not isinstance(context, LabelContext):
        raise TypeError("context must be a LabelContext.")
    if config.semantics != "action":
        raise ValueError("Breakout labels require semantics='action'.")
    expected = {"window", "buy_buffer", "sell_buffer", "alternating"}
    extras = sorted(set(config.parameters) - expected)
    if extras:
        raise ValueError(f"Unknown breakout parameters: {extras}.")
    parameters = {
        **LabelConfig.breakout().parameters,
        **config.parameters,
    }
    if context.group_col is None:
        labeled = generate_breakout_labels(
            frame,
            parameters["window"],
            price_col=context.price_col,
            date_col=context.date_col,
            buy_buffer=parameters["buy_buffer"],
            sell_buffer=parameters["sell_buffer"],
            alternating=parameters["alternating"],
        )
    else:
        if context.group_col not in frame.columns:
            raise ValueError(f"Missing breakout group column: {context.group_col}")
        labeled = generate_breakout_labels_by_ticker(
            frame,
            parameters["window"],
            price_col=context.price_col,
            group_col=context.group_col,
            date_col=context.date_col,
            buy_buffer=parameters["buy_buffer"],
            sell_buffer=parameters["sell_buffer"],
            alternating=parameters["alternating"],
        )
    counts = label_statistics(labeled)
    return LabelResult(
        frame=labeled,
        known_mask=np.ones(len(labeled), dtype=bool),
        class_names=("Sell", "Hold", "Buy"),
        semantics="action",
        metadata={
            "method": "breakout",
            "objective": config.objective,
            "semantics": config.semantics,
            "parameters": dict(parameters),
            "class_counts": counts,
            "action_rate": (counts["Buy"] + counts["Sell"]) / len(labeled),
        },
    )


# Compatibility names used by existing scripts and notebooks.
add_labels = generate_breakout_labels


def labelling(frame: pd.DataFrame, window: int, price_col: str = "adj_close"):
    labeled = generate_breakout_labels(frame, window, price_col=price_col)
    return labeled, label_statistics(labeled)


def labelling_all(frame: pd.DataFrame, window: int, price_col: str = "adj_close"):
    return generate_breakout_labels_by_ticker(frame, window, price_col=price_col)


__all__ = [
    "add_labels",
    "build_breakout_label_result",
    "enforce_alternating_signals",
    "generate_breakout_labels",
    "generate_breakout_labels_by_ticker",
    "label_statistics",
    "labelling",
    "labelling_all",
]
