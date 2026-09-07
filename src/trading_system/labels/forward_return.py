from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .schema import LABEL_ID_TO_NAME, TradeLabel


def _validate_forward_return_parameters(
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
) -> tuple[int, float, float]:
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(
        horizon, (int, np.integer)
    ):
        raise TypeError("horizon must be an integer.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    for name, value in (
        ("buy_threshold", buy_threshold),
        ("sell_threshold", sell_threshold),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be numeric.")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    return int(horizon), float(buy_threshold), float(sell_threshold)


def build_forward_return_labels(
    frame: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    horizon: int = 1,
    buy_threshold: float = 0.002,
    sell_threshold: float = 0.002,
    date_col: str = "date",
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Build the historical M1 labels for one chronological price series."""

    horizon, buy_threshold, sell_threshold = _validate_forward_return_parameters(
        horizon,
        buy_threshold,
        sell_threshold,
    )
    if frame is None or frame.empty:
        raise ValueError("Cannot label an empty frame.")
    missing = [
        column for column in (date_col, price_col) if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing forward-label columns: {missing}")

    out = frame.sort_values(date_col).copy()
    price = pd.to_numeric(out[price_col], errors="coerce")
    price_values = price.to_numpy(dtype=np.float64)
    if not np.isfinite(price_values).all() or (price_values <= 0.0).any():
        raise ValueError(f"{price_col} contains invalid values.")
    forward_return = (price.shift(-horizon) / price) - 1.0
    label_ids = np.full(len(out), TradeLabel.HOLD.value, dtype=np.int64)
    label_ids[forward_return > buy_threshold] = TradeLabel.BUY.value
    label_ids[forward_return < -sell_threshold] = TradeLabel.SELL.value
    out["fwd_ret"] = forward_return
    out["label_score"] = forward_return
    out["Label_id"] = label_ids
    out["Label"] = out["Label_id"].map(LABEL_ID_TO_NAME)
    known = forward_return.notna()
    report: dict[str, int | float] = {
        "horizon": horizon,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "n_rows": int(len(out)),
        "n_known": int(known.sum()),
        "n_unknown": int((~known).sum()),
        # Historical counts include the unknown tail represented as Hold.
        "n_buy": int((out["Label_id"] == TradeLabel.BUY.value).sum()),
        "n_hold": int((out["Label_id"] == TradeLabel.HOLD.value).sum()),
        "n_sell": int((out["Label_id"] == TradeLabel.SELL.value).sum()),
    }
    return out, report


def build_forward_return_labels_by_ticker(
    frame: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    horizon: int = 1,
    buy_threshold: float = 0.002,
    sell_threshold: float = 0.002,
    group_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    """Build forward-return targets independently for every ticker."""

    if group_col not in frame.columns:
        return build_forward_return_labels(
            frame,
            price_col=price_col,
            horizon=horizon,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            date_col=date_col,
        )[0]
    parts = [
        build_forward_return_labels(
            group,
            price_col=price_col,
            horizon=horizon,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            date_col=date_col,
        )[0]
        for _, group in frame.groupby(group_col, sort=False, dropna=False)
    ]
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values([group_col, date_col])
        .reset_index(drop=True)
    )


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["Label"].value_counts()
    return {name: int(counts.get(name, 0)) for name in ("Buy", "Hold", "Sell")}


def build_forward_return_label_result(frame, config, context):
    """Registry adapter with per-ticker and per-split leakage barriers."""

    from .config import LabelConfig
    from .registry import LabelContext, LabelResult

    if not isinstance(config, LabelConfig) or config.method != "forward_return":
        raise ValueError("Forward-return builder requires method='forward_return'.")
    if not isinstance(context, LabelContext):
        raise TypeError("context must be a LabelContext.")
    if config.semantics != "action":
        raise ValueError("Forward-return labels require semantics='action'.")
    expected = {"horizon", "buy_threshold", "sell_threshold"}
    extras = sorted(set(config.parameters) - expected)
    if extras:
        raise ValueError(f"Unknown forward-return parameters: {extras}.")
    parameters = {
        **LabelConfig.forward_return().parameters,
        **config.parameters,
    }
    if context.group_col is not None and context.group_col not in frame.columns:
        raise ValueError(f"Missing forward-return group column: {context.group_col}")

    segment_columns: list[str] = []
    if context.group_col is not None:
        segment_columns.append(context.group_col)
    if context.split_col is not None and context.split_col in frame.columns:
        segment_columns.append(context.split_col)
    if segment_columns:
        grouper = segment_columns[0] if len(segment_columns) == 1 else segment_columns
        parts = [
            build_forward_return_labels(
                segment,
                price_col=context.price_col,
                date_col=context.date_col,
                **parameters,
            )[0]
            for _, segment in frame.groupby(grouper, sort=False, dropna=False)
        ]
        labeled = pd.concat(parts, ignore_index=True)
        sort_columns = [
            column
            for column in (context.group_col, context.date_col)
            if column is not None
        ]
        labeled = labeled.sort_values(sort_columns).reset_index(drop=True)
    else:
        labeled = build_forward_return_labels(
            frame,
            price_col=context.price_col,
            date_col=context.date_col,
            **parameters,
        )[0].reset_index(drop=True)

    known_mask = labeled["fwd_ret"].notna().to_numpy(dtype=bool, copy=True)
    if "_label_known" in labeled.columns:
        existing = labeled["_label_known"]
        if existing.isna().any():
            raise ValueError("_label_known contains missing values.")
        known_mask &= existing.to_numpy(dtype=bool)
    labeled["_label_known"] = known_mask
    known_counts = _class_counts(labeled.loc[known_mask])
    all_counts = _class_counts(labeled)
    n_known = int(known_mask.sum())
    return LabelResult(
        frame=labeled,
        known_mask=known_mask,
        class_names=("Sell", "Hold", "Buy"),
        semantics="action",
        metadata={
            "method": "forward_return",
            "objective": config.objective,
            "semantics": config.semantics,
            "parameters": dict(parameters),
            "class_counts": known_counts,
            "all_class_counts": all_counts,
            "n_rows": int(len(labeled)),
            "n_known": n_known,
            "n_unknown": int(len(labeled) - n_known),
            "action_rate": (
                (known_counts["Buy"] + known_counts["Sell"]) / n_known
                if n_known
                else 0.0
            ),
        },
    )


__all__ = [
    "build_forward_return_label_result",
    "build_forward_return_labels",
    "build_forward_return_labels_by_ticker",
]
