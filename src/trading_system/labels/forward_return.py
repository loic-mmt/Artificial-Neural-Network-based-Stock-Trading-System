from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import LABEL_ID_TO_NAME, TradeLabel


def build_forward_return_labels(
    frame: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    horizon: int = 1,
    buy_threshold: float = 0.002,
    sell_threshold: float = 0.002,
    date_col: str = "date",
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if buy_threshold < 0 or sell_threshold < 0:
        raise ValueError("buy_threshold and sell_threshold must be non-negative.")
    if frame is None or frame.empty:
        raise ValueError("Cannot label an empty frame.")
    missing = [
        column for column in (date_col, price_col) if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing forward-label columns: {missing}")

    out = frame.sort_values(date_col).copy()
    price = pd.to_numeric(out[price_col], errors="coerce")
    if price.isna().any():
        raise ValueError(f"{price_col} contains invalid values.")
    forward_return = (price.shift(-horizon) / price) - 1.0
    label_ids = np.full(len(out), TradeLabel.HOLD.value, dtype=np.int64)
    label_ids[forward_return > buy_threshold] = TradeLabel.BUY.value
    label_ids[forward_return < -sell_threshold] = TradeLabel.SELL.value
    out["fwd_ret"] = forward_return
    out["Label_id"] = label_ids
    out["Label"] = out["Label_id"].map(LABEL_ID_TO_NAME)
    report: dict[str, int | float] = {
        "horizon": int(horizon),
        "buy_threshold": float(buy_threshold),
        "sell_threshold": float(sell_threshold),
        "n_rows": int(len(out)),
        "n_buy": int((out["Label_id"] == TradeLabel.BUY.value).sum()),
        "n_hold": int((out["Label_id"] == TradeLabel.HOLD.value).sum()),
        "n_sell": int((out["Label_id"] == TradeLabel.SELL.value).sum()),
    }
    return out, report


__all__ = ["build_forward_return_labels"]
