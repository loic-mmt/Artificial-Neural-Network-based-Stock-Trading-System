from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

try:
    import talib
except ImportError:  # pragma: no cover - fallback supports lightweight environments
    talib = None


TECHNICAL_FEATURE_COLUMNS = [
    "open_ret",
    "high_ret",
    "low_ret",
    "close_ret",
    "adj_close_ret",
    "volume_ret",
    "rsi",
    "macd",
    "williams",
    "range_log",
    "body_log",
    "upper_wick_log",
    "lower_wick_log",
    "volume_relatif",
    "volatility_10",
]


def compute_returns(
    frame: pd.DataFrame,
    cols: Sequence[str] | None = None,
    *,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Append log returns, optionally calculated independently per group."""

    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    out = frame.copy()
    columns = (
        list(cols)
        if cols is not None
        else ["open", "high", "low", "close", "adj_close", "volume"]
    )

    def one(series: pd.Series, column: str) -> pd.Series:
        if column == "volume":
            logged = np.log1p(series.clip(lower=0))
            return logged.diff()
        safe = series.clip(lower=1e-12)
        return np.log(safe / safe.shift(1))

    for column in columns:
        if column not in out.columns:
            continue
        target = "volume_ret" if column == "volume" else f"{column}_ret"
        if group_col is not None and group_col in out.columns:
            out[target] = out.groupby(group_col, sort=False, dropna=False)[
                column
            ].transform(lambda series, current=column: one(series, current))
        else:
            out[target] = one(out[column], column)
    return out


def normalize_prices(
    frame: pd.DataFrame,
    cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Append log-price columns."""

    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    out = frame.copy()
    columns = (
        list(cols)
        if cols is not None
        else ["open", "high", "low", "close", "adj_close"]
    )
    for column in columns:
        if column in out.columns:
            out[f"log_{column}"] = np.log(out[column].clip(lower=1e-12))
    return out


def _fallback_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = (
        delta.clip(lower=0)
        .ewm(alpha=1 / window, adjust=False, min_periods=window)
        .mean()
    )
    losses = (
        (-delta.clip(upper=0))
        .ewm(alpha=1 / window, adjust=False, min_periods=window)
        .mean()
    )
    relative_strength = gains / losses.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _compute_one(frame: pd.DataFrame, *, date_col: str) -> pd.DataFrame:
    work = frame.sort_values(date_col).copy()
    required = ["open", "high", "low", "close", "adj_close", "volume"]
    missing = [column for column in required if column not in work.columns]
    if missing:
        raise ValueError(f"Missing technical feature columns: {missing}")
    work = compute_returns(work, required)
    work = normalize_prices(work, ["open", "high", "low", "close", "adj_close"])

    if talib is not None:
        macd, _, _ = talib.MACD(work["log_adj_close"].to_numpy(dtype=float))
        work["rsi"] = talib.RSI(work["log_adj_close"].to_numpy(dtype=float))
        work["macd"] = macd
        work["williams"] = talib.WILLR(
            work["log_high"].to_numpy(dtype=float),
            work["log_low"].to_numpy(dtype=float),
            work["log_close"].to_numpy(dtype=float),
        )
    else:
        ema_fast = (
            work["log_adj_close"].ewm(span=12, adjust=False, min_periods=12).mean()
        )
        ema_slow = (
            work["log_adj_close"].ewm(span=26, adjust=False, min_periods=26).mean()
        )
        work["macd"] = ema_fast - ema_slow
        work["rsi"] = _fallback_rsi(work["log_adj_close"])
        rolling_high = work["log_high"].rolling(14).max()
        rolling_low = work["log_low"].rolling(14).min()
        work["williams"] = -100.0 * (
            (rolling_high - work["log_close"])
            / (rolling_high - rolling_low).replace(0.0, np.nan)
        )

    work["range_log"] = work["log_high"] - work["log_low"]
    work["body_log"] = work["log_close"] - work["log_open"]
    work["upper_wick_log"] = work["log_high"] - np.maximum(
        work["log_open"], work["log_close"]
    )
    work["lower_wick_log"] = (
        np.minimum(work["log_open"], work["log_close"]) - work["log_low"]
    )
    work["volume_relatif"] = work["volume"] / work["volume"].rolling(10).mean()
    work["volatility_10"] = work["adj_close_ret"].rolling(10).std()
    return work


def compute_technical_features(
    frame: pd.DataFrame,
    *,
    group_col: str | None = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    """Build compact technical features without mixing symbols."""

    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    if date_col not in frame.columns:
        raise ValueError(f"Missing date column: {date_col}")
    if group_col is not None and group_col in frame.columns:
        parts = [
            _compute_one(group, date_col=date_col)
            for _, group in frame.groupby(group_col, sort=False, dropna=False)
        ]
        return (
            pd.concat(parts, ignore_index=True)
            .sort_values([group_col, date_col])
            .reset_index(drop=True)
        )
    return (
        _compute_one(frame, date_col=date_col)
        .sort_values(date_col)
        .reset_index(drop=True)
    )


compute_features = compute_technical_features


__all__ = [
    "TECHNICAL_FEATURE_COLUMNS",
    "compute_features",
    "compute_returns",
    "compute_technical_features",
    "normalize_prices",
]
