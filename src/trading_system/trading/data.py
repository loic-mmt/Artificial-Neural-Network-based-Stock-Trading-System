"""Strict bar/target contracts; timestamps express information availability."""

import numpy as np
import pandas as pd

from trading_system.backtest.positions import labels_to_positions
from trading_system.training.financial_loss import probabilities_to_positions


def targets_from_probabilities(probabilities, position_mode="long_short"):
    return probabilities_to_positions(probabilities, position_mode)


def targets_from_labels(labels, *, label_semantics, position_mode="long_short"):
    return labels_to_positions(labels, label_semantics=label_semantics, position_mode=position_mode)


def prepare_bars(bars, config):
    required = {"date", "ticker", "open", "high", "low", "close"}
    if not required.issubset(bars):
        raise ValueError(f"Bars lack columns: {sorted(required - set(bars))}.")
    work = bars.copy().reset_index(drop=True)
    if work.empty or work.ticker.isna().any():
        raise ValueError("Bars must be non-empty with non-missing tickers.")
    work["ticker"] = work.ticker.astype(str)
    if work.ticker.str.strip().eq("").any():
        raise ValueError("Empty ticker.")
    work["date"] = pd.to_datetime(work.date, utc=True, errors="raise")
    if work.date.isna().any() or work.duplicated(["date", "ticker"]).any():
        raise ValueError("Each ticker/date must be valid and unique.")
    work = work.sort_values(["date", "ticker"]).reset_index(drop=True)
    tickers = sorted(work.ticker.unique())
    calendar = pd.DatetimeIndex(work.date.unique()).sort_values()
    if len(calendar) < 2 or not work.groupby("date").size().eq(len(tickers)).all():
        raise ValueError("All tickers require identical complete calendars and at least two bars.")
    prices = work[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError("OHLC prices must be finite and positive.")
    if ((prices.low > prices[["open", "close"]].min(axis=1)) |
        (prices.high < prices[["open", "close"]].max(axis=1))).any():
        raise ValueError("Invalid OHLC ordering.")
    work[prices.columns] = prices
    supplied_times = {"open_time", "close_time"}.intersection(work.columns)
    if supplied_times and len(supplied_times) != 2:
        raise ValueError("Supply both open_time and close_time.")
    if supplied_times:
        for name in supplied_times:
            work[name] = pd.to_datetime(work[name], utc=True, errors="raise")
    elif config.bar_mode == "intraday":
        raise ValueError("Intraday bars require explicit open_time and close_time.")
    else:
        if not work.date.eq(work.date.dt.normalize()).all():
            raise ValueError("Daily date must be a midnight session label, or supply explicit bar times.")
        sessions = work.date.dt.strftime("%Y-%m-%d")
        for name, clock in (("open_time", config.session_open), ("close_time", config.session_close)):
            local = pd.to_datetime(sessions + " " + clock).dt.tz_localize(config.timezone, ambiguous="raise", nonexistent="raise")
            work[name] = local.dt.tz_convert("UTC")
    if work[["open_time", "close_time"]].isna().any().any() or (work.open_time >= work.close_time).any():
        raise ValueError("Bar times must be valid, with open_time < close_time.")
    if not work.groupby("date")[["open_time", "close_time"]].nunique().eq(1).all().all():
        raise ValueError("Portfolio bar times must be synchronized.")
    schedule = work.drop_duplicates("date")
    if (schedule.open_time.iloc[1:].to_numpy() < schedule.close_time.iloc[:-1].to_numpy()).any():
        raise ValueError("Bars must not overlap.")
    for name in ("dividends", "stock_splits"):
        if name not in work:
            work[name] = 0.0
        work[name] = pd.to_numeric(work[name], errors="raise")
        if not np.isfinite(work[name]).all() or (work[name] < 0).any():
            raise ValueError(f"{name} must be finite and non-negative.")
    work["sector"] = work["sector"] if "sector" in work else None
    if config.enabled and config.max_sector_weight is not None and work.sector.isna().any():
        raise ValueError("Sector limits require a sector for every bar.")
    # Indicators are in a continuous split-adjusted reference unit. Raw-mode
    # history uses only split actions observed up to each bar, never future ones.
    for _, part in work.groupby("ticker", sort=False):
        units = part.stock_splits.replace(0, 1).cumprod() if config.price_basis == "raw" else pd.Series(1., index=part.index)
        px = part[["open", "high", "low", "close"]].mul(units, axis=0)
        prev = px.close.shift(1)
        tr = pd.concat([px.high - px.low, (px.high - prev).abs(), (px.low - prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(config.atr_window, min_periods=config.atr_window).mean()
        vol = px.close.pct_change().rolling(config.volatility_window, min_periods=config.volatility_window).std(ddof=0) * np.sqrt(config.annualization)
        work.loc[part.index, "atr_open"] = atr.shift(1) / units
        work.loc[part.index, "atr_close"] = atr / units
        work.loc[part.index, "vol_open"] = vol.shift(1)
        work.loc[part.index, "vol_close"] = vol
    return work, tickers, supplied_times != set()


def prepare_targets(bars, targets, config):
    if isinstance(targets, pd.DataFrame):
        name = "target_position" if "target_position" in targets else "position"
        if not {"date", "ticker", name}.issubset(targets):
            raise ValueError("Target frame requires date, ticker, target_position (or position).")
        values = targets[["date", "ticker", name] + (["signal_available_at"] if "signal_available_at" in targets else [])].copy()
        values["date"] = pd.to_datetime(values.date, utc=True, errors="raise")
        values["ticker"] = values.ticker.astype(str)
        values = values.rename(columns={name: "target_position"})
        if values.duplicated(["date", "ticker"]).any() or len(values) != len(bars):
            raise ValueError("Targets must have exactly one row per bar.")
        aligned = bars[["date", "ticker"]].merge(values, on=["date", "ticker"], how="left", validate="one_to_one")
    else:
        # Arrays follow the ORIGINAL input row order; engine converts them to a
        # keyed frame before sorting, so callers cannot silently misalign assets.
        raise TypeError("Internal targets must be a keyed DataFrame.")
    q = pd.to_numeric(aligned.target_position, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(q).all() or (np.abs(q) > 1).any():
        raise ValueError("Targets must be finite, aligned and in [-1, 1].")
    if config.position_mode == "long_only" and (q < 0).any():
        raise ValueError("Negative target in long_only mode.")
    minimum = bars["open_time"] if config.signal_timing == "after_open" else bars["close_time"]
    if "signal_available_at" not in aligned:
        aligned["signal_available_at"] = minimum + pd.Timedelta(1, "ns")
    else:
        aligned["signal_available_at"] = pd.to_datetime(aligned.signal_available_at, utc=True, errors="raise")
        if aligned.signal_available_at.isna().any() or (aligned.signal_available_at <= minimum).any():
            raise ValueError("signal_available_at must follow the declared observation time.")
    return aligned
