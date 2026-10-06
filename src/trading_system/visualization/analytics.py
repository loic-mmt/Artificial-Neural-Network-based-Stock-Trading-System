"""Display-window statistics, separate from the recorded engine metrics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .schemas import RunData


def _slice_dates(frame: pd.DataFrame, start, end, columns: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    column = next((name for name in columns if name in frame), None)
    if column is None:
        return frame.copy()
    dates = pd.to_datetime(frame[column], utc=True, errors="coerce")
    keep = dates.notna()
    if start is not None:
        boundary = pd.Timestamp(start)
        boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
        keep &= dates >= boundary
    if end is not None:
        boundary = pd.Timestamp(end)
        boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
        keep &= dates < boundary + pd.Timedelta(days=1)
    return frame.loc[keep].copy().sort_values(column).reset_index(drop=True)


def slice_run(data: RunData, start=None, end=None) -> RunData:
    """Use the first displayed equity as reference, without inventing a prior bar.

    Saved metrics and saved per-bar returns are retained independently. The
    display return/drawdown describe transitions inside the visible window.
    Trades are filtered by exit date, positions by observation date.
    """
    equity = _slice_dates(data.equity, start, end, ("date", "timestamp"))
    if "equity" in equity and not equity.empty:
        values = pd.to_numeric(equity["equity"], errors="coerce")
        if "drawdown" in equity:
            equity["source_drawdown"] = equity["drawdown"]
        if "net_return" in equity:
            equity["source_net_return"] = equity["net_return"]
        equity["drawdown"] = values / values.cummax() - 1
        equity["net_return"] = values.pct_change(fill_method=None).fillna(0.0)
    return replace(
        data,
        equity=equity,
        positions=_slice_dates(data.positions, start, end, ("date", "timestamp")),
        trades=_slice_dates(data.trades, start, end, ("exit_time", "exit_date", "date", "timestamp")),
        market=_slice_dates(data.market, start, end, ("date", "timestamp")),
        orders=_slice_dates(data.orders, start, end, ("timestamp", "date")),
    )


def window_statistics(data: RunData) -> dict[str, float | int]:
    """Return first-to-last observed performance, never a replacement Sharpe."""
    if data.equity.empty or "equity" not in data.equity:
        return {}
    values = pd.to_numeric(data.equity["equity"], errors="coerce").dropna()
    if values.empty or not np.isfinite(values).all() or (values <= 0).any():
        return {}
    output: dict[str, float | int] = {
        "observed_return": float(values.iloc[-1] / values.iloc[0] - 1),
        "window_drawdown": float((values / values.cummax() - 1).min()),
        "last_equity": float(values.iloc[-1]),
        "points": len(values),
    }
    if "benchmark_equity" in data.equity:
        benchmark = pd.to_numeric(data.equity["benchmark_equity"], errors="coerce").dropna()
        if benchmark.index.equals(values.index) and np.isfinite(benchmark).all() and (benchmark > 0).all():
            output["benchmark_return"] = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)
            output["excess_return"] = output["observed_return"] - output["benchmark_return"]
    if "gross_exposure" in data.equity:
        exposure = pd.to_numeric(data.equity["gross_exposure"], errors="coerce")
        if exposure.notna().any():
            output["mean_gross_exposure"] = float(exposure.mean())
    return output
