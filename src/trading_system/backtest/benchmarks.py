from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(prices: np.ndarray) -> np.ndarray:
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("prices must be a non-empty 1D array.")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("prices must be finite and positive.")
    returns = np.zeros(len(values), dtype=np.float64)
    if len(values) > 1:
        returns[:-1] = (values[1:] / values[:-1]) - 1.0
    return returns


def buy_and_hold_curve(
    prices: np.ndarray, initial_capital: float = 10_000.0
) -> np.ndarray:
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    return float(initial_capital) * np.cumprod(1.0 + forward_returns(prices))


def compute_benchmark(
    frame: pd.DataFrame,
    capital: float = 10_000.0,
    *,
    price_col: str = "adj_close",
) -> float:
    if price_col not in frame.columns:
        raise ValueError(f"Missing price column: {price_col}")
    if frame.empty:
        return float(capital)
    curve = buy_and_hold_curve(frame[price_col].to_numpy(dtype=np.float64), capital)
    return float(curve[-1])


def evaluate_buy_hold_only(
    frame: pd.DataFrame,
    initial_capital: float = 10_000.0,
    price_col: str = "adj_close",
) -> dict[str, float]:
    final = compute_benchmark(frame, initial_capital, price_col=price_col)
    return {
        "initial_capital": float(initial_capital),
        "buy_hold_final_capital": final,
        "buy_hold_pnl": final - float(initial_capital),
    }


__all__ = [
    "buy_and_hold_curve",
    "compute_benchmark",
    "evaluate_buy_hold_only",
    "forward_returns",
]
