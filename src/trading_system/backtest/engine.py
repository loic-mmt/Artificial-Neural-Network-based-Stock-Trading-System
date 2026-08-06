from __future__ import annotations

import numpy as np
import pandas as pd

from .benchmarks import buy_and_hold_curve, forward_returns
from .positions import (
    PositionMode,
    apply_execution_delay,
    labels_to_positions,
    position_turnover,
)


def run_label_backtest(
    prices: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    initial_capital: float = 10_000.0,
    fee_per_trade: float = 0.0,
    position_mode: PositionMode = "long_short",
    execution_delay: int = 1,
) -> dict[str, object]:
    """Backtest labels with explicit timing and turnover semantics."""

    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if fee_per_trade < 0:
        raise ValueError("fee_per_trade must be non-negative.")
    values = np.asarray(prices, dtype=np.float64)
    labels = np.asarray(predicted_labels, dtype=np.int64)
    if len(values) != len(labels):
        raise ValueError("Prediction count does not match price count.")
    if len(values) < 2:
        raise ValueError("At least two prices are required for a backtest.")

    returns = forward_returns(values)
    targets = labels_to_positions(labels, position_mode=position_mode)
    executed = apply_execution_delay(targets, execution_delay)
    turnover = position_turnover(executed)
    strategy_returns = executed * returns
    model_curve = np.empty(len(values), dtype=np.float64)
    capital = float(initial_capital)
    for index, strategy_return in enumerate(strategy_returns):
        capital *= 1.0 + float(strategy_return)
        capital -= float(fee_per_trade) * float(turnover[index])
        capital = max(capital, 0.0)
        model_curve[index] = capital
    benchmark_curve = buy_and_hold_curve(values, initial_capital)
    return {
        "model_curve": model_curve,
        "buy_hold_curve": benchmark_curve,
        "forward_returns": returns,
        "target_positions": targets,
        "executed_positions": executed,
        "turnover": turnover,
        "strategy_returns": strategy_returns,
    }


def _summarize_backtest(
    result: dict[str, object], initial_capital: float
) -> dict[str, float]:
    model_curve = np.asarray(result["model_curve"], dtype=np.float64)
    benchmark_curve = np.asarray(result["buy_hold_curve"], dtype=np.float64)
    model_final = float(model_curve[-1])
    benchmark_final = float(benchmark_curve[-1])
    return {
        "initial_capital": float(initial_capital),
        "model_final_capital": model_final,
        "buy_hold_final_capital": benchmark_final,
        "model_pnl": model_final - float(initial_capital),
        "buy_hold_pnl": benchmark_final - float(initial_capital),
        "outperformance": model_final - benchmark_final,
    }


def evaluate_strategy_vs_buy_hold(
    test_frame: pd.DataFrame,
    predicted_labels: np.ndarray,
    initial_capital: float = 10_000.0,
    price_col: str = "adj_close",
    fee_per_trade: float = 0.0,
    position_mode: PositionMode = "long_short",
    execution_delay: int = 1,
    *,
    group_col: str | None = None,
    date_col: str = "date",
) -> dict[str, float]:
    """Evaluate one series or an equal-capital grouped portfolio."""

    labels = np.asarray(predicted_labels, dtype=np.int64)
    if len(test_frame) != len(labels):
        raise ValueError("Prediction count does not match test rows.")
    if group_col is None or group_col not in test_frame.columns:
        result = run_label_backtest(
            test_frame[price_col].to_numpy(dtype=np.float64),
            labels,
            initial_capital=initial_capital,
            fee_per_trade=fee_per_trade,
            position_mode=position_mode,
            execution_delay=execution_delay,
        )
        return _summarize_backtest(result, initial_capital)

    work = test_frame.reset_index(drop=True).copy()
    work["_prediction_index"] = np.arange(len(work), dtype=np.int64)
    groups = [
        group
        for _, group in work.groupby(group_col, sort=False, dropna=False)
        if len(group) >= 2
    ]
    if not groups:
        raise ValueError("No group has at least two test rows.")
    capital_per_group = float(initial_capital) / len(groups)
    model_total = 0.0
    benchmark_total = 0.0
    for group in groups:
        group = group.sort_values(date_col)
        group_labels = labels[group["_prediction_index"].to_numpy(dtype=np.int64)]
        result = run_label_backtest(
            group[price_col].to_numpy(dtype=np.float64),
            group_labels,
            initial_capital=capital_per_group,
            fee_per_trade=fee_per_trade,
            position_mode=position_mode,
            execution_delay=execution_delay,
        )
        model_total += float(np.asarray(result["model_curve"])[-1])
        benchmark_total += float(np.asarray(result["buy_hold_curve"])[-1])
    return {
        "initial_capital": float(initial_capital),
        "model_final_capital": model_total,
        "buy_hold_final_capital": benchmark_total,
        "model_pnl": model_total - float(initial_capital),
        "buy_hold_pnl": benchmark_total - float(initial_capital),
        "outperformance": model_total - benchmark_total,
    }


def run_backtest_from_labels(*args, **kwargs):
    from .lib import run_backtest_from_labels as implementation

    return implementation(*args, **kwargs)


def execute_first_check_pipeline(*args, **kwargs):
    from .lib import execute_first_check_pipeline as implementation

    return implementation(*args, **kwargs)


def execute_first_check_pipeline_external(*args, **kwargs):
    from .lib import execute_first_check_pipeline_external as implementation

    return implementation(*args, **kwargs)


def __getattr__(name: str):
    if name == "BacktestConfig":
        from .lib import BacktestConfig

        return BacktestConfig
    raise AttributeError(name)


__all__ = [
    "execute_first_check_pipeline",
    "execute_first_check_pipeline_external",
    "evaluate_strategy_vs_buy_hold",
    "run_label_backtest",
    "run_backtest_from_labels",
]
