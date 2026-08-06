import numpy as np
import pandas as pd

from trading_system.backtest.engine import (
    evaluate_strategy_vs_buy_hold,
    run_label_backtest,
)


def test_prediction_executes_next_bar():
    result = run_label_backtest(
        np.asarray([100.0, 110.0, 121.0]),
        np.asarray([2, 1, 1]),
        initial_capital=100.0,
        position_mode="long_only",
        execution_delay=1,
    )
    assert result["executed_positions"].tolist() == [0.0, 1.0, 1.0]
    assert np.isclose(result["model_curve"][-1], 110.0)


def test_long_short_flip_counts_two_units_of_turnover():
    result = run_label_backtest(
        np.asarray([100.0, 100.0, 100.0, 100.0]),
        np.asarray([2, 0, 1, 1]),
        initial_capital=100.0,
        fee_per_trade=1.0,
        position_mode="long_short",
        execution_delay=1,
    )
    assert result["turnover"].tolist() == [0.0, 1.0, 2.0, 0.0]
    assert result["model_curve"][-1] == 97.0


def test_grouped_backtest_allocates_total_capital_once():
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "adj_close": [100.0, 110.0, 100.0, 90.0],
        }
    )
    result = evaluate_strategy_vs_buy_hold(
        frame,
        np.asarray([2, 1, 2, 1]),
        initial_capital=100.0,
        position_mode="long_only",
        group_col="ticker",
    )
    assert result["initial_capital"] == 100.0
