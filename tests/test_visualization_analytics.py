from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.visualization.analytics import slice_run, window_statistics
from trading_system.visualization.schemas import RunData, RunRecord


def test_window_references_first_visible_point_and_preserves_engine_metrics():
    record = RunRecord("example", "Example", "trading", Path("example"), metrics={"net_return": -0.25})
    data = RunData(record, equity=pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=4, tz="UTC"),
        "equity": [200.0, 100.0, 80.0, 120.0],
        "drawdown": [0.0, -0.5, -0.6, -0.4],
        "net_return": [0.0, -0.5, -0.2, 0.5],
    }))
    visible = slice_run(data, "2025-01-02", "2025-01-04")
    stats = window_statistics(visible)
    assert np.isclose(stats["observed_return"], 0.2)
    assert np.isclose(stats["window_drawdown"], -0.2)
    assert np.isclose(visible.equity.drawdown.iloc[-1], 0.0)
    assert visible.equity.source_drawdown.iloc[-1] == -0.4
    assert visible.equity.net_return.iloc[0] == 0.0
    assert record.metrics["net_return"] == -0.25
    assert len(data.equity) == 4


def test_date_end_includes_whole_utc_day_and_trades_use_exit_date():
    record = RunRecord("one", "One", "trading", Path("one"))
    data = RunData(record,
        equity=pd.DataFrame({"date": pd.to_datetime(["2025-01-01T23:30Z", "2025-01-02T00:00Z"]), "equity": [100, 110]}),
        trades=pd.DataFrame({"exit_time": pd.to_datetime(["2025-01-01T20:00Z", "2025-01-02T20:00Z"]), "net_pnl": [1, 2]}))
    visible = slice_run(data, "2025-01-01", "2025-01-01")
    assert len(visible.equity) == 1
    assert visible.trades.net_pnl.tolist() == [1]


def test_metrics_only_and_empty_window_are_supported():
    data = RunData(RunRecord("one", "One", "benchmark", Path("one")))
    assert window_statistics(slice_run(data)) == {}


def test_single_point_has_zero_observed_return_without_inventing_initial_capital():
    data = RunData(RunRecord("one", "One", "mt5", Path("one"), metadata={"initial_capital": 100}),
        equity=pd.DataFrame({"date": pd.to_datetime(["2025-01-01"], utc=True), "equity": [90]}))
    assert window_statistics(slice_run(data))["observed_return"] == 0.0


def test_buy_hold_return_uses_same_visible_observations():
    data = RunData(RunRecord("one", "One", "mt5", Path("one")),
        equity=pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3, tz="UTC"),
                             "equity": [100., 90., 108.], "benchmark_equity": [100., 105., 110.]}))
    stats = window_statistics(slice_run(data, "2025-01-02", "2025-01-03"))
    assert np.isclose(stats["benchmark_return"], 110 / 105 - 1)
    assert np.isclose(stats["excess_return"], .2 - (110 / 105 - 1))
    data.equity.loc[1, "benchmark_equity"] = np.nan
    assert "benchmark_return" not in window_statistics(data)


def test_market_and_executed_orders_follow_window_without_mutating_source():
    dates = pd.date_range("2025-01-01", periods=3, tz="UTC")
    data = RunData(RunRecord("one", "One", "trading", Path("one")),
        market=pd.DataFrame({"date": dates, "close": [10, 11, 12]}),
        orders=pd.DataFrame({"timestamp": dates, "quantity": [1, -1, 1]}))
    visible = slice_run(data, "2025-01-02", "2025-01-02")
    assert visible.market.close.tolist() == [11]
    assert visible.orders.quantity.tolist() == [-1]
    assert len(data.market) == len(data.orders) == 3
