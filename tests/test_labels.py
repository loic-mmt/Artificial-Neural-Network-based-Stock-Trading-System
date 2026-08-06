import pandas as pd

from trading_system.labels.breakout import (
    enforce_alternating_signals,
    generate_breakout_labels,
)
from trading_system.labels.forward_return import build_forward_return_labels
from trading_system.labels.oracle_dp import build_oracle_labels_train_only


def test_alternating_breakout_actions():
    assert enforce_alternating_signals(["Buy", "Buy", "Hold", "Sell", "Sell"]) == [
        "Buy",
        "Hold",
        "Hold",
        "Sell",
        "Hold",
    ]
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=7),
            "adj_close": [10.0, 9.0, 8.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    labeled = generate_breakout_labels(frame, 2)
    assert set(labeled["Label_id"].unique()).issubset({0, 1, 2})


def test_forward_return_tail_is_hold():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "adj_close": [10.0, 11.0, 10.0, 12.0, 12.0],
        }
    )
    labeled, report = build_forward_return_labels(
        frame,
        horizon=2,
        buy_threshold=0.01,
        sell_threshold=0.01,
    )
    assert labeled["Label_id"].iloc[-2:].tolist() == [1, 1]
    assert report["n_rows"] == 5


def test_oracle_dp_and_backtest_agree():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8),
            "adj_close": [100.0, 102.0, 99.0, 105.0, 101.0, 108.0, 103.0, 110.0],
        }
    )
    _, report = build_oracle_labels_train_only(frame, fee_per_trade=0.5)
    assert report["dp_eval_abs_gap"] < 1e-8
