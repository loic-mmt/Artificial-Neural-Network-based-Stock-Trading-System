"""Post-open features must not read the day's unfinished high/low/close."""

import numpy as np
import pandas as pd

from trading_system.data.post_open import build_post_open_frame
from trading_system.experiments.post_open_pilot import execution_sensitivity
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel


def test_post_open_features_ignore_unfinished_bar():
    dates = pd.date_range("2020-01-01", periods=85, freq="B")
    base = pd.DataFrame({
        "date": dates, "ticker": "A", "open": 100 + np.arange(len(dates), dtype=float),
        "high": 101 + np.arange(len(dates), dtype=float),
        "low": 99 + np.arange(len(dates), dtype=float),
        "close": 100.5 + np.arange(len(dates), dtype=float),
        "adj_close": 100.5 + np.arange(len(dates), dtype=float),
        "volume": 1000 + np.arange(len(dates), dtype=float),
    })
    original, columns = build_post_open_frame(base, groups=("technical",))
    changed = base.copy()
    changed.loc[changed.index[-1], ["high", "low", "close", "adj_close", "volume"]] = [500, 1, 300, 300, 999999]
    updated, _ = build_post_open_frame(changed, groups=("technical",))
    assert original.loc[original.index[-1], "night_pct"] == updated.loc[updated.index[-1], "night_pct"]
    pd.testing.assert_series_equal(original.loc[original.index[-1], list(columns)],
                                   updated.loc[updated.index[-1], list(columns)])
    assert original.loc[original.index[-1], "completed_bar_date"] == dates[-2]


def test_post_open_gap_uses_known_open_and_prior_close():
    dates = pd.date_range("2020-01-01", periods=85, freq="B")
    frame = pd.DataFrame({"date": dates, "ticker": "A", "open": 101., "high": 102.,
                          "low": 99., "close": 100., "adj_close": 100., "volume": 1000.})
    output, _ = build_post_open_frame(frame, groups=("technical",))
    assert np.isclose(output.iloc[-1]["night_pct"], 0.01)


def test_explicit_same_session_panel_earns_open_to_next_open():
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4),
                          "adj_open_target": [100., 110., 121., 133.1]})
    panel = ReturnPanel(frame, price_col="adj_open_target", execution_delay=0,
                        allow_same_session=True)
    net, executed, _, _, _ = panel.path(np.array([1., 0., 0., 0.]),
                                         FinancialLossConfig("sharpe", cost_bps=0))
    assert np.allclose(net, [0.1, 0., 0.])
    assert np.allclose(executed, [[1., 0., 0.]])


def test_ohlc_stress_bounds_open_proxy_for_long_trade():
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4),
                          "ticker": ["A"] * 4, "open": [100., 110., 120., 130.],
                          "high": [105., 115., 125., 135.],
                          "low": [95., 105., 115., 125.],
                          "adj_open_target": [100., 110., 120., 130.]})
    panel = ReturnPanel(frame, price_col="adj_open_target", group_col="ticker",
                        execution_delay=0, allow_same_session=True)
    report = execution_sensitivity(panel, frame, np.array([1., 1., 1., 1.]),
                                   FinancialLossConfig("sharpe", cost_bps=0), draws=10, seed=1)
    cases = report["deterministic"]
    assert cases["high_low_worst"]["net_return"] <= cases["open_proxy"]["net_return"]
    assert cases["open_proxy"]["net_return"] <= cases["high_low_best"]["net_return"]
