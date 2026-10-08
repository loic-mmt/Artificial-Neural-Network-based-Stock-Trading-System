import numpy as np
import pandas as pd
import pytest

from trading_system.training.financial_loss import FinancialLossConfig
from trading_system.training.post_open_panel import PostOpenReturnPanel


def quotes(n=10, ticker="A"):
    rng = np.random.default_rng(19)
    opening = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.02, n)))
    close = opening * np.exp(rng.normal(0.001, 0.01, n))
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "ticker": ticker,
                         "open": opening, "close": close, "adj_close": close})


@pytest.mark.parametrize("protocol", ["intraday", "overnight"])
@pytest.mark.parametrize("objective", ["pnl", "sharpe", "combined"])
def test_exact_global_gradient_with_costs_and_signal_subsets(protocol, objective):
    data = pd.concat([quotes(8), quotes(8, "B").assign(open=lambda f: f.open * 1.1)], ignore_index=True)
    signals = data.drop(index=[2, 5, 11]).sample(frac=1, random_state=8)
    panel = PostOpenReturnPanel(data, protocol=protocol, tickers=["B", "A", "C"], signal_frame=signals)
    config = FinancialLossConfig(objective, cost_bps=17, combined_pnl_weight=0.25)
    positions = np.random.default_rng(4).uniform(-0.8, 0.8, len(signals))
    loss, gradient = panel.loss_and_gradient(positions, config)
    numerical = np.zeros_like(positions)
    for i in range(len(positions)):
        step = np.zeros_like(positions)
        step[i] = 1e-6
        numerical[i] = (panel.loss_and_gradient(positions + step, config)[0]
                        - panel.loss_and_gradient(positions - step, config)[0]) / 2e-6
    assert np.isfinite(loss)
    np.testing.assert_allclose(gradient, numerical, atol=1e-7, rtol=2e-5)
    assert panel.tickers == ("B", "A", "C")
    assert np.all(panel.indices[2] == -1)
    assert np.all(panel.path(positions, config)[1][2] == 0)


def test_intraday_pays_entry_and_close_exit_for_each_day_even_unchanged_side():
    data = quotes(3).assign(open=100., close=110., adj_close=55.)
    panel = PostOpenReturnPanel(data, protocol="intraday")
    net, executed, _, turnover, costs = panel.path(np.ones(3), FinancialLossConfig(cost_bps=100))
    np.testing.assert_allclose(net, [0.08] * 3)
    np.testing.assert_array_equal(executed, [[1, 1, 1]])
    np.testing.assert_array_equal(turnover, [[2, 2, 2]])
    np.testing.assert_allclose(costs, [[0.02] * 3])


def test_overnight_open_returns_rebalance_and_terminal_liquidation():
    data = quotes(4).assign(adj_open_target=[100., 110., 99., 99.])
    panel = PostOpenReturnPanel(data, protocol="overnight")
    net, executed, delta, turnover, _ = panel.path([1., -1., -1., 1.], FinancialLossConfig(cost_bps=100))
    np.testing.assert_allclose(net, [0.09, 0.08, 0., -0.01])
    np.testing.assert_array_equal(executed, [[1, -1, -1, 0]])
    np.testing.assert_array_equal(delta, [[1, -2, 0, 1]])
    np.testing.assert_array_equal(turnover, [[1, 2, 0, 1]])
    _, gradient = panel.loss_and_gradient([1., -1., -1., 1.], FinancialLossConfig())
    assert gradient[-1] == 0
    assert panel.metadata["overnight_terminal_day"] == "zero_return_liquidation_only"


def test_missing_signal_is_cash_with_actual_exit_cost_not_calendar_compression():
    data = quotes(4).assign(adj_open_target=[100., 110., 121., 133.1])
    signals = data.drop(index=1).iloc[[2, 0, 1]]
    panel = PostOpenReturnPanel(data, protocol="overnight", signal_frame=signals)
    net, executed, _, turnover, _ = panel.path([1., 1., 1.], FinancialLossConfig(cost_bps=100))
    np.testing.assert_array_equal(panel.indices, [[1, -1, 2, 0]])
    np.testing.assert_allclose(net, [0.09, -0.01, 0.09, -0.01])
    np.testing.assert_array_equal(executed, [[1, 0, 1, 0]])
    np.testing.assert_array_equal(turnover, [[1, 1, 1, 1]])
    assert len(net) == 4


def test_leading_trailing_absences_use_fixed_slots_and_last_quoted_open_liquidation():
    a = quotes(5).assign(adj_open_target=[100., 110., 121., 133.1, 146.41])
    b = a.iloc[1:4].assign(ticker="B")
    data = pd.concat([a, b], ignore_index=True)
    panel = PostOpenReturnPanel(data, protocol="overnight", tickers=["A", "B"])
    net, executed, _, turnover, _ = panel.path(np.ones(len(data)), FinancialLossConfig(cost_bps=100))
    np.testing.assert_array_equal(executed[1], [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(turnover[1], [0, 1, 0, 1, 0])
    assert net[0] == pytest.approx(0.045)  # second ticker's slot is cash
    metrics = panel.metrics(np.ones(len(data)), FinancialLossConfig())
    assert metrics["assets"] == 2
    assert metrics["mean_cash_allocation"] > 0


@pytest.mark.parametrize("protocol", ["intraday", "overnight"])
def test_internal_quote_holes_are_rejected_including_explicit_calendar(protocol):
    data = quotes(6)
    with pytest.raises(ValueError, match="internal missing"):
        PostOpenReturnPanel(data.drop(index=2), protocol=protocol, calendar=data.date)
    other = data.assign(ticker="B")
    with pytest.raises(ValueError, match="internal missing"):
        PostOpenReturnPanel(pd.concat([data.drop(index=2), other]), protocol=protocol)


def test_invalid_dates_signal_keys_prices_and_flat_metrics():
    data = quotes(5)
    with pytest.raises(ValueError, match="unique"):
        PostOpenReturnPanel(pd.concat([data, data.iloc[:1]]), protocol="intraday")
    with pytest.raises(ValueError, match="positive"):
        PostOpenReturnPanel(data.assign(open=0), protocol="intraday")
    with pytest.raises(ValueError, match="Signal keys"):
        PostOpenReturnPanel(data, protocol="intraday", signal_frame=data.assign(ticker="UNKNOWN"))
    panel = PostOpenReturnPanel(data, protocol="intraday")
    loss, gradient = panel.loss_and_gradient(np.zeros(5), FinancialLossConfig("sharpe"))
    assert loss == 0
    assert np.isfinite(gradient).all()
    assert panel.metrics(np.zeros(5), FinancialLossConfig())["net_sharpe"] == 0
    assert panel.metrics(np.zeros(5), FinancialLossConfig())["mean_cash_allocation"] == 1
