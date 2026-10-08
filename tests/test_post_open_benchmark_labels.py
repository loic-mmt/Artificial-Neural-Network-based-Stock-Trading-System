import numpy as np
import pandas as pd
import pytest

from trading_system.labels.post_open_benchmark import build_post_open_labels


def quotes(n=50, ticker="A"):
    rng = np.random.default_rng(43)
    close = 100 * np.exp(np.cumsum(rng.normal(0.005, 0.02, n)))
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "ticker": ticker,
                         "open": close * 0.99, "close": close, "adj_close": close})


def test_intraday_uses_raw_prices_unknown_is_not_flat_and_rows_are_aligned():
    data = quotes(4).assign(open=[100., 100., 100., np.nan], close=[101., 99., 100., 100.],
                            adj_close=[50., 49.5, 50., 50.]).iloc[[3, 1, 0, 2]]
    result = build_post_open_labels(data, "intraday-return")
    assert result.index.tolist() == data.index.tolist()
    assert result.Label.tolist() == ["Flat", "Short", "Long", "Flat"]
    assert result._label_known.tolist() == [False, True, True, True]
    np.testing.assert_allclose(result.label_score.iloc[1:], [-0.01, 0.01, 0.])
    assert result.label_score.iloc[0] != result.label_score.iloc[0]
    assert result.attrs["post_open_label_contract"]["horizon_sessions"] == 1


def test_horizon_counts_j_through_j_plus_nine_and_adjustment_is_coherent():
    data = quotes(12).assign(open=200., close=200., adj_close=100.)
    data.loc[9, "adj_close"] = 105.
    result = build_post_open_labels(data, "forward_return", horizon=10)
    assert result.fwd_ret.iloc[0] == pytest.approx(0.05)
    assert result.label_end_date.iloc[0] == pd.Timestamp(data.date.iloc[9], tz="UTC")
    assert result._label_known.tolist() == [True, True, True] + [False] * 9
    assert result.Label.iloc[0] == "Long"
    explicit = build_post_open_labels(data.assign(adj_open_target=100.), "forward_return", horizon=10)
    assert explicit.fwd_ret.iloc[0] == result.fwd_ret.iloc[0]


def test_forward_threshold_neutral_is_flat_not_previous_position():
    data = quotes(4).assign(open=100., close=[101., 100.1, 99.9, 99.], adj_close=[101., 100.1, 99.9, 99.])
    result = build_post_open_labels(data, "forward_return", horizon=1)
    assert result.target_position.tolist() == [1, 0, 0, -1]
    assert result.Label.tolist() == ["Long", "Flat", "Flat", "Short"]


@pytest.mark.parametrize("method", ["intraday_return", "forward_return", "volatility_position"])
def test_partition_boundary_blocks_future_and_prior_state(method):
    data = quotes(60)
    start, end = data.date.iloc[25], data.date.iloc[44]
    result = build_post_open_labels(data, method, horizon=4, volatility_window=5,
                                    partition_start=start, partition_end=end)
    assert not result._label_known.iloc[:25].any()
    assert not result._label_known.iloc[45:].any()
    last = 44 if method == "intraday_return" else 41
    assert not result._label_known.iloc[last + 1:].any()
    assert (result.loc[result._label_known, "label_end_date"] <= pd.Timestamp(end, tz="UTC")).all()
    changed = data.copy()
    changed.loc[45:, ["open", "close", "adj_close"]] *= 100
    changed_result = build_post_open_labels(changed, method, horizon=4, volatility_window=5,
                                            partition_start=start, partition_end=end)
    pd.testing.assert_series_equal(result.Label_id, changed_result.Label_id)
    np.testing.assert_allclose(result.label_score, changed_result.label_score, equal_nan=True)


def test_historical_volatility_stops_at_previous_close():
    data = quotes(50)
    baseline = build_post_open_labels(data, "volatility_position", horizon=3, volatility_window=5)
    changed = data.copy()
    changed.loc[20:, ["close", "adj_close"]] *= 2
    observed = build_post_open_labels(changed, "volatility_position", horizon=3, volatility_window=5)
    assert baseline.historical_volatility.iloc[20] == observed.historical_volatility.iloc[20]
    assert baseline.historical_volatility.iloc[21] != observed.historical_volatility.iloc[21]
    assert not baseline._label_known.iloc[:6].any()


def test_volatility_state_resets_at_partition_start():
    data = quotes(20).assign(open=100., close=np.linspace(100., 120., 20), adj_close=np.linspace(100., 120., 20))
    # Extremely long minimum holding would retain an early Long forever if the
    # historical split state were accidentally inherited.
    data.loc[12:, "open"] = data.loc[12:, "close"] * 10
    full = build_post_open_labels(data, "volatility_position", horizon=1, volatility_window=2,
                                  min_holding_period=100)
    split = build_post_open_labels(data, "volatility_position", horizon=1, volatility_window=2,
                                   min_holding_period=100, partition_start=data.date.iloc[12])
    assert full.target_position.iloc[12] == 1
    assert split.target_position.iloc[12] == -1


def test_global_calendar_hole_does_not_shorten_ticker_horizon():
    a = quotes(8)
    b = quotes(8, "B").drop(index=2)
    data = pd.concat([a, b], ignore_index=True).sample(frac=1, random_state=2)
    result = build_post_open_labels(data, "forward_return", horizon=3)
    bresult = result[result.ticker == "B"].sort_values("date")
    assert not bresult._label_known.iloc[:2].any()
    assert bresult.label_end_date.iloc[0] == pd.Timestamp(a.date.iloc[2], tz="UTC")
    assert bresult._label_known.iloc[2]


def test_duplicates_and_invalid_parameters_rejected():
    data = quotes(10)
    with pytest.raises(ValueError, match="unique"):
        build_post_open_labels(pd.concat([data, data.iloc[:1]]), "forward_return")
    with pytest.raises(ValueError, match="horizon"):
        build_post_open_labels(data, "forward_return", horizon=0)
    with pytest.raises(ValueError, match="partition_start"):
        build_post_open_labels(data, "forward_return", partition_start="2021-01-01", partition_end="2020-01-01")
