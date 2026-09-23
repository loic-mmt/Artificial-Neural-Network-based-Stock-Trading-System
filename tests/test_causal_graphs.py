"""Graph source dates and edge semantics must survive future data changes."""

import numpy as np
import pandas as pd
import pytest

from trading_system.data.causal_graphs import GraphBuildConfig, build_graph_snapshots


def _prices():
    dates = pd.date_range("2020-01-01", periods=12, tz="UTC")
    a = np.array([100, 101, 103, 102, 105, 108, 107, 109, 111, 110, 114, 116], float)
    b = a * 2
    c = np.array([90, 92, 91, 94, 95, 93, 96, 94, 97, 96, 98, 99], float)
    return pd.DataFrame([
        {"date": date, "ticker": ticker, "adj_close": price,
         "sector": "industrial" if ticker in ("A", "B") else "energy"}
        for date, values in zip(dates, zip(a, b, c))
        for ticker, price in zip(("A", "B", "C"), values)
    ])


def _build(frame, mode, sessions):
    return build_graph_snapshots(
        frame, tickers=("A", "B", "C"), prediction_sessions=sessions,
        training_end="2020-01-09T00:00:00Z",
        config=GraphBuildConfig(mode, lookback=4, threshold=.7),
    )


def test_sector_and_identity_controls_use_independent_node_slots():
    frame = _prices()
    sessions = ["2020-01-07", "2020-01-10"]
    assert _build(frame, "identity", sessions) == ()
    graphs = _build(frame, "sector", sessions)
    assert len(graphs) == 2
    np.testing.assert_array_equal(graphs[0].edge_index, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(graphs[0].edge_weight, [1, 1])


def test_pearson_graphs_are_causal_and_static_graph_stays_frozen():
    frame = _prices()
    sessions = ["2020-01-07", "2020-01-10"]
    static = _build(frame, "train_pearson", sessions)
    rolling = _build(frame, "rolling_pearson", sessions)
    assert static[0].source_end == static[1].source_end
    assert rolling[0].source_end < rolling[1].source_end
    np.testing.assert_array_equal(static[0].edge_index, static[1].edge_index)
    later = frame.copy()
    later.loc[later.date.ge("2020-01-10"), "adj_close"] *= [1, 5, 2, 1, 3, 4, 2, 1, 5]
    before = _build(later, "rolling_pearson", ["2020-01-07"])[0]
    np.testing.assert_array_equal(rolling[0].edge_index, before.edge_index)
    np.testing.assert_allclose(rolling[0].edge_weight, before.edge_weight)
    assert static[0].source_start == pd.Timestamp("2020-01-01T00:00:00Z")
    assert rolling[0].source_start == pd.Timestamp("2020-01-03T00:00:00Z")
    assert static[0].source_start < static[0].source_end < static[0].session


def test_graph_builder_rejects_missing_history_and_future_prediction():
    frame = _prices()
    with pytest.raises(ValueError, match="calibration prefix"):
        _build(frame, "train_pearson", ["2020-01-03"])
    with pytest.raises(ValueError, match="Prediction sessions"):
        _build(frame, "sector", ["2020-01-15"])
