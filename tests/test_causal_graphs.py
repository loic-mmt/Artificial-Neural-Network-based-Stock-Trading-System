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


def test_topk_graph_has_neighbors_and_honors_rebalance_schedule():
    frame = _prices()
    graphs = build_graph_snapshots(
        frame,
        tickers=("A", "B", "C"),
        prediction_sessions=["2020-01-07", "2020-01-08", "2020-01-10"],
        training_end="2020-01-09T00:00:00Z",
        config=GraphBuildConfig(
            "rolling_topk", lookback=4, weight_mode="absolute",
            neighbors=1, rebalance_bars=2,
        ),
    )
    for graph in graphs:
        assert np.bincount(graph.edge_index[1], minlength=3).min() >= 1
        edges = set(map(tuple, graph.edge_index.T))
        assert all((destination, source) in edges for source, destination in edges)
    assert graphs[1].source_end == graphs[0].source_end
    assert graphs[2].source_end > graphs[1].source_end


def test_residual_topk_uses_only_causal_market_and_sector_context():
    frame = _prices()
    dates = pd.date_range("2020-01-01", periods=12, tz="UTC")
    context = pd.DataFrame({
        "date": dates,
        "spy_close": 100 + np.arange(12) + np.sin(np.arange(12)),
        "industrial_close": 80 + .7 * np.arange(12) + np.cos(np.arange(12)),
        "energy_close": 70 + .4 * np.arange(12) + np.sin(np.arange(12) / 2),
    })
    options = dict(
        tickers=("A", "B", "C"), prediction_sessions=["2020-01-07"],
        training_end="2020-01-09T00:00:00Z",
        config=GraphBuildConfig(
            "rolling_residual_topk", lookback=4, weight_mode="absolute", neighbors=1,
        ),
        context_frame=context,
        sector_context_columns={
            "industrial": "industrial_close", "energy": "energy_close",
        },
    )
    original = build_graph_snapshots(frame, **options)[0]
    future = context.copy()
    future.loc[future.date.ge("2020-01-08"), "spy_close"] *= 10
    unchanged = build_graph_snapshots(frame, **{**options, "context_frame": future})[0]
    np.testing.assert_array_equal(original.edge_index, unchanged.edge_index)
    np.testing.assert_allclose(original.edge_weight, unchanged.edge_weight)
    assert np.bincount(original.edge_index[1], minlength=3).min() == 1
