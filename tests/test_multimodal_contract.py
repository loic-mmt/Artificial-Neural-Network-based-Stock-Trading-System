import numpy as np
import pandas as pd
import pytest

from trading_system.data.multimodal import (
    GraphSnapshot,
    build_multimodal_dataset,
    validate_calendar_partitions,
)
from trading_system.models.multimodal_contract import BranchOutput


def prepared_frames():
    history = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01"],
            "ticker": ["A", "B"],
            "signal": [10.0, 20.0],
            "node_signal": [100.0, 200.0],
            "market_signal": [0.1, 0.1],
        }
    )
    # Deliberately shuffled: neither ticker nor date input order defines a slot.
    target = pd.DataFrame(
        {
            "date": ["2020-01-04", "2020-01-03", "2020-01-02", "2020-01-02"],
            "ticker": ["B", "A", "B", "A"],
            "signal": [24.0, 13.0, 22.0, 12.0],
            "node_signal": [204.0, 103.0, 202.0, 102.0],
            "market_signal": [0.4, 0.3, 0.2, 0.2],
            "Label_id": [2, 1, 1, 0],
            "_label_known": [True, False, True, True],
        }
    )
    return target, history


def build(target=None, history=None, **overrides):
    default_target, default_history = prepared_frames()
    options = {
        "tickers": ["B", "A"],
        "context_len": 2,
        "temporal_columns": ["signal"],
        "node_columns": ["node_signal"],
        "market_columns": ["market_signal"],
        "history_frame": default_history if history is None else history,
    }
    options.update(overrides)
    return build_multimodal_dataset(
        default_target if target is None else target, **options
    )


def test_date_ticker_alignment_masks_and_backtest_rows():
    target, _ = prepared_frames()
    graph = GraphSnapshot(
        session="2020-01-02",
        source_end="2020-01-02T16:30:00Z",
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_weight=np.array([0.7, 0.7]),
    )
    dataset = build(graphs=[graph])
    batch = dataset.batch([0, 1, 2])

    assert len(dataset) == 3
    assert batch.tickers == ("B", "A")
    assert tuple(str(day.date()) for day in batch.sessions) == (
        "2020-01-02", "2020-01-03", "2020-01-04"
    )
    assert batch.temporal.shape == (3, 2, 2, 1)
    assert batch.node.shape == (3, 2, 1)
    assert batch.market.shape == (3, 1)
    assert batch.sentiment.shape == (3, 2, 0)
    np.testing.assert_array_equal(batch.asset_mask, [[1, 1], [0, 1], [1, 0]])
    np.testing.assert_array_equal(batch.temporal_mask, batch.asset_mask)
    np.testing.assert_array_equal(batch.node_mask, batch.asset_mask)
    np.testing.assert_array_equal(batch.graph_mask, [[1, 1], [0, 0], [0, 0]])
    np.testing.assert_array_equal(batch.market_mask, [1, 1, 1])
    np.testing.assert_array_equal(batch.sentiment_mask, np.zeros((3, 2), bool))
    np.testing.assert_array_equal(batch.label_mask, [[1, 1], [0, 0], [1, 0]])
    np.testing.assert_array_equal(batch.labels, [[1, 0], [-1, -1], [2, -1]])
    np.testing.assert_array_equal(batch.row_positions, [[2, 3], [-1, 1], [0, -1]])
    np.testing.assert_array_equal(batch.temporal[0, 0, :, 0], [20, 22])
    np.testing.assert_array_equal(batch.temporal[0, 1, :, 0], [10, 12])
    np.testing.assert_array_equal(batch.temporal[1, 1, :, 0], [12, 13])
    np.testing.assert_array_equal(batch.temporal[2, 0, :, 0], [22, 24])
    assert batch.graphs == (graph, None, None)
    for date_index, ticker_index in zip(*np.where(batch.asset_mask)):
        row = target.iloc[batch.row_positions[date_index, ticker_index]]
        assert row.ticker == batch.tickers[ticker_index]
        assert pd.Timestamp(row.date, tz="UTC") == batch.sessions[date_index]


def test_order_independence_and_on_demand_batching():
    target, history = prepared_frames()
    reordered = target.sample(frac=1, random_state=7).reset_index(drop=True)
    first = build(target=target, history=history).batch([0, 1, 2])
    second = build(target=reordered, history=history).batch([0, 1, 2])
    np.testing.assert_array_equal(first.temporal, second.temporal)
    np.testing.assert_array_equal(first.node, second.node)
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.asset_mask, second.asset_mask)
    batches = list(build().iter_batches(2))
    assert [len(batch.sessions) for batch in batches] == [2, 1]
    with pytest.raises(ValueError, match="positive integer"):
        list(build().iter_batches(0))


def test_separate_market_sequence_uses_its_own_history_and_cutoffs():
    market = pd.DataFrame({
        "date": ["2020-01-03", "2020-01-01", "2020-01-02"],
        "macro_signal": [3.0, 1.0, 2.0],
        "vix_signal": [30.0, 10.0, 20.0],
        "available_at": [
            "2020-01-02T12:00:00Z", "2019-12-31T12:00:00Z",
            "2020-01-01T12:00:00Z",
        ],
        "source_end": [
            "2020-01-03T16:00:00Z", "2020-01-01T16:00:00Z",
            "2020-01-02T16:00:00Z",
        ],
    })
    options = dict(
        market_frame=market,
        market_context_len=2,
        market_publication_columns=["macro_signal"],
        market_close_columns=["vix_signal"],
    )
    batch = build(**options).batch([0, 1, 2])
    assert batch.market_sequence.shape == (3, 2, 2)
    np.testing.assert_array_equal(batch.market_sequence_mask, [1, 1, 0])
    np.testing.assert_array_equal(batch.market_sequence[0], [[1, 10], [2, 20]])
    np.testing.assert_array_equal(batch.market_sequence[1], [[2, 20], [3, 30]])
    assert not batch.market_sequence[2].any()
    reordered = build(**{**options, "market_frame": market.iloc[::-1]}).batch([0, 1, 2])
    np.testing.assert_array_equal(batch.market_sequence, reordered.market_sequence)

    late_macro = market.copy()
    late_macro.loc[late_macro.date.eq("2020-01-02"), "available_at"] = "2020-01-02T00:00:00Z"
    with pytest.raises(ValueError, match="future information"):
        build(**{**options, "market_frame": late_macro})
    late_close = market.copy()
    late_close.loc[late_close.date.eq("2020-01-02"), "source_end"] = "2020-01-03T00:00:00Z"
    with pytest.raises(ValueError, match="future information"):
        build(**{**options, "market_frame": late_close})
    with pytest.raises(ValueError, match="must be separate"):
        build(**{**options, "temporal_columns": ["macro_signal"]})


def test_global_calendar_partitions_and_history_are_strict():
    target, history = prepared_frames()
    validate_calendar_partitions(history, target)
    overlap = history.assign(date="2020-01-02")
    with pytest.raises(ValueError, match="overlap"):
        validate_calendar_partitions(overlap, target)
    with pytest.raises(ValueError, match="overlap"):
        build(history=overlap)
    future = history.assign(date="2020-01-05")
    with pytest.raises(ValueError, match="precede"):
        build(history=future)


def test_duplicate_keys_and_unknown_tickers_are_rejected():
    target, _ = prepared_frames()
    duplicate = pd.concat([target, target.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate session/ticker"):
        build(target=duplicate)
    with pytest.raises(ValueError, match="outside selection"):
        build(target=target.assign(ticker=["B", "A", "C", "A"]))
    same_session = target.copy()
    same_session.loc[1, "date"] = "2020-01-02T18:00:00Z"
    with pytest.raises(ValueError, match="duplicate session/ticker"):
        build(target=same_session)


def test_sentiment_coverage_is_distinct_from_neutral_score():
    sentiment = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-02", "2020-01-04"],
            "ticker": ["B", "A", "B"],
            "sentiment_score": [0.0, 0.0, 0.0],
            "news_count": [0.0, 0.0, 0.0],
            "source_available": [True, False, False],
            "available_at": ["2020-01-01T20:00:00Z", None, None],
        }
    )
    batch = build(
        sentiment_frame=sentiment,
        sentiment_columns=["sentiment_score", "news_count"],
    ).batch([0, 1, 2])
    np.testing.assert_array_equal(batch.sentiment_mask, [[1, 0], [0, 0], [0, 0]])
    np.testing.assert_array_equal(batch.sentiment[0, 0], [0.0, 0.0])
    np.testing.assert_array_equal(batch.sentiment[0, 1], [0.0, 0.0])
    leaked = sentiment.copy()
    leaked.loc[0, "available_at"] = "2020-01-02T00:00:00Z"
    with pytest.raises(ValueError, match="session midnight"):
        build(sentiment_frame=leaked, sentiment_columns=["sentiment_score"])
    absent = build(sentiment_columns=["sentiment_score"]).batch([0])
    assert absent.sentiment.shape == (1, 2, 1)
    assert not absent.sentiment_mask.any()


def test_fundamental_publication_cutoff_is_strict():
    target, _ = prepared_frames()
    target["fund_available_at"] = "2020-01-01T12:00:00Z"
    target.loc[0, "fund_available_at"] = "2020-01-04T00:00:00Z"
    with pytest.raises(ValueError, match="fundamentals"):
        build(target=target)


def test_graph_provenance_edges_and_absent_nodes_are_checked():
    with pytest.raises(ValueError, match="after its prediction session"):
        GraphSnapshot(
            "2020-01-02",
            "2020-01-03T00:00:00Z",
            np.empty((2, 0), dtype=np.int64),
            np.empty(0),
        )
    edge = np.array([[0], [1]], dtype=np.int64)
    weights = np.array([1.0])
    future_node = GraphSnapshot("2020-01-03", "2020-01-03T16:00:00Z", edge, weights)
    with pytest.raises(ValueError, match="absent or invalid nodes"):
        build(graphs=[future_node])
    outside = GraphSnapshot(
        "2020-01-02", "2020-01-02T16:00:00Z", np.array([[0], [2]]), weights
    )
    with pytest.raises(ValueError, match="exceeds ticker slots"):
        build(graphs=[outside])
    edge[0, 0] = 9
    assert future_node.edge_index[0, 0] == 0
    target, _ = prepared_frames()
    target.loc[target.ticker.eq("A") & target.date.eq("2020-01-02"), "node_signal"] = np.nan
    invalid_node = GraphSnapshot(
        "2020-01-02", "2020-01-02T16:00:00Z", np.array([[0], [1]]), weights
    )
    with pytest.raises(ValueError, match="absent or invalid nodes"):
        build(target=target, graphs=[invalid_node])


def test_conflicting_market_values_are_not_silently_selected():
    target, _ = prepared_frames()
    target.loc[target.ticker.eq("A") & target.date.eq("2020-01-02"), "market_signal"] = 0.9
    with pytest.raises(ValueError, match="Conflicting market features"):
        build(target=target)


def test_independent_branches_and_unknown_inference_labels():
    target, _ = prepared_frames()
    unlabeled = target.drop(columns=["Label_id", "_label_known"])
    graph_only = build(
        target=unlabeled, temporal_columns=(), node_columns=["node_signal"]
    ).batch([0])
    assert graph_only.temporal.shape == (1, 2, 2, 0)
    assert not graph_only.temporal_mask.any()
    assert not graph_only.label_mask.any()
    np.testing.assert_array_equal(graph_only.labels, [[-1, -1]])
    temporal_only = build(target=unlabeled, node_columns=()).batch([0])
    assert temporal_only.node.shape == (1, 2, 0)
    assert not temporal_only.node_mask.any()
    assert temporal_only.temporal_mask.all()


def test_incomplete_temporal_history_does_not_remove_graph_node_or_label():
    target, history = prepared_frames()
    one_date = target[target.date.eq("2020-01-02")].copy()
    dataset = build(target=one_date, history=history, context_len=3)
    batch = dataset.batch([0])
    np.testing.assert_array_equal(batch.asset_mask, [[1, 1]])
    np.testing.assert_array_equal(batch.node_mask, [[1, 1]])
    np.testing.assert_array_equal(batch.label_mask, [[1, 1]])
    assert not batch.temporal_mask.any()
    assert not batch.temporal.any()


def test_branch_output_contract_accepts_tensor_like_values_without_copying():
    logits = np.zeros((2, 3, 3), dtype=np.float32)
    representation = np.zeros((2, 3, 5), dtype=np.float32)
    availability = np.ones((2, 3), dtype=bool)
    output = BranchOutput(logits, representation, availability)
    output.validate_shape(batch_dates=2, assets=3)
    assert output.logits is logits
    with pytest.raises(ValueError, match="logits"):
        BranchOutput(logits[:, :, :2], representation, availability).validate_shape(
            batch_dates=2, assets=3
        )
