"""Independent branch semantics with tiny synthetic, date-aligned batches."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from trading_system.data.multimodal import GraphSnapshot, build_multimodal_dataset
from trading_system.models.multimodal_branches import (
    GNNBranch,
    GRUBranch,
    MarketTransformerBranch,
    SentimentBranch,
    masked_branch_cross_entropy,
)
from trading_system.models.neural.config import GRUConfig, TransformerConfig
from trading_system.models.multimodal_router import BranchRouter, BranchSelection


def _graph(*, edge=True, weight=1.0):
    return GraphSnapshot(
        session="2020-01-02",
        source_end="2020-01-02T16:00:00Z",
        edge_index=np.array([[0], [1]], dtype=np.int64) if edge else np.empty((2, 0), np.int64),
        edge_weight=np.array([weight], dtype=np.float32) if edge else np.empty(0, np.float32),
    )


def _market():
    return pd.DataFrame({
        "date": ["2020-01-02", "2020-01-02", "2020-01-03"],
        "ticker": ["A", "B", "A"],
        "temporal_signal": [1.0, 2.0, 3.0],
        "node_signal": [1.0, 2.0, 3.0],
        "Label_id": [0, 2, 1],
        "_label_known": [True, True, False],
    })


def _sentiment():
    return pd.DataFrame({
        "date": ["2020-01-02", "2020-01-02", "2020-01-03"],
        "ticker": ["A", "B", "A"],
        "sentiment_score": [0.0, 0.0, 0.0],
        "sentiment_score_present": [1.0, 0.0, 0.0],
        "news_count": [1, 0, 0],
        "source_available": [True, True, False],
        "coverage_status": ["covered", "covered", "unknown"],
        "available_at": ["2020-01-01T20:00:00Z", None, None],
    })


def _market_state():
    return pd.DataFrame({
        "date": ["2020-01-02", "2020-01-03"],
        "macro_signal": [0.1, 0.2],
        "vix_signal": [10.0, 20.0],
        "available_at": ["2020-01-01T12:00:00Z", "2020-01-02T12:00:00Z"],
        "source_end": ["2020-01-02T16:00:00Z", "2020-01-03T16:00:00Z"],
    })


def _batch(*, market=None, graph=None, sentiment=None, market_state=None):
    if market is None:
        market = _market()
    if sentiment is None:
        sentiment = _sentiment()
    if market_state is None:
        market_state = _market_state()
    return build_multimodal_dataset(
        market,
        tickers=["A", "B"],
        context_len=1,
        temporal_columns=["temporal_signal"],
        node_columns=["node_signal"],
        market_frame=market_state,
        market_context_len=1,
        market_publication_columns=["macro_signal"],
        market_close_columns=["vix_signal"],
        sentiment_frame=sentiment,
        sentiment_columns=["sentiment_score", "sentiment_score_present", "news_count"],
        graphs=[] if graph is None else [graph],
    ).batch([0, 1])


def test_gru_reuses_encoder_and_trains_only_valid_stock_windows():
    torch.manual_seed(5)
    batch = _batch()
    branch = GRUBranch(1, 1, GRUConfig(hidden_size=4, temporal_pooling="attention"))
    branch.eval()
    output = branch(batch)
    assert output.logits.shape == (2, 2, 3)
    assert output.representation.shape[:2] == (2, 2)
    assert output.availability.tolist() == [[True, True], [True, False]]
    assert not output.logits[1, 1].any()
    valid = torch.as_tensor(batch.temporal[batch.temporal_mask])
    torch.testing.assert_close(output.logits[output.availability], branch.encoder(valid))
    loss = masked_branch_cross_entropy(output, batch)
    loss.backward()
    assert any(parameter.grad is not None for parameter in branch.parameters())


def test_market_transformer_is_separate_and_broadcasts_only_available_dates():
    torch.manual_seed(5)
    batch = _batch()
    branch = MarketTransformerBranch(2, 1, TransformerConfig(
        d_model=4, n_heads=2, num_layers=1, dim_feedforward=8, dropout=0.0,
    ))
    branch.eval()
    output = branch(batch)
    assert output.logits.shape == (2, 2, 3)
    assert output.availability.tolist() == [[True, True], [True, False]]
    torch.testing.assert_close(output.logits[0, 0], output.logits[0, 1])
    assert not output.logits[1, 1].any()
    direct = branch.encoder(torch.as_tensor(batch.market_sequence))
    torch.testing.assert_close(output.logits[:, 0], direct)
    changed_stock = _market()
    changed_stock["temporal_signal"] *= 100
    torch.testing.assert_close(output.logits, branch(_batch(market=changed_stock)).logits)
    changed_market = _market_state()
    changed_market["vix_signal"] *= 2
    assert not torch.allclose(
        output.logits, branch(_batch(market_state=changed_market)).logits
    )
    masked_branch_cross_entropy(output, batch).backward()
    assert any(parameter.grad is not None for parameter in branch.parameters())


def test_gnn_is_independent_of_temporal_features_and_uses_sparse_neighbors():
    torch.manual_seed(7)
    model = GNNBranch(1, hidden_size=4, graph_mode="provided")
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.5)
    isolated = model(_batch(graph=_graph(edge=False)))
    connected_batch = _batch(graph=_graph())
    connected = model(connected_batch)
    assert connected.availability.tolist() == [[True, True], [False, False]]
    assert not connected.logits[1].any()
    assert not torch.allclose(isolated.logits[0, 1], connected.logits[0, 1])

    changed = _market()
    changed["temporal_signal"] *= 100
    unchanged_graph = model(_batch(market=changed, graph=_graph()))
    torch.testing.assert_close(connected.logits, unchanged_graph.logits)
    masked_branch_cross_entropy(connected, connected_batch).backward()
    assert model.convolutions[0].weight.grad is not None


def test_multimodal_gru_keeps_configured_benchmark_head():
    branch = GRUBranch(
        1, 1,
        GRUConfig(
            hidden_size=4, temporal_pooling="attention",
            head_type="layernorm_mlp", head_hidden_size=3,
        ),
    )
    assert isinstance(branch.encoder.head.norm, torch.nn.LayerNorm)
    assert branch.encoder.head.hidden.out_features == 3
    assert branch(_batch()).logits.shape == (2, 2, 3)


def test_identity_graph_control_works_without_graph_and_rejects_signed_weights():
    batch = _batch()
    identity = GNNBranch(1, graph_mode="identity")(batch)
    provided = GNNBranch(1, graph_mode="provided")(batch)
    assert identity.availability.tolist() == [[True, True], [True, False]]
    assert not provided.availability.any()
    with pytest.raises(ValueError, match="non-negative"):
        GNNBranch(1, graph_mode="provided")(_batch(graph=_graph(weight=-1.0)))


def test_sentiment_branch_distinguishes_observed_neutral_zero_news_and_unknown():
    torch.manual_seed(9)
    batch = _batch()
    output = SentimentBranch(3, hidden_size=4)(batch)
    assert output.availability.tolist() == [[True, True], [False, False]]
    assert batch.sentiment[0, 0, 0] == batch.sentiment[0, 1, 0] == 0
    assert batch.sentiment[0, 0, 1] == 1
    assert batch.sentiment[0, 1, 1] == 0
    assert not output.logits[1].any()
    masked_branch_cross_entropy(output, batch).backward()


def test_masked_loss_rejects_batches_without_known_available_labels():
    batch = _batch()
    output = GNNBranch(1, graph_mode="provided")(batch)
    with pytest.raises(ValueError, match="No known labels"):
        masked_branch_cross_entropy(output, batch)


def test_branch_router_selects_modalities_independently():
    batch = _batch(graph=_graph())
    selected = BranchSelection(
        enabled=("gru", "market_transformer", "gnn", "sentiment"),
        gru=GRUConfig(hidden_size=4),
        market_transformer=TransformerConfig(
            d_model=4, n_heads=2, num_layers=1, dim_feedforward=8, dropout=0.0,
        ),
        gnn_graph_mode="provided", gnn_hidden_size=4,
    )
    outputs = BranchRouter(batch, selected)(batch)
    assert tuple(outputs) == selected.enabled
    assert all(output.logits.shape == (2, 2, 3) for output in outputs.values())
    assert outputs["market_transformer"].availability.tolist() == [[1, 1], [1, 0]]
    assert outputs["gnn"].availability.tolist() == [[1, 1], [0, 0]]
    assert outputs["sentiment"].availability.tolist() == [[1, 1], [0, 0]]
    only_gnn = BranchRouter(batch, BranchSelection(enabled=("gnn",)))(batch)
    assert tuple(only_gnn) == ("gnn",)


def test_branch_router_refuses_implicit_or_missing_inputs():
    batch = _batch()
    with pytest.raises(ValueError, match="unique"):
        BranchSelection(enabled=("gru", "gru"))
    with pytest.raises(ValueError, match="Unknown"):
        BranchSelection(enabled=("gru", "mystery"))
    market_absent = build_multimodal_dataset(
        _market(), tickers=["A", "B"], context_len=1,
        temporal_columns=["temporal_signal"],
    ).batch([0])
    with pytest.raises(ValueError, match="market sequence"):
        BranchRouter(market_absent, BranchSelection(enabled=("market_transformer",)))
