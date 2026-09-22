"""Modular market-gate and fusion controls, with no historical corpus."""

import pytest

torch = pytest.importorskip("torch")

from test_multimodal_branches import _batch, _graph, _market, _market_state, _sentiment
from trading_system.data.multimodal import build_multimodal_dataset
from trading_system.models.multimodal_branches import masked_branch_cross_entropy
from trading_system.models.multimodal_router import BranchSelection
from trading_system.models.multimodal_system import (
    MarketFeatureGate,
    MultimodalOptions,
    MultimodalSystem,
)
from trading_system.models.neural.config import GRUConfig, TransformerConfig


def _selection(*, graph_mode="provided"):
    return BranchSelection(
        enabled=("gru", "market_transformer", "gnn", "sentiment"),
        gru=GRUConfig(hidden_size=4, temporal_pooling="attention"),
        market_transformer=TransformerConfig(
            d_model=4, n_heads=2, num_layers=1, dim_feedforward=8, dropout=0.0,
        ),
        gnn_graph_mode=graph_mode,
        gnn_hidden_size=4,
        sentiment_hidden_size=4,
    )


def _richer_batch(*, graph=None):
    stock = _market().assign(
        temporal_extra=[2.0, 1.0, 4.0],
        node_extra=[4.0, 1.0, 2.0],
    )
    return build_multimodal_dataset(
        stock,
        tickers=["A", "B"],
        context_len=1,
        temporal_columns=["temporal_signal", "temporal_extra"],
        node_columns=["node_signal", "node_extra"],
        market_frame=_market_state(),
        market_context_len=1,
        market_publication_columns=["macro_signal"],
        market_close_columns=["vix_signal"],
        sentiment_frame=_sentiment(),
        sentiment_columns=["sentiment_score", "sentiment_score_present", "news_count"],
        graphs=[] if graph is None else [graph],
    ).batch([0, 1])


def test_market_transformer_can_be_context_only_and_missing_branches_renormalize():
    batch = _batch(graph=_graph())
    options = MultimodalOptions(
        selection=_selection(),
        prediction_branches=("gru", "gnn", "sentiment"),
        fusion="mean",
    )
    system = MultimodalSystem(batch, options)
    result = system(batch)
    assert tuple(result.branches) == options.selection.enabled
    assert result.weight_branches == options.prediction_branches
    torch.testing.assert_close(result.weights[0, 0], torch.tensor([1 / 3] * 3))
    torch.testing.assert_close(result.weights[1, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert not result.fused.availability[1, 1]
    assert not result.fused.logits[1, 1].any()
    assert result.fused.representation.shape[-1] == sum(
        result.branches[name].representation.shape[-1]
        for name in options.prediction_branches
    )


def test_market_gate_is_identity_initially_and_can_be_enabled_independently():
    batch = _richer_batch()
    selection = _selection(graph_mode="identity")
    torch.manual_seed(11)
    baseline = MultimodalSystem(batch, MultimodalOptions(
        selection=selection, prediction_branches=("gru",),
    ))
    torch.manual_seed(11)
    gated = MultimodalSystem(batch, MultimodalOptions(
        selection=selection, prediction_branches=("gru",), gate_gru="market",
    ))
    baseline.eval()
    gated.eval()
    torch.testing.assert_close(baseline(batch).fused.logits, gated(batch).fused.logits)
    with torch.no_grad():
        gated.gru_gate.market_projection.weight[0, 0] = 2.0
    assert not torch.allclose(baseline(batch).fused.logits, gated(batch).fused.logits)

    gate = MarketFeatureGate(2, mode="market", market_width=2)
    values = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    with torch.no_grad():
        gate.market_projection.weight[0].fill_(1.0)
    result = gate(
        values,
        market_state=torch.tensor([[2.0, 1.0], [2.0, 1.0]]),
        market_available=torch.tensor([True, False]),
    )
    torch.testing.assert_close(result[1], values[1])


def test_static_fusion_and_market_guided_gnn_receive_gradients():
    batch = _richer_batch(graph=_graph())
    model = MultimodalSystem(batch, MultimodalOptions(
        selection=_selection(), prediction_branches=("gru", "gnn"),
        gate_gru="market", gate_gnn="market", fusion="static",
    ))
    prediction = model(batch)
    loss = masked_branch_cross_entropy(prediction.fused, batch)
    loss.backward()
    assert model.fusion.branch_logits.grad is not None
    assert model.gru_gate.market_projection.weight.grad is not None
    assert model.gnn_gate.market_projection.weight.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in model.router.branches["market_transformer"].parameters()
    )


def test_configuration_refuses_hidden_dependencies_or_unfitted_confidence():
    selection = _selection()
    with pytest.raises(ValueError, match="exactly one"):
        MultimodalOptions(selection=selection)
    with pytest.raises(ValueError, match="enabled"):
        MultimodalOptions(selection=selection, prediction_branches=("other",))
    with pytest.raises(ValueError, match="unique"):
        MultimodalOptions(selection=selection, prediction_branches=("gru", "gru"))
    with pytest.raises(ValueError, match="validation-fitted"):
        MultimodalOptions(
            selection=selection, prediction_branches=("gru", "gnn"),
            fusion="confidence",
        )
    no_market = BranchSelection(enabled=("gru",))
    with pytest.raises(ValueError, match="market Transformer"):
        MultimodalOptions(selection=no_market, gate_gru="market")


def test_confidence_fusion_uses_explicit_validation_temperatures():
    batch = _batch(graph=_graph())
    model = MultimodalSystem(batch, MultimodalOptions(
        selection=_selection(), prediction_branches=("gru", "gnn"),
        fusion="confidence",
        confidence_temperatures={"gru": 1.5, "gnn": 2.0},
    ))
    result = model(batch)
    torch.testing.assert_close(result.weights[1, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(result.weights[0, 0].sum(), torch.tensor(1.0))
