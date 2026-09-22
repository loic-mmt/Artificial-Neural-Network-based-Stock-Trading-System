"""Explicit branch selection and feature routing for multimodal experiments.

This module selects independently testable models. It deliberately returns a
mapping of branch outputs rather than silently fusing them or changing the
legacy 3D experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import nn

from trading_system.data.multimodal import MultimodalBatch
from trading_system.models.multimodal_contract import BranchOutput
from trading_system.models.neural.config import GRUConfig, TransformerConfig
from trading_system.models.multimodal_branches import (
    GNNBranch,
    GRUBranch,
    MarketTransformerBranch,
    SentimentBranch,
)


BRANCH_NAMES = ("gru", "market_transformer", "gnn", "sentiment")


@dataclass(frozen=True)
class BranchSelection:
    """Independent switches and model parameters; no implicit branch coupling."""

    enabled: tuple[str, ...] = ("gru",)
    gru: GRUConfig | None = None
    market_transformer: TransformerConfig | None = None
    gnn_graph_mode: Literal["identity", "provided"] = "identity"
    gnn_hidden_size: int = 32
    gnn_layers: int = 1
    gnn_dropout: float = 0.0
    sentiment_hidden_size: int = 16

    def __post_init__(self) -> None:
        if isinstance(self.enabled, str) or not self.enabled:
            raise ValueError("enabled must name at least one branch.")
        if len(self.enabled) != len(set(self.enabled)):
            raise ValueError("enabled branches must be unique.")
        unknown = sorted(set(self.enabled) - set(BRANCH_NAMES))
        if unknown:
            raise ValueError(f"Unknown multimodal branches: {unknown}")
        if self.gru is not None and not isinstance(self.gru, GRUConfig):
            raise TypeError("gru must be a GRUConfig.")
        if self.market_transformer is not None and not isinstance(
            self.market_transformer, TransformerConfig
        ):
            raise TypeError("market_transformer must be a TransformerConfig.")


class BranchRouter(nn.Module):
    """Build only requested branches from a prepared sample batch."""

    def __init__(self, sample: MultimodalBatch, selection: BranchSelection) -> None:
        super().__init__()
        if not isinstance(sample, MultimodalBatch):
            raise TypeError("sample must be a MultimodalBatch.")
        if not isinstance(selection, BranchSelection):
            raise TypeError("selection must be BranchSelection.")
        self.selection = selection
        models: dict[str, nn.Module] = {}
        for name in selection.enabled:
            if name == "gru":
                _, _, context_len, features = sample.temporal.shape
                if not context_len or not features:
                    raise ValueError("GRU requires stock temporal features.")
                models[name] = GRUBranch(features, context_len, selection.gru)
            elif name == "market_transformer":
                _, context_len, features = sample.market_sequence.shape
                if not context_len or not features:
                    raise ValueError("Market Transformer requires a market sequence.")
                models[name] = MarketTransformerBranch(
                    features, context_len, selection.market_transformer
                )
            elif name == "gnn":
                features = sample.node.shape[-1]
                if not features:
                    raise ValueError("GNN requires stock node features.")
                models[name] = GNNBranch(
                    features,
                    hidden_size=selection.gnn_hidden_size,
                    num_layers=selection.gnn_layers,
                    dropout=selection.gnn_dropout,
                    graph_mode=selection.gnn_graph_mode,
                )
            elif name == "sentiment":
                features = sample.sentiment.shape[-1]
                if not features:
                    raise ValueError("Sentiment branch requires numeric sentiment features.")
                models[name] = SentimentBranch(
                    features, hidden_size=selection.sentiment_hidden_size
                )
        self.branches = nn.ModuleDict(models)

    def forward(self, batch: MultimodalBatch) -> dict[str, BranchOutput]:
        return {name: model(batch) for name, model in self.branches.items()}


__all__ = ["BRANCH_NAMES", "BranchRouter", "BranchSelection"]
