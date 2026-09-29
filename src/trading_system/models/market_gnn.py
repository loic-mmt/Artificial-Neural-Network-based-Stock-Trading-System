"""Independent GNN optionally conditioned by a market-only Transformer."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from trading_system.data.multimodal import MultimodalBatch
from trading_system.models.multimodal_branches import GNNBranch, MarketTransformerBranch
from trading_system.models.multimodal_system import MarketFeatureGate
from trading_system.models.neural.config import TransformerConfig


class MarketGNNControl(nn.Module):
    """Predict with the GNN; the Transformer only gates node features.

    This is deliberately not logit fusion.  It isolates whether a global
    market state helps the relational branch select useful node attributes.
    """

    def __init__(
        self,
        node_width: int,
        market_width: int,
        market_context_len: int,
        transformer_config: TransformerConfig,
        *,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        graph_mode: str = "provided",
        market_gate: bool = True,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.market_gate = bool(market_gate)
        self.gnn = GNNBranch(
            node_width,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            graph_mode=graph_mode,
        )
        self.market_transformer = None
        self.gate = None
        if self.market_gate:
            if market_width <= 0:
                raise ValueError("Market-guided GNN needs market features.")
            self.market_transformer = MarketTransformerBranch(
                market_width, market_context_len, transformer_config,
            )
            self.gate = MarketFeatureGate(
                node_width,
                mode="market",
                market_width=self.market_transformer.encoder.embedding_size,
                temperature=gate_temperature,
            )

    def forward(self, batch: MultimodalBatch):
        if not self.market_gate:
            return self.gnn(batch)
        _, market_state = self.market_transformer.forward_with_state(batch)
        available = torch.as_tensor(
            batch.market_sequence_mask,
            dtype=torch.bool,
            device=market_state.device,
        )
        if not bool(available.all()):
            raise ValueError("Matched market/GNN benchmark requires complete market windows.")
        node = torch.as_tensor(
            batch.node, dtype=torch.float32, device=market_state.device,
        )
        gated = self.gate(
            node, market_state=market_state, market_available=available,
        )
        return self.gnn(replace(batch, node=gated))


__all__ = ["MarketGNNControl"]
