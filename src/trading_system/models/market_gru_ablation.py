"""Single-voter market-context controls for the GRU benchmark.

No GNN or late fusion is involved. The market encoder is either a cheap pooled
context or the independent market Transformer; both see only market_sequence.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from trading_system.data.multimodal import MultimodalBatch
from trading_system.models.multimodal_branches import GRUBranch, MarketTransformerBranch
from trading_system.models.multimodal_system import MarketFeatureGate
from trading_system.models.neural.config import GRUConfig, TransformerConfig


CANDIDATES = (
    "gru", "concat", "static_gate", "market_gate_simple",
    "market_gate_transformer", "market_transformer_only",
)


class MarketGRUControl(nn.Module):
    """One prediction branch; market context may alter GRU inputs, never fuse votes."""

    def __init__(self, candidate: str, stock_width: int, market_width: int,
                 context_len: int, gru_config: GRUConfig,
                 transformer_config: TransformerConfig) -> None:
        super().__init__()
        if candidate not in CANDIDATES or stock_width <= 0 or market_width <= 0:
            raise ValueError("Unknown candidate or empty stock/market feature group.")
        self.candidate = candidate
        self.gru = None if candidate == "market_transformer_only" else GRUBranch(
            stock_width + (market_width if candidate == "concat" else 0),
            context_len, gru_config,
        )
        self.market_transformer = (
            MarketTransformerBranch(market_width, context_len, transformer_config)
            if candidate in ("market_gate_transformer", "market_transformer_only") else None
        )
        self.simple_encoder = (
            nn.Sequential(nn.Linear(market_width, 16), nn.Tanh())
            if candidate == "market_gate_simple" else None
        )
        gate_mode = "static" if candidate == "static_gate" else "market"
        self.gate = (
            MarketFeatureGate(
                stock_width, mode=gate_mode,
                market_width=(16 if candidate == "market_gate_simple" else transformer_config.d_model),
            ) if candidate in ("static_gate", "market_gate_simple", "market_gate_transformer")
            else None
        )

    def forward(self, batch: MultimodalBatch):
        if self.candidate == "market_transformer_only":
            return self.market_transformer(batch)
        if self.candidate == "gru":
            return self.gru(batch)
        device = next(self.parameters()).device
        market = torch.as_tensor(batch.market_sequence, dtype=torch.float32, device=device)
        available = torch.as_tensor(batch.market_sequence_mask, dtype=torch.bool, device=device)
        if not bool(available.all()):
            raise ValueError("Matched market benchmark requires complete market windows.")
        stock = torch.as_tensor(batch.temporal, dtype=torch.float32, device=device)
        if self.candidate == "concat":
            expanded = market[:, None].expand(-1, stock.shape[1], -1, -1)
            return self.gru(replace(batch, temporal=torch.cat((stock, expanded), dim=-1)))
        if self.candidate == "static_gate":
            state = None
        elif self.candidate == "market_gate_simple":
            state = self.simple_encoder(market.mean(dim=1))
        else:
            _, state = self.market_transformer.forward_with_state(batch)
        gated = self.gate(stock, market_state=state, market_available=available)
        return self.gru(replace(batch, temporal=gated))


__all__ = ["CANDIDATES", "MarketGRUControl"]
