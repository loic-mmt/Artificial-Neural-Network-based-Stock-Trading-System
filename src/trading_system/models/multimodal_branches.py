"""Independent, date-aligned neural branches; no gating or fusion.

Import this module only when PyTorch is installed. The existing 3D experiment
runner and its model registry are intentionally untouched.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trading_system.data.multimodal import MultimodalBatch
from trading_system.models.multimodal_contract import BranchOutput
from trading_system.models.neural.config import GRUConfig, TransformerConfig
from trading_system.models.neural.gru_base import (
    build_gru_head_factory,
    build_gru_module,
)
from trading_system.models.neural.transformer import build_transformer_module
from trading_system.models.specs import ModelBuildContext


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_batch(batch: MultimodalBatch) -> tuple[int, int]:
    if not isinstance(batch, MultimodalBatch):
        raise TypeError("Expected a MultimodalBatch.")
    dates, assets = batch.asset_mask.shape
    if len(batch.sessions) != dates or len(batch.tickers) != assets:
        raise ValueError("Batch dates or ticker slots do not match masks.")
    return dates, assets


def _sequence_output(
    batch: MultimodalBatch,
    *,
    module: nn.Module,
    input_size: int,
    context_len: int,
) -> BranchOutput:
    dates, assets = _validate_batch(batch)
    if batch.temporal.shape != (dates, assets, context_len, input_size):
        raise ValueError("Temporal batch dimensions do not match the branch.")
    device = next(module.parameters()).device
    values = torch.as_tensor(batch.temporal, dtype=torch.float32, device=device)
    availability = torch.as_tensor(
        batch.asset_mask & batch.temporal_mask, dtype=torch.bool, device=device
    )
    flat_mask = availability.reshape(-1)
    representation = values.new_zeros((dates * assets, module.embedding_size))
    logits = values.new_zeros((dates * assets, 3))
    if flat_mask.any():
        positions = torch.nonzero(flat_mask, as_tuple=False).flatten()
        encoded = module.encode(values.reshape(dates * assets, context_len, input_size)[positions])
        representation = representation.index_copy(0, positions, encoded)
        logits = logits.index_copy(0, positions, module.head(encoded))
    output = BranchOutput(
        logits=logits.reshape(dates, assets, 3),
        representation=representation.reshape(dates, assets, -1),
        availability=availability,
    )
    output.validate_shape(batch_dates=dates, assets=assets)
    return output


class GRUBranch(nn.Module):
    """Apply the existing GRU encoder independently to valid asset windows."""

    def __init__(
        self, input_size: int, context_len: int, config: GRUConfig | None = None
    ) -> None:
        super().__init__()
        _positive_int(input_size, "input_size")
        _positive_int(context_len, "context_len")
        self.input_size = input_size
        self.context_len = context_len
        self.config = config if config is not None else GRUConfig()
        if not isinstance(self.config, GRUConfig):
            raise TypeError("config must be GRUConfig.")
        context = ModelBuildContext(input_size=input_size, context_len=context_len)
        self.encoder = build_gru_module(
            context, self.config, torch, nn,
            head_factory=build_gru_head_factory(self.config, nn),
        )

    def forward(self, batch: MultimodalBatch) -> BranchOutput:
        return _sequence_output(
            batch, module=self.encoder, input_size=self.input_size,
            context_len=self.context_len,
        )


class MarketTransformerBranch(nn.Module):
    """Encode a global macro/micro/VIX sequence, separate from stock GRU data.

    The market representation is shared across present tickers. Its standalone
    three-class head is consequently a market-state control, not a stock-specific
    prediction. Later gating/fusion may reuse the date-level representation.
    """

    def __init__(
        self, input_size: int, context_len: int,
        config: TransformerConfig | None = None,
    ) -> None:
        super().__init__()
        _positive_int(input_size, "input_size")
        _positive_int(context_len, "context_len")
        self.input_size = input_size
        self.context_len = context_len
        self.config = config if config is not None else TransformerConfig()
        if not isinstance(self.config, TransformerConfig):
            raise TypeError("config must be TransformerConfig.")
        context = ModelBuildContext(input_size=input_size, context_len=context_len)
        self.encoder = build_transformer_module(context, self.config, torch, nn)

    def forward_with_state(
        self, batch: MultimodalBatch
    ) -> tuple[BranchOutput, torch.Tensor]:
        dates, assets = _validate_batch(batch)
        if batch.market_sequence.shape != (dates, self.context_len, self.input_size):
            raise ValueError("Market sequence dimensions do not match the Transformer.")
        device = next(self.parameters()).device
        values = torch.as_tensor(batch.market_sequence, dtype=torch.float32, device=device)
        market_available = torch.as_tensor(
            batch.market_sequence_mask, dtype=torch.bool, device=device
        )
        asset_available = torch.as_tensor(batch.asset_mask, dtype=torch.bool, device=device)
        representation = values.new_zeros((dates, self.encoder.embedding_size))
        logits = values.new_zeros((dates, 3))
        if market_available.any():
            positions = torch.nonzero(market_available, as_tuple=False).flatten()
            encoded = self.encoder.encode(values[positions])
            representation = representation.index_copy(0, positions, encoded)
            logits = logits.index_copy(0, positions, self.encoder.head(encoded))
        availability = market_available[:, None] & asset_available
        output = BranchOutput(
            logits=logits[:, None, :].expand(-1, assets, -1) * availability.unsqueeze(-1),
            representation=representation[:, None, :].expand(-1, assets, -1) * availability.unsqueeze(-1),
            availability=availability,
        )
        output.validate_shape(batch_dates=dates, assets=assets)
        return output, representation

    def forward(self, batch: MultimodalBatch) -> BranchOutput:
        output, _ = self.forward_with_state(batch)
        return output


# The old name is retained as an import alias, but now routes to market data.
TransformerBranch = MarketTransformerBranch


class GNNBranch(nn.Module):
    """Sparse non-negative GCN over stock-node features, never GRU embeddings.

    Edge row 0 is source and row 1 is destination. Inputs must omit self-edges;
    active nodes receive exactly one unit self-loop here. `identity` is the
    node-MLP control; `provided` requires a dated graph for every usable date.
    """

    def __init__(
        self,
        input_size: int,
        *,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        graph_mode: Literal["identity", "provided"] = "provided",
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_size", input_size), ("hidden_size", hidden_size),
            ("num_layers", num_layers),
        ):
            _positive_int(value, name)
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if graph_mode not in ("identity", "provided"):
            raise ValueError("graph_mode must be 'identity' or 'provided'.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.graph_mode = graph_mode
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.convolutions = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 3)

    @staticmethod
    def _normalized_edges(
        batch: MultimodalBatch,
        date_index: int,
        active: torch.Tensor,
        *,
        graph_mode: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = torch.nonzero(active, as_tuple=False).flatten()
        if graph_mode == "identity":
            src = slots
            dst = slots
            weight = torch.ones(len(slots), dtype=torch.float32, device=device)
        else:
            snapshot = batch.graphs[date_index]
            if snapshot is None:
                raise ValueError("A provided graph is missing for an available date.")
            edges = np.asarray(snapshot.edge_index)
            weights = np.asarray(snapshot.edge_weight)
            if (weights < 0).any():
                raise ValueError("Standard GCN requires non-negative edge weights.")
            if edges.shape[1] and (edges[0] == edges[1]).any():
                raise ValueError("Provided graphs must omit self-edges.")
            if edges.shape[1] and len(set(map(tuple, edges.T))) != edges.shape[1]:
                raise ValueError("Provided graph edges must be unique.")
            # GraphSnapshot deliberately freezes its NumPy arrays. Copy before
            # handing them to PyTorch, which requires writable backing memory.
            src = torch.tensor(edges[0].copy(), dtype=torch.long, device=device)
            dst = torch.tensor(edges[1].copy(), dtype=torch.long, device=device)
            weight = torch.tensor(weights.copy(), dtype=torch.float32, device=device)
            if len(src) and not (active[src] & active[dst]).all():
                raise ValueError("Graph edges reference unavailable nodes.")
            src = torch.cat((src, slots))
            dst = torch.cat((dst, slots))
            weight = torch.cat((weight, torch.ones(len(slots), device=device)))
        degree = torch.zeros(len(active), dtype=torch.float32, device=device)
        degree = degree.index_add(0, dst, weight)
        normalizer = weight * degree[src].rsqrt() * degree[dst].rsqrt()
        return src, dst, normalizer

    @staticmethod
    def _dense_normalized_adjacency(
        batch: MultimodalBatch,
        date_index: int,
        active: torch.Tensor,
        *,
        graph_mode: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the same directed GCN operator without MPS scatter operations."""
        active_slots = active.detach().cpu().numpy().astype(bool)
        slots = np.flatnonzero(active_slots)
        if graph_mode == "identity":
            src = dst = slots
            weight = np.ones(len(slots), dtype=np.float32)
        else:
            snapshot = batch.graphs[date_index]
            if snapshot is None:
                raise ValueError("A provided graph is missing for an available date.")
            edges = np.asarray(snapshot.edge_index)
            weights = np.asarray(snapshot.edge_weight)
            if (weights < 0).any():
                raise ValueError("Standard GCN requires non-negative edge weights.")
            if edges.shape[1] and (edges[0] == edges[1]).any():
                raise ValueError("Provided graphs must omit self-edges.")
            if edges.shape[1] and len(set(map(tuple, edges.T))) != edges.shape[1]:
                raise ValueError("Provided graph edges must be unique.")
            if edges.shape[1] and not (active_slots[edges[0]] & active_slots[edges[1]]).all():
                raise ValueError("Graph edges reference unavailable nodes.")
            src = np.concatenate((edges[0], slots))
            dst = np.concatenate((edges[1], slots))
            weight = np.concatenate((weights, np.ones(len(slots), dtype=np.float32)))
        degree = np.bincount(dst, weights=weight, minlength=len(active_slots))
        normalized = weight / np.sqrt(degree[src] * degree[dst])
        adjacency = np.zeros((len(active_slots), len(active_slots)), dtype=np.float32)
        adjacency[dst, src] = normalized.astype(np.float32)
        return torch.as_tensor(adjacency, dtype=torch.float32, device=device)

    def forward(self, batch: MultimodalBatch) -> BranchOutput:
        dates, assets = _validate_batch(batch)
        if batch.node.shape != (dates, assets, self.input_size):
            raise ValueError("Node batch dimensions do not match the GNN branch.")
        device = next(self.parameters()).device
        values = torch.as_tensor(batch.node, dtype=torch.float32, device=device)
        node_mask = torch.as_tensor(
            batch.asset_mask & batch.node_mask, dtype=torch.bool, device=device
        )
        availability = (
            node_mask
            if self.graph_mode == "identity"
            else node_mask & torch.as_tensor(batch.graph_mask, dtype=torch.bool, device=device)
        )
        representations: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        for date_index in range(dates):
            active = availability[date_index]
            if not active.any():
                representations.append(values.new_zeros((assets, self.hidden_size)))
                logits.append(values.new_zeros((assets, 3)))
                continue
            # MPS has no deterministic index_add implementation. The graphs in
            # this benchmark are small, so dense matmul preserves determinism.
            if device.type == "mps":
                adjacency = self._dense_normalized_adjacency(
                    batch, date_index, active,
                    graph_mode=self.graph_mode, device=device,
                )
            else:
                src, dst, weight = self._normalized_edges(
                    batch, date_index, active,
                    graph_mode=self.graph_mode, device=device,
                )
            hidden = F.relu(self.input_projection(values[date_index]))
            hidden = hidden * active.unsqueeze(-1)
            for convolution in self.convolutions:
                if device.type == "mps":
                    aggregated = adjacency @ hidden
                else:
                    aggregated = torch.zeros_like(hidden).index_add(
                        0, dst, hidden[src] * weight.unsqueeze(-1)
                    )
                hidden = self.dropout(F.relu(convolution(aggregated)))
                hidden = hidden * active.unsqueeze(-1)
            representations.append(hidden)
            logits.append(self.head(hidden) * active.unsqueeze(-1))
        output = BranchOutput(
            logits=torch.stack(logits),
            representation=torch.stack(representations),
            availability=availability,
        )
        output.validate_shape(batch_dates=dates, assets=assets)
        return output


class SentimentBranch(nn.Module):
    """Optional classifier over precomputed FinBERT aggregates, not raw text."""

    def __init__(self, input_size: int, *, hidden_size: int = 16) -> None:
        super().__init__()
        _positive_int(input_size, "input_size")
        _positive_int(hidden_size, "hidden_size")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, batch: MultimodalBatch) -> BranchOutput:
        dates, assets = _validate_batch(batch)
        if batch.sentiment.shape != (dates, assets, self.input_size):
            raise ValueError("Sentiment batch dimensions do not match the branch.")
        device = next(self.parameters()).device
        values = torch.as_tensor(batch.sentiment, dtype=torch.float32, device=device)
        availability = torch.as_tensor(
            batch.asset_mask & batch.sentiment_mask, dtype=torch.bool, device=device
        )
        hidden = self.encoder(values) * availability.unsqueeze(-1)
        logits = self.head(hidden) * availability.unsqueeze(-1)
        output = BranchOutput(logits, hidden, availability)
        output.validate_shape(batch_dates=dates, assets=assets)
        return output


def masked_branch_cross_entropy(output: BranchOutput, batch: MultimodalBatch) -> torch.Tensor:
    """Standalone supervised loss on known labels and available branch slots."""

    dates, assets = _validate_batch(batch)
    output.validate_shape(batch_dates=dates, assets=assets)
    available = output.availability
    if not isinstance(available, torch.Tensor) or available.dtype != torch.bool:
        raise TypeError("Branch availability must be a boolean torch tensor.")
    known = torch.as_tensor(batch.label_mask, dtype=torch.bool, device=available.device)
    selected = available & known
    if not selected.any():
        raise ValueError("No known labels are available for this branch batch.")
    labels = torch.as_tensor(batch.labels, dtype=torch.long, device=available.device)
    return F.cross_entropy(output.logits[selected], labels[selected])


__all__ = [
    "GNNBranch",
    "GRUBranch",
    "MarketTransformerBranch",
    "SentimentBranch",
    "TransformerBranch",
    "masked_branch_cross_entropy",
]
