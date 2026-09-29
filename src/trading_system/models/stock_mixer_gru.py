"""Optional same-session cross-asset context on top of the shared GRU encoder."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from trading_system.models.neural.config import GRUConfig
from trading_system.models.neural.gru_base import build_gru_module
from trading_system.models.specs import ModelBuildContext


AssetMixer = Literal["none", "mean", "stock_mixer", "attention_self", "attention"]
MIXERS: tuple[AssetMixer, ...] = (
    "none", "mean", "stock_mixer", "attention_self", "attention",
)


class StockMixerGRU(nn.Module):
    """Return per-asset logits from a date-grouped ``[B, N, T, F]`` input.

    The encoder is shared across assets. No cross-date communication occurs.
    ``stock_mixer`` has fixed asset slots; the attention variants do not.
    """

    def __init__(self, *, input_size: int, context_len: int, assets: int,
                 config: GRUConfig, mixer: AssetMixer, market_states: int = 3,
                 attention_heads: int = 1) -> None:
        super().__init__()
        if mixer not in MIXERS:
            raise ValueError(f"Unknown asset mixer: {mixer}")
        if min(input_size, context_len, assets, market_states, attention_heads) < 1:
            raise ValueError("Model dimensions must be positive.")
        if market_states >= assets and mixer == "stock_mixer":
            raise ValueError("StockMixer bottleneck must be smaller than asset count.")
        self.mixer = mixer
        self.assets = assets
        self.context_len = context_len
        self.input_size = input_size
        self.encoder = build_gru_module(
            ModelBuildContext(input_size=input_size, context_len=context_len),
            config, torch, nn,
        )
        width = self.encoder.embedding_size
        # Preserve the exact standalone head for S0. Relation variants use a
        # shared two-input head; the self-only attention variant controls its
        # added capacity without exchanging information across assets.
        standalone_head = self.encoder.head
        self.encoder.head = nn.Identity()
        self.head = standalone_head if mixer == "none" else nn.Linear(2 * width, 3)
        if mixer == "stock_mixer":
            self.norm = nn.LayerNorm(width)
            self.compress = nn.Parameter(torch.empty(market_states, assets))
            self.expand = nn.Parameter(torch.empty(assets, market_states))
            nn.init.xavier_uniform_(self.compress)
            nn.init.xavier_uniform_(self.expand)
        elif mixer in ("attention_self", "attention"):
            if width % attention_heads:
                raise ValueError("GRU embedding width must divide attention_heads.")
            self.norm = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(
                width, attention_heads, dropout=0.0, batch_first=True,
            )

    def forward(self, sequences: torch.Tensor, asset_mask: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 4 or sequences.shape[1:] != (
            self.assets, self.context_len, self.input_size
        ):
            raise ValueError("Expected [dates, assets, time, features] input.")
        if asset_mask.shape != sequences.shape[:2] or asset_mask.dtype != torch.bool:
            raise ValueError("Expected boolean [dates, assets] mask.")
        dates = len(sequences)
        # Keep the autograd path dense: advanced indexing's backward uses
        # index_put(accumulate=True), which is not deterministic on MPS.
        safe_sequences = torch.where(asset_mask[:, :, None, None], sequences, 0.0)
        flattened = safe_sequences.reshape(dates * self.assets, self.context_len, self.input_size)
        hidden = self.encoder.encode(flattened).reshape(dates, self.assets, -1)
        hidden = hidden * asset_mask.unsqueeze(-1)
        zeros = torch.zeros_like(hidden)
        if self.mixer == "none":
            return self.head(hidden) * asset_mask.unsqueeze(-1)
        if self.mixer == "mean":
            total = hidden.sum(dim=1, keepdim=True)
            others = (asset_mask.sum(dim=1, keepdim=True) - 1).clamp_min(1)
            context = (total - hidden) / others.unsqueeze(-1)
            context = torch.where(
                (asset_mask.sum(dim=1, keepdim=True) > 1).unsqueeze(-1), context, zeros,
            )
        elif self.mixer == "stock_mixer":
            normalized = self.norm(hidden) * asset_mask.unsqueeze(-1)
            market = torch.einsum("mn,bnd->bmd", self.compress, normalized)
            market = torch.nn.functional.gelu(market)
            context = torch.einsum("nm,bmd->bnd", self.expand, market)
        else:
            normalized = self.norm(hidden)
            if self.mixer == "attention_self":
                # Length-one sequences are exactly diagonal attention and
                # avoid fully masked queries for absent assets.
                single = normalized.reshape(dates * self.assets, 1, -1)
                context, _ = self.attention(single, single, single, need_weights=False)
                context = context.reshape(dates, self.assets, -1)
            else:
                # Empty dates receive dummy keys, then all their outputs are
                # masked below. This prevents all-masked softmax NaNs.
                safe_keys = asset_mask | ~asset_mask.any(dim=1, keepdim=True)
                context, _ = self.attention(
                    normalized, normalized, normalized,
                    key_padding_mask=~safe_keys, need_weights=False,
                )
        context = context * asset_mask.unsqueeze(-1)
        logits = self.head(torch.cat((hidden, context), dim=-1))
        return logits * asset_mask.unsqueeze(-1)
