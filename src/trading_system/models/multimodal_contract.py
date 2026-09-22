"""Output shape contract for independent multimodal branches.

No model or fusion is implemented here. Tensor-like values remain untouched so
future PyTorch branches can retain gradients through logits and representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BranchOutput:
    """Per-asset logits, embeddings and availability for one model branch."""

    logits: Any
    representation: Any
    availability: Any

    def validate_shape(self, *, batch_dates: int, assets: int) -> None:
        if tuple(self.logits.shape) != (batch_dates, assets, 3):
            raise ValueError("Branch logits must have shape (dates, assets, 3).")
        shape = tuple(self.representation.shape)
        if len(shape) != 3 or shape[:2] != (batch_dates, assets) or shape[2] < 1:
            raise ValueError("Branch representation must have shape (dates, assets, d>0).")
        if tuple(self.availability.shape) != (batch_dates, assets):
            raise ValueError("Branch availability must have shape (dates, assets).")


__all__ = ["BranchOutput"]
