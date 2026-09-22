"""Configurable market gating and late fusion over independent branches.

Inspired by MASTER's market-conditioned feature scaling and the separate-path
late fusion in FusionLSTM-CNF. This is an ablatable implementation, not a claim
that either paper's reported trading results transfer to this dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from trading_system.data.multimodal import MultimodalBatch
from trading_system.models.multimodal_contract import BranchOutput
from trading_system.models.multimodal_router import BranchRouter, BranchSelection


GateMode = Literal["none", "static", "market"]
FusionMode = Literal["single", "mean", "static", "confidence"]


@dataclass(frozen=True)
class MultimodalOptions:
    """Choose encoders, prediction voters, gates and fusion independently."""

    selection: BranchSelection
    prediction_branches: tuple[str, ...] | None = None
    gate_gru: GateMode = "none"
    gate_gnn: GateMode = "none"
    gate_temperature: float = 1.0
    fusion: FusionMode = "single"
    confidence_temperatures: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.selection, BranchSelection):
            raise TypeError("selection must be BranchSelection.")
        voters = (
            self.selection.enabled
            if self.prediction_branches is None else self.prediction_branches
        )
        if isinstance(voters, str) or not voters or len(voters) != len(set(voters)):
            raise ValueError("prediction_branches must contain unique branch names.")
        if not set(voters).issubset(self.selection.enabled):
            raise ValueError("Prediction branches must be enabled in selection.")
        object.__setattr__(self, "prediction_branches", tuple(voters))
        if self.fusion not in ("single", "mean", "static", "confidence"):
            raise ValueError("Unknown fusion mode.")
        if self.fusion == "single" and len(voters) != 1:
            raise ValueError("single fusion requires exactly one prediction branch.")
        if self.gate_gru not in ("none", "static", "market") or self.gate_gnn not in (
            "none", "static", "market"
        ):
            raise ValueError("Unknown market gate mode.")
        if self.gate_gru != "none" and "gru" not in self.selection.enabled:
            raise ValueError("GRU gate requires an enabled GRU branch.")
        if self.gate_gnn != "none" and "gnn" not in self.selection.enabled:
            raise ValueError("GNN gate requires an enabled GNN branch.")
        if (self.gate_gru == "market" or self.gate_gnn == "market") and (
            "market_transformer" not in self.selection.enabled
        ):
            raise ValueError("Market-guided gates require an enabled market Transformer.")
        if not isinstance(self.gate_temperature, (int, float)) or not math.isfinite(
            self.gate_temperature
        ) or self.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")
        if self.fusion == "confidence":
            temperatures = self.confidence_temperatures
            if temperatures is None or set(temperatures) != set(voters) or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                for value in temperatures.values()
            ):
                raise ValueError(
                    "Confidence fusion needs positive validation-fitted temperatures "
                    "for every prediction branch."
                )


class MarketFeatureGate(nn.Module):
    """MASTER-style F*softmax scaling; uniform initialization is identity."""

    def __init__(
        self,
        features: int,
        *,
        mode: Literal["static", "market"],
        market_width: int = 0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if features <= 0 or temperature <= 0:
            raise ValueError("Feature count and temperature must be positive.")
        if mode not in ("static", "market"):
            raise ValueError("MarketFeatureGate mode must be static or market.")
        self.features = features
        self.mode = mode
        self.temperature = temperature
        if mode == "static":
            self.gate_logits = nn.Parameter(torch.zeros(features))
            self.market_projection = None
        else:
            if market_width <= 0:
                raise ValueError("Market gate requires a market representation width.")
            self.register_parameter("gate_logits", None)
            self.market_projection = nn.Linear(market_width, features)
            nn.init.zeros_(self.market_projection.weight)
            nn.init.zeros_(self.market_projection.bias)

    def forward(
        self,
        values: torch.Tensor,
        *,
        market_state: torch.Tensor | None = None,
        market_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.ndim not in (3, 4) or values.shape[-1] != self.features:
            raise ValueError("Gate inputs need [dates, assets, (time), features].")
        dates = values.shape[0]
        if self.mode == "static":
            logits = self.gate_logits.expand(dates, -1)
        else:
            if market_state is None or market_available is None or (
                market_state.ndim != 2 or market_state.shape[0] != dates
                or market_available.shape != (dates,)
            ):
                raise ValueError("Market gate needs aligned market state and availability.")
            logits = self.market_projection(market_state)
        coefficients = self.features * F.softmax(logits / self.temperature, dim=-1)
        if self.mode == "market":
            coefficients = torch.where(
                market_available[:, None], coefficients, torch.ones_like(coefficients)
            )
        if values.ndim == 4:
            coefficients = coefficients[:, None, None, :]
        else:
            coefficients = coefficients[:, None, :]
        return values * coefficients


class MaskedLogitFusion(nn.Module):
    """Renormalize all weights over available branches, never over missing ones."""

    def __init__(
        self,
        branches: tuple[str, ...],
        *,
        mode: FusionMode,
        confidence_temperatures: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.branches = tuple(branches)
        self.mode = mode
        if mode == "static":
            self.branch_logits = nn.Parameter(torch.zeros(len(branches)))
        else:
            self.register_parameter("branch_logits", None)
        if mode == "confidence":
            if confidence_temperatures is None or set(confidence_temperatures) != set(branches):
                raise ValueError("Confidence fusion requires branch temperatures.")
            self.register_buffer(
                "temperatures",
                torch.tensor([confidence_temperatures[name] for name in branches], dtype=torch.float32),
            )
        else:
            self.register_buffer("temperatures", None)

    def forward(
        self, outputs: Mapping[str, BranchOutput]
    ) -> tuple[BranchOutput, torch.Tensor]:
        if any(name not in outputs for name in self.branches):
            raise ValueError("Fusion is missing a selected branch output.")
        selected = [outputs[name] for name in self.branches]
        dates, assets = selected[0].availability.shape
        for output in selected:
            output.validate_shape(batch_dates=dates, assets=assets)
        masks = torch.stack([output.availability for output in selected], dim=-1)
        logits = torch.stack([output.logits for output in selected], dim=-2)
        if self.mode in ("single", "mean"):
            scores = torch.ones_like(masks, dtype=logits.dtype)
        elif self.mode == "static":
            scores = self.branch_logits.softmax(dim=0).expand_as(masks)
        else:
            calibrated = logits / self.temperatures.view(1, 1, -1, 1)
            scores = calibrated.softmax(dim=-1).amax(dim=-1).detach()
        scores = scores * masks
        weights = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        availability = masks.any(dim=-1)
        fused_logits = (logits * weights.unsqueeze(-1)).sum(dim=-2)
        fused_representation = torch.cat(
            [output.representation * weights[..., index, None]
             for index, output in enumerate(selected)],
            dim=-1,
        )
        fused = BranchOutput(fused_logits, fused_representation, availability)
        fused.validate_shape(batch_dates=dates, assets=assets)
        return fused, weights


@dataclass(frozen=True)
class MultimodalPrediction:
    branches: dict[str, BranchOutput]
    fused: BranchOutput
    weights: torch.Tensor
    weight_branches: tuple[str, ...]


class MultimodalSystem(nn.Module):
    """Composable branches, optional market feature gates and explicit fusion."""

    def __init__(self, sample: MultimodalBatch, options: MultimodalOptions) -> None:
        super().__init__()
        if not isinstance(options, MultimodalOptions):
            raise TypeError("options must be MultimodalOptions.")
        self.options = options
        self.router = BranchRouter(sample, options.selection)
        market_width = (
            self.router.branches["market_transformer"].encoder.embedding_size
            if "market_transformer" in self.router.branches else 0
        )
        self.gru_gate = (
            MarketFeatureGate(
                sample.temporal.shape[-1], mode=options.gate_gru,
                market_width=market_width, temperature=options.gate_temperature,
            ) if options.gate_gru != "none" else None
        )
        self.gnn_gate = (
            MarketFeatureGate(
                sample.node.shape[-1], mode=options.gate_gnn,
                market_width=market_width, temperature=options.gate_temperature,
            ) if options.gate_gnn != "none" else None
        )
        self.fusion = MaskedLogitFusion(
            options.prediction_branches,
            mode=options.fusion,
            confidence_temperatures=options.confidence_temperatures,
        )

    def forward(self, batch: MultimodalBatch) -> MultimodalPrediction:
        outputs: dict[str, BranchOutput] = {}
        market_state = None
        market_available = None
        if "market_transformer" in self.router.branches:
            market_model = self.router.branches["market_transformer"]
            outputs["market_transformer"], market_state = market_model.forward_with_state(batch)
            market_available = torch.as_tensor(
                batch.market_sequence_mask, dtype=torch.bool, device=market_state.device
            )
        for name, branch in self.router.branches.items():
            if name == "market_transformer":
                continue
            branch_batch = batch
            gate = self.gru_gate if name == "gru" else self.gnn_gate if name == "gnn" else None
            if gate is not None:
                field = "temporal" if name == "gru" else "node"
                values = torch.as_tensor(
                    getattr(batch, field), dtype=torch.float32,
                    device=next(gate.parameters()).device,
                )
                transformed = gate(
                    values, market_state=market_state, market_available=market_available
                )
                branch_batch = replace(batch, **{field: transformed})
            outputs[name] = branch(branch_batch)
        outputs = {name: outputs[name] for name in self.options.selection.enabled}
        fused, weights = self.fusion(outputs)
        return MultimodalPrediction(
            branches=outputs,
            fused=fused,
            weights=weights,
            weight_branches=self.options.prediction_branches,
        )


__all__ = [
    "GateMode",
    "FusionMode",
    "MarketFeatureGate",
    "MaskedLogitFusion",
    "MultimodalOptions",
    "MultimodalPrediction",
    "MultimodalSystem",
]
