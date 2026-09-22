"""Optional, decision-causal normalization of an observed input window.

The expanding, rolling and Gaussian-score filters restart at the beginning of
each context window. They never inspect a target or a later observation.
"""

from __future__ import annotations

from typing import Any


def build_window_normalizer(config: Any, features: int, torch: Any, nn: Any) -> Any:
    class WindowNormalizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            indices = config.normalization_feature_indices
            if indices is not None and any(index >= features for index in indices):
                raise ValueError("normalization_feature_indices exceed input features.")
            mask = torch.zeros(features, dtype=torch.bool)
            mask[list(range(features)) if indices is None else list(indices)] = True
            self.register_buffer("feature_mask", mask, persistent=False)
            if config.input_normalization in ("revin", "revin_side"):
                self.affine_weight = nn.Parameter(torch.ones(features))
                self.affine_bias = nn.Parameter(torch.zeros(features))

        def forward(self, values: Any) -> tuple[Any, Any | None]:
            mode = config.input_normalization
            if mode == "none":
                return values, None
            epsilon = config.normalization_epsilon
            side = None
            if mode in ("revin", "revin_side"):
                mean = values.mean(dim=1, keepdim=True).detach()
                variance = values.var(dim=1, keepdim=True, unbiased=False).detach()
                scale = torch.sqrt(variance + epsilon)
                normalized = (values - mean) / scale
                normalized = normalized * self.affine_weight + self.affine_bias
                if mode == "revin_side":
                    mask = self.feature_mask.to(values.dtype).view(1, -1)
                    side = torch.cat((mean[:, 0] * mask, torch.log(scale[:, 0]) * mask), dim=-1)
            elif mode in ("expanding", "rolling"):
                length = values.shape[1]
                if mode == "expanding":
                    starts = torch.zeros(length, dtype=torch.long, device=values.device)
                else:
                    starts = torch.arange(length, device=values.device).sub(
                        config.normalization_window - 1
                    ).clamp_min(0)
                ends = torch.arange(1, length + 1, device=values.device)
                totals = torch.cat((torch.zeros_like(values[:, :1]), values.cumsum(dim=1)), dim=1)
                squares = torch.cat((torch.zeros_like(values[:, :1]), values.square().cumsum(dim=1)), dim=1)
                count = (ends - starts).to(values.dtype).view(1, -1, 1)
                mean = (totals[:, ends] - totals[:, starts]) / count
                variance = ((squares[:, ends] - squares[:, starts]) / count - mean.square()).clamp_min(0)
                normalized = (values - mean) / torch.sqrt(variance + epsilon)
            elif mode == "gas":
                # Gaussian score updates: mu += alpha*(x-mu),
                # variance += alpha*((x-mu)^2-variance). Current x is
                # normalized using the preceding state, then assimilated.
                mean = torch.zeros_like(values[:, 0])
                variance = torch.ones_like(mean)
                outputs = []
                for current in values.unbind(dim=1):
                    innovation = current - mean
                    outputs.append(innovation / torch.sqrt(variance + epsilon))
                    mean = mean + config.normalization_rate * innovation
                    variance = variance + config.normalization_rate * (innovation.square() - variance)
                    variance = variance.clamp_min(epsilon)
                normalized = torch.stack(outputs, dim=1)
            else:
                raise ValueError(f"Unknown input normalization: {mode}")
            return torch.where(self.feature_mask.view(1, 1, -1), normalized, values), side

    return WindowNormalizer()


__all__ = ["build_window_normalizer"]
