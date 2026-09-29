"""Train-only, pointwise denoising of standardized continuous features.

This is a small DAE/feature-attention DAE adaptation, not an AGRUA replica.
Categorical/calendar columns bypass the network and the decoder is linear.
"""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch
from torch import nn

from trading_system.features.market import SECTOR_ONE_HOT_FEATURES
from trading_system.models.neural.trainer import resolve_device, seed_torch_run


@dataclass(frozen=True)
class DenoisingConfig:
    hidden_size: int = 64
    latent_size: int = 8
    epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    noise_std: float = 0.1

    def __post_init__(self):
        for name in ("hidden_size", "latent_size", "epochs", "patience", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(self.noise_std) or self.noise_std <= 0:
            raise ValueError("noise_std must be finite and positive.")


def continuous_indices(columns):
    """Exclude discrete features explicitly, without learning from test values."""
    discrete = set(SECTOR_ONE_HOT_FEATURES) | {
        "day_of_week", "day", "week", "month", "year", "quarter",
        "trend_regime", "vol_regime", "sector_surge_breadth",
        "is_month_end", "is_month_start", "is_quarter_end", "is_year_end",
    }
    return tuple(i for i, column in enumerate(columns)
                 if column.removeprefix("lag1_") not in discrete
                 and not column.endswith(("_available", "_missing", "_mask")))


class FeatureAutoencoder(nn.Module):
    def __init__(self, width, config, attention=False):
        super().__init__()
        latent = min(config.latent_size, max(1, width // 2))
        self.encoder = nn.Sequential(nn.Linear(width, config.hidden_size), nn.ReLU(),
                                     nn.Linear(config.hidden_size, latent), nn.ReLU())
        self.attention = nn.Linear(latent, latent) if attention else None
        self.decoder = nn.Sequential(nn.Linear(latent, config.hidden_size), nn.ReLU(),
                                     nn.Linear(config.hidden_size, width))

    def forward(self, values):
        latent = self.encoder(values)
        if self.attention is not None:
            latent = latent * torch.softmax(self.attention(latent), dim=-1)
        return self.decoder(latent)


class FrozenDenoiser:
    def __init__(self, model, columns, indices, device, report):
        self.model = model.eval().requires_grad_(False)
        self.columns = tuple(columns)
        self.indices = tuple(indices)
        self.device = device
        self.report = report

    def transform(self, values):
        if values.ndim not in (2, 3) or values.shape[-1] != len(self.columns):
            raise ValueError("Denoiser feature dimensions do not match.")
        if not np.isfinite(values).all():
            raise ValueError("Denoiser requires finite standardized features.")
        output = np.array(values, dtype=np.float32, copy=True, order="C")
        flat = output.reshape(-1, output.shape[-1])
        with torch.no_grad():
            for start in range(0, len(flat), 8192):
                block = flat[start:start + 8192]
                # Selection happens in NumPy, never in the MPS autograd graph.
                tensor = torch.as_tensor(block[:, self.indices], device=self.device)
                reconstructed = self.model(tensor).cpu().numpy()
                if not np.isfinite(reconstructed).all():
                    raise FloatingPointError("Non-finite denoiser output.")
                block[:, self.indices] = reconstructed
        return output

    def checkpoint(self):
        return {"state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "columns": self.columns, "indices": self.indices, "report": self.report}


def fit_denoiser(train, inner, columns, *, config, attention=False, seed=1, device="auto"):
    """Fit on unique train window endpoints, select epoch by clean inner MSE.

    Each endpoint is one ticker/session, not a duplicated overlapping window.
    No outer-fold inputs or target returns are accepted by this function.
    """
    for values in (train, inner):
        if (values.ndim != 3 or not len(values) or values.shape[-1] != len(columns)
                or not np.isfinite(values).all()):
            raise ValueError("Expected finite, non-empty standardized windows.")
    indices = continuous_indices(columns)
    if not indices:
        raise ValueError("No continuous features to denoise.")
    started = perf_counter()
    resolved = resolve_device(device, torch)
    seed_torch_run(seed, True, torch)
    model = FeatureAutoencoder(len(indices), config, attention).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.)
    clean_train = np.asarray(train[:, -1, :][:, indices], dtype=np.float32)
    clean_inner = np.asarray(inner[:, -1, :][:, indices], dtype=np.float32)
    rng = np.random.default_rng(seed)
    best, state, best_epoch, stale = np.inf, None, 0, 0
    history = []
    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(len(clean_train))
        for start in range(0, len(order), config.batch_size):
            clean = clean_train[order[start:start + config.batch_size]]
            noise = rng.normal(0, config.noise_std, clean.shape).astype(np.float32)
            target = torch.as_tensor(clean, device=resolved)
            prediction = model(torch.as_tensor(clean + noise, device=resolved))
            objective = torch.nn.functional.mse_loss(prediction, target)
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("Non-finite reconstruction loss.")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        total = 0.
        with torch.no_grad():
            for start in range(0, len(clean_inner), config.batch_size):
                target = torch.as_tensor(clean_inner[start:start + config.batch_size], device=resolved)
                total += float(torch.nn.functional.mse_loss(model(target), target, reduction="sum"))
        mse = total / clean_inner.size
        if not np.isfinite(mse):
            raise FloatingPointError("Non-finite validation reconstruction loss.")
        history.append(mse)
        if mse < best - 1e-6:
            best, best_epoch, stale = mse, epoch + 1, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(state)
    report = {"config": asdict(config), "attention": attention, "seed": seed,
              "device": str(resolved), "best_epoch": best_epoch,
              "inner_mse": history, "best_inner_mse": best,
              "fit_rows": len(clean_train), "selection_rows": len(clean_inner),
              "continuous_columns": [columns[i] for i in indices],
              "passthrough_columns": [c for i, c in enumerate(columns) if i not in indices],
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "training_seconds": perf_counter() - started}
    return FrozenDenoiser(model, columns, indices, resolved, report)
