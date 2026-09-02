from __future__ import annotations

import importlib
import math
import random
from collections.abc import Mapping
from functools import partial
from time import perf_counter
from typing import Any

import numpy as np

from trading_system.models.base import FitResult, TrainingHistory
from trading_system.training.weights import compute_class_weights

from .config import CommonTrainingConfig


def require_torch() -> tuple[Any, Any]:
    """Import PyTorch lazily so base package imports remain lightweight."""

    try:
        torch_module = importlib.import_module("torch")
        nn_module = importlib.import_module("torch.nn")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'PyTorch models require `pip install -e ".[neural]"`.'
        ) from error
    return torch_module, nn_module


def resolve_device(requested: str, torch_module: Any) -> Any:
    if requested not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'.")
    cuda_available = bool(torch_module.cuda.is_available())
    mps_backend = getattr(torch_module.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    if requested == "auto":
        requested = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    elif requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is unavailable.")
    elif requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is unavailable.")
    return torch_module.device(requested)


def seed_torch_run(seed: int, deterministic: bool, torch_module: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(bool(deterministic))
    cudnn = getattr(torch_module.backends, "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = bool(deterministic)
        cudnn.benchmark = not bool(deterministic)


def _seed_loader_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_tensor_loader(
    X: np.ndarray,
    y: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    torch_module: Any,
) -> Any:
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 3 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("X must be a non-empty finite float32 (N, T, F) array.")
    values = np.ascontiguousarray(values)
    tensors: list[Any] = [torch_module.from_numpy(values)]
    if y is not None:
        labels = np.asarray(y)
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("y must contain integer labels.")
        labels = labels.astype(np.int64, copy=False)
        if labels.ndim != 1 or len(labels) != len(values):
            raise ValueError("y must be a 1D array aligned with X.")
        tensors.append(torch_module.from_numpy(np.ascontiguousarray(labels)))
    elif shuffle:
        raise ValueError("Unlabeled prediction loaders cannot shuffle.")
    if batch_size <= 0 or num_workers < 0 or seed < 0:
        raise ValueError("Invalid loader batch_size, num_workers, or seed.")
    dataset = torch_module.utils.data.TensorDataset(*tensors)
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    return torch_module.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=partial(_seed_loader_worker, base_seed=seed),
    )


def build_weighted_cross_entropy(
    y_train: np.ndarray,
    num_classes: int,
    device: Any,
    torch_module: Any,
    nn_module: Any,
) -> Any:
    labels = np.asarray(y_train, dtype=np.int64)
    weights = compute_class_weights(labels, num_classes)
    tensor = torch_module.as_tensor(weights, dtype=torch_module.float32, device=device)
    return nn_module.CrossEntropyLoss(weight=tensor)


def train_one_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    *,
    gradient_clip_norm: float | None,
    torch_module: Any,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        if not bool(torch_module.isfinite(loss).item()):
            raise FloatingPointError("Training produced a non-finite loss.")
        loss.backward()
        if gradient_clip_norm is not None:
            torch_module.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
        optimizer.step()
        batch_size = int(len(features))
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Training loader is empty.")
    return total_loss / total_samples


def evaluate_loss(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    *,
    torch_module: Any,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch_module.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            loss = criterion(model(features), labels)
            if not bool(torch_module.isfinite(loss).item()):
                raise FloatingPointError("Validation produced a non-finite loss.")
            batch_size = int(len(features))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Validation loader is empty.")
    return total_loss / total_samples


def clone_torch_state_dict(model: Any, torch_module: Any) -> dict[str, Any]:
    del torch_module
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def fit_torch_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    *,
    num_classes: int,
    config: CommonTrainingConfig,
) -> FitResult:
    started = perf_counter()
    torch_module, nn_module = require_torch()
    device = resolve_device(config.device, torch_module)
    seed_torch_run(config.seed, config.deterministic, torch_module)
    model.to(device)
    train_loader = build_tensor_loader(
        X_train,
        y_train,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
        torch_module=torch_module,
    )
    validation_loader = None
    if X_val is not None and y_val is not None:
        validation_loader = build_tensor_loader(
            X_val,
            y_val,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed,
            num_workers=config.num_workers,
            torch_module=torch_module,
        )
    criterion = build_weighted_cross_entropy(
        y_train, num_classes, device, torch_module, nn_module
    )
    optimizer = torch_module.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history = TrainingHistory()
    best_state: dict[str, Any] | None = None
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    stop_reason = "max_epochs"
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip_norm=config.gradient_clip_norm,
            torch_module=torch_module,
        )
        history.train_loss.append(train_loss)
        selection_loss = train_loss
        if validation_loader is not None:
            selection_loss = evaluate_loss(
                model,
                validation_loader,
                criterion,
                device,
                torch_module=torch_module,
            )
            history.val_loss.append(selection_loss)
        if selection_loss < best_loss - config.early_stopping_min_delta:
            best_loss = selection_loss
            best_epoch = epoch + 1
            best_state = clone_torch_state_dict(model, torch_module)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            stop_reason = "early_stopping"
            break
    if best_state is None:
        raise RuntimeError("Training failed to record model state.")
    restore_torch_state_dict(model, best_state, torch_module=torch_module)
    return FitResult(
        best_epoch,
        stop_reason,
        history,
        training_duration_seconds=perf_counter() - started,
        parameter_count=int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        seed=config.seed,
        device=str(device),
    )


def predict_torch_probabilities(
    model: Any,
    X: np.ndarray,
    *,
    batch_size: int,
    device: Any,
    torch_module: Any,
) -> np.ndarray:
    loader = build_tensor_loader(
        X,
        None,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=0,
        torch_module=torch_module,
    )
    model.eval()
    outputs = []
    with torch_module.no_grad():
        for (features,) in loader:
            logits = model(features.to(device))
            outputs.append(
                torch_module.softmax(logits, dim=1).detach().cpu().numpy()
            )
    probabilities = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
        raise RuntimeError("Model returned invalid probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Model probabilities do not sum to one.")
    return probabilities


def restore_torch_state_dict(
    model: Any,
    state: Mapping[str, Any],
    *,
    torch_module: Any,
) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("Torch state must be a mapping.")
    current = model.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    if missing or unexpected:
        raise ValueError(
            f"Invalid torch state keys; missing={missing}, unexpected={unexpected}."
        )
    restored = {}
    for name, target in current.items():
        source = state[name]
        tensor = (
            source.detach()
            if isinstance(source, torch_module.Tensor)
            else torch_module.as_tensor(source)
        )
        if tuple(tensor.shape) != tuple(target.shape):
            raise ValueError(
                f"Torch state {name} has shape {tuple(tensor.shape)}; "
                f"expected {tuple(target.shape)}."
            )
        restored[name] = tensor.to(device=target.device, dtype=target.dtype)
    model.load_state_dict(restored, strict=True)
    model.eval()


__all__ = [
    "build_tensor_loader",
    "build_weighted_cross_entropy",
    "clone_torch_state_dict",
    "evaluate_loss",
    "fit_torch_model",
    "predict_torch_probabilities",
    "require_torch",
    "resolve_device",
    "restore_torch_state_dict",
    "seed_torch_run",
    "train_one_epoch",
]
