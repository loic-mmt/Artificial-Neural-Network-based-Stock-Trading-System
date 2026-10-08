"""Isolated full-path CE, financial and hybrid trainer for label benchmarks.

Weights are frozen during both chronological passes of an epoch. The first
pass keeps only scalar losses and one position per row. The second replays the
same dropout stream and accumulates parameter gradients from bounded blocks.
Exactly one AdamW step follows each epoch, regardless of the objective family.
Existing sequence and position trainers deliberately remain unchanged.
"""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np

from trading_system.models.base import FitResult, TrainingHistory
from trading_system.models.neural.base import TorchSequenceClassifier
from trading_system.models.neural.trainer import (
    clone_torch_state_dict,
    restore_torch_state_dict,
    seed_torch_run,
)
from trading_system.training.financial_loss import FinancialLossConfig
from trading_system.training.weights import compute_class_weights


def _mask(values, rows, name, *, default=True):
    if values is None:
        return np.full(rows, default, dtype=bool)
    result = np.asarray(values)
    if result.shape != (rows,) or result.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean vector aligned with X.")
    return result


def _supervision(y, known, available, name):
    labels = np.asarray(y)
    if labels.shape != available.shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{name} must be an aligned integer label vector.")
    selected = _mask(known, len(labels), f"known_{name}") & available
    if not selected.any():
        raise ValueError(f"{name} requires at least one known available label.")
    if np.any((labels[selected] < 0) | (labels[selected] >= 3)):
        raise ValueError(f"{name} known labels must belong to [0, 2].")
    # Unknown labels may use any sentinel. They never enter CE or its weights.
    return labels.astype(np.int64, copy=False), selected


def _validate_panel(panel, rows, name):
    if panel is None or getattr(panel, "rows", None) != rows:
        raise ValueError(f"{name} must expose an aligned rows count.")
    if not callable(getattr(panel, "loss_and_gradient", None)):
        raise TypeError(f"{name} must implement loss_and_gradient.")


def _financial(panel, positions, config):
    value, gradient = panel.loss_and_gradient(positions, config)
    gradient = np.asarray(gradient, dtype=np.float64)
    if not np.isfinite(value) or gradient.shape != positions.shape or not np.isfinite(gradient).all():
        raise FloatingPointError("Non-finite or misaligned financial loss/gradient.")
    return float(value), gradient


def _norm(buffers, torch):
    # Do not flatten/copy all parameters into one additional tensor.
    total = sum(float(value.detach().to(device="cpu", dtype=torch.float64).square().sum().item()) for value in buffers)
    return float(np.sqrt(total))


def fit_label_loss_model(
    model,
    X_train,
    y_train,
    known_train,
    X_val,
    y_val,
    known_val,
    *,
    objective,
    train_panel=None,
    val_panel=None,
    loss_config=None,
    hybrid_ce_weight=0.5,
    available_train=None,
    available_val=None,
    progress_callback=None,
):
    """Fit one three-class sequence model, selecting only on inner validation.

    ``financial`` intentionally never inspects ``y_*`` or ``known_*``. All
    rows remain in the return chronology, including unavailable/fallback rows,
    whose targets and gradients are forced to zero. For CE, the known mask and
    availability mask jointly select supervision; class weights use TRAIN only.

    ``hybrid_ce_weight`` combines globally normalized CE with the financial
    objective. It is independent of ``loss_config.combined_pnl_weight``.
    ``progress_callback``, when supplied, receives one JSON-safe epoch record.
    """
    started = perf_counter()
    if not isinstance(model, TorchSequenceClassifier):
        raise TypeError("Label loss training requires TorchSequenceClassifier.")
    if objective not in ("cross_entropy", "financial", "hybrid"):
        raise ValueError("objective must be cross_entropy, financial, or hybrid.")
    if not np.isfinite(hybrid_ce_weight) or not 0 <= hybrid_ce_weight <= 1:
        raise ValueError("hybrid_ce_weight must be finite and in [0, 1].")
    use_ce = objective in ("cross_entropy", "hybrid")
    use_financial = objective in ("financial", "hybrid")
    ce_coefficient = 1.0 if objective == "cross_entropy" else float(hybrid_ce_weight) if use_ce else 0.0
    financial_coefficient = 1.0 if objective == "financial" else 1.0 - float(hybrid_ce_weight) if use_financial else 0.0
    if use_financial:
        if loss_config is None:
            loss_config = FinancialLossConfig("combined", combined_pnl_weight=0.25)
        if loss_config.objective == "cross_entropy":
            raise ValueError("Financial component requires a financial loss_config.")
    train = model._validate_sequences(X_train)
    validation = model._validate_sequences(X_val)
    availability = (
        _mask(available_train, len(train), "available_train"),
        _mask(available_val, len(validation), "available_val"),
    )
    supervised = (None, None)
    weights = None
    if use_ce:
        supervised = (
            _supervision(y_train, known_train, availability[0], "y_train"),
            _supervision(y_val, known_val, availability[1], "y_val"),
        )
        weights = compute_class_weights(supervised[0][0][supervised[0][1]], 3)
    if use_financial:
        _validate_panel(train_panel, len(train), "train_panel")
        _validate_panel(val_panel, len(validation), "val_panel")

    config, torch, module, device = model.config, model.torch, model.module, model.device
    seed_torch_run(config.seed, config.deterministic, torch)
    module.to(device)
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters.")
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    class_weights = None if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=device)
    denominators = (
        None if not use_ce else float(np.sum(weights[supervised[0][0][supervised[0][1]]], dtype=np.float64)),
        None if not use_ce else float(np.sum(weights[supervised[1][0][supervised[1][1]]], dtype=np.float64)),
    )
    coefficients = torch.as_tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device=device)

    def block_loss(logits, split, start, end):
        labels, selected = supervised[split]
        local = selected[start:end]
        if not local.any():
            return None
        index = torch.as_tensor(np.flatnonzero(local), device=device)
        targets = torch.as_tensor(labels[start:end][local], dtype=torch.long, device=device)
        return torch.nn.functional.cross_entropy(
            logits[index], targets, weight=class_weights, reduction="sum"
        ) / denominators[split]

    def objective_pass(values, split, panel):
        ce_loss = 0.0 if use_ce else None
        positions = np.empty(len(values), dtype=np.float64) if use_financial else None
        with torch.no_grad():
            for start in range(0, len(values), config.batch_size):
                end = min(start + config.batch_size, len(values))
                logits = module(torch.as_tensor(values[start:end], device=device))
                if use_ce:
                    loss = block_loss(logits, split, start, end)
                    if loss is not None:
                        ce_loss += float(loss.item())
                if use_financial:
                    q = (torch.softmax(logits, dim=1) @ coefficients).cpu().numpy()
                    positions[start:end] = np.where(availability[split][start:end], q, 0.0)
        financial_loss, gradient = (None, None) if not use_financial else _financial(panel, positions, loss_config)
        total = (0.0 if ce_loss is None else ce_coefficient * ce_loss) + (
            0.0 if financial_loss is None else financial_coefficient * financial_loss
        )
        if not np.isfinite(total):
            raise FloatingPointError("Non-finite combined label objective.")
        return {"ce_loss": ce_loss, "financial_loss": financial_loss, "total_loss": float(total)}, gradient

    history = TrainingHistory()
    trace = []
    best_loss, best_epoch, best_state, stale = np.inf, 0, None, 0
    stop_reason = "max_epochs"
    for epoch in range(config.epochs):
        epoch_started = perf_counter()
        epoch_seed = (config.seed + epoch + 1) % (2**32)
        module.train()
        seed_torch_run(epoch_seed, config.deterministic, torch)
        before, financial_gradient = objective_pass(train, 0, train_panel)
        ce_gradients = [torch.zeros_like(parameter) for parameter in parameters] if use_ce else None
        financial_gradients = [torch.zeros_like(parameter) for parameter in parameters] if use_financial else None
        optimizer.zero_grad(set_to_none=True)
        # Replaying the first pass prevents dropout from applying the derivative
        # of the financial path to a different set of stochastic predictions.
        seed_torch_run(epoch_seed, config.deterministic, torch)
        for start in range(0, len(train), config.batch_size):
            end = min(start + config.batch_size, len(train))
            logits = module(torch.as_tensor(train[start:end], device=device))
            ce = block_loss(logits, 0, start, end) if use_ce else None
            if ce is not None:
                gradients = torch.autograd.grad(ce, parameters, retain_graph=use_financial, allow_unused=True)
                for accumulator, gradient in zip(ce_gradients, gradients):
                    if gradient is not None:
                        accumulator.add_(gradient.detach())
            if use_financial:
                q = torch.softmax(logits, dim=1) @ coefficients
                upstream = financial_gradient[start:end] * availability[0][start:end]
                gradients = torch.autograd.grad(
                    q, parameters,
                    grad_outputs=torch.as_tensor(upstream, dtype=q.dtype, device=device),
                    allow_unused=True,
                )
                for accumulator, gradient in zip(financial_gradients, gradients):
                    if gradient is not None:
                        accumulator.add_(gradient.detach())
        for index, parameter in enumerate(parameters):
            parameter.grad = torch.zeros_like(parameter)
            if use_ce:
                parameter.grad.add_(ce_gradients[index], alpha=ce_coefficient)
            if use_financial:
                parameter.grad.add_(financial_gradients[index], alpha=financial_coefficient)
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("Non-finite full-path parameter gradient.")
        ce_norm = None if ce_gradients is None else _norm(ce_gradients, torch)
        financial_norm = None if financial_gradients is None else _norm(financial_gradients, torch)
        combined_norm = _norm([parameter.grad for parameter in parameters], torch)
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
        optimizer.step()
        module.eval()
        train_losses, _ = objective_pass(train, 0, train_panel)
        val_losses, _ = objective_pass(validation, 1, val_panel)
        history.train_loss.append(train_losses["total_loss"])
        history.val_loss.append(val_losses["total_loss"])
        selection_loss = val_losses["total_loss"]
        if selection_loss < best_loss - config.early_stopping_min_delta:
            best_loss, best_epoch, stale = selection_loss, epoch + 1, 0
            best_state = clone_torch_state_dict(module, torch)
        else:
            stale += 1
        record = {
            "epoch": epoch + 1, "optimizer_updates": epoch + 1,
            "train": train_losses, "validation": val_losses,
            "stochastic_train_before_update": before,
            "ce_gradient_norm": ce_norm, "financial_gradient_norm": financial_norm,
            "weighted_ce_gradient_norm": None if ce_norm is None else ce_coefficient * ce_norm,
            "weighted_financial_gradient_norm": None if financial_norm is None else financial_coefficient * financial_norm,
            "combined_gradient_norm_preclip": combined_norm,
            "best_epoch": best_epoch, "stale_epochs": stale,
            "duration_seconds": float(perf_counter() - epoch_started),
        }
        trace.append(record)
        if progress_callback is not None:
            progress_callback(record)
        if stale >= config.early_stopping_patience:
            stop_reason = "early_stopping"
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    restore_torch_state_dict(module, best_state, torch_module=torch)
    module.eval()
    model.fitted_ = True
    result = FitResult(
        best_epoch, stop_reason, history,
        training_duration_seconds=perf_counter() - started,
        parameter_count=model.parameter_count(), seed=config.seed, device=str(device),
    )
    model.fit_result_ = result
    # A cap reached during recent meaningful validation progress is a budget
    # warning, not a conclusion that the labels are impossible to learn.
    recent = history.val_loss[-5:]
    improving = len(recent) >= 2 and recent[-1] < min(recent[:-1]) - config.early_stopping_min_delta
    model.learning_trace_ = trace
    model.learning_diagnostics_ = {
        "format_version": 1, "objective": objective,
        "optimizer": "AdamW", "update_cadence": "one_global_step_per_epoch",
        "optimizer_updates": len(trace), "max_epochs": config.epochs,
        "best_epoch": best_epoch, "best_validation_loss": float(best_loss),
        "stop_reason": stop_reason,
        "budget_insufficient": bool(stop_reason == "max_epochs" and improving),
        "checkpoint_selection": "inner_validation_total_loss",
        "hybrid_ce_weight": float(hybrid_ce_weight) if objective == "hybrid" else None,
        "ce_coefficient": ce_coefficient, "financial_coefficient": financial_coefficient,
        "financial_loss_config": asdict(loss_config) if use_financial else None,
        "class_weights": None if weights is None else weights.astype(float).tolist(),
        "class_counts": None if not use_ce else np.bincount(
            supervised[0][0][supervised[0][1]], minlength=3
        ).tolist(),
        "supervised_train_rows": None if not use_ce else int(supervised[0][1].sum()),
        "supervised_validation_rows": None if not use_ce else int(supervised[1][1].sum()),
        "available_train_rows": int(availability[0].sum()),
        "available_validation_rows": int(availability[1].sum()),
        "trace": trace,
    }
    return result


__all__ = ["fit_label_loss_model"]
