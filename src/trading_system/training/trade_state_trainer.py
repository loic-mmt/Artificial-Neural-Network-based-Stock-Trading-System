"""Closed-loop detached-state rollouts and one full-path update per epoch."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from trading_system.models.base import FitResult, TrainingHistory
from trading_system.models.neural.trade_state_gru import TradeStateGRUClassifier
from trading_system.models.neural.trainer import clone_torch_state_dict, restore_torch_state_dict, seed_torch_run
from trading_system.training.label_loss_trainer import _financial, _norm, _supervision, _validate_panel
from trading_system.training.trade_state import OvernightTradeBook, TradeStateConfig
from trading_system.training.weights import compute_class_weights


@dataclass
class TradeStateRollout:
    probabilities: np.ndarray
    log_probabilities: np.ndarray
    positions: np.ndarray
    states: np.ndarray
    raw_states: np.ndarray
    net_returns: np.ndarray


def chronological_batches(panel, batch_size):
    for day in range(len(panel.dates)):
        assets = np.flatnonzero(panel.indices[:, day] >= 0)
        for start in range(0, len(assets), batch_size):
            selected = assets[start:start + batch_size]
            yield day, selected, panel.indices[selected, day]


def rollout_trade_state_model(model, X, panel, *, state_config=None, loss_config, decoder="continuous"):
    """Fresh book each call; all assets observe the same pre-decision snapshot.

    The caller controls train/eval mode. Returned NumPy states carry no graph.
    They are replayed unchanged during backward, with frozen model weights.
    """
    if not isinstance(model, TradeStateGRUClassifier):
        raise TypeError("Closed-loop rollouts require TradeStateGRUClassifier.")
    if decoder not in ("continuous", "sign"):
        raise ValueError("decoder must be continuous or sign.")
    values = model._validate_sequences(X)
    _validate_panel(panel, len(values), "panel")
    if panel.protocol != "overnight":
        raise ValueError("Trade states currently require the overnight protocol.")
    state_config = state_config or TradeStateConfig()
    torch, device = model.torch, model.device
    n, width = len(values), model.state_size
    probabilities = np.empty((n, 3), np.float32)
    log_probabilities = np.empty((n, 3), np.float32)
    states, raw = np.empty((n, width), np.float32), np.empty((n, width), np.float64)
    positions, net = np.zeros(n), np.zeros(len(panel.dates))
    book = OvernightTradeBook(len(panel.tickers))
    visited = np.zeros(n, dtype=bool)
    with torch.no_grad():
        # Market encodings do not depend on book state. Encode bounded blocks
        # on the GPU once, then run the tiny deterministic state head on CPU.
        # This avoids thousands of small GRU launches/device round trips while
        # storing only [rows, hidden], never a graph spanning market sessions.
        embeddings = np.empty((n, model.module.encoder.embedding_size), np.float32)
        for start in range(0, n, model.config.batch_size):
            end = min(start + model.config.batch_size, n)
            embeddings[start:end] = model.module.encoder.encode(
                torch.as_tensor(values[start:end], device=device)).cpu().numpy()
        head_weight = model.module.head.weight.detach().cpu()
        head_bias = model.module.head.bias.detach().cpu()
        for day in range(len(panel.dates)):
            observations = book.observe()
            encoded = state_config.encode(observations)
            targets = np.zeros(len(panel.tickers))
            assets = np.flatnonzero(panel.indices[:, day] >= 0)
            for start in range(0, len(assets), model.config.batch_size):
                selected = assets[start:start + model.config.batch_size]
                rows = panel.indices[selected, day]
                states[rows], raw[rows] = encoded[selected], observations[selected]
                inputs = torch.cat((torch.from_numpy(embeddings[rows]), torch.from_numpy(states[rows])), dim=1)
                logits = torch.nn.functional.linear(inputs, head_weight, head_bias)
                log_p = torch.log_softmax(logits, dim=1)
                p = log_p.exp().cpu().numpy()
                probabilities[rows], log_probabilities[rows] = p, log_p.cpu().numpy()
                q = p[:, 2].astype(float) - p[:, 0].astype(float)
                if decoder == "sign":
                    q = np.sign(q)
                positions[rows] = q
                targets[selected] = np.where(panel.available[selected, day], q, 0.)
                visited[rows] = True
            # J->J+1 returns never enter observations for J.
            net[day] = book.advance(targets, panel.returns[:, day], loss_config.cost_bps)
    if not visited.all():
        raise ValueError("Panel omitted signal rows during chronological rollout.")
    expected = panel.path(positions, loss_config)[0]
    if not np.allclose(net, expected, rtol=1e-10, atol=1e-12):
        raise RuntimeError("State book and financial execution paths disagree.")
    return TradeStateRollout(probabilities, log_probabilities, positions, states, raw, net)


def fit_trade_state_model(model, X_train, y_train, known_train, X_val, y_val, known_val,
                          *, objective, train_panel, val_panel, loss_config,
                          state_config=None, progress_callback=None):
    """Semi-gradient training: exact current-decision gradients, detached history.

    Positions are continuous during both objective families' training. CE uses
    TRAIN-only class weights; the financial objective never consumes labels.
    Early stopping uses independently reset inner-validation books only.
    """
    started = perf_counter()
    if not isinstance(model, TradeStateGRUClassifier):
        raise TypeError("Trade-state training requires TradeStateGRUClassifier.")
    if objective not in ("cross_entropy", "financial"):
        raise ValueError("Only cross_entropy and financial are included in S0-S3.")
    if loss_config.objective == "cross_entropy":
        raise ValueError("A financial config is needed to simulate execution costs in every family.")
    state_config = state_config or TradeStateConfig()
    train, val = model._validate_sequences(X_train), model._validate_sequences(X_val)
    for panel, values in ((train_panel, train), (val_panel, val)):
        _validate_panel(panel, len(values), "panel")
    use_ce = objective == "cross_entropy"
    supervised, weights = None, None
    if use_ce:
        supervised = (_supervision(y_train, known_train, np.ones(len(train), bool), "y_train"),
                      _supervision(y_val, known_val, np.ones(len(val), bool), "y_val"))
        weights = compute_class_weights(supervised[0][0][supervised[0][1]], 3)
    config, torch, module, device = model.config, model.torch, model.module, model.device
    seed_torch_run(config.seed, config.deterministic, torch)
    parameters = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    class_weights = None if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=device)
    denominators = None if weights is None else [float(weights[y[m]].sum(dtype=np.float64)) for y, m in supervised]

    def objective_pass(values, split, panel):
        rollout = rollout_trade_state_model(model, values, panel, state_config=state_config, loss_config=loss_config)
        if use_ce:
            y, known = supervised[split]
            ce = float(-np.sum(rollout.log_probabilities[known, y[known]] * weights[y[known]], dtype=np.float64)
                       / denominators[split])
            losses, gradient = {"ce_loss": ce, "financial_loss": None, "total_loss": ce}, None
        else:
            financial, gradient = _financial(panel, rollout.positions, loss_config)
            losses = {"ce_loss": None, "financial_loss": financial, "total_loss": financial}
        if not np.isfinite(losses["total_loss"]):
            raise FloatingPointError("Non-finite trade-state objective.")
        return losses, gradient, rollout

    history, trace = TrainingHistory(), []
    best_loss, best_epoch, best_state, stale = np.inf, 0, None, 0
    stop_reason = "max_epochs"
    for epoch in range(config.epochs):
        epoch_started = perf_counter()
        epoch_seed = (config.seed + epoch + 1) % (2**32)
        module.train()
        seed_torch_run(epoch_seed, config.deterministic, torch)
        before, financial_gradient, cached = objective_pass(train, 0, train_panel)
        optimizer.zero_grad(set_to_none=True)
        # Identical encoder block shapes, dropout stream and frozen weights;
        # cached states already follow the chronological book. The linear head
        # has no stochastic operations, so replay can batch across sessions.
        seed_torch_run(epoch_seed, config.deterministic, torch)
        for start in range(0, len(train), config.batch_size):
            rows = np.arange(start, min(start + config.batch_size, len(train)))
            logits = module(torch.as_tensor(train[rows], device=device),
                            torch.as_tensor(cached.states[rows], device=device))
            if use_ce:
                y, known = supervised[0]
                local = known[rows]
                if not local.any():
                    continue
                index = torch.as_tensor(np.flatnonzero(local), device=device)
                targets = torch.as_tensor(y[rows][local], dtype=torch.long, device=device)
                loss = torch.nn.functional.cross_entropy(logits[index], targets,
                    weight=class_weights, reduction="sum") / denominators[0]
                loss.backward()
            else:
                p = torch.softmax(logits, dim=1)
                q = p[:, 2] - p[:, 0]
                q.backward(torch.as_tensor(financial_gradient[rows], dtype=q.dtype, device=device))
        del cached
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError("Missing/non-finite trade-state gradients.")
        norm = _norm(gradients, torch)
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
        optimizer.step()
        module.eval()
        train_losses, _, train_rollout = objective_pass(train, 0, train_panel)
        val_losses, _, val_rollout = objective_pass(val, 1, val_panel)
        del train_rollout, val_rollout
        history.train_loss.append(train_losses["total_loss"])
        history.val_loss.append(val_losses["total_loss"])
        if val_losses["total_loss"] < best_loss - config.early_stopping_min_delta:
            best_loss, best_epoch, stale = val_losses["total_loss"], epoch + 1, 0
            best_state = clone_torch_state_dict(module, torch)
        else:
            stale += 1
        record = {"epoch": epoch + 1, "optimizer_updates": epoch + 1,
            "train": train_losses, "validation": val_losses, "stochastic_train_before_update": before,
            "ce_gradient_norm": norm if use_ce else None,
            "financial_gradient_norm": None if use_ce else norm,
            "weighted_ce_gradient_norm": norm if use_ce else None,
            "weighted_financial_gradient_norm": None if use_ce else norm,
            "combined_gradient_norm_preclip": norm, "best_epoch": best_epoch, "stale_epochs": stale,
            "duration_seconds": float(perf_counter() - epoch_started)}
        trace.append(record)
        if progress_callback is not None:
            progress_callback(record)
        if stale >= config.early_stopping_patience:
            stop_reason = "early_stopping"
            break
    if best_state is None:
        raise RuntimeError("No finite inner-validation checkpoint.")
    restore_torch_state_dict(module, best_state, torch_module=torch)
    module.eval()
    model.fitted_ = True
    result = FitResult(best_epoch, stop_reason, history, training_duration_seconds=perf_counter() - started,
                      parameter_count=model.parameter_count(), seed=config.seed, device=str(device))
    model.fit_result_ = result
    recent = history.val_loss[-5:]
    improving = len(recent) >= 2 and recent[-1] < min(recent[:-1]) - config.early_stopping_min_delta
    model.learning_trace_ = trace
    model.learning_diagnostics_ = {"format_version": 1, "objective": objective,
        "state_config": asdict(state_config), "optimizer": "AdamW",
        "update_cadence": "one_global_step_per_epoch", "optimizer_updates": len(trace),
        "max_epochs": config.epochs, "best_epoch": best_epoch, "best_validation_loss": float(best_loss),
        "stop_reason": stop_reason, "budget_insufficient": bool(stop_reason == "max_epochs" and improving),
        "checkpoint_selection": "inner_validation_total_loss", "training_decoder": "continuous",
        "class_weights": None if weights is None else weights.astype(float).tolist(),
        "class_counts": None if not use_ce else np.bincount(supervised[0][0][supervised[0][1]], minlength=3).tolist(),
        "financial_loss_config": asdict(loss_config) if not use_ce else None,
        "book_reset": "cash_at_each_partition", "gradient_contract": "current_decision_only_detached_state_history",
        "trace": trace}
    return result
