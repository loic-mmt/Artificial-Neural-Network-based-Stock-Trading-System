"""Detached chronological binary policies; bounded memory and one update/epoch."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from trading_system.models.base import FitResult, TrainingHistory
from trading_system.models.neural.short_decision_gru import ShortDecisionGRUClassifier
from trading_system.models.neural.trade_state_gru import TradeStateGRUClassifier
from trading_system.models.neural.trainer import clone_torch_state_dict, restore_torch_state_dict, seed_torch_run
from trading_system.training.label_loss_trainer import _financial, _norm, _supervision, _validate_panel
from trading_system.training.trade_state import OvernightTradeBook, TradeStateConfig

SHORT_POLICIES = ("long_short", "long_flat", "short_flat")


@dataclass(frozen=True)
class ShortPolicyConfig:
    long_activation_threshold: float = .5
    short_activation_threshold: float = .5
    conflict_policy: str = "flat"

    def __post_init__(self):
        if any(not np.isfinite(v) or not 0 < v < 1 for v in
               (self.long_activation_threshold, self.short_activation_threshold)):
            raise ValueError("Activation thresholds must be finite and in (0, 1).")
        if self.conflict_policy not in ("flat", "margin"):
            raise ValueError("Conflict policy must be flat or margin.")


def project_policy(q, policy):
    if policy == "long_short":
        return q
    if policy == "long_flat":
        return q.clamp_min(0) if hasattr(q, "clamp_min") else np.maximum(q, 0)
    if policy == "short_flat":
        return q.clamp_max(0) if hasattr(q, "clamp_max") else np.minimum(q, 0)
    raise ValueError("Unknown execution policy.")


def decode_short_scores(scores, architecture, *, decoder="continuous", policy="long_short", config=None):
    """Torch/NumPy parity; conflicts/ties are fixed before OUTER, never fitted there.

    Head order is Short for a detector; Long, Short for the other variants.
    Long/Flat and Short/Flat project the same directional decision. A closed-loop
    rerun can change future scores; the benchmark separately freezes alerts for
    its pure Short-versus-Flat counterfactual.
    """
    config = config or ShortPolicyConfig()
    if decoder not in ("continuous", "sign"):
        raise ValueError("Decoder must be continuous or sign.")
    width = 1 if architecture == "short_detector" else 2
    if architecture not in ("short_detector", "dual_head", "separate_grus") or scores.ndim != 2 or scores.shape[1] != width:
        raise ValueError("Scores do not match the short architecture.")
    is_tensor = hasattr(scores, "detach")
    if not is_tensor and (not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Finite activation scores in [0, 1] required.")
    short = scores[:, -1]
    if decoder == "continuous":
        q = 1 - 2 * short if width == 1 else scores[:, 0] - short
    else:
        alert = short > config.short_activation_threshold
        if width == 1:
            q = 1 - 2 * (alert.to(scores.dtype) if is_tensor else alert.astype(float))
        else:
            long = scores[:, 0] > config.long_activation_threshold
            q = (long & ~alert).to(scores.dtype) - (alert & ~long).to(scores.dtype) if is_tensor else (long & ~alert).astype(float) - (alert & ~long).astype(float)
            if config.conflict_policy == "margin":
                margin = ((scores[:, 0] - config.long_activation_threshold) / (1 - config.long_activation_threshold)
                          - (short - config.short_activation_threshold) / (1 - config.short_activation_threshold))
                if is_tensor:
                    import torch
                    q = torch.where(long & alert, margin.sign(), q)
                else:
                    q = np.where(long & alert, np.sign(margin), q)
    return project_policy(q, policy)


def binary_supervision(labels, known, head_names):
    """Unknown stays masked; Short-vs-not-Short is explicit, not a Flat label."""
    y, selected = _supervision(labels, known, np.ones(len(labels), bool), "labels")
    targets = np.column_stack([y == (0 if name == "short" else 2) for name in head_names]).astype(np.float32)
    return targets, selected


def binary_class_weights(targets, known):
    """Balanced TRAIN-only weights per head, including single-class TRAIN."""
    result = []
    for column in targets.T:
        counts = np.bincount(column[known].astype(int), minlength=2)
        weights = np.divide(counts.sum(), np.count_nonzero(counts) * counts,
                            out=np.zeros(2, dtype=float), where=counts > 0)
        result.append(weights)
    return np.asarray(result)


@dataclass
class ShortDecisionRollout:
    scores: np.ndarray
    logits: np.ndarray
    positions: np.ndarray
    states: np.ndarray
    raw_states: np.ndarray
    net_returns: np.ndarray


def rollout_short_decisions(model, X, panel, *, state_config, loss_config, decoder="continuous",
                            policy="long_short", policy_config=None):
    if not isinstance(model, (ShortDecisionGRUClassifier, TradeStateGRUClassifier)):
        raise TypeError("Short or reference trade-state GRU required.")
    if state_config.gradient_mode != "detached" or panel.protocol != "overnight":
        raise ValueError("Short study fixes detached overnight states.")
    if decoder not in ("continuous", "sign") or policy not in SHORT_POLICIES:
        raise ValueError("Invalid decoder or policy.")
    values = model._validate_sequences(X)
    _validate_panel(panel, len(values), "panel")
    torch, module = model.torch, model.module
    reference = isinstance(model, TradeStateGRUClassifier)
    head_count = 3 if reference else len(model.head_names)
    scores = np.empty((len(values), head_count), np.float32)
    logits = np.empty_like(scores)
    states, raw = np.empty((len(values), 10), np.float32), np.empty((len(values), 10))
    positions, net, visited = np.zeros(len(values)), np.zeros(len(panel.dates)), np.zeros(len(values), bool)
    book = OvernightTradeBook(len(panel.tickers))
    with torch.no_grad():
        # Encode without graphs in memory blocks; only [rows, branches, hidden]
        # survives the pass, not the recurrent activations for the entire book.
        embeddings = []
        for start in range(0, len(values), model.config.batch_size):
            end = min(start + model.config.batch_size, len(values))
            sequences = torch.as_tensor(values[start:end], device=model.device)
            encoded = module.encoder.encode(sequences) if reference else module.encode(sequences)
            if start == 0:
                embeddings = np.empty((len(values), *encoded.shape[1:]), np.float32)
            embeddings[start:end] = encoded.cpu().numpy()
        heads = [module.head] if reference else module.heads
        weights = [(h.weight.detach().cpu(), h.bias.detach().cpu()) for h in heads]
        for day in range(len(panel.dates)):
            observations = book.observe(); encoded = state_config.encode(observations)
            assets = np.flatnonzero(panel.indices[:, day] >= 0); targets = np.zeros(len(panel.tickers))
            for start in range(0, len(assets), model.config.batch_size):
                selected = assets[start:start + model.config.batch_size]; rows = panel.indices[selected, day]
                states[rows], raw[rows] = encoded[selected], observations[selected]
                if reference:
                    inputs = torch.cat((torch.from_numpy(embeddings[rows]), torch.from_numpy(states[rows])), dim=1)
                    local = torch.nn.functional.linear(inputs, *weights[0])
                    p = torch.log_softmax(local, dim=1).exp().numpy(); q = p[:, 2].astype(float) - p[:, 0].astype(float)
                    if decoder == "sign": q = np.sign(q)
                    q = project_policy(q, policy)
                else:
                    branch_count = embeddings.shape[1]
                    local = torch.cat([torch.nn.functional.linear(torch.cat((
                        torch.from_numpy(embeddings[rows, i if branch_count == 2 else 0]),
                        torch.from_numpy(states[rows])), dim=1), *weight)
                        for i, weight in enumerate(weights)], dim=1)
                    p = torch.sigmoid(local).numpy()
                    q = decode_short_scores(p, model.config.architecture, decoder=decoder,
                                            policy=policy, config=policy_config)
                logits[rows], scores[rows], positions[rows] = local.numpy(), p, q
                targets[selected] = np.where(panel.available[selected, day], q, 0.)
                visited[rows] = True
            net[day] = book.advance(targets, panel.returns[:, day], loss_config.cost_bps)
    if not visited.all() or not np.allclose(net, panel.path(positions, loss_config)[0], rtol=1e-10, atol=1e-12):
        raise RuntimeError("Short policy book and execution panel disagree.")
    return ShortDecisionRollout(scores, logits, positions, states, raw, net)


def fit_short_decision_model(model, X_train, y_train, known_train, X_val, y_val, known_val,
                             *, objective, train_panel, val_panel, loss_config, state_config,
                             progress_callback=None):
    if not isinstance(model, ShortDecisionGRUClassifier) or objective not in ("financial", "cross_entropy"):
        raise ValueError("Short GRU and financial or binary CE objective required.")
    started = perf_counter()
    train, val = model._validate_sequences(X_train), model._validate_sequences(X_val)
    torch, module, cfg, device = model.torch, model.module, model.config, model.device
    use_ce = objective == "cross_entropy"
    supervised, weights, denominators = None, None, None
    if use_ce:
        supervised = [binary_supervision(y, known, model.head_names)
                      for y, known in ((y_train, known_train), (y_val, known_val))]
        weights = binary_class_weights(*supervised[0])
        denominators = [weights[np.arange(len(weights))[None, :], y[known].astype(int)].sum(axis=0)
                        for y, known in supervised]
        if any(np.any(d <= 0) for d in denominators):
            raise ValueError("INNER targets have no TRAIN-supported weighted examples.")
    seed_torch_run(cfg.seed, cfg.deterministic, torch)
    parameters = list(module.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def ce_tensor(logits, targets, known, split):
        if not known.any(): return None
        local = logits[torch.as_tensor(known, device=logits.device)]
        y = torch.as_tensor(targets[known], device=logits.device)
        w = torch.as_tensor(weights, dtype=y.dtype, device=y.device)
        row_weights = torch.where(y > 0, w[:, 1], w[:, 0])
        losses = torch.nn.functional.binary_cross_entropy_with_logits(local, y, reduction="none")
        denominator = torch.as_tensor(denominators[split], device=y.device, dtype=y.dtype)
        return ((losses * row_weights).sum(dim=0) / denominator).mean()

    def objective_pass(values, panel, split):
        rolled = rollout_short_decisions(model, values, panel, state_config=state_config, loss_config=loss_config)
        if use_ce:
            y, known = supervised[split]
            ce = float(ce_tensor(torch.as_tensor(rolled.logits), y, known, split))
            losses, gradient = {"ce_loss": ce, "financial_loss": None, "total_loss": ce}, None
        else:
            financial, gradient = _financial(panel, rolled.positions, loss_config)
            losses = {"ce_loss": None, "financial_loss": financial, "total_loss": financial}
        if not np.isfinite(losses["total_loss"]): raise FloatingPointError("Non-finite short objective.")
        return losses, gradient, rolled

    history, trace = TrainingHistory(), []
    best, best_state, best_epoch, stale, stop = np.inf, None, 0, 0, "max_epochs"
    for epoch in range(cfg.epochs):
        epoch_started = perf_counter(); epoch_seed = (cfg.seed + epoch + 1) % (2**32)
        module.train(); seed_torch_run(epoch_seed, cfg.deterministic, torch)
        before, gradient, cached = objective_pass(train, train_panel, 0)
        optimizer.zero_grad(set_to_none=True); seed_torch_run(epoch_seed, cfg.deterministic, torch)
        for start in range(0, len(train), cfg.batch_size):
            end = min(start + cfg.batch_size, len(train))
            logits = module(torch.as_tensor(train[start:end], device=device),
                            torch.as_tensor(cached.states[start:end], device=device))
            if use_ce:
                y, known = supervised[0]; loss = ce_tensor(logits, y[start:end], known[start:end], 0)
                if loss is not None: loss.backward()
            else:
                q = decode_short_scores(torch.sigmoid(logits), cfg.architecture)
                q.backward(torch.as_tensor(gradient[start:end], device=device, dtype=q.dtype))
        buffers = [p.grad for p in parameters if p.grad is not None]
        if not buffers or any(not bool(torch.isfinite(g).all()) for g in buffers):
            raise FloatingPointError("Missing/non-finite short gradients.")
        norm = _norm(buffers, torch)
        if cfg.gradient_clip_norm is not None: torch.nn.utils.clip_grad_norm_(parameters, cfg.gradient_clip_norm)
        optimizer.step(); del cached
        module.eval()
        train_losses, _, _ = objective_pass(train, train_panel, 0)
        val_losses, _, _ = objective_pass(val, val_panel, 1)
        history.train_loss.append(train_losses["total_loss"]); history.val_loss.append(val_losses["total_loss"])
        if val_losses["total_loss"] < best - cfg.early_stopping_min_delta:
            best, best_epoch, stale = val_losses["total_loss"], epoch + 1, 0
            best_state = clone_torch_state_dict(module, torch)
        else: stale += 1
        record = {"epoch": epoch + 1, "optimizer_updates": epoch + 1, "train": train_losses,
            "validation": val_losses, "stochastic_train_before_update": before,
            "weighted_ce_gradient_norm": norm if use_ce else None,
            "weighted_financial_gradient_norm": None if use_ce else norm,
            "combined_gradient_norm_preclip": norm, "best_epoch": best_epoch, "stale_epochs": stale,
            "duration_seconds": perf_counter() - epoch_started}
        trace.append(record)
        if progress_callback: progress_callback(record)
        if stale >= cfg.early_stopping_patience: stop = "early_stopping"; break
    if best_state is None: raise RuntimeError("No finite inner-validation short checkpoint.")
    restore_torch_state_dict(module, best_state, torch_module=torch); module.eval(); model.fitted_ = True
    result = FitResult(best_epoch, stop, history, training_duration_seconds=perf_counter() - started,
                      parameter_count=model.parameter_count(), seed=cfg.seed, device=str(device))
    model.fit_result_, model.learning_trace_ = result, trace
    recent = history.val_loss[-5:]
    model.learning_diagnostics_ = {"objective": objective, "architecture": cfg.architecture,
        "binary_head_names": model.head_names, "binary_train_class_weights": None if weights is None else weights.tolist(),
        "binary_target_contract": "Short vs not-Short; Long vs not-Long; unknown excluded",
        "state_config": asdict(state_config), "state_gradient_mode": "detached",
        "training_policy": "long_short", "training_decoder": "continuous", "optimizer": "AdamW",
        "update_cadence": "one_global_step_per_epoch", "optimizer_updates": len(trace),
        "best_epoch": best_epoch, "best_validation_loss": best, "stop_reason": stop,
        "budget_insufficient": bool(stop == "max_epochs" and len(recent) >= 2 and recent[-1] < min(recent[:-1]) - cfg.early_stopping_min_delta),
        "financial_loss_config": None if use_ce else asdict(loss_config), "trace": trace}
    return result
