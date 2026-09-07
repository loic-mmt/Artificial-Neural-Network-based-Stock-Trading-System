"""Exact full-path gradients with bounded activation memory.

Two passes replay identical dropout randomness at frozen weights. The first
retains only positions; the second recomputes one block's activations and applies
the global objective's position gradient. One optimizer step follows all blocks.
"""

from time import perf_counter

import numpy as np

from trading_system.models.base import FitResult, TrainingHistory
from trading_system.models.manual_ann.sequence_adapter import ManualANNSequenceAdapter
from trading_system.models.manual_ann.manual_nn import dropout_mask, softmax
from .financial_loss import position_coefficients, probabilities_to_positions


def predict_positions(model, values, position_mode):
    result = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), model.config.batch_size):
        end = start + model.config.batch_size
        result[start:end] = probabilities_to_positions(model.predict_proba(values[start:end]), position_mode)
    return result


class _NumpyBackend:
    def __init__(self, model, X):
        self.model = model
        self.config = model.config
        self.flat = model.estimator
        rng = np.random.default_rng(self.config.seed)
        self.flat.W0_ = (0.01 * rng.standard_normal((X.shape[1] * X.shape[2], self.config.hidden_size))).astype(np.float32)
        self.flat.b0_ = np.zeros((1, self.config.hidden_size), np.float32)
        self.flat.W1_ = (0.01 * rng.standard_normal((self.config.hidden_size, 3))).astype(np.float32)
        self.flat.b1_ = np.zeros((1, 3), np.float32)
        model.context_len_, model.feature_count_ = X.shape[1:]

    def reset_randomness(self, seed):
        self.rng = np.random.default_rng(seed)

    def forward(self, values, coefficients, *, backward=False, gradient=None):
        x = values.reshape(len(values), -1)
        W0, b0, W1, b1 = self.flat._state()
        hidden = x @ W0 + b0
        active = hidden > 0
        np.maximum(hidden, 0, out=hidden)
        mask = None
        if self.config.dropout_probability:
            mask = dropout_mask(hidden.shape, self.config.dropout_probability, self.rng)
            hidden *= mask
        probabilities = softmax(hidden @ W1 + b1)
        positions = probabilities @ coefficients
        if backward:
            logits_gradient = (probabilities * (coefficients - positions[:, None]) * gradient[:, None]).astype(np.float32)
            d_hidden = logits_gradient @ W1.T
            if mask is not None:
                d_hidden *= mask
            d_hidden *= active
            for accumulator, value in zip(self.gradients, (x.T @ d_hidden, d_hidden.sum(axis=0, keepdims=True), hidden.T @ logits_gradient, logits_gradient.sum(axis=0, keepdims=True))):
                accumulator += value
        return positions

    def zero_grad(self):
        self.gradients = [np.zeros_like(value) for value in self.flat._state()]

    def step(self):
        state = self.flat._state()
        for parameter, gradient in zip(state, self.gradients):
            if not np.isfinite(gradient).all():
                raise FloatingPointError("Non-finite ANN financial gradient.")
            parameter -= self.config.learning_rate * gradient
        if self.config.weight_decay:
            decay = 1.0 - self.config.learning_rate * self.config.weight_decay
            state[0] *= decay
            state[2] *= decay


class _TorchBackend:
    def __init__(self, model, X):
        from trading_system.models.neural.base import TorchSequenceClassifier
        if not isinstance(model, TorchSequenceClassifier):
            raise TypeError("Financial loss supports Manual ANN and built-in PyTorch models.")
        self.model = model
        self.torch = model.torch
        self.optimizer = self.torch.optim.AdamW(model.module.parameters(), lr=model.config.learning_rate, weight_decay=model.config.weight_decay)
        model.fitted_ = True

    def reset_randomness(self, seed):
        from trading_system.models.neural.trainer import seed_torch_run
        seed_torch_run(seed, self.model.config.deterministic, self.torch)
        self.model.module.train()

    def forward(self, values, coefficients, *, backward=False, gradient=None):
        torch = self.torch
        with torch.set_grad_enabled(backward):
            x = torch.as_tensor(values, device=self.model.device)
            probabilities = torch.softmax(self.model.module(x), dim=1)
            positions = probabilities @ torch.as_tensor(coefficients, dtype=probabilities.dtype, device=self.model.device)
            if backward:
                positions.backward(torch.as_tensor(gradient, dtype=positions.dtype, device=self.model.device))
        return positions.detach().cpu().numpy()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        parameters = list(self.model.module.parameters())
        if any(parameter.grad is not None and not bool(self.torch.isfinite(parameter.grad).all()) for parameter in parameters):
            raise FloatingPointError("Non-finite PyTorch financial gradient.")
        if self.model.config.gradient_clip_norm is not None:
            self.torch.nn.utils.clip_grad_norm_(parameters, self.model.config.gradient_clip_norm)
        self.optimizer.step()


def fit_position_model(model, X_train, train_panel, X_val, val_panel, loss_config, position_mode):
    """Optimize net return or regularized Sharpe, selecting by validation loss."""
    started = perf_counter()
    for values, panel in ((X_train, train_panel), (X_val, val_panel)):
        if values.ndim != 3 or len(values) != panel.rows or not np.isfinite(values).all():
            raise ValueError("Financial training requires finite aligned (N, T, F) arrays.")
    if X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("Train and validation sequence dimensions differ.")
    if loss_config.objective == "cross_entropy":
        raise ValueError("Use the unchanged classifier trainer for cross_entropy.")
    backend = _NumpyBackend(model, X_train) if isinstance(model, ManualANNSequenceAdapter) else _TorchBackend(model, X_train)
    config = model.config
    coefficients = position_coefficients(position_mode).astype(np.float32)
    history = TrainingHistory()
    best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0
    stop_reason = "max_epochs"
    for epoch in range(config.epochs):
        seed = (config.seed + epoch + 1) % (2**32)
        backend.reset_randomness(seed)
        positions = np.empty(len(X_train), dtype=np.float64)
        for start in range(0, len(X_train), config.batch_size):
            end = start + config.batch_size
            positions[start:end] = backend.forward(X_train[start:end], coefficients)
        _, gradient = train_panel.loss_and_gradient(positions, loss_config)
        backend.zero_grad()
        backend.reset_randomness(seed)
        for start in range(0, len(X_train), config.batch_size):
            end = start + config.batch_size
            backend.forward(X_train[start:end], coefficients, backward=True, gradient=gradient[start:end])
        backend.step()
        # Evaluation has dropout disabled and uses the entire chronological path.
        train_loss, _ = train_panel.loss_and_gradient(predict_positions(model, X_train, position_mode), loss_config)
        val_loss, _ = val_panel.loss_and_gradient(predict_positions(model, X_val, position_mode), loss_config)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        if val_loss < best_loss - config.early_stopping_min_delta:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = model.state_dict()
        else:
            stale += 1
        if stale >= config.early_stopping_patience:
            stop_reason = "early_stopping"
            break
    if best_state is None:
        raise RuntimeError("No finite position model checkpoint.")
    model.load_state_dict(best_state)
    result = FitResult(best_epoch, stop_reason, history, training_duration_seconds=perf_counter() - started,
                       parameter_count=model.parameter_count(), seed=config.seed,
                       device=str(getattr(model, "device", "cpu")))
    model.fit_result_ = result
    if isinstance(model, ManualANNSequenceAdapter):
        model.estimator.fit_result_ = result
    return result
