"""Bounded state-history VJPs, without retaining GRU graphs across sessions.

The forward trading policy and discrete episode accounting are unchanged.
Gradients cross previous positions, exposures and marked equity/drawdown only.
Sign changes, entry anchors, ages and underwater counters remain piecewise
constant. This is partial, truncated BPTT, not a differentiable order engine.
"""

import numpy as np

from trading_system.training.trade_state import OvernightTradeBook

GRADIENT_STATE_COLUMNS = (
    "previous_position", "portfolio_gross_exposure", "portfolio_net_exposure",
    "portfolio_cash_fraction", "portfolio_drawdown",
)


class DifferentiableOvernightTradeBook:
    """Exact forward book, with a graph on continuous state channels on CPU."""

    def __init__(self, assets):
        import torch

        self.torch = torch
        self.reference = OvernightTradeBook(assets)
        self.position = torch.zeros(assets, dtype=torch.float64)
        self.equity = torch.ones((), dtype=torch.float64)
        self.peak = torch.ones((), dtype=torch.float64)

    def observe(self, config):
        torch = self.torch
        observations = self.reference.observe()
        raw = torch.as_tensor(observations, dtype=torch.float64)
        gross, net = self.position.abs().mean(), self.position.mean()
        drawdown = self.equity / self.peak - 1
        columns = [self.position, raw[:, 1], torch.log1p(raw[:, 2]) / np.log1p(config.age_scale),
            torch.tanh(raw[:, 3] / config.return_scale), torch.tanh(raw[:, 4] / config.return_scale),
            torch.log1p(raw[:, 5]) / np.log1p(config.age_scale), gross.expand(len(raw)),
            net.expand(len(raw)), (1 - gross).expand(len(raw)), drawdown.expand(len(raw))]
        encoded = torch.stack([value if i < config.active_width else torch.zeros_like(value)
                               for i, value in enumerate(columns)], dim=1)
        encoded = encoded.to(dtype=torch.float32)
        # Preserve exactly the existing NumPy transforms in the forward pass.
        # Tiny tanh/reduction roundoff must not trigger different hard episodes.
        canonical = torch.from_numpy(config.encode(observations))
        return canonical + (encoded - encoded.detach())

    def advance(self, targets, next_returns, cost_bps):
        torch = self.torch
        q = targets.to(dtype=torch.float64)
        returns = torch.as_tensor(next_returns, dtype=torch.float64)
        if q.shape != self.position.shape:
            raise ValueError("Differentiable book targets must match the universe.")
        # The NumPy reference validates bounds/solvency and owns all discrete
        # episode events. Detaching those events does not detach q's amplitude.
        expected = self.reference.advance(q.detach().numpy(), next_returns, cost_bps)
        net = (q * returns - cost_bps * 1e-4 * (q - self.position).abs()).mean()
        if not np.isclose(float(net.detach()), expected, rtol=1e-10, atol=1e-12):
            raise RuntimeError("Differentiable and reference trading books disagree.")
        self.position = q
        self.equity = self.equity * (1 + net)
        # At ties use the old peak, matching the zero drawdown subgradient.
        self.peak = torch.where(self.equity > self.peak, self.equity, self.peak)
        return net

    def detach(self):
        """Keep actual book values, but cut credit assignment at a boundary."""
        self.position = self.position.detach()
        self.equity = self.equity.detach()
        self.peak = self.peak.detach()


def state_history_vjp(model, panel, cached, upstream, state_config, loss_config, steps):
    """Replay the tiny CPU head chronologically; return encoder/head gradients.

    ``upstream`` is d(full financial loss)/d(q) on the complete calendar, not
    a separate Sharpe per chunk. Windows cut only the policy-state graph.
    Encoder outputs are leaf inputs: the expensive GRU is replayed afterward
    in the same original blocks/dropout stream, using their accumulated VJP.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("TBPTT steps must be a positive number of global sessions.")
    if cached.embeddings is None:
        raise ValueError("State-history replay requires cached encoder embeddings.")
    torch = model.torch
    weight = model.module.head.weight.detach().cpu().clone().requires_grad_()
    bias = model.module.head.bias.detach().cpu().clone().requires_grad_()
    embedding_gradient = np.zeros_like(cached.embeddings)
    book = DifferentiableOvernightTradeBook(len(panel.tickers))
    maximum_error, windows = 0., 0
    for first in range(0, len(panel.dates), steps):
        terms, leaves = [], []
        for day in range(first, min(first + steps, len(panel.dates))):
            states = book.observe(state_config)
            targets = torch.zeros(len(panel.tickers), dtype=torch.float64)
            assets = np.flatnonzero(panel.indices[:, day] >= 0)
            for start in range(0, len(assets), model.config.batch_size):
                selected = assets[start:start + model.config.batch_size]
                rows = panel.indices[selected, day]
                embedding = torch.from_numpy(cached.embeddings[rows]).requires_grad_()
                inputs = torch.cat((embedding, states[selected]), dim=1)
                p = torch.log_softmax(torch.nn.functional.linear(inputs, weight, bias), dim=1).exp()
                q = p[:, 2].double() - p[:, 0].double()
                error = float(np.max(np.abs(q.detach().numpy() - cached.positions[rows])))
                maximum_error = max(maximum_error, error)
                if not np.allclose(q.detach().numpy(), cached.positions[rows], rtol=1e-5, atol=2e-6):
                    raise RuntimeError("TBPTT replay changed the frozen forward policy.")
                # Numerical value alignment only; this is not a surrogate
                # gradient for sign/argmax. Continuous policy credit is kept.
                q = torch.from_numpy(cached.positions[rows]) + (q - q.detach())
                terms.append((q * torch.as_tensor(upstream[rows], dtype=q.dtype)).sum())
                leaves.append((rows, embedding))
                active = torch.as_tensor(panel.available[selected, day])
                targets = targets.index_copy(0, torch.as_tensor(selected), torch.where(active, q, 0.))
            book.advance(targets, panel.returns[:, day], loss_config.cost_bps)
        if terms:
            torch.stack(terms).sum().backward()
            for rows, embedding in leaves:
                if embedding.grad is not None:
                    embedding_gradient[rows] = embedding.grad.numpy()
        book.detach()
        windows += 1
        # Release the window before constructing the next one. No encoder
        # graph is present here, even when the requested window is large.
        del terms, leaves
    head_gradients = tuple(torch.zeros_like(value) if value.grad is None else value.grad.detach()
                           for value in (weight, bias))
    return embedding_gradient, head_gradients, {
        "tbptt_windows": windows, "tbptt_max_graph_sessions": min(steps, len(panel.dates)),
        "state_replay_max_position_error": maximum_error,
    }
