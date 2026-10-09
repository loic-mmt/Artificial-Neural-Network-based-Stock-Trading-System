"""Causal, detached observations of an equal-slot overnight trading book."""

from dataclasses import dataclass

import numpy as np

STATE_COLUMNS = (
    "previous_position", "is_open", "trade_age_sessions",
    "signed_return_since_entry", "trade_return_giveback",
    "consecutive_underwater_sessions", "portfolio_gross_exposure",
    "portfolio_net_exposure", "portfolio_cash_fraction", "portfolio_drawdown",
)
STATE_VARIANTS = ("S0", "S1", "S2", "S3")
_WIDTH = {"S0": 0, "S1": 3, "S2": 6, "S3": 10}


@dataclass(frozen=True)
class TradeStateConfig:
    variant: str = "S0"
    return_scale: float = .1
    age_scale: float = 252.
    gradient_mode: str = "detached"

    def __post_init__(self):
        if self.variant not in STATE_VARIANTS:
            raise ValueError("variant must be S0, S1, S2 or S3.")
        if self.gradient_mode != "detached":
            raise ValueError("Only detached states are implemented; TBPTT is a later benchmark.")
        if any(not np.isfinite(v) or v <= 0 for v in (self.return_scale, self.age_scale)):
            raise ValueError("State return/age scales must be finite and positive.")

    def encode(self, observations):
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(STATE_COLUMNS) or not np.isfinite(values).all():
            raise ValueError("Finite [assets, 10] state observations required.")
        encoded = values.copy()
        encoded[:, [2, 5]] = np.log1p(values[:, [2, 5]]) / np.log1p(self.age_scale)
        encoded[:, [3, 4]] = np.tanh(values[:, [3, 4]] / self.return_scale)
        encoded[:, _WIDTH[self.variant]:] = 0.
        return encoded.astype(np.float32)


class OvernightTradeBook:
    """Observe before J's decision; mark J->J+1 only after that decision.

    Resizing without a sign change preserves the episode's entry anchor and
    age. Entry return is a signed adjusted-price return, not a quantity-weighted
    accounting PnL. Gross/net/cash describe the previous q/N targets, not a
    broker margin balance. No oracle labels, stop rule or duration cap is used.
    """

    def __init__(self, assets):
        if not isinstance(assets, int) or assets <= 0:
            raise ValueError("A positive asset count is required.")
        self.position = np.zeros(assets)
        self.price = np.ones(assets)
        self.entry = np.ones(assets)
        self.age = np.zeros(assets, dtype=np.int64)
        self.best = np.zeros(assets)
        self.underwater = np.zeros(assets, dtype=np.int64)
        self.equity = self.peak = 1.

    def observe(self):
        opened = self.position != 0
        signed = np.where(opened, np.sign(self.position) * (self.price / self.entry - 1), 0.)
        gross, net = np.abs(self.position).mean(), self.position.mean()
        return np.column_stack((self.position, opened, self.age, signed,
            np.where(opened, np.maximum(self.best - signed, 0), 0), self.underwater,
            np.full(len(opened), gross), np.full(len(opened), net),
            np.full(len(opened), 1 - gross), np.full(len(opened), self.equity / self.peak - 1)))

    def advance(self, targets, next_returns, cost_bps):
        """Execute simultaneously, then advance to the next decision instant."""
        q, returns = np.asarray(targets, float), np.asarray(next_returns, float)
        if q.shape != self.position.shape or returns.shape != q.shape:
            raise ValueError("Book targets and returns must align with its asset universe.")
        if (not np.isfinite(q).all() or not np.isfinite(returns).all()
                or (np.abs(q) > 1 + 1e-6).any() or (returns <= -1).any()
                or not np.isfinite(cost_bps) or cost_bps < 0):
            raise ValueError("Invalid book positions, returns or costs.")
        q = np.clip(q, -1, 1)
        reset = (np.sign(q) != np.sign(self.position)) | (q == 0)
        self.entry[reset] = self.price[reset]
        self.age[reset] = self.underwater[reset] = 0
        self.best[reset] = 0.
        net = float(np.mean(q * returns - cost_bps * 1e-4 * np.abs(q - self.position)))
        if not np.isfinite(net) or net <= -1:
            raise FloatingPointError("Trade-state portfolio became insolvent.")
        self.position = q.copy()
        self.price *= 1 + returns
        opened = q != 0
        self.age[opened] += 1
        signed = np.sign(q) * (self.price / self.entry - 1)
        self.best = np.where(opened, np.maximum(self.best, signed), 0.)
        self.underwater = np.where(opened & (signed < 0), self.underwater + 1, 0)
        self.equity *= 1 + net
        self.peak = max(self.peak, self.equity)
        return net
