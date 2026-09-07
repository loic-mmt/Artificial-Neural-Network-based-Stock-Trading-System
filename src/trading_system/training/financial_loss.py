"""Chronological position objectives, shared by NumPy, PyTorch and evaluation."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FinancialLossConfig:
    objective: str = "pnl"
    cost_bps: float = 5.0
    annualization: int = 252
    sharpe_epsilon: float = 1e-4

    def __post_init__(self):
        if self.objective not in ("cross_entropy", "pnl", "sharpe"):
            raise ValueError("objective must be cross_entropy, pnl, or sharpe.")
        if not math.isfinite(self.cost_bps) or not 0 <= self.cost_bps < 10000:
            raise ValueError("cost_bps must be finite and in [0, 10000).")
        if isinstance(self.annualization, bool) or not isinstance(self.annualization, int) or self.annualization <= 0:
            raise ValueError("annualization must be a positive integer.")
        if not math.isfinite(self.sharpe_epsilon) or self.sharpe_epsilon <= 0:
            raise ValueError("sharpe_epsilon must be finite and positive.")


def position_coefficients(position_mode):
    if position_mode == "long_short":
        return np.array([-1., 0., 1.])
    if position_mode == "long_only":
        return np.array([0., 0., 1.])
    raise ValueError("position_mode must be long_only or long_short.")


def probabilities_to_positions(probabilities, position_mode):
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite (N, 3) probabilities.")
    if (values < 0).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Probabilities must be non-negative and sum to one.")
    return values @ position_coefficients(position_mode)


class ReturnPanel:
    """Complete synchronized asset calendars; no invented returns for missing bars.

    A signal at close t is executed at close t+delay, earns the following
    close-to-close return, and cannot cross a supplied split boundary. Each split
    starts flat and liquidates at its final observed close. No terminal entry.
    """

    def __init__(self, frame, *, price_col="adj_close", date_col="date",
                 group_col=None, execution_delay=1):
        if isinstance(execution_delay, bool) or not isinstance(execution_delay, int) or execution_delay < 1:
            raise ValueError("Position experiments require execution_delay >= 1.")
        work = frame.reset_index(drop=True).copy()
        work[date_col] = pd.to_datetime(work[date_col], utc=True, errors="raise")
        if work[date_col].isna().any():
            raise ValueError("Position dates cannot be missing.")
        if group_col is not None and (group_col not in work or work[group_col].isna().any()):
            raise ValueError("Position tickers must be present and non-missing.")
        groups = [part for _, part in work.groupby(group_col, sort=True)] if group_col else [work]
        indices, prices, calendar = [], [], None
        for part in groups:
            part = part.sort_values(date_col)
            dates = part[date_col].to_numpy()
            if len(dates) < execution_delay + 3 or part[date_col].duplicated().any():
                raise ValueError("Each asset needs unique dates and at least delay + 3 rows.")
            if calendar is not None and not np.array_equal(calendar, dates):
                raise ValueError("Position portfolios require identical asset calendars; align data explicitly.")
            calendar = dates
            price = part[price_col].to_numpy(dtype=np.float64)
            if not np.isfinite(price).all() or (price <= 0).any():
                raise ValueError("Position prices must be finite and positive.")
            indices.append(part.index.to_numpy())
            prices.append(price)
        if not indices:
            raise ValueError("Empty position panel.")
        self.indices = np.stack(indices)
        prices = np.stack(prices)
        self.returns = prices[:, 1:] / prices[:, :-1] - 1
        self.delay = execution_delay
        self.rows = len(work)
        self.dates = calendar[1:]

    def path(self, positions, config):
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (self.rows,) or not np.isfinite(positions).all() or (np.abs(positions) > 1 + 1e-6).any():
            raise ValueError("Positions must be finite, aligned and bounded by [-1, 1].")
        target = positions[self.indices]
        executed = np.zeros_like(self.returns)
        executed[:, self.delay:] = target[:, :self.returns.shape[1] - self.delay]
        delta = np.diff(executed, axis=1, prepend=0.)
        turnover = np.abs(delta)
        # Liquidation is charged at the last known close, with no extra return.
        turnover[:, -1] += np.abs(executed[:, -1])
        costs = config.cost_bps * 1e-4 * turnover
        net = executed * self.returns - costs
        return net.mean(axis=0), executed, delta, turnover, costs

    def loss_and_gradient(self, positions, config):
        if config.objective == "cross_entropy":
            raise ValueError("Cross entropy requires labeled classifier training.")
        net, executed, delta, _, _ = self.path(positions, config)
        count = len(net)
        mean = net.mean()
        if config.objective == "pnl":
            loss = -mean
            upstream = np.full(count, -1. / count)
        else:
            centered = net - mean
            scale = np.sqrt(np.mean(centered ** 2) + config.sharpe_epsilon ** 2)
            annual = np.sqrt(config.annualization)
            loss = -annual * mean / scale
            upstream = -annual / count * (1 / scale - mean * centered / scale ** 3)
        upstream = upstream / len(self.indices)
        rate = config.cost_bps * 1e-4
        signs = np.sign(delta)  # zero subgradient at the absolute-value kink
        gradient = upstream * (self.returns - rate * signs)
        gradient[:, :-1] += rate * signs[:, 1:] * upstream[1:]
        gradient[:, -1] -= rate * np.sign(executed[:, -1]) * upstream[-1]
        target_gradient = np.zeros(self.indices.shape, dtype=np.float64)
        target_gradient[:, :self.returns.shape[1] - self.delay] = gradient[:, self.delay:]
        aligned_gradient = np.zeros(self.rows, dtype=np.float64)
        aligned_gradient[self.indices] = target_gradient
        if not np.isfinite(loss) or not np.isfinite(aligned_gradient).all():
            raise FloatingPointError("Non-finite financial objective or gradient.")
        return float(loss), aligned_gradient

    def metrics(self, positions, config, initial_capital=10000.):
        if not math.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("initial_capital must be finite and positive.")
        net, executed, _, turnover, costs = self.path(positions, config)
        if (net <= -1).any():
            raise ValueError("Position portfolio became insolvent (net return <= -1).")
        wealth = np.r_[1., np.cumprod(1 + net)]
        std = net.std(ddof=0)
        # Flat portfolios have zero Sharpe; nonzero constant returns have no
        # unregularized Sharpe and are explicitly unavailable in the report.
        sharpe = np.sqrt(config.annualization) * net.mean() / std if std > 0 else (0. if np.all(net == 0) else None)
        return {
            "net_pnl": float(initial_capital * (wealth[-1] - 1)),
            "net_return": float(wealth[-1] - 1),
            "mean_net_return": float(net.mean()),
            "net_sharpe": None if sharpe is None else float(sharpe),
            "regularized_sharpe": float(np.sqrt(config.annualization) * net.mean() / np.sqrt(std ** 2 + config.sharpe_epsilon ** 2)),
            "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
            "turnover": float(turnover.mean(axis=0).sum()),
            "cost_return_sum": float(costs.mean(axis=0).sum()),
            "mean_abs_position": float(np.abs(executed).mean()),
            "periods": len(net),
            "assets": len(self.indices),
        }
