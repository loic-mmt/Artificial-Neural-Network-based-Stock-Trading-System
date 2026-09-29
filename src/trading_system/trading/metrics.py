"""Metrics use one self-financing portfolio, including initial capital."""

import numpy as np
import pandas as pd


def portfolio_metrics(equity, orders, trades, config):
    wealth = np.r_[config.initial_capital, equity.equity.to_numpy(dtype=float)]
    returns = wealth[1:] / wealth[:-1] - 1
    excess = returns - ((1 + config.risk_free_rate) ** (1 / config.annualization) - 1)
    std = excess.std(ddof=0)
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    contributions = {}
    if not trades.empty:
        contributions = {str(k): float(v) for k, v in trades.groupby("exit_reason").net_pnl.sum().items()}
    return {
        "initial_capital": config.initial_capital,
        "final_capital": float(wealth[-1]),
        "net_pnl": float(wealth[-1] - wealth[0]),
        "net_return": float(wealth[-1] / wealth[0] - 1),
        "sharpe": float(np.sqrt(config.annualization) * excess.mean() / std) if std > 0 else (0. if np.all(excess == 0) else None),
        "sortino": float(np.sqrt(config.annualization) * excess.mean() / downside) if downside > 0 else None,
        "max_drawdown": float(drawdown.min()),
        "turnover": float(orders.turnover.sum()) if not orders.empty else 0.,
        "fees": float(orders.fee.sum()) if not orders.empty else 0.,
        "slippage_cost": float(orders.slippage_cost.sum()) if not orders.empty else 0.,
        "order_count": len(orders), "trade_count": len(trades),
        "mean_gross_exposure": float(equity.gross_exposure.mean()),
        "mean_net_exposure": float(equity.net_exposure.mean()),
        "win_rate": float(trades.net_pnl.gt(0).mean()) if not trades.empty else None,
        "pnl_by_exit_reason": contributions,
    }


def frame_fingerprint(frame):
    import hashlib
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()
