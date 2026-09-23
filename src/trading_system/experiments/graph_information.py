"""Exploratory, date-blocked diagnostics for already trained graph controls.

These statistics never fit or select a trading rule. Each seed/fold is reported
separately because seeds share the same realized market path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.training.financial_loss import ReturnPanel


def moving_block_interval(values, *, block_length=20, samples=1000, seed=1):
    """Percentile interval for a daily mean, preserving local serial dependence."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a non-empty finite daily series.")
    if isinstance(block_length, bool) or not isinstance(block_length, int) or block_length < 1:
        raise ValueError("block_length must be a positive integer.")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 100:
        raise ValueError("samples must be an integer >= 100.")
    length = len(values)
    width = min(block_length, length)
    blocks = (length + width - 1) // width
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        starts = rng.integers(0, length - width + 1, size=blocks)
        selected = (starts[:, None] + np.arange(width)).ravel()[:length]
        means[index] = values[selected].mean()
    lower, upper = np.quantile(means, [.025, .975])
    return {"mean": float(values.mean()), "ci_95": [float(lower), float(upper)]}


def _conditional_block_interval(values, condition, *, block_length, samples, seed):
    """Bootstrap a regime mean by resampling the full calendar in date blocks."""
    values = np.asarray(values, dtype=np.float64)
    condition = np.asarray(condition, dtype=bool)
    if values.shape != condition.shape or not condition.any():
        return None
    length = len(values)
    width = min(block_length, length)
    blocks = (length + width - 1) // width
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        starts = rng.integers(0, length - width + 1, size=blocks)
        selected = (starts[:, None] + np.arange(width)).ravel()[:length]
        selected_mask = condition[selected]
        if selected_mask.any():
            means.append(values[selected][selected_mask].mean())
    if not means:
        return None
    lower, upper = np.quantile(means, [.025, .975])
    return {"dates": int(condition.sum()), "mean_bps_per_day": float(values[condition].mean() * 1e4),
            "ci_95_bps_per_day": [float(lower * 1e4), float(upper * 1e4)]}


def paired_graph_information(
    paired: pd.DataFrame,
    loss,
    *,
    execution_delay=1,
    block_length=20,
    samples=1000,
    seed=1,
    signal_threshold=.05,
):
    """Compare one GNN against its matched GRU on a single outer fold/seed.

    `paired` has one row per date/ticker and the columns `gru_position`,
    `gnn_position`, `adj_close`, `gnn_degree`, `high_volatility`. Conditional signal attribution
    is gross; only the daily portfolio contrast includes turnover costs.
    """
    required = {"date", "ticker", "adj_close", "gru_position", "gnn_position",
                "gnn_degree", "high_volatility"}
    if required - set(paired):
        raise ValueError(f"Missing paired columns: {sorted(required - set(paired))}")
    work = paired.sort_values(["date", "ticker"]).reset_index(drop=True).copy()
    if work.duplicated(["date", "ticker"]).any():
        raise ValueError("Paired predictions must have unique ticker/date rows.")
    if not 0 <= signal_threshold < 1:
        raise ValueError("signal_threshold must be in [0, 1).")
    gru = work.gru_position.to_numpy(dtype=np.float64)
    gnn = work.gnn_position.to_numpy(dtype=np.float64)
    degree = work.gnn_degree.to_numpy(dtype=np.float64)
    if not np.isfinite(np.stack((gru, gnn, degree))).all() or (degree < 0).any():
        raise ValueError("Positions and graph degrees must be finite; degrees non-negative.")
    panel = ReturnPanel(work, price_col="adj_close", date_col="date",
                        group_col="ticker", execution_delay=execution_delay)
    gru_net, _, _, _, _ = panel.path(gru, loss)
    gnn_net, _, _, _, _ = panel.path(gnn, loss)
    daily_net_delta = gnn_net - gru_net
    # Signal t executes at t+delay and earns the subsequent close-to-close return.
    future = np.full(panel.indices.shape, np.nan, dtype=np.float64)
    eligible = panel.returns.shape[1] - panel.delay
    future[:, :eligible] = panel.returns[:, panel.delay:]
    forward_return = np.empty(len(work), dtype=np.float64)
    forward_return[panel.indices] = future
    valid = np.isfinite(forward_return)
    opposed = valid & (np.abs(gru) >= signal_threshold) & (np.abs(gnn) >= signal_threshold) & (gru * gnn < 0)
    connected = valid & (degree > 0)
    # The mean over all assets gives a daily portfolio contribution; zeros for
    # non-selected assets keep degree/disagreement groups comparable.
    gross_delta = np.where(valid, (gnn - gru) * np.nan_to_num(forward_return), 0.)
    disagreement_daily = np.zeros_like(gross_delta)
    disagreement_daily[opposed] = gross_delta[opposed]
    connected_daily = np.zeros_like(gross_delta)
    connected_daily[connected] = gross_delta[connected]
    dates = pd.to_datetime(work.date, utc=True)
    per_day = pd.DataFrame({"date": dates[valid], "disagreement": disagreement_daily[valid],
                            "connected": connected_daily[valid]}).groupby("date", sort=True).mean()
    regime = work.high_volatility.to_numpy(dtype=bool)[panel.indices]
    if not (regime == regime[0]).all():
        raise ValueError("Volatility regime must be the same for all assets on each date.")
    outcome_indices = np.arange(len(daily_net_delta)) - panel.delay
    regime_eligible = outcome_indices >= 0
    outcome_regime = np.zeros(len(daily_net_delta), dtype=bool)
    outcome_regime[regime_eligible] = regime[0, outcome_indices[regime_eligible]]
    # Rows without an observable future return are not signal opportunities.
    corr = float(np.corrcoef(gru[valid], gnn[valid])[0, 1]) if (
        valid.sum() > 1 and np.std(gru[valid]) > 0 and np.std(gnn[valid]) > 0
    ) else None
    def bps(series, offset):
        result = moving_block_interval(series, block_length=block_length,
                                       samples=samples, seed=seed + offset)
        return {"mean_bps_per_day": result["mean"] * 1e4,
                "ci_95_bps_per_day": [value * 1e4 for value in result["ci_95"]]}
    return {
        "dates": int(len(panel.dates)), "assets": int(len(panel.indices)),
        "position_correlation": corr,
        "mean_abs_position_difference": float(np.mean(np.abs(gnn[valid] - gru[valid]))),
        "opposite_signal_rows": int(opposed.sum()),
        "opposite_signal_fraction": float(opposed.sum() / valid.sum()),
        "connected_signal_fraction": float(connected.sum() / valid.sum()),
        "net_gnn_minus_gru": bps(daily_net_delta, 0),
        "opposite_signal_gross_contribution": bps(per_day.disagreement.to_numpy(), 1),
        "connected_node_gross_contribution": bps(per_day.connected.to_numpy(), 2),
        "net_delta_high_volatility": _conditional_block_interval(
            daily_net_delta, regime_eligible & outcome_regime,
            block_length=block_length, samples=samples, seed=seed + 3,
        ),
        "net_delta_low_volatility": _conditional_block_interval(
            daily_net_delta, regime_eligible & ~outcome_regime,
            block_length=block_length, samples=samples, seed=seed + 4,
        ),
        "note": "Conditional contributions exclude costs; intervals are exploratory and per seed/fold.",
    }
