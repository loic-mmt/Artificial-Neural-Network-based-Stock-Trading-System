"""OUTER-only alert, episode and causal trend diagnostics. No selection."""

import numpy as np
import pandas as pd

from trading_system.training.short_decision import binary_supervision


def binary_head_metrics(labels, known, scores, names, thresholds):
    labels, known, scores = np.asarray(labels), np.asarray(known), np.asarray(scores)
    if (labels.ndim != 1 or known.shape != labels.shape or known.dtype != np.bool_
            or scores.shape != (len(labels), len(names)) or len(thresholds) != len(names)
            or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Aligned known masks, labels and bounded head scores required.")
    if not known.any():
        return [{"head": name, "threshold": thresholds[i], "n_known": 0, "n_unknown": len(labels),
                 "positive_support": 0, "predicted_positive_fraction": None, "accuracy": None,
                 "balanced_accuracy": None, "positive_precision": None, "positive_recall": None,
                 "roc_auc": None, "confusion_matrix": [[0, 0], [0, 0]], "target": f"{name} vs not-{name}"}
                for i, name in enumerate(names)]
    targets, known = binary_supervision(labels, known, names)
    rows = []
    from sklearn.metrics import roc_auc_score
    for i, name in enumerate(names):
        y = targets[known, i].astype(bool); p = np.asarray(scores)[known, i]
        pred = p > thresholds[i]
        tn, fp, fn, tp = (int(x.sum()) for x in (~y & ~pred, ~y & pred, y & ~pred, y & pred))
        recall = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        rows.append({"head": name, "threshold": thresholds[i], "n_known": int(known.sum()),
            "n_unknown": int((~known).sum()), "positive_support": int(y.sum()),
            "predicted_positive_fraction": float(pred.mean()), "accuracy": (tp + tn) / len(y),
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "positive_precision": tp / (tp + fp) if tp + fp else None, "positive_recall": recall,
            "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
            "confusion_matrix": [[tn, fp], [fn, tp]], "target": f"{name} vs not-{name}"})
    return rows


def short_alert_diagnostics(positions, trades, states, executed_positions):
    """Daily alert accuracy is not episode win rate; open-ended exits are censored."""
    positions, trades = pd.DataFrame(positions), pd.DataFrame(trades)
    short = positions.loc[positions.position < 0]
    episodes = trades.loc[trades.side.eq("Short")] if not trades.empty else trades
    losses = episodes.loc[episodes.net_return < 0] if not episodes.empty else episodes
    before = states.raw_states
    underwater = (before[:, 0] < 0) & (before[:, 5] > 0)
    closed = underwater & (np.asarray(executed_positions) >= 0)
    model_closed = closed & (states.positions >= 0)
    forced = closed & (states.positions < 0)
    return {"short_sessions": len(short), "short_daily_down_fraction": float((short.asset_return < 0).mean()) if len(short) else None,
        "short_daily_up_fraction": float((short.asset_return > 0).mean()) if len(short) else None,
        "short_episodes": len(episodes), "short_episode_win_fraction": float((episodes.net_return > 0).mean()) if len(episodes) else None,
        "short_duration_mean": float(episodes.duration_sessions.mean()) if len(episodes) else None,
        "short_duration_median": float(episodes.duration_sessions.median()) if len(episodes) else None,
        "short_duration_max": int(episodes.duration_sessions.max()) if len(episodes) else None,
        "losing_shorts_100_sessions": int((losses.duration_sessions >= 100).sum()) if len(losses) else 0,
        "losing_shorts_252_sessions": int((losses.duration_sessions >= 252).sum()) if len(losses) else 0,
        "underwater_short_closures": int(closed.sum()),
        "underwater_short_model_closures": int(model_closed.sum()),
        "forced_underwater_short_liquidations": int(forced.sum()),
        "mean_consecutive_underwater_at_model_closure": float(before[model_closed, 5].mean()) if model_closed.any() else None,
        "mean_consecutive_underwater_at_closure": float(before[closed, 5].mean()) if closed.any() else None,
        "closure_scope": "observed_closures_only_includes_forced_final_liquidation; not_all_losing_trades"}


def causal_asset_trends(prices, window):
    """Close J-1 versus its SMA(window), independently per asset; no regime filter."""
    keys = ["date", "ticker"]
    work = prices.sort_values(["ticker", "date"]).copy()
    work["date"] = pd.to_datetime(work.date, utc=True)
    if work.duplicated(keys).any(): raise ValueError("Unique ticker/date trend keys required.")
    work["asset_trend"] = "unknown"
    if window:
        price = "adj_close" if "adj_close" in work else "close"
        for _, group in work.groupby("ticker", sort=False):
            previous = group[price].shift(1)
            mean = previous.rolling(window, min_periods=window).mean()
            work.loc[group.index, "asset_trend"] = np.where(mean.isna(), "unknown",
                np.where(previous >= mean, "bullish", "bearish"))
    return work[keys + ["asset_trend"]]


def trend_attribution(positions, trends):
    """Additive PnL contributions, not fictitious independently funded portfolios."""
    work = pd.DataFrame(positions).copy(); work["date"] = pd.to_datetime(work.date, utc=True)
    work = work.merge(trends, on=["date", "ticker"], how="left", validate="many_to_one")
    if work.asset_trend.isna().any(): raise ValueError("Missing causal trend keys.")
    work["side"] = np.where(work.position < 0, "Short", np.where(work.position > 0, "Long", "Flat"))
    return work.groupby(["asset_trend", "side"], dropna=False).agg(
        rows=("position", "size"), pnl_contribution=("pnl_contribution", "sum"),
        mean_abs_position=("position", lambda q: q.abs().mean()),
        underlying_up_fraction=("asset_return", lambda x: (x > 0).mean())).reset_index()
