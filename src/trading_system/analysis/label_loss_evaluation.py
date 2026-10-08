"""Comparable post-open benchmark diagnostics, without fitting or selection.

Returns are slot returns: each ticker owns 1/N of the portfolio and its target
position is bounded by [-1, 1]. Attribution uses wealth before each session,
so ticker and Long/Short dollar contributions add exactly to portfolio PnL.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.training.financial_loss import probabilities_to_positions


def decode_probabilities(probabilities, decoder="continuous"):
    """Decode Short/Flat/Long probabilities; exact zero is genuinely Flat."""
    values = np.asarray(probabilities, dtype=np.float64)
    continuous = probabilities_to_positions(values, "long_short")
    if decoder == "continuous":
        return continuous
    if decoder == "sign":
        return np.sign(continuous)
    if decoder == "argmax":
        # NumPy's first-class tie rule is deterministic and is recorded here.
        return np.argmax(values, axis=1).astype(np.float64) - 1.0
    raise ValueError("decoder must be continuous, sign, or argmax.")


def _class_scores(labels, predicted):
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    recall = np.divide(true_positive, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * true_positive, support + predicted_support,
                   out=np.zeros(3), where=support + predicted_support > 0)
    count = int(confusion.sum())
    present = support > 0
    return {
        "accuracy": float(true_positive.sum() / count) if count else None,
        # Fixed three-class averaging deliberately counts absent classes as 0.
        "balanced_accuracy": float(recall.mean()) if count else None,
        "macro_f1": float(f1.mean()) if count else None,
        "balanced_accuracy_fixed3": float(recall.mean()) if count else None,
        "macro_f1_fixed3": float(f1.mean()) if count else None,
        "balanced_accuracy_present_classes": float(recall[present].mean()) if count else None,
        "macro_f1_present_classes": float(f1[present].mean()) if count else None,
        "n_present_classes": int(present.sum()),
        "confusion_matrix": confusion.tolist(),
        "class_support": support.tolist(),
        "predicted_class_support": predicted_support.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
    }


def classification_metrics(y, probabilities, known, *, majority_class=None):
    """Evaluate only known targets; unknown placeholders never become Flat.

    Pass the TRAIN majority class for an out-of-sample baseline. If omitted,
    the majority of this evaluation sample is reported as a descriptive upper
    baseline, explicitly marked as such, not as a fitted predictor.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    decode_probabilities(probabilities)
    labels = np.asarray(y)
    mask = np.asarray(known)
    if labels.shape != (len(probabilities),) or mask.shape != labels.shape or mask.dtype != np.bool_:
        raise ValueError("Labels and boolean known mask must align with probabilities.")
    try:
        observed = labels[mask].astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Known labels must be class IDs 0, 1, 2.") from exc
    if not np.isin(observed, [0, 1, 2]).all():
        raise ValueError("Known labels must be class IDs 0, 1, 2.")
    observed = observed.astype(np.int64)
    if majority_class is not None and (isinstance(majority_class, (bool, np.bool_))
                                      or majority_class not in (0, 1, 2)):
        raise ValueError("majority_class must be an integer class ID 0, 1, 2.")
    result = _class_scores(observed, np.argmax(probabilities[mask], axis=1))
    result.update({
        "n_known": int(mask.sum()), "n_unknown": int((~mask).sum()),
        "class_names": ["Short", "Flat", "Long"],
        "absent_class_policy": "fixed_three_class_zero_recall_and_f1",
        "cross_entropy": (float(-np.log(np.clip(probabilities[mask, observed], 1e-15, 1)).mean())
                          if len(observed) else None),
    })
    source = "train" if majority_class is not None else "evaluation_sample_descriptive"
    majority = (int(majority_class) if majority_class is not None else
                int(np.argmax(np.bincount(observed, minlength=3))) if len(observed) else None)
    baseline = (_class_scores(observed, np.full(len(observed), majority, dtype=np.int64))
                if majority is not None else _class_scores(observed, np.empty(0, dtype=np.int64)))
    baseline.update({"class_id": majority, "class_source": source})
    result["majority_baseline"] = baseline
    return result


def _path_metrics(net, positions, turnover, costs, config, capital, *, allow_slot_insolvency=False):
    net = np.asarray(net, dtype=np.float64)
    insolvent = bool((net <= -1).any())
    if not len(net) or not np.isfinite(net).all() or (insolvent and not allow_slot_insolvency):
        raise ValueError("Evaluation requires a finite, nonempty, solvent daily return path.")
    wealth = None if insolvent else np.r_[1., np.cumprod(1 + net)]
    standard_deviation = float(net.std(ddof=0))
    sharpe = (float(np.sqrt(config.annualization) * net.mean() / standard_deviation)
              if standard_deviation > 0 else 0. if np.all(net == 0) else None)
    exposure = np.abs(positions).mean(axis=0)
    return {
        "net_pnl": None if wealth is None else float(capital * (wealth[-1] - 1)),
        "net_return": None if wealth is None else float(wealth[-1] - 1),
        "mean_net_return": float(net.mean()),
        "net_sharpe": sharpe,
        "regularized_sharpe": float(np.sqrt(config.annualization) * net.mean()
                                    / np.sqrt(standard_deviation ** 2 + config.sharpe_epsilon ** 2)),
        "max_drawdown": None if wealth is None else float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
        "standalone_slot_insolvent": insolvent,
        "mean_abs_position": float(exposure.mean()),
        "mean_long_exposure": float(np.maximum(positions, 0).mean()),
        "mean_short_exposure": float(np.maximum(-positions, 0).mean()),
        "mean_cash_fraction": float(np.maximum(1 - exposure, 0).mean()),
        "long_fraction": float((positions > 0).mean()),
        "short_fraction": float((positions < 0).mean()),
        "flat_fraction": float((positions == 0).mean()),
        "turnover": float(turnover.mean(axis=0).sum()),
        "cost_return_sum": float(costs.mean(axis=0).sum()),
        "periods": int(len(net)), "assets": int(len(positions)),
    }


def _side_paths(executed, returns, costs, protocol):
    long = np.maximum(executed, 0)
    short = np.minimum(executed, 0)
    if protocol == "intraday":
        long_units, short_units = 2 * long, -2 * short
    else:
        long_units = np.abs(np.diff(long, axis=1, prepend=0.))
        short_units = np.abs(np.diff(short, axis=1, prepend=0.))
        # Some panels liquidate on a dedicated final (zero-position) session,
        # others charge final liquidation on the last active session.
        long_units[:, -1] += long[:, -1]
        short_units[:, -1] += -short[:, -1]
    total_units = long_units + short_units
    long_costs = np.divide(costs * long_units, total_units,
                           out=np.zeros_like(costs), where=total_units > 0)
    short_costs = costs - long_costs
    return long * returns - long_costs, short * returns - short_costs, long_costs, short_costs


def _trade_records(dates, tickers, executed, gross, long_costs, short_costs, protocol):
    records = []
    for asset, ticker in enumerate(tickers):
        direction = np.sign(executed[asset])
        index = 0
        while index < len(dates):
            if not direction[index]:
                index += 1
                continue
            start = index
            if protocol != "intraday":
                while index + 1 < len(dates) and direction[index + 1] == direction[start]:
                    index += 1
            end = index
            side_costs = long_costs if direction[start] > 0 else short_costs
            episode_cost = side_costs[asset, start:end + 1].copy()
            if protocol != "intraday" and end + 1 < len(dates):
                episode_cost[-1] += side_costs[asset, end + 1]
            episode_net = gross[asset, start:end + 1] - episode_cost
            records.append({
                "ticker": str(ticker), "side": "Long" if direction[start] > 0 else "Short",
                "entry_date": dates[start].isoformat(), "last_return_date": dates[end].isoformat(),
                "exit_date": dates[min(end + int(protocol != "intraday"), len(dates) - 1)].isoformat(),
                "duration_sessions": int(end - start + 1),
                "mean_abs_position": float(np.abs(executed[asset, start:end + 1]).mean()),
                "gross_return_sum": float(gross[asset, start:end + 1].sum()),
                "net_return_sum": float(episode_net.sum()),
                "net_return": None if (episode_net <= -1).any() else float(np.prod(1 + episode_net) - 1),
                "standalone_slot_insolvent": bool((episode_net <= -1).any()),
                "cost_return_sum": float(episode_cost.sum()),
                "protocol": protocol,
            })
            index += 1
    return records


def _records_keys(keys, panel):
    if keys is None:
        return {}, None
    if isinstance(keys, Mapping) and all(np.isscalar(value) or value is None for value in keys.values()):
        return dict(keys), None
    frame = pd.DataFrame(keys).reset_index(drop=True)
    if len(frame) != panel.rows:
        raise ValueError("keys must have exactly one record per panel input row.")
    shared = {column: frame.iloc[0][column] for column in frame
              if column not in {"date", "ticker", "row_index"} and len(frame)
              and frame[column].nunique(dropna=False) == 1}
    return shared, frame


def evaluate_positions(panel, positions, loss_config, *, initial_capital=10000., keys=None):
    """Evaluate a PostOpenReturnPanel under its own execution/cost contract.

    Missing input rows remain calendar slots, rather than being compressed.
    Position records include terminal/absence liquidation rows (row_index None)
    so exported costs still reconcile exactly with portfolio PnL.
    """
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive.")
    net, executed, delta, turnover, costs = panel.path(positions, loss_config)
    returns = np.asarray(panel.returns, dtype=np.float64)
    indices = np.asarray(panel.indices)
    executed, turnover, costs = (np.asarray(values, dtype=np.float64)
                                for values in (executed, turnover, costs))
    if returns.ndim != 2 or any(values.shape != returns.shape for values in (executed, turnover, costs, indices)):
        raise ValueError("Post-open panel paths must align as [tickers, sessions].")
    if not all(np.isfinite(values).all() for values in (returns, executed, turnover, costs)):
        raise ValueError("Post-open paths must be finite.")
    net = np.asarray(net, dtype=np.float64)
    if net.ndim == 2:
        net = net.mean(axis=0)
    asset_net = executed * returns - costs
    if net.shape != (returns.shape[1],) or not np.allclose(net, asset_net.mean(axis=0), atol=1e-12):
        raise ValueError("Daily portfolio path does not reconcile with equal-slot asset returns.")
    dates = pd.DatetimeIndex(pd.to_datetime(panel.dates, utc=True))
    tickers = tuple(panel.tickers)
    if len(dates) != returns.shape[1] or len(tickers) != returns.shape[0] or dates.has_duplicates:
        raise ValueError("Panel calendar and ticker labels must be unique and aligned.")
    protocol = panel.protocol
    global_metadata, key_frame = _records_keys(keys, panel)
    metrics = _path_metrics(net, executed, turnover, costs, loss_config, initial_capital)
    wealth = np.r_[initial_capital, initial_capital * np.cumprod(1 + net)]
    scale = wealth[:-1] / len(tickers)
    gross = executed * returns
    long_net, short_net, long_costs, short_costs = _side_paths(executed, returns, costs, protocol)
    metrics.update({
        "protocol": protocol,
        "long_pnl_contribution": float((long_net * scale).sum()),
        "short_pnl_contribution": float((short_net * scale).sum()),
        "cost_pnl": float((costs * scale).sum()),
        "allocation": "fixed_equal_ticker_slots_no_leverage",
    })
    per_ticker = []
    for asset, ticker in enumerate(tickers):
        ticker_metrics = _path_metrics(asset_net[asset], executed[asset:asset + 1],
                                      turnover[asset:asset + 1], costs[asset:asset + 1],
                                      loss_config, initial_capital / len(tickers), allow_slot_insolvency=True)
        ticker_metrics.update({
            "ticker": str(ticker), "protocol": protocol,
            "portfolio_pnl_contribution": float((asset_net[asset] * scale).sum()),
            "long_pnl_contribution": float((long_net[asset] * scale).sum()),
            "short_pnl_contribution": float((short_net[asset] * scale).sum()),
        })
        per_ticker.append(ticker_metrics)
    daily = [{
        "date": date.isoformat(), "net_return": float(net[index]),
        "gross_return": float(gross[:, index].mean()),
        "cost": float(costs[:, index].mean()),
        "turnover": float(turnover[:, index].mean()),
        "exposure": float(np.abs(executed[:, index]).mean()),
        "long_exposure": float(np.maximum(executed[:, index], 0).mean()),
        "short_exposure": float(np.maximum(-executed[:, index], 0).mean()),
        "cash_fraction": float(max(0., 1 - np.abs(executed[:, index]).mean())),
        "equity": float(wealth[index + 1]),
        "long_pnl_contribution": float((long_net[:, index] * scale[index]).sum()),
        "short_pnl_contribution": float((short_net[:, index] * scale[index]).sum()),
    } for index, date in enumerate(dates)]
    records = []
    for asset, ticker in enumerate(tickers):
        for session, date in enumerate(dates):
            row = int(indices[asset, session])
            metadata = dict(global_metadata)
            if key_frame is not None and row >= 0:
                metadata.update(key_frame.iloc[row].to_dict())
            if row >= 0 and key_frame is not None:
                if "ticker" in metadata and str(metadata["ticker"]) != str(ticker):
                    raise ValueError("Prediction ticker does not match its panel slot.")
                if "date" in metadata and pd.to_datetime(metadata["date"], utc=True) != date:
                    raise ValueError("Prediction date does not match its panel slot.")
            metadata.update({
                "row_index": row if row >= 0 else None, "date": date.isoformat(),
                "ticker": str(ticker), "position": float(executed[asset, session]),
                "asset_return": float(returns[asset, session]),
                "gross_return": float(gross[asset, session]),
                "net_return": float(asset_net[asset, session]),
                "cost": float(costs[asset, session]), "turnover": float(turnover[asset, session]),
                "exposure": float(abs(executed[asset, session]) / len(tickers)),
                "allocation_weight": float(1 / len(tickers)),
                "pnl_contribution": float(asset_net[asset, session] * scale[session]),
            })
            records.append(metadata)
    trades = _trade_records(dates, tickers, executed, gross, long_costs, short_costs, protocol)
    for collection in (per_ticker, daily, trades):
        for record in collection:
            for name, value in global_metadata.items():
                record.setdefault(name, value)
    durations = [record["duration_sessions"] for record in trades]
    metrics["trades"] = len(trades)
    metrics["duration_sessions"] = {
        "min": int(min(durations)) if durations else None,
        "mean": float(np.mean(durations)) if durations else None,
        "max": int(max(durations)) if durations else None,
    }
    baseline_positions = np.ones(panel.rows)
    baseline = panel.path(baseline_positions, loss_config)
    baseline_exposure = float(np.abs(baseline[1]).mean())
    matched_scale = (metrics["mean_abs_position"] / baseline_exposure
                     if baseline_exposure > 0 else 0.)
    if matched_scale > 1 + 1e-10:
        raise ValueError("Model exposure exceeds available always-long slots; no leverage control allowed.")
    controls = {}
    control_equities = {}
    for name, targets in (("always_long", baseline_positions),
                          ("always_long_mean_exposure_matched", baseline_positions * min(matched_scale, 1.)),
                          ("cash", np.zeros(panel.rows))):
        control_net, control_q, _, control_turnover, control_costs = panel.path(targets, loss_config)
        controls[name] = _path_metrics(control_net, control_q, control_turnover, control_costs,
                                      loss_config, initial_capital)
        control_equities[name] = initial_capital * np.cumprod(1 + np.asarray(control_net))
    for session, record in enumerate(daily):
        record.update({
            "baseline_equity": float(control_equities["always_long"][session]),
            "exposure_matched_baseline_equity": float(control_equities["always_long_mean_exposure_matched"][session]),
            "cash_equity": float(control_equities["cash"][session]),
        })
    controls["matching"] = {
        "scale": float(matched_scale), "target_mean_exposure": metrics["mean_abs_position"],
        "same_protocol_calendar_costs": True,
        "caution": "Mean-exposure matching does not equalize exposure timing or constitute an alpha estimate.",
    }
    metrics["outperformance_vs_always_long"] = metrics["net_pnl"] - controls["always_long"]["net_pnl"]
    metrics["outperformance_vs_exposure_matched_long"] = (
        metrics["net_pnl"] - controls["always_long_mean_exposure_matched"]["net_pnl"])
    return {"metrics": metrics, "per_ticker": per_ticker, "daily_paths": daily,
            "position_records": records, "trades": trades, "exposure_controls": controls}


def compare_opposite_positions(frame):
    """Pair models only within identical fold/seed/protocol/decoder calendars.

    Opposed episodes break on Flat, a changed direction, or a missing global
    session. No intersection-only comparison hides a candidate's missing rows.
    """
    work = pd.DataFrame(frame).copy()
    columns = ["candidate", "fold", "seed", "protocol", "decoder", "date", "ticker", "position"]
    if not set(columns).issubset(work):
        raise ValueError(f"Position comparison requires columns: {columns}.")
    if work.empty:
        return {"summary": [], "episodes": []}
    work["date"] = pd.to_datetime(work["date"], utc=True, errors="raise")
    if (work[columns].isna().any().any() or not np.isfinite(work["position"]).all()
            or (work["position"].abs() > 1 + 1e-6).any()):
        raise ValueError("Comparison keys and positions must be present and finite.")
    if work.duplicated(columns[:-1]).any():
        raise ValueError("Duplicate prediction key in position comparison.")
    summaries, episodes = [], []
    grouping = ["fold", "seed", "protocol", "decoder"]
    for values, group in work.groupby(grouping, sort=True, dropna=False):
        metadata = dict(zip(grouping, values))
        calendar = sorted(group["date"].unique())
        ordinal = {date: index for index, date in enumerate(calendar)}
        slot_count = group["ticker"].nunique()
        for first, second in combinations(sorted(group["candidate"].unique()), 2):
            left = group[group["candidate"].eq(first)].set_index(["ticker", "date"]).sort_index()
            right = group[group["candidate"].eq(second)].set_index(["ticker", "date"]).sort_index()
            if not left.index.equals(right.index):
                raise ValueError("Compared candidates must have exactly the same ticker/session calendar.")
            pair = left[["position"]].rename(columns={"position": "position_a"}).join(
                right[["position"]].rename(columns={"position": "position_b"}))
            pair["opposed"] = pair.position_a * pair.position_b < 0
            active = (pair.position_a != 0) & (pair.position_b != 0)
            pair["weight"] = (pair.position_a.abs() + pair.position_b.abs()) / (2 * slot_count)
            comparable_weight = float(pair.loc[active, "weight"].sum())
            record = {**metadata, "candidate_a": first, "candidate_b": second,
                      "n_rows": len(pair), "both_nonflat": int(active.sum()),
                      "opposed_rows": int(pair.opposed.sum()),
                      "opposition_rate_both_nonflat": float(pair.opposed.sum() / active.sum()) if active.any() else None,
                      "exposure_weighted_opposition_rate": float(pair.loc[pair.opposed, "weight"].sum() / comparable_weight) if comparable_weight else None,
                      "opposed_exposure_sum": float(pair.loc[pair.opposed, "weight"].sum())}
            for field in ("gross_return", "net_return", "pnl_contribution"):
                if field in left and field in right:
                    for suffix, candidate in (("a", left), ("b", right)):
                        weights = 1 if field == "pnl_contribution" else candidate.get("allocation_weight", 1 / slot_count)
                        pair[f"{field}_{suffix}"] = candidate[field] * weights
                        record[f"opposed_{field}_contribution_{suffix}"] = float(pair.loc[pair.opposed, f"{field}_{suffix}"].sum())
            summaries.append(record)
            for ticker, ticker_pair in pair.reset_index().groupby("ticker", sort=True):
                ticker_pair = ticker_pair.sort_values("date").reset_index(drop=True)
                index = 0
                while index < len(ticker_pair):
                    if not ticker_pair.at[index, "opposed"]:
                        index += 1
                        continue
                    start = index
                    while index + 1 < len(ticker_pair):
                        previous, following = ticker_pair.iloc[index], ticker_pair.iloc[index + 1]
                        if (not following.opposed or np.sign(following.position_a) != np.sign(previous.position_a)
                                or ordinal[following.date] != ordinal[previous.date] + 1):
                            break
                        index += 1
                    block = ticker_pair.iloc[start:index + 1]
                    episode = {**metadata, "candidate_a": first, "candidate_b": second,
                               "ticker": str(ticker), "start_date": block.iloc[0].date.isoformat(),
                               "end_date": block.iloc[-1].date.isoformat(), "duration_sessions": len(block),
                               "side_a": int(np.sign(block.iloc[0].position_a)),
                               "side_b": int(np.sign(block.iloc[0].position_b)),
                               "mean_exposure": float(block.weight.mean())}
                    for column in pair.columns:
                        if column.endswith(("_a", "_b")) and column.startswith(("gross_return", "net_return", "pnl_contribution")):
                            episode[column] = float(block[column].sum())
                    episodes.append(episode)
                    index += 1
    return {"summary": summaries, "episodes": episodes}


def _plot_price_contract(prices, protocol):
    """Choose the price coordinate matching the executed protocol, not a feature."""
    frame = pd.DataFrame(prices).copy()
    if protocol == "overnight":
        from trading_system.labels.post_open_benchmark import adjusted_open_prices
        if "adj_open_target" not in frame and "open" not in frame:
            raise ValueError("Overnight position plots require open or adj_open_target, not closing prices alone.")
        if "adj_open_target" not in frame and "adj_close" in frame and "close" not in frame:
            raise ValueError("Adjusting overnight opens requires both raw close and adj_close.")
        frame["_plot_price"] = adjusted_open_prices(frame)
        return frame, "Adjusted open", "Open J to open J+1; color = position after open J"
    if protocol == "intraday":
        column = next((name for name in ("close", "adj_close") if name in frame), None)
        if column is None:
            raise ValueError("Intraday position plots require a close price curve.")
        frame["_plot_price"] = pd.to_numeric(frame[column], errors="coerce")
        return (frame, column,
                "Close curve colored by session positions; trades are open to close, not close to close")
    raise ValueError("Position plots require an explicit intraday or overnight protocol.")


def plot_evaluation(destination, daily_paths, position_frame=None, prices=None, *, max_position_charts=1):
    """Optional bounded plots; never render one chart per grid configuration.

    At most eight equity trajectories and max_position_charts ticker charts are
    emitted in deterministic key order, not selected by achieved return.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.dates as mdates

    destination = Path(destination)
    if isinstance(max_position_charts, bool) or not isinstance(max_position_charts, int) or max_position_charts < 0:
        raise ValueError("max_position_charts must be a non-negative integer.")
    destination.mkdir(parents=True, exist_ok=True)
    work = pd.DataFrame(daily_paths).copy()
    if work.empty:
        return []
    if not {"date", "equity"}.issubset(work):
        raise ValueError("Equity plots require date and equity.")
    work["date"] = pd.to_datetime(work.date, utc=True)
    groups = [column for column in ("candidate", "protocol", "decoder", "fold", "seed") if column in work]
    trajectories = list(work.groupby(groups, sort=True)) if groups else [("portfolio", work)]
    figure, axes = plt.subplots(figsize=(11, 5))
    for name, trajectory in trajectories[:8]:
        trajectory = trajectory.sort_values("date")
        axes.plot(trajectory.date, trajectory.equity, label=str(name))
    if len(trajectories) == 1:
        trajectory = trajectories[0][1].sort_values("date")
        for column, label in (("baseline_equity", "Always Long (same protocol)"),
                              ("exposure_matched_baseline_equity", "Always Long (matched mean exposure)"),
                              ("cash_equity", "Cash")):
            if column in trajectory:
                axes.plot(trajectory.date, trajectory[column], label=label, linestyle="--", alpha=.8)
    axes.set(title=f"Portfolio equity ({min(8, len(trajectories))}/{len(trajectories)} trajectories)",
             xlabel="Session", ylabel="Capital")
    axes.legend(fontsize=7)
    axes.grid(alpha=.2)
    figure.tight_layout()
    path = destination / "portfolio-equity.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    outputs = [str(path)]
    if position_frame is None or prices is None or max_position_charts == 0:
        return outputs
    positions, price_frame = pd.DataFrame(position_frame).copy(), pd.DataFrame(prices).copy()
    if positions.empty or price_frame.empty:
        return outputs
    if not {"date", "ticker"}.issubset(price_frame):
        raise ValueError("Position plots require ticker/date and a price column.")
    positions["date"] = pd.to_datetime(positions.date, utc=True)
    price_frame["date"] = pd.to_datetime(price_frame.date, utc=True)
    if price_frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Price plots require unique ticker/session prices.")
    position_groups = [column for column in (*groups, "ticker") if column in positions]
    for chart, (name, group) in enumerate(positions.groupby(position_groups, sort=True)):
        if chart == max_position_charts:
            break
        if "protocol" not in group or group.protocol.nunique() != 1:
            raise ValueError("Position plots require one explicit protocol per trajectory.")
        plot_prices, price_label, explanation = _plot_price_contract(price_frame, group.iloc[0].protocol)
        price_column = "_plot_price"
        group = group.drop(columns=[price_column], errors="ignore").merge(
            plot_prices[["date", "ticker", price_column]], on=["date", "ticker"],
            how="left", validate="many_to_one").sort_values("date")
        valid = np.isfinite(pd.to_numeric(group[price_column], errors="coerce"))
        group = group[valid]
        if len(group) < 2:
            continue
        x = mdates.date2num(group.date.to_numpy())
        y = group[price_column].to_numpy(dtype=float)
        points = np.column_stack([x, y])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        q = group.position.to_numpy(dtype=float)[:-1]
        colors = np.where(q > 0, "#16835a", np.where(q < 0, "#c44343", "#8b8b8b"))
        figure, axes = plt.subplots(figsize=(11, 4))
        axes.add_collection(LineCollection(segments, colors=colors, linewidths=1.6))
        axes.autoscale()
        axes.xaxis_date()
        axes.set(title=f"{name}: green Long, red Short, grey Flat",
                 xlabel=explanation, ylabel=price_label)
        figure.autofmt_xdate()
        figure.tight_layout()
        path = destination / f"positions-{chart + 1:02d}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        outputs.append(str(path))
    return outputs


def plot_oppositions(destination, position_frame, prices, *, max_tickers=2):
    """Stack all candidates' colored price paths on the same comparison chart.

    Exactly one fold/seed/protocol/decoder is required, preventing charts from
    visually merging unrelated experiments. Tickers are chosen alphabetically,
    not by achieved profit. No position arrows are drawn.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    if isinstance(max_tickers, bool) or not isinstance(max_tickers, int) or max_tickers < 0:
        raise ValueError("max_tickers must be a non-negative integer.")
    positions, price_frame = pd.DataFrame(position_frame).copy(), pd.DataFrame(prices).copy()
    if positions.empty or price_frame.empty or max_tickers == 0:
        return []
    for key in ("fold", "seed", "protocol", "decoder"):
        if key not in positions or positions[key].nunique(dropna=False) != 1:
            raise ValueError("Opposition plots require one common fold/seed/protocol/decoder.")
    # This also validates the exact candidate calendars and duplicate keys.
    compare_opposite_positions(positions)
    if not {"date", "ticker"}.issubset(price_frame):
        raise ValueError("Opposition plots require ticker/date and a price column.")
    positions["date"] = pd.to_datetime(positions.date, utc=True)
    price_frame["date"] = pd.to_datetime(price_frame.date, utc=True)
    if price_frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Opposition prices require unique ticker/session rows.")
    price_frame, price_label, explanation = _plot_price_contract(price_frame, positions.iloc[0].protocol)
    price_column = "_plot_price"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    candidates = sorted(positions.candidate.unique())
    context = ", ".join(f"{key}={positions.iloc[0][key]}" for key in ("fold", "seed", "protocol", "decoder"))
    for chart, ticker in enumerate(sorted(positions.ticker.unique())[:max_tickers]):
        figure, axes = plt.subplots(len(candidates), 1, figsize=(12, max(3, 2.1 * len(candidates))),
                                    sharex=True, sharey=True, squeeze=False)
        for axis, candidate in zip(axes[:, 0], candidates):
            group = positions[positions.ticker.eq(ticker) & positions.candidate.eq(candidate)]
            group = group.drop(columns=[price_column], errors="ignore").merge(
                price_frame[["date", "ticker", price_column]], on=["date", "ticker"],
                how="left", validate="one_to_one").sort_values("date")
            y = pd.to_numeric(group[price_column], errors="coerce").to_numpy(dtype=float)
            x = mdates.date2num(group.date.to_numpy())
            q = group.position.to_numpy(dtype=float)
            if len(group) >= 2:
                points = np.column_stack([x, y])
                segments = np.stack([points[:-1], points[1:]], axis=1)
                valid = np.isfinite(segments).all(axis=(1, 2))
                colors = np.where(q[:-1] > 0, "#16835a", np.where(q[:-1] < 0, "#c44343", "#8b8b8b"))
                axis.add_collection(LineCollection(segments[valid], colors=colors[valid], linewidths=1.5))
                axis.autoscale()
                axis.xaxis_date()
            axis.set(title=str(candidate), ylabel=price_label)
            axis.grid(alpha=.15)
        axes[-1, 0].set_xlabel(explanation)
        figure.suptitle(f"{ticker}: comparable decisions\n{context}", fontsize=10)
        figure.legend(handles=[Line2D([0], [0], color=color, label=side)
                               for side, color in (("Long", "#16835a"), ("Short", "#c44343"), ("Flat", "#8b8b8b"))],
                      loc="upper right", fontsize=8)
        figure.autofmt_xdate()
        figure.tight_layout(rect=(0, 0, 1, .95))
        path = destination / f"oppositions-{chart + 1:02d}.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        outputs.append(str(path))
    return outputs


def plot_learning_trace(destination, trace):
    """Plot recorded losses and weighted gradients, without recomputation."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    records = list(trace)
    if not records:
        return []
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in records]
    outputs = []
    components = [component for component in ("total_loss", "ce_loss", "financial_loss")
                  if any(record.get(partition, {}).get(component) is not None
                         for record in records for partition in ("train", "validation"))]
    figure, axes = plt.subplots(1, max(1, len(components)), figsize=(4 * max(1, len(components)), 3.6), squeeze=False)
    for axis, component in zip(axes[0], components):
        for partition in ("train", "validation"):
            values = [record.get(partition, {}).get(component) for record in records]
            if any(value is not None for value in values):
                axis.plot(epochs, [np.nan if value is None else value for value in values], label=partition)
        axis.set(title=component, xlabel="Optimizer updates")
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(fontsize=8)
    if not components:
        axes[0, 0].text(.5, .5, "No recorded loss components", ha="center", va="center")
        axes[0, 0].set_axis_off()
    figure.tight_layout()
    path = destination / "learning-losses.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    outputs.append(str(path))
    figure, axis = plt.subplots(figsize=(8, 3.6))
    for component in ("weighted_ce_gradient_norm", "weighted_financial_gradient_norm", "combined_gradient_norm_preclip"):
        values = [record.get(component) for record in records]
        if any(value is not None for value in values):
            axis.plot(epochs, [np.nan if value is None else value for value in values], label=component)
    if axis.lines:
        axis.set(yscale="symlog", xlabel="Optimizer updates", ylabel="Gradient norm", title="Gradient components before clipping")
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
        figure.tight_layout()
        path = destination / "learning-gradients.png"
        figure.savefig(path, dpi=130)
        outputs.append(str(path))
    plt.close(figure)
    return outputs
