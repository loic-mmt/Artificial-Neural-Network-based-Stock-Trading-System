"""Retrospective label diagnostics, without training or executable-PnL claims.

Raw label runs, decoded positions and overlapping native label events are
different objects. This module deliberately reports them separately. Common
horizon probes compare the direction of each label, not carried positions.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import pandas as pd

from trading_system.backtest.positions import labels_to_positions
from trading_system.labels.config import LabelConfig
from trading_system.labels.registry import LabelResult


_RUN_COLUMNS = [
    "ticker", "label", "label_id", "start_date", "end_date", "length_sessions",
    "left_censored", "right_censored",
]
_TRADE_COLUMNS = [
    "ticker", "side", "entry_date", "exit_date", "duration_sessions",
    "gross_return", "net_return", "turnover_units", "left_censored",
    "right_censored", "protocol",
]
_EVENT_COLUMNS = [
    "ticker", "label_id", "side", "start_date", "end_date", "duration_sessions",
    "event_return", "gross_return", "net_return", "directed", "overlapping",
]
_TRANSITION_COLUMNS = ["ticker", "from_label", "to_label", "count", "rate"]
_PROBE_COLUMNS = [
    "ticker", "horizon", "threshold", "n_rows", "n_active", "n_neutral",
    "side_accuracy", "small_move_fraction_active", "n_opportunities",
    "opportunity_direction_accuracy", "opportunity_capture_rate",
    "wrong_way_count", "missed_neutral_count", "missed_neutral_rate",
    "mean_directed_return",
]


def _stats(values: Sequence[float] | np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "min": None, "mean": None, "median": None,
                "p90": None, "max": None}
    return {
        "count": int(len(array)), "min": float(array.min()),
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)), "max": float(array.max()),
    }


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _return_stats(table: pd.DataFrame) -> dict[str, object]:
    net = table["net_return"].to_numpy(dtype=np.float64)
    positive = float(net[net > 0].sum())
    negative = float(-net[net < 0].sum())
    return {
        "count": int(len(table)),
        "duration_sessions": _stats(table["duration_sessions"]),
        "gross_return": _stats(table["gross_return"]),
        "net_return": _stats(net),
        "win_rate": _mean(net > 0),
        "profit_factor": positive / negative if negative > 0 else None,
        "no_net_losses": bool(len(net) and negative == 0),
    }


def _numeric_prices(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"Missing diagnostic price column: {column}")
    prices = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"{column} must contain finite, positive prices.")
    return prices


def _validated_inputs(result, config, ticker, common_mask):
    if not isinstance(result, LabelResult) or not isinstance(config, LabelConfig):
        raise TypeError("result and config must be LabelResult and LabelConfig.")
    if result.semantics != config.semantics or result.class_names != config.class_names:
        raise ValueError("Label result and configuration contracts do not match.")
    if result.semantics not in ("action", "target_position"):
        raise ValueError("Directional diagnostics require three-class labels.")
    if not isinstance(ticker, str) or not ticker:
        raise ValueError("ticker must be a non-empty string.")
    frame = result.frame.copy()
    if "date" not in frame:
        raise ValueError("Label diagnostics require a date column.")
    if "ticker" in frame and not frame["ticker"].eq(ticker).all():
        raise ValueError("Analyze one ticker at a time, without mixing price series.")
    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Dates must be valid and unique per ticker.")
    frame["date"] = dates
    order = np.argsort(dates.to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    known = result.known_mask[order].copy()
    common = np.ones(len(frame), dtype=bool)
    if common_mask is not None:
        raw_common = np.asarray(common_mask)
        if raw_common.dtype != np.bool_ or raw_common.ndim != 1 or len(raw_common) != len(frame):
            raise ValueError("common_mask must be an aligned one-dimensional boolean array.")
        common = raw_common[order].copy()
    ids = pd.to_numeric(frame["Label_id"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isin(ids[known], [0, 1, 2]).all():
        raise ValueError("Known labels must have integer IDs 0, 1 or 2.")
    ids = np.where(known, ids, 1).astype(np.int64)
    price_col = result.metadata.get("price_col", "adj_close")
    if price_col not in frame and price_col == "adj_close":
        price_col = "close"
    prices = _numeric_prices(frame, price_col)
    return frame, known, ids, prices, common, price_col


def _positions(ids: np.ndarray, known: np.ndarray, semantics: str) -> np.ndarray:
    """Decode each known block independently; unknown labels reset state."""
    positions = np.zeros(len(ids), dtype=np.float64)
    boundaries = np.flatnonzero(np.r_[True, known[1:] != known[:-1], True])
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        if known[lo]:
            positions[lo:hi] = labels_to_positions(
                ids[lo:hi], position_mode="long_short", label_semantics=semantics,
            )
    return positions


def _label_runs(frame, known, ids, classes, ticker, first):
    records = []
    index = 0
    while index < len(frame):
        if not known[index]:
            index += 1
            continue
        lo = index
        while index + 1 < len(frame) and known[index + 1] and ids[index + 1] == ids[lo]:
            index += 1
        hi = index
        if hi >= first:
            visible_lo = max(lo, first)
            records.append({
                "ticker": ticker, "label": classes[ids[lo]], "label_id": int(ids[lo]),
                "start_date": frame.at[visible_lo, "date"], "end_date": frame.at[hi, "date"],
                "length_sessions": hi - visible_lo + 1,
                "left_censored": bool(lo < first or lo == 0 or not known[lo - 1]),
                "right_censored": bool(hi == len(frame) - 1 or not known[hi + 1]),
            })
        index += 1
    return pd.DataFrame(records, columns=_RUN_COLUMNS)


def _position_trades(frame, known, positions, prices, ticker, first, fee, intraday):
    records = []
    if intraday:
        opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=np.float64)
        closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
        active = known & (positions != 0)
        if (not np.isfinite(opens[active]).all() or (opens[active] <= 0).any()
                or not np.isfinite(closes[active]).all() or (closes[active] <= 0).any()):
            raise ValueError("Known intraday trades require finite, positive open and close.")
        for index in np.flatnonzero(active & (np.arange(len(frame)) >= first)):
            gross = float(positions[index] * (closes[index] / opens[index] - 1.0))
            records.append({
                "ticker": ticker, "side": int(positions[index]),
                "entry_date": frame.at[index, "date"], "exit_date": frame.at[index, "date"],
                "duration_sessions": 1, "gross_return": gross, "net_return": gross - 2 * fee,
                "turnover_units": 2.0, "left_censored": False, "right_censored": False,
                "protocol": "retrospective_same_day_open_close",
            })
        return pd.DataFrame(records, columns=_TRADE_COLUMNS)

    current = 0.0
    entry = first
    left_censored = False
    for index in range(first, len(frame)):
        next_position = float(positions[index])
        if current and (next_position != current or index == len(frame) - 1):
            gross = float(current * (prices[index] / prices[entry] - 1.0))
            records.append({
                "ticker": ticker, "side": int(current),
                "entry_date": frame.at[entry, "date"], "exit_date": frame.at[index, "date"],
                "duration_sessions": index - entry,
                "gross_return": gross, "net_return": gross - 2 * fee,
                "turnover_units": 2.0, "left_censored": left_censored,
                "right_censored": bool(not known[index] or
                                       (next_position == current and index == len(frame) - 1)),
                "protocol": "retrospective_close_to_close_no_delay",
            })
            current = 0.0
        if next_position and next_position != current and index < len(frame) - 1:
            current, entry = next_position, index
            left_censored = bool(index == first and first > 0
                                 and known[first - 1] and positions[first - 1] == current)
    return pd.DataFrame(records, columns=_TRADE_COLUMNS)


def _native_events(frame, known, ids, config, ticker, first, fee):
    records = []
    dates = frame["date"]
    date_indices = {value: index for index, value in enumerate(dates)}
    method = config.method
    if method == "triple_barrier":
        required = {"label_event_id", "label_end_date", "event_return"}
        if not required.issubset(frame):
            raise ValueError(f"Triple-barrier diagnostics require columns: {sorted(required)}")
        eligible = known & frame["label_event_id"].notna().to_numpy(dtype=bool)
        market_returns = pd.to_numeric(frame["event_return"], errors="coerce").to_numpy()
        end_dates = pd.to_datetime(frame["label_end_date"], errors="coerce", utc=True)
    elif method in ("forward_return", "volatility_position"):
        if "fwd_ret" not in frame:
            raise ValueError("Forward-label diagnostics require fwd_ret.")
        horizon = int(config.parameters.get("horizon", 1 if method == "forward_return" else 10))
        market_returns = pd.to_numeric(frame["fwd_ret"], errors="coerce").to_numpy()
        eligible = known & (np.arange(len(frame)) + horizon < len(frame))
        end_dates = dates.shift(-horizon)
    elif method == "intraday_return":
        if "intraday_ret" not in frame:
            raise ValueError("Intraday diagnostics require intraday_ret.")
        market_returns = pd.to_numeric(frame["intraday_ret"], errors="coerce").to_numpy()
        eligible = known.copy()
        end_dates = dates
    else:
        return pd.DataFrame(columns=_EVENT_COLUMNS)
    indices = np.flatnonzero(eligible & (np.arange(len(frame)) >= first))
    previous_end = -1
    previous_record = None
    for index in indices:
        end_date = end_dates.iloc[index]
        if end_date not in date_indices or not np.isfinite(market_returns[index]):
            raise ValueError("Known native event must end inside the provided price history.")
        end_index = date_indices[end_date]
        if end_index < index or (end_index == index and method != "intraday_return"):
            raise ValueError("Native event end must follow its start.")
        side = int(ids[index] - 1)
        directed = side != 0
        overlap = bool(index < previous_end and method != "intraday_return")
        if overlap and previous_record is not None:
            records[previous_record]["overlapping"] = True
        records.append({
            "ticker": ticker, "label_id": int(ids[index]), "side": side,
            "start_date": dates.iloc[index], "end_date": end_date,
            "duration_sessions": 1 if method == "intraday_return" else end_index - index,
            "event_return": float(market_returns[index]),
            "gross_return": float(side * market_returns[index]) if directed else np.nan,
            "net_return": float(side * market_returns[index] - 2 * fee) if directed else np.nan,
            "directed": directed, "overlapping": overlap,
        })
        if end_index > previous_end:
            previous_end, previous_record = end_index, len(records) - 1
    return pd.DataFrame(records, columns=_EVENT_COLUMNS)


def _transitions(ids, known, classes, ticker, first):
    counts = np.zeros((3, 3), dtype=np.int64)
    for index in range(first + 1, len(ids)):
        if known[index - 1] and known[index]:
            counts[ids[index - 1], ids[index]] += 1
    total = int(counts.sum())
    table = pd.DataFrame([
        {"ticker": ticker, "from_label": source, "to_label": target,
         "count": int(counts[i, j]), "rate": float(counts[i, j] / total) if total else 0.0}
        for i, source in enumerate(classes) for j, target in enumerate(classes)
    ], columns=_TRANSITION_COLUMNS)
    return table, total, int(counts.sum() - counts.trace())


def _horizon_probes(ids, known, prices, common, ticker, first, horizons, thresholds):
    records = []
    for horizon in horizons:
        count = max(len(prices) - horizon, 0)
        eligible = known[:count] & common[:count] & (np.arange(count) >= first)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            future_returns = prices[horizon:] / prices[:count] - 1.0 if count else np.array([])
        eligible &= np.isfinite(future_returns)
        returns = future_returns[eligible]
        sides = ids[:count][eligible] - 1
        active, neutral = sides != 0, sides == 0
        correct = sides * returns > 0
        for threshold in thresholds:
            large = np.abs(returns) >= threshold
            relevant = active & large
            opportunities = int(large.sum())
            missed = int((neutral & large).sum())
            records.append({
                "ticker": ticker, "horizon": horizon, "threshold": threshold,
                "n_rows": int(len(returns)), "n_active": int(active.sum()),
                "n_neutral": int(neutral.sum()), "side_accuracy": _mean(correct[active]),
                "small_move_fraction_active": _mean((np.abs(returns) < threshold)[active]),
                "n_opportunities": opportunities,
                "opportunity_direction_accuracy": _mean(correct[relevant]),
                "opportunity_capture_rate": float((correct & relevant).sum() / opportunities)
                if opportunities else None,
                "wrong_way_count": int((relevant & ~correct).sum()),
                "missed_neutral_count": missed,
                "missed_neutral_rate": missed / opportunities if opportunities else None,
                "mean_directed_return": _mean((sides * returns)[active]),
            })
    return pd.DataFrame(records, columns=_PROBE_COLUMNS)


def analyze_label_result(
    result: LabelResult,
    config: LabelConfig,
    *,
    ticker: str,
    fee_bps: float = 5.0,
    noise_thresholds: Sequence[float] = (0.002, 0.005, 0.01, 0.015),
    probe_horizons: Sequence[int] = (1, 5, 10),
    start: str | None = None,
    common_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Analyze one ticker, keeping unknown rows and horizon boundaries explicit.

    ``start`` crops diagnostics after decoding the original history. Ongoing
    positions are marked left-censored and rebased to the first visible close.
    The caller must cap the input before a holdout or requested end: no future
    label event is allowed to extend outside this history. ``common_mask``
    restricts horizon probes only, never modifies the decoded position path.

    Fees are one-way units of traded notional: an entry and exit each cost
    ``fee_bps``. Position returns assume fixed initial notional per episode,
    not daily rebalancing. Native events may overlap and are never compounded.
    """
    if isinstance(fee_bps, bool) or not math.isfinite(float(fee_bps)) or fee_bps < 0:
        raise ValueError("fee_bps must be finite and non-negative.")
    thresholds = tuple(float(value) for value in noise_thresholds)
    if not thresholds or any(not math.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("noise_thresholds must contain finite, positive fractions.")
    horizons = tuple(probe_horizons)
    if not horizons or any(isinstance(value, (bool, np.bool_))
                           or not isinstance(value, (int, np.integer)) or value <= 0
                           for value in horizons):
        raise ValueError("probe_horizons must contain positive integers.")
    frame, known, ids, prices, common, price_col = _validated_inputs(
        result, config, ticker, common_mask,
    )
    first = 0
    if start is not None:
        try:
            start_date = pd.Timestamp(start)
            start_date = start_date.tz_localize("UTC") if start_date.tz is None else start_date.tz_convert("UTC")
        except (TypeError, ValueError) as error:
            raise ValueError("start must be a valid date.") from error
        if pd.isna(start_date):
            raise ValueError("start must be a valid date.")
        first = int(frame["date"].searchsorted(start_date))
    fee = float(fee_bps) / 10_000.0
    positions = _positions(ids, known, result.semantics)
    runs = _label_runs(frame, known, ids, result.class_names, ticker, first)
    intraday = config.method == "intraday_return"
    trades = _position_trades(frame, known, positions, prices, ticker, first, fee, intraday)
    events = _native_events(frame, known, ids, config, ticker, first, fee)
    transitions, n_adjacent, n_changes = _transitions(ids, known, result.class_names, ticker, first)
    probes = _horizon_probes(ids, known, prices, common, ticker, first, horizons, thresholds)

    visible_known = known[first:]
    counts = {name: int(((ids[first:] == index) & visible_known).sum())
              for index, name in enumerate(result.class_names)}
    n_known = int(visible_known.sum())
    proportions = {name: count / n_known if n_known else 0.0 for name, count in counts.items()}
    entropy = -sum(value * math.log2(value) for value in proportions.values() if value > 0)
    lengths = runs["length_sessions"].to_numpy(dtype=np.int64)
    trade_stats = _return_stats(trades)
    trade_stats["n_left_censored"] = int(trades["left_censored"].sum())
    trade_stats["n_right_censored"] = int(trades["right_censored"].sum())
    directed = events.loc[events["directed"].astype(bool)]
    native_stats = _return_stats(directed)
    native_stats["n_events_including_neutral"] = int(len(events))
    native_stats["n_overlapping_events"] = int(events["overlapping"].sum())
    native_stats["duration_all_events"] = _stats(events["duration_sessions"])
    if config.method == "triple_barrier" and "barrier_touch" in frame:
        visible_events = known & frame["label_event_id"].notna().to_numpy(dtype=bool)
        visible_events &= np.arange(len(frame)) >= first
        touches = frame.loc[visible_events, "barrier_touch"].value_counts()
        native_stats["touch_counts"] = {
            name: int(touches.get(name, 0)) for name in ("profit", "stop", "vertical")
        }
    native_stats["small_move_fraction"] = {
        str(value): _mean((directed["event_return"].abs().to_numpy() < value))
        for value in thresholds
    }
    exposure_stop = len(frame) if intraday else max(len(frame) - 1, first)
    exposure_positions = positions[first:exposure_stop]
    exposure_known = known[first:exposure_stop]
    denominator = len(exposure_positions)
    turnover = float(trades["turnover_units"].sum())
    summary = {
        "ticker": ticker, "method": config.method, "semantics": result.semantics,
        "price_col": price_col, "n_rows": int(len(frame) - first), "n_known": n_known,
        "n_unknown": int(len(frame) - first - n_known),
        "class_counts": counts, "class_proportions": proportions,
        "label_entropy_bits": float(entropy),
        "label_run_stats": {**_stats(lengths), "singleton_fraction": _mean(lengths == 1)},
        "label_run_stats_by_class": {
            name: _stats(runs.loc[runs["label"] == name, "length_sessions"])
            for name in result.class_names
        },
        "n_adjacent_known_pairs": n_adjacent, "label_transition_count": n_changes,
        "label_transition_rate": n_changes / n_adjacent if n_adjacent else None,
        "position_trade_stats": trade_stats, "native_event_stats": native_stats,
        "position_trade_stats_by_side": {
            name: _return_stats(trades.loc[trades["side"] == side])
            for side, name in ((1, "long"), (-1, "short"))
        },
        "turnover_units": turnover,
        "turnover_per_session": turnover / denominator if denominator else None,
        "n_exposure_sessions": int(denominator),
        "n_known_exposure_sessions": int(exposure_known.sum()),
        "long_exposure": float((exposure_positions > 0).sum() / denominator) if denominator else None,
        "short_exposure": float((exposure_positions < 0).sum() / denominator) if denominator else None,
        "flat_exposure": float(((exposure_positions == 0) & exposure_known).sum() / denominator) if denominator else None,
        "unknown_exposure": float((~exposure_known).sum() / denominator) if denominator else None,
        "protocol": "retrospective_same_day_open_close" if intraday else "retrospective_close_to_close_no_delay",
        "fees_bps": float(fee_bps), "noise_thresholds": list(thresholds),
        "probe_horizons": [int(value) for value in horizons],
        "position_return_convention": "side*(exit/entry-1), fixed initial notional per episode; entry+exit fees",
        "native_events_compounded": False,
        "probe_convention": "Label_id-1 at close J, close J->close J+h; Hold/Flat neutral, not carried",
        "missed_neutral_rate_denominator": "all known common rows with absolute horizon return >= threshold",
        "interpretation": "Retrospective targets only. Turnover and profits do not establish learnability or executable performance.",
    }
    return {"summary": summary, "label_runs": runs, "position_trades": trades,
            "native_events": events, "transitions": transitions, "horizon_probes": probes}


__all__ = ["analyze_label_result"]
