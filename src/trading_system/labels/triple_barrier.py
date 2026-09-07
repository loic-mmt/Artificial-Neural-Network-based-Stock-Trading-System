from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .schema import LABEL_ID_TO_NAME, POSITION_ID_TO_NAME

_EVENT_FILTERS = {"all", "cusum"}
_BETWEEN_EVENT_POLICIES = {"hold", "flat", "carry"}
VOLATILITY_ESTIMATORS = ("rolling_std", "atr", "bollinger")
_VOLATILITY_FLOOR = np.finfo(np.float64).eps
_DEFAULT_PARAMETERS = {
    "max_holding": 10,
    "volatility_window": 20,
    "volatility_estimator": "rolling_std",
    "profit_barrier": 1.0,
    "stop_barrier": 1.0,
    "event_filter": "all",
    "cusum_threshold": 0.5,
    "cost_bps": 5.0,
    "between_event_policy": "hold",
}


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _finite_float(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    invalid = result <= 0.0 if positive else result < 0.0
    if not math.isfinite(result) or invalid:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return result


def _validate_parameters(
    *,
    max_holding: int,
    volatility_window: int,
    volatility_estimator: str,
    profit_barrier: float,
    stop_barrier: float,
    event_filter: str,
    cusum_threshold: float,
    cost_bps: float,
    between_event_policy: str,
) -> dict[str, object]:
    parameters: dict[str, object] = {
        "max_holding": _positive_integer("max_holding", max_holding),
        "volatility_window": _positive_integer(
            "volatility_window", volatility_window
        ),
        "profit_barrier": _finite_float("profit_barrier", profit_barrier),
        "stop_barrier": _finite_float("stop_barrier", stop_barrier),
        "cusum_threshold": _finite_float(
            "cusum_threshold", cusum_threshold, positive=True
        ),
        "cost_bps": _finite_float("cost_bps", cost_bps),
    }
    if parameters["profit_barrier"] == 0.0 and parameters["stop_barrier"] == 0.0:
        raise ValueError("At least one horizontal barrier must be enabled.")
    if event_filter not in _EVENT_FILTERS:
        raise ValueError("event_filter must be 'all' or 'cusum'.")
    if volatility_estimator not in VOLATILITY_ESTIMATORS:
        raise ValueError(
            "volatility_estimator must be rolling_std, atr, or bollinger."
        )
    parameters["volatility_estimator"] = volatility_estimator
    if between_event_policy not in _BETWEEN_EVENT_POLICIES:
        raise ValueError("between_event_policy must be hold, flat, or carry.")
    parameters["event_filter"] = event_filter
    parameters["between_event_policy"] = between_event_policy
    return parameters


def first_barrier_touch(
    path_returns: Sequence[float] | np.ndarray,
    *,
    upper_barrier: float | None,
    lower_barrier: float | None,
) -> tuple[int, int, str]:
    """Return direction, one-based touch offset, and first barrier name."""

    path = np.asarray(path_returns, dtype=np.float64)
    if path.ndim != 1 or not len(path):
        raise ValueError("path_returns must be a non-empty 1D array.")
    if not np.isfinite(path).all():
        raise ValueError("path_returns must contain only finite values.")
    if upper_barrier is None and lower_barrier is None:
        raise ValueError("At least one horizontal barrier must be enabled.")
    if upper_barrier is not None and (
        not math.isfinite(float(upper_barrier)) or float(upper_barrier) <= 0.0
    ):
        raise ValueError("upper_barrier must be positive or None.")
    if lower_barrier is not None and (
        not math.isfinite(float(lower_barrier)) or float(lower_barrier) >= 0.0
    ):
        raise ValueError("lower_barrier must be negative or None.")

    upper_hits = (
        np.flatnonzero(path >= float(upper_barrier))
        if upper_barrier is not None
        else np.empty(0, dtype=np.int64)
    )
    lower_hits = (
        np.flatnonzero(path <= float(lower_barrier))
        if lower_barrier is not None
        else np.empty(0, dtype=np.int64)
    )
    upper_index = int(upper_hits[0]) if len(upper_hits) else len(path)
    lower_index = int(lower_hits[0]) if len(lower_hits) else len(path)
    if upper_index < lower_index and upper_index < len(path):
        return 1, upper_index + 1, "profit"
    if lower_index < upper_index and lower_index < len(path):
        return -1, lower_index + 1, "stop"
    return 0, len(path), "vertical"


def symmetric_cusum_events(
    returns: Sequence[float] | np.ndarray,
    volatility: Sequence[float] | np.ndarray,
    *,
    threshold_multiplier: float = 0.5,
) -> np.ndarray:
    """Select causal event rows with a symmetric volatility-scaled CUSUM."""

    threshold_multiplier = _finite_float(
        "threshold_multiplier", threshold_multiplier, positive=True
    )
    values = np.asarray(returns, dtype=np.float64)
    scales = np.asarray(volatility, dtype=np.float64)
    if values.ndim != 1 or scales.ndim != 1 or len(values) != len(scales):
        raise ValueError("returns and volatility must be aligned 1D arrays.")
    events = np.zeros(len(values), dtype=bool)
    positive_sum = 0.0
    negative_sum = 0.0
    for index, (value, scale) in enumerate(zip(values, scales)):
        if not np.isfinite(value) or not np.isfinite(scale):
            positive_sum = 0.0
            negative_sum = 0.0
            continue
        threshold = threshold_multiplier * max(float(scale), _VOLATILITY_FLOOR)
        positive_sum = max(0.0, positive_sum + float(value))
        negative_sum = min(0.0, negative_sum + float(value))
        if positive_sum > threshold or negative_sum < -threshold:
            events[index] = True
            positive_sum = 0.0
            negative_sum = 0.0
    return events


def _validated_prices(
    frame: pd.DataFrame,
    *,
    price_col: str,
    date_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if frame is None or frame.empty:
        raise ValueError("Cannot label an empty frame.")
    missing = [column for column in (date_col, price_col) if column not in frame]
    if missing:
        raise ValueError(f"Missing triple-barrier columns: {missing}")
    work = frame.sort_values(date_col).reset_index(drop=True).copy()
    prices = pd.to_numeric(work[price_col], errors="coerce")
    values = prices.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError(f"{price_col} contains invalid values.")
    dates = pd.to_datetime(work[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{date_col} contains invalid values.")
    work[date_col] = dates
    return work, prices


def _historical_volatility(prices: pd.Series, window: int) -> pd.Series:
    returns = prices.pct_change(fill_method=None)
    return returns.rolling(window, min_periods=window).std(ddof=0)


def _barrier_scale(
    frame: pd.DataFrame,
    prices: pd.Series,
    volatility: pd.Series,
    parameters: dict[str, object],
) -> pd.Series:
    """Causal, dimensionless barrier width, frozen at each event's start.

    ATR uses a simple rolling mean of adjusted true ranges / current price.
    Bollinger uses normalized half-width of bands at two population stds.
    All estimators share the return-volatility warmup and CUSUM sampling.
    """

    estimator = parameters["volatility_estimator"]
    if estimator == "rolling_std":
        return volatility
    window = int(parameters["volatility_window"])
    if estimator == "bollinger":
        rolling = prices.rolling(window, min_periods=window)
        scale = 2.0 * rolling.std(ddof=0) / rolling.mean()
    else:
        required = ("high", "low", "close")
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"ATR requires OHLC columns: {missing}")
        ohlc = frame.loc[:, required].apply(pd.to_numeric, errors="coerce")
        values = ohlc.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError("ATR high, low, and close must be finite and positive.")
        if ((ohlc["low"] > ohlc["close"]) | (ohlc["close"] > ohlc["high"])).any():
            raise ValueError("ATR requires low <= close <= high.")
        # OHLC may be raw while the label price is split/dividend-adjusted.
        adjustment = prices / ohlc["close"]
        high = ohlc["high"] * adjustment
        low = ohlc["low"] * adjustment
        previous = prices.shift(1)
        true_range = pd.concat(
            [high - low, (high - previous).abs(), (low - previous).abs()],
            axis=1,
        ).max(axis=1).where(previous.notna())
        scale = true_range.rolling(window, min_periods=window).mean() / prices
    return scale.where(volatility.notna())


def _event_candidates(
    prices: pd.Series,
    volatility: pd.Series,
    parameters: dict[str, object],
) -> np.ndarray:
    eligible = volatility.notna().to_numpy(dtype=bool)
    if parameters["event_filter"] == "all":
        return eligible
    returns = prices.pct_change(fill_method=None).to_numpy(dtype=np.float64)
    return symmetric_cusum_events(
        returns,
        volatility.to_numpy(dtype=np.float64),
        threshold_multiplier=float(parameters["cusum_threshold"]),
    ) & eligible


def _direction_aware_cost_adjustment(value: float, cost_fraction: float) -> float:
    return math.copysign(max(abs(value) - cost_fraction, 0.0), value)


def _label_segment(
    frame: pd.DataFrame,
    historical_volatility: pd.Series,
    historical_barrier_scale: pd.Series,
    *,
    price_col: str,
    date_col: str,
    parameters: dict[str, object],
) -> pd.DataFrame:
    out = frame.copy()
    prices = pd.to_numeric(out[price_col], errors="raise")
    volatility = historical_volatility.reindex(out.index)
    barrier_scale = historical_barrier_scale.reindex(out.index)
    event_candidates = _event_candidates(prices, volatility, parameters) & (
        barrier_scale.notna().to_numpy(dtype=bool)
    )
    n_rows = len(out)
    max_holding = int(parameters["max_holding"])
    cost_fraction = float(parameters["cost_bps"]) / 10_000.0

    event_ids = np.full(n_rows, -1, dtype=np.int64)
    event_directions = np.zeros(n_rows, dtype=np.int8)
    event_known = np.zeros(n_rows, dtype=bool)
    end_dates = np.full(
        n_rows, np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    touches = np.full(n_rows, "not_event", dtype=object)
    event_returns = np.full(n_rows, np.nan, dtype=np.float64)
    net_event_returns = np.full(n_rows, np.nan, dtype=np.float64)
    scores = np.full(n_rows, np.nan, dtype=np.float64)
    upper_barriers = np.full(n_rows, np.nan, dtype=np.float64)
    lower_barriers = np.full(n_rows, np.nan, dtype=np.float64)
    event_number = 0

    price_values = prices.to_numpy(dtype=np.float64)
    date_values = out[date_col].to_numpy(dtype="datetime64[ns]")
    vol_values = barrier_scale.to_numpy(dtype=np.float64)
    for index in np.flatnonzero(event_candidates):
        event_ids[index] = event_number
        event_number += 1
        if index + max_holding >= n_rows:
            touches[index] = "unknown"
            continue
        scale = max(float(vol_values[index]), _VOLATILITY_FLOOR)
        profit_multiple = float(parameters["profit_barrier"])
        stop_multiple = float(parameters["stop_barrier"])
        upper = (
            profit_multiple * scale + cost_fraction
            if profit_multiple > 0.0
            else None
        )
        lower = (
            -(stop_multiple * scale + cost_fraction)
            if stop_multiple > 0.0
            else None
        )
        upper_barriers[index] = upper if upper is not None else np.nan
        lower_barriers[index] = lower if lower is not None else np.nan
        path = (
            price_values[index + 1 : index + max_holding + 1]
            / price_values[index]
            - 1.0
        )
        direction, offset, touch = first_barrier_touch(
            path,
            upper_barrier=upper,
            lower_barrier=lower,
        )
        end_index = index + offset
        realized = float(path[offset - 1])
        net_realized = _direction_aware_cost_adjustment(realized, cost_fraction)
        event_directions[index] = direction
        event_known[index] = True
        end_dates[index] = date_values[end_index]
        touches[index] = touch
        event_returns[index] = realized
        net_event_returns[index] = net_realized
        scores[index] = net_realized / scale

    base_known = barrier_scale.notna().to_numpy(dtype=bool, copy=True)
    if "_label_known" in out:
        existing = out["_label_known"]
        if existing.isna().any():
            raise ValueError("_label_known contains missing values.")
        base_known &= existing.to_numpy(dtype=bool)
    known = base_known.copy()
    known[event_candidates] = event_known[event_candidates]

    policy = str(parameters["between_event_policy"])
    label_ids = np.ones(n_rows, dtype=np.int64)
    if policy in ("hold", "flat"):
        label_ids[event_known] = event_directions[event_known].astype(np.int64) + 1
    else:
        current = 0
        unresolved = False
        for index in range(n_rows):
            if event_candidates[index]:
                if event_known[index]:
                    current = int(event_directions[index])
                    unresolved = False
                else:
                    unresolved = True
            label_ids[index] = current + 1
            if unresolved:
                known[index] = False

    names = LABEL_ID_TO_NAME if policy == "hold" else POSITION_ID_TO_NAME
    out["label_event_id"] = pd.Series(event_ids, index=out.index).mask(event_ids < 0).astype("Int64")
    out["label_end_date"] = pd.to_datetime(end_dates)
    out["barrier_touch"] = touches
    out["barrier_scale"] = barrier_scale
    out["upper_barrier"] = upper_barriers
    out["lower_barrier"] = lower_barriers
    out["event_return"] = event_returns
    out["net_event_return"] = net_event_returns
    out["label_score"] = scores
    out["target_position"] = (
        label_ids - 1 if policy != "hold" else np.full(n_rows, np.nan)
    )
    out["Label_id"] = label_ids
    out["Label"] = pd.Series(label_ids).map(names).to_numpy()
    out["_label_known"] = known
    return out


def generate_triple_barrier_labels(
    frame: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    date_col: str = "date",
    max_holding: int = 10,
    volatility_window: int = 20,
    volatility_estimator: str = "rolling_std",
    profit_barrier: float = 1.0,
    stop_barrier: float = 1.0,
    event_filter: str = "all",
    cusum_threshold: float = 0.5,
    cost_bps: float = 5.0,
    between_event_policy: str = "hold",
) -> pd.DataFrame:
    """Generate leakage-aware triple-barrier labels for one price series."""

    parameters = _validate_parameters(
        max_holding=max_holding,
        volatility_window=volatility_window,
        volatility_estimator=volatility_estimator,
        profit_barrier=profit_barrier,
        stop_barrier=stop_barrier,
        event_filter=event_filter,
        cusum_threshold=cusum_threshold,
        cost_bps=cost_bps,
        between_event_policy=between_event_policy,
    )
    work, prices = _validated_prices(frame, price_col=price_col, date_col=date_col)
    volatility = _historical_volatility(
        prices, int(parameters["volatility_window"])
    )
    return _label_segment(
        work,
        volatility,
        _barrier_scale(work, prices, volatility, parameters),
        price_col=price_col,
        date_col=date_col,
        parameters=parameters,
    )


def _generate_for_context(
    frame: pd.DataFrame,
    *,
    price_col: str,
    date_col: str,
    group_col: str | None,
    split_col: str | None,
    parameters: dict[str, object],
) -> pd.DataFrame:
    if group_col is not None and group_col not in frame:
        raise ValueError(f"Missing triple-barrier group column: {group_col}")
    ticker_groups = (
        frame.groupby(group_col, sort=False, dropna=False)
        if group_col is not None
        else [(None, frame)]
    )
    parts: list[pd.DataFrame] = []
    for _, ticker_frame in ticker_groups:
        work, prices = _validated_prices(
            ticker_frame, price_col=price_col, date_col=date_col
        )
        volatility = _historical_volatility(
            prices, int(parameters["volatility_window"])
        )
        barrier_scale = _barrier_scale(work, prices, volatility, parameters)
        segments = (
            work.groupby(split_col, sort=False, dropna=False)
            if split_col is not None and split_col in work
            else [(None, work)]
        )
        for _, segment in segments:
            parts.append(
                _label_segment(
                    segment,
                    volatility,
                    barrier_scale,
                    price_col=price_col,
                    date_col=date_col,
                    parameters=parameters,
                )
            )
    sort_columns = [column for column in (group_col, date_col) if column is not None]
    combined = (
        pd.concat(parts, ignore_index=True)
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )
    event_mask = combined["label_event_id"].notna()
    combined.loc[event_mask, "label_event_id"] = np.arange(
        int(event_mask.sum()), dtype=np.int64
    )
    combined["label_event_id"] = combined["label_event_id"].astype("Int64")
    return combined


def generate_triple_barrier_labels_by_ticker(
    frame: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    group_col: str = "ticker",
    date_col: str = "date",
    **parameters,
) -> pd.DataFrame:
    validated = _validate_parameters(**{**_DEFAULT_PARAMETERS, **parameters})
    return _generate_for_context(
        frame,
        price_col=price_col,
        date_col=date_col,
        group_col=group_col if group_col in frame else None,
        split_col=None,
        parameters=validated,
    )


def build_triple_barrier_label_result(frame, config, context):
    """Registry adapter for M3 path-dependent event labels."""

    from .config import LabelConfig
    from .registry import LabelContext, LabelResult

    if not isinstance(config, LabelConfig) or config.method != "triple_barrier":
        raise ValueError("Triple-barrier builder requires method='triple_barrier'.")
    if not isinstance(context, LabelContext):
        raise TypeError("context must be a LabelContext.")
    expected = set(LabelConfig.triple_barrier().parameters)
    extras = sorted(set(config.parameters) - expected)
    if extras:
        raise ValueError(f"Unknown triple-barrier parameters: {extras}.")
    parameters = _validate_parameters(
        **{**LabelConfig.triple_barrier().parameters, **config.parameters}
    )
    expected_semantics = (
        "action"
        if parameters["between_event_policy"] == "hold"
        else "target_position"
    )
    if config.semantics != expected_semantics:
        raise ValueError(
            "Triple-barrier semantics do not match between_event_policy."
        )
    labeled = _generate_for_context(
        frame,
        price_col=context.price_col,
        date_col=context.date_col,
        group_col=context.group_col,
        split_col=context.split_col,
        parameters=parameters,
    )
    known_mask = labeled["_label_known"].to_numpy(dtype=bool)
    class_names = config.class_names
    counts = labeled.loc[known_mask, "Label"].value_counts()
    event_mask = labeled["label_event_id"].notna()
    known_events = event_mask & labeled["_label_known"]
    touch_counts = labeled.loc[known_events, "barrier_touch"].value_counts()
    event_durations = (
        pd.to_datetime(labeled.loc[known_events, "label_end_date"])
        - pd.to_datetime(labeled.loc[known_events, context.date_col])
    ).dt.days
    return LabelResult(
        frame=labeled,
        known_mask=known_mask,
        class_names=class_names,
        semantics=config.semantics,
        metadata={
            "method": "triple_barrier",
            "objective": config.objective,
            "semantics": config.semantics,
            "parameters": dict(parameters),
            "class_counts": {
                name: int(counts.get(name, 0)) for name in class_names
            },
            "n_rows": int(len(labeled)),
            "n_known": int(known_mask.sum()),
            "n_unknown": int(len(labeled) - known_mask.sum()),
            "n_events": int(event_mask.sum()),
            "n_known_events": int(known_events.sum()),
            "event_rate": float(event_mask.mean()),
            "touch_counts": {
                name: int(touch_counts.get(name, 0))
                for name in ("profit", "stop", "vertical")
            },
            "mean_event_duration_days": (
                float(event_durations.mean()) if len(event_durations) else 0.0
            ),
        },
    )


__all__ = [
    "VOLATILITY_ESTIMATORS",
    "build_triple_barrier_label_result",
    "first_barrier_touch",
    "generate_triple_barrier_labels",
    "generate_triple_barrier_labels_by_ticker",
    "symmetric_cusum_events",
]
