from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .schema import POSITION_ID_TO_NAME

_POSITION_MODES = {"long_flat", "long_short"}
_SCORE_VOLATILITY_FLOOR = np.finfo(np.float64).eps
_DEFAULT_PARAMETERS = {
    "horizon": 10,
    "volatility_window": 20,
    "long_threshold": 1.0,
    "short_threshold": 1.5,
    "exit_threshold": 0.25,
    "min_holding_period": 5,
    "cooldown": 0,
    "cost_bps": 5.0,
    "position_mode": "long_flat",
}


def _integer_parameter(name: str, value: object, *, positive: bool) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")
    integer = int(value)
    minimum = 1 if positive else 0
    if integer < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}.")
    return integer


def _float_parameter(
    name: str,
    value: object,
    *,
    positive: bool,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
        qualifier = "positive" if positive else "finite and non-negative"
        raise ValueError(f"{name} must be {qualifier}.")
    return number


def _validate_volatility_position_parameters(
    *,
    horizon: int,
    volatility_window: int,
    long_threshold: float,
    short_threshold: float,
    exit_threshold: float,
    min_holding_period: int,
    cooldown: int,
    cost_bps: float,
    position_mode: str,
) -> dict[str, object]:
    parameters: dict[str, object] = {
        "horizon": _integer_parameter("horizon", horizon, positive=True),
        "volatility_window": _integer_parameter(
            "volatility_window", volatility_window, positive=True
        ),
        "long_threshold": _float_parameter(
            "long_threshold", long_threshold, positive=True
        ),
        "short_threshold": _float_parameter(
            "short_threshold", short_threshold, positive=True
        ),
        "exit_threshold": _float_parameter(
            "exit_threshold", exit_threshold, positive=False
        ),
        "min_holding_period": _integer_parameter(
            "min_holding_period", min_holding_period, positive=False
        ),
        "cooldown": _integer_parameter("cooldown", cooldown, positive=False),
        "cost_bps": _float_parameter("cost_bps", cost_bps, positive=False),
    }
    if position_mode not in _POSITION_MODES:
        raise ValueError(
            "position_mode must be 'long_flat' or 'long_short'."
        )
    parameters["position_mode"] = position_mode
    if parameters["exit_threshold"] >= min(
        parameters["long_threshold"], parameters["short_threshold"]
    ):
        raise ValueError(
            "exit_threshold must be lower than long_threshold and short_threshold."
        )
    return parameters


def build_persistent_positions(
    label_scores: Sequence[float] | np.ndarray,
    *,
    known_mask: Sequence[bool] | np.ndarray | None = None,
    long_threshold: float = 1.0,
    short_threshold: float = 1.5,
    exit_threshold: float = 0.25,
    min_holding_period: int = 5,
    cooldown: int = 0,
    position_mode: str = "long_flat",
) -> np.ndarray:
    """Convert continuous scores into persistent Short/Flat/Long states."""

    parameters = _validate_volatility_position_parameters(
        horizon=1,
        volatility_window=1,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        exit_threshold=exit_threshold,
        min_holding_period=min_holding_period,
        cooldown=cooldown,
        cost_bps=0.0,
        position_mode=position_mode,
    )
    scores = np.asarray(label_scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("label_scores must be a 1D array.")
    if known_mask is None:
        known = np.isfinite(scores)
    else:
        known = np.asarray(known_mask, dtype=bool)
        if known.ndim != 1 or len(known) != len(scores):
            raise ValueError("known_mask must align one-to-one with label_scores.")
        if not np.isfinite(scores[known]).all():
            raise ValueError("Known label scores must be finite.")

    positions = np.zeros(len(scores), dtype=np.int8)
    current = 0
    last_transition: int | None = None
    for index, score in enumerate(scores):
        if not known[index]:
            positions[index] = current
            continue

        if score > parameters["long_threshold"]:
            desired = 1
        elif score < -parameters["short_threshold"]:
            desired = -1 if position_mode == "long_short" else 0
        elif abs(score) < parameters["exit_threshold"]:
            desired = 0
        else:
            desired = current

        if desired != current and last_transition is not None:
            distance = index - last_transition
            holding_blocked = (
                current != 0 and distance < parameters["min_holding_period"]
            )
            cooldown_blocked = distance <= parameters["cooldown"]
            if holding_blocked or cooldown_blocked:
                desired = current
        if desired != current:
            current = desired
            last_transition = index
        positions[index] = current
    return positions


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
        raise ValueError(f"Missing volatility-position columns: {missing}")
    work = frame.sort_values(date_col).reset_index(drop=True).copy()
    price = pd.to_numeric(work[price_col], errors="coerce")
    values = price.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError(f"{price_col} contains invalid values.")
    return work, price


def _historical_volatility(price: pd.Series, window: int) -> pd.Series:
    daily_returns = price.pct_change(fill_method=None)
    return daily_returns.rolling(window, min_periods=window).std(ddof=0)


def _label_segment(
    frame: pd.DataFrame,
    historical_volatility: pd.Series,
    *,
    price_col: str,
    parameters: dict[str, object],
) -> pd.DataFrame:
    out = frame.copy()
    price = pd.to_numeric(out[price_col], errors="raise")
    horizon = int(parameters["horizon"])
    forward_return = (price.shift(-horizon) / price) - 1.0
    cost_fraction = float(parameters["cost_bps"]) / 10_000.0
    cost_adjusted_return = np.sign(forward_return) * np.maximum(
        forward_return.abs() - cost_fraction,
        0.0,
    )
    denominator = historical_volatility.reindex(out.index) * math.sqrt(horizon)
    denominator = denominator.clip(lower=_SCORE_VOLATILITY_FLOOR)
    score = cost_adjusted_return / denominator
    known = forward_return.notna() & historical_volatility.reindex(out.index).notna()
    score = score.where(known)
    if "_label_known" in out:
        existing = out["_label_known"]
        if existing.isna().any():
            raise ValueError("_label_known contains missing values.")
        known &= existing.astype(bool)
        score = score.where(known)

    target_positions = build_persistent_positions(
        score.to_numpy(dtype=np.float64),
        known_mask=known.to_numpy(dtype=bool),
        long_threshold=float(parameters["long_threshold"]),
        short_threshold=float(parameters["short_threshold"]),
        exit_threshold=float(parameters["exit_threshold"]),
        min_holding_period=int(parameters["min_holding_period"]),
        cooldown=int(parameters["cooldown"]),
        position_mode=str(parameters["position_mode"]),
    )
    out["fwd_ret"] = forward_return
    out["historical_volatility"] = historical_volatility.reindex(out.index)
    out["cost_adjusted_fwd_ret"] = cost_adjusted_return
    out["label_score"] = score
    out["target_position"] = target_positions
    out["Label_id"] = target_positions.astype(np.int64) + 1
    out["Label"] = out["Label_id"].map(POSITION_ID_TO_NAME)
    out["_label_known"] = known.to_numpy(dtype=bool)
    return out


def generate_volatility_position_labels(
    frame: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    date_col: str = "date",
    horizon: int = 10,
    volatility_window: int = 20,
    long_threshold: float = 1.0,
    short_threshold: float = 1.5,
    exit_threshold: float = 0.25,
    min_holding_period: int = 5,
    cooldown: int = 0,
    cost_bps: float = 5.0,
    position_mode: str = "long_flat",
) -> pd.DataFrame:
    """Generate leakage-aware persistent position labels for one price series."""

    parameters = _validate_volatility_position_parameters(
        horizon=horizon,
        volatility_window=volatility_window,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        exit_threshold=exit_threshold,
        min_holding_period=min_holding_period,
        cooldown=cooldown,
        cost_bps=cost_bps,
        position_mode=position_mode,
    )
    work, price = _validated_prices(frame, price_col=price_col, date_col=date_col)
    volatility = _historical_volatility(price, int(parameters["volatility_window"]))
    return _label_segment(
        work,
        volatility,
        price_col=price_col,
        parameters=parameters,
    )


def _position_diagnostics(frame: pd.DataFrame) -> tuple[int, list[int]]:
    known_positions = frame.loc[frame["_label_known"], "target_position"].to_numpy()
    if not len(known_positions):
        return 0, []
    changes = np.flatnonzero(known_positions[1:] != known_positions[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [len(known_positions)]))
    lengths = np.diff(boundaries).astype(int).tolist()
    return int(len(changes)), lengths


def _generate_for_context(
    frame: pd.DataFrame,
    *,
    price_col: str,
    date_col: str,
    group_col: str | None,
    split_col: str | None,
    parameters: dict[str, object],
) -> tuple[pd.DataFrame, int, list[int]]:
    if group_col is not None and group_col not in frame:
        raise ValueError(f"Missing volatility-position group column: {group_col}")
    ticker_groups = (
        frame.groupby(group_col, sort=False, dropna=False)
        if group_col is not None
        else [(None, frame)]
    )
    parts: list[pd.DataFrame] = []
    transition_count = 0
    regime_lengths: list[int] = []
    for _, ticker_frame in ticker_groups:
        work, price = _validated_prices(
            ticker_frame,
            price_col=price_col,
            date_col=date_col,
        )
        volatility = _historical_volatility(
            price,
            int(parameters["volatility_window"]),
        )
        segments = (
            work.groupby(split_col, sort=False, dropna=False)
            if split_col is not None and split_col in work
            else [(None, work)]
        )
        for _, segment in segments:
            labeled = _label_segment(
                segment,
                volatility,
                price_col=price_col,
                parameters=parameters,
            )
            transitions, lengths = _position_diagnostics(labeled)
            transition_count += transitions
            regime_lengths.extend(lengths)
            parts.append(labeled)
    sort_columns = [column for column in (group_col, date_col) if column is not None]
    combined = (
        pd.concat(parts, ignore_index=True)
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )
    return combined, transition_count, regime_lengths


def generate_volatility_position_labels_by_ticker(
    frame: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    group_col: str = "ticker",
    date_col: str = "date",
    **parameters,
) -> pd.DataFrame:
    validated = _validate_volatility_position_parameters(
        **{**_DEFAULT_PARAMETERS, **parameters}
    )
    labeled, _, _ = _generate_for_context(
        frame,
        price_col=price_col,
        date_col=date_col,
        group_col=group_col if group_col in frame else None,
        split_col=None,
        parameters=validated,
    )
    return labeled


def build_volatility_position_label_result(frame, config, context):
    """Registry adapter for M2 volatility-adjusted position states."""

    from .config import LabelConfig
    from .registry import LabelContext, LabelResult

    if not isinstance(config, LabelConfig) or config.method != "volatility_position":
        raise ValueError(
            "Volatility-position builder requires method='volatility_position'."
        )
    if not isinstance(context, LabelContext):
        raise TypeError("context must be a LabelContext.")
    if config.semantics != "target_position":
        raise ValueError(
            "Volatility-position labels require semantics='target_position'."
        )
    expected = set(LabelConfig.volatility_position().parameters)
    extras = sorted(set(config.parameters) - expected)
    if extras:
        raise ValueError(f"Unknown volatility-position parameters: {extras}.")
    parameters = _validate_volatility_position_parameters(
        **{
            **LabelConfig.volatility_position().parameters,
            **config.parameters,
        }
    )
    labeled, transition_count, regime_lengths = _generate_for_context(
        frame,
        price_col=context.price_col,
        date_col=context.date_col,
        group_col=context.group_col,
        split_col=context.split_col,
        parameters=parameters,
    )
    known_mask = labeled["_label_known"].to_numpy(dtype=bool)
    known_labels = labeled.loc[known_mask, "Label"]
    counts = known_labels.value_counts()
    class_counts = {
        name: int(counts.get(name, 0)) for name in ("Short", "Flat", "Long")
    }
    n_known = int(known_mask.sum())
    return LabelResult(
        frame=labeled,
        known_mask=known_mask,
        class_names=("Short", "Flat", "Long"),
        semantics="target_position",
        metadata={
            "method": "volatility_position",
            "objective": config.objective,
            "semantics": config.semantics,
            "parameters": dict(parameters),
            "class_counts": class_counts,
            "n_rows": int(len(labeled)),
            "n_known": n_known,
            "n_unknown": int(len(labeled) - n_known),
            "transition_count": transition_count,
            "transition_rate": (
                transition_count / max(n_known - 1, 1) if n_known else 0.0
            ),
            "mean_regime_length": (
                float(np.mean(regime_lengths)) if regime_lengths else 0.0
            ),
            "median_regime_length": (
                float(np.median(regime_lengths)) if regime_lengths else 0.0
            ),
        },
    )


__all__ = [
    "build_persistent_positions",
    "build_volatility_position_label_result",
    "generate_volatility_position_labels",
    "generate_volatility_position_labels_by_ticker",
]
