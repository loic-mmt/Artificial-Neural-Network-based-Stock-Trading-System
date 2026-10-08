from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LabelConfig
from .registry import LabelContext, LabelResult
from .schema import POSITION_ID_TO_NAME, PositionLabel


def build_intraday_return_labels(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Label one session Long/Short/Flat from its raw open-to-close return.

    The close is future information for a decision made just after the open.
    These labels are retrospective targets, not causal input features. Invalid
    prices remain unknown rather than being treated as genuine Flat sessions.
    """

    if frame is None or frame.empty:
        raise ValueError("Cannot label an empty frame.")
    missing = [column for column in (date_col, "open", "close") if column not in frame]
    if missing:
        raise ValueError(f"Missing intraday-return columns: {missing}")
    out = frame.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{date_col} contains invalid dates.")
    if dates.duplicated().any():
        raise ValueError("Intraday-return labels require unique dates per ticker.")
    out[date_col] = dates
    out = out.sort_values(date_col, kind="stable")
    opens = pd.to_numeric(out["open"], errors="coerce").to_numpy(dtype=np.float64)
    closes = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=np.float64)
    known = np.isfinite(opens) & np.isfinite(closes) & (opens > 0) & (closes > 0)
    returns = np.full(len(out), np.nan, dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        returns[known] = closes[known] / opens[known] - 1.0
    known &= np.isfinite(returns)
    returns[~known] = np.nan
    if "_label_known" in out:
        existing = out["_label_known"]
        if existing.isna().any():
            raise ValueError("_label_known contains missing values.")
        known &= existing.to_numpy(dtype=bool)
    returns[~known] = np.nan
    ids = np.full(len(out), PositionLabel.FLAT.value, dtype=np.int64)
    ids[known & (closes > opens)] = PositionLabel.LONG.value
    ids[known & (closes < opens)] = PositionLabel.SHORT.value
    out["intraday_ret"] = returns
    out["label_score"] = returns
    out["Label_id"] = ids
    out["Label"] = out["Label_id"].map(POSITION_ID_TO_NAME)
    out["_label_known"] = known
    counts = out.loc[known, "Label"].value_counts()
    report = {
        "n_rows": int(len(out)),
        "n_known": int(known.sum()),
        "n_unknown": int((~known).sum()),
        "n_long": int(counts.get("Long", 0)),
        "n_flat": int(counts.get("Flat", 0)),
        "n_short": int(counts.get("Short", 0)),
    }
    return out, report


def build_intraday_return_label_result(
    frame: pd.DataFrame,
    config: LabelConfig,
    context: LabelContext,
) -> LabelResult:
    """Registry adapter; no future row or partition is used by the target."""

    if not isinstance(config, LabelConfig) or config.method != "intraday_return":
        raise ValueError("Intraday-return builder requires method='intraday_return'.")
    if not isinstance(context, LabelContext):
        raise TypeError("context must be a LabelContext.")
    if config.semantics != "target_position":
        raise ValueError("Intraday-return labels require semantics='target_position'.")
    if config.parameters:
        raise ValueError(f"Unknown intraday-return parameters: {sorted(config.parameters)}.")
    if context.group_col is not None and context.group_col not in frame:
        raise ValueError(f"Missing intraday-return group column: {context.group_col}")
    if context.group_col is None:
        labeled = build_intraday_return_labels(frame, date_col=context.date_col)[0]
    else:
        if frame.empty:
            raise ValueError("Cannot label an empty frame.")
        parts = [
            build_intraday_return_labels(group, date_col=context.date_col)[0]
            for _, group in frame.groupby(context.group_col, sort=False, dropna=False)
        ]
        labeled = pd.concat(parts, ignore_index=True).sort_values(
            [context.group_col, context.date_col], kind="stable"
        )
    labeled = labeled.reset_index(drop=True)
    known = labeled["_label_known"].to_numpy(dtype=bool)
    counts = labeled.loc[known, "Label"].value_counts()
    class_counts = {name: int(counts.get(name, 0)) for name in config.class_names}
    n_known = int(known.sum())
    return LabelResult(
        frame=labeled,
        known_mask=known,
        class_names=config.class_names,
        semantics=config.semantics,
        metadata={
            "method": "intraday_return",
            "objective": config.objective,
            "semantics": config.semantics,
            "parameters": {},
            "class_counts": class_counts,
            "n_rows": int(len(labeled)),
            "n_known": n_known,
            "n_unknown": int(len(labeled) - n_known),
            "action_rate": (
                (class_counts["Long"] + class_counts["Short"]) / n_known
                if n_known else 0.0
            ),
            "return_column": "intraday_ret",
            "entry_price": "raw open J",
            "exit_price": "raw close J",
            "target_available_at": "close J",
            "execution_note": (
                "Retrospective open-to-close target. An open-J decision must not "
                "use close/high/low/volume J as input. The existing runner's "
                "execution timing is unchanged."
            ),
        },
    )


__all__ = ["build_intraday_return_labels", "build_intraday_return_label_result"]
