from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def _validate_three_way_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")


def _sort_frame(frame: pd.DataFrame, date_col: str | None) -> pd.DataFrame:
    if date_col is None:
        return frame.reset_index(drop=True).copy()
    if date_col not in frame.columns:
        raise ValueError(f"Missing date column: {date_col}")
    return frame.sort_values(date_col).reset_index(drop=True).copy()


def _split_one(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    date_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = _sort_frame(frame, date_col)
    if len(work) < 3:
        raise ValueError(
            "At least three rows are required for train/validation/test split."
        )
    train_end = min(max(int(len(work) * train_ratio), 1), len(work) - 2)
    val_end = min(
        max(int(len(work) * (train_ratio + val_ratio)), train_end + 1),
        len(work) - 1,
    )
    return (
        work.iloc[:train_end].copy(),
        work.iloc[train_end:val_end].copy(),
        work.iloc[val_end:].copy(),
    )


def chronological_train_val_test_split(
    frame: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    *,
    group_col: str | None = None,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split for one series or independently per group."""

    _validate_three_way_ratios(train_ratio, val_ratio)
    if frame is None or frame.empty:
        raise ValueError("Cannot split an empty frame.")
    if group_col is None or group_col not in frame.columns:
        return _split_one(frame, train_ratio, val_ratio, date_col)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(group_col, sort=False, dropna=False):
        if len(group) < 3:
            continue
        train, val, test = _split_one(group, train_ratio, val_ratio, date_col)
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
    if not train_parts:
        raise ValueError("No group contains enough rows for splitting.")

    sort_cols = [group_col, date_col]

    def combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
        return (
            pd.concat(parts, ignore_index=True)
            .sort_values(sort_cols)
            .reset_index(drop=True)
        )

    return combine(train_parts), combine(val_parts), combine(test_parts)


def chronological_train_val_split(
    frame: pd.DataFrame,
    val_ratio: float = 0.15,
    *,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if frame is None or len(frame) < 2:
        raise ValueError("At least two rows are required for train/validation split.")
    work = _sort_frame(frame, date_col)
    split_at = min(max(int(len(work) * (1.0 - val_ratio)), 1), len(work) - 1)
    return work.iloc[:split_at].copy(), work.iloc[split_at:].copy()


def split_by_calendar_boundaries(
    frame: pd.DataFrame,
    val_start: object,
    test_start: object,
    *,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if date_col not in frame.columns:
        raise ValueError(f"Missing date column: {date_col}")
    work = frame.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    if work[date_col].isna().any():
        raise ValueError(f"Invalid values found in {date_col}.")
    val_boundary = pd.Timestamp(val_start)
    test_boundary = pd.Timestamp(test_start)
    if val_boundary >= test_boundary:
        raise ValueError("val_start must precede test_start.")
    return (
        work[work[date_col] < val_boundary].copy(),
        work[
            (work[date_col] >= val_boundary) & (work[date_col] < test_boundary)
        ].copy(),
        work[work[date_col] >= test_boundary].copy(),
    )


def prepare_feature_split_frames(
    labeled_frame: pd.DataFrame,
    *,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    feature_columns: Sequence[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    label_col: str = "Label_id",
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stable calendar splits, then fit imputation values on train only."""

    base = _sort_frame(labeled_frame, date_col)
    _, raw_val, raw_test = chronological_train_val_test_split(
        base, train_ratio=train_ratio, val_ratio=val_ratio, date_col=date_col
    )
    featured = feature_builder(base).sort_values(date_col).reset_index(drop=True).copy()
    missing = [
        column
        for column in [*feature_columns, label_col]
        if column not in featured.columns
    ]
    if missing:
        raise ValueError(f"Missing feature split columns: {missing}")
    featured.loc[:, feature_columns] = featured[list(feature_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    featured.loc[:, feature_columns] = featured[list(feature_columns)].replace(
        [np.inf, -np.inf], np.nan
    )
    featured = featured.dropna(subset=[label_col]).copy()
    train, val, test = split_by_calendar_boundaries(
        featured,
        raw_val[date_col].iloc[0],
        raw_test[date_col].iloc[0],
        date_col=date_col,
    )
    train = train.dropna(subset=[*feature_columns, label_col]).copy()
    if train.empty or val.empty or test.empty:
        raise ValueError(
            "Feature split produced an empty train, validation, or test frame."
        )
    fill_values = train[list(feature_columns)].median(numeric_only=True).fillna(0.0)
    for split in (val, test):
        split.loc[:, feature_columns] = (
            split[list(feature_columns)].fillna(fill_values).fillna(0.0)
        )
    return (
        train.sort_values(date_col).reset_index(drop=True),
        val.sort_values(date_col).reset_index(drop=True),
        test.sort_values(date_col).reset_index(drop=True),
        raw_test.sort_values(date_col).reset_index(drop=True),
    )


__all__ = [
    "chronological_train_val_split",
    "chronological_train_val_test_split",
    "prepare_feature_split_frames",
    "split_by_calendar_boundaries",
]
