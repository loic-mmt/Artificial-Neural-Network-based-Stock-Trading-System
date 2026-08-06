from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _validate_context_len(context_len: int) -> int:
    if not isinstance(context_len, (int, np.integer)):
        raise TypeError("context_len must be an integer.")
    if context_len <= 0:
        raise ValueError("context_len must be positive.")
    return int(context_len)


def build_sequence_dataset(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    context_len: int,
    target_start: int = 0,
    return_indices: bool = False,
    *,
    label_col: str = "Label_id",
):
    """Build labeled 3D windows with shape ``(samples, time, features)``."""

    # Keep the label outside the feature tensor: including it would leak the
    # answer directly into every training sequence.
    context_len = _validate_context_len(context_len)
    if isinstance(feature_columns, str):
        raise TypeError("feature_columns must be a sequence of column names.")
    columns = list(feature_columns)
    if not columns:
        raise ValueError("feature_columns must not be empty.")
    if any(not isinstance(column, str) or not column for column in columns):
        raise TypeError("feature_columns must contain non-empty strings.")
    if len(columns) != len(set(columns)):
        raise ValueError("feature_columns must not contain duplicates.")
    if label_col in columns:
        raise ValueError("label_col must not be included in feature_columns.")
    missing = [
        column
        for column in dict.fromkeys([*columns, label_col])
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing sequence columns: {missing}")

    # target_start is positional, not a DataFrame index. The history-aware builder
    # uses it to prevent prefix rows from becoming returned targets.
    if isinstance(target_start, (bool, np.bool_)) or not isinstance(
        target_start, (int, np.integer)
    ):
        raise TypeError("target_start must be an integer.")
    target_start = int(target_start)
    if target_start < 0:
        raise ValueError("target_start must be non-negative.")

    # Preserve the full 3D contract even when no sample can be constructed.
    empty_x = np.empty((0, context_len, len(columns)), dtype=np.float32)
    empty_y = np.empty((0,), dtype=np.int64)
    empty_idx = np.empty((0,), dtype=np.int64)
    if len(frame) < context_len:
        return (empty_x, empty_y, empty_idx) if return_indices else (empty_x, empty_y)

    # Convert features once before slicing. Non-finite inputs would later poison
    # the scaler and model, so they fail at this data-layer boundary.
    values = frame[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Sequence features must contain only finite values.")
    numeric_labels = pd.to_numeric(frame[label_col], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric_labels).all():
        raise ValueError("Sequence labels must contain only finite numeric values.")
    if not np.equal(numeric_labels, np.floor(numeric_labels)).all():
        raise ValueError("Sequence labels must be integer class identifiers.")
    labels = numeric_labels.astype(np.int64)

    # A target requires T-1 earlier rows. A larger target_start deliberately skips
    # complete windows belonging to an historical prefix.
    first_target = max(target_start, context_len - 1)
    indices = np.arange(first_target, len(frame), dtype=np.int64)
    if len(indices) == 0:
        return (empty_x, empty_y, empty_idx) if return_indices else (empty_x, empty_y)
    # Each slice ends on its target row and never reads a future row. np.stack
    # preserves `(time, features)` instead of flattening the two dimensions.
    windows = np.stack(
        [values[index - context_len + 1 : index + 1] for index in indices],
        axis=0,
    ).astype(np.float32, copy=False)
    # The target and positional index both refer to the final row of each window.
    # These indices can later align predictions with `frame.iloc` and prices.
    targets = labels[indices]
    if not (len(windows) == len(targets) == len(indices)):
        raise RuntimeError("Sequence windows, labels and indices are misaligned.")
    return (windows, targets, indices) if return_indices else (windows, targets)


def build_sequence_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    context_len: int,
    target_start: int = 0,
    return_indices: bool = False,
):
    """Build unlabeled 3D windows for inference."""

    context_len = _validate_context_len(context_len)
    if isinstance(feature_columns, str):
        raise TypeError("feature_columns must be a sequence of column names.")
    columns = list(feature_columns)
    if not columns:
        raise ValueError("feature_columns must not be empty.")
    if any(not isinstance(column, str) or not column for column in columns):
        raise TypeError("feature_columns must contain non-empty strings.")
    if len(columns) != len(set(columns)):
        raise ValueError("feature_columns must not contain duplicates.")

    # Inference does not require a label column. This allows the same function to
    # process live or walk-forward chunks whose true class is not yet available.
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing sequence feature columns: {missing}")

    # target_start is a positional boundary. A history prefix can supply context,
    # while this boundary prevents the prefix itself from receiving predictions.
    if isinstance(target_start, (bool, np.bool_)) or not isinstance(
        target_start, (int, np.integer)
    ):
        raise TypeError("target_start must be an integer.")
    target_start = int(target_start)
    if target_start < 0:
        raise ValueError("target_start must be non-negative.")

    # Preserve `(samples, time, features)` even for an empty result. Downstream
    # models can inspect dimensions without adding special cases.
    empty_x = np.empty((0, context_len, len(columns)), dtype=np.float32)
    empty_idx = np.empty((0,), dtype=np.int64)
    if len(frame) < context_len:
        return (empty_x, empty_idx) if return_indices else empty_x

    # The function respects the row order supplied by the caller. Chronological
    # sorting belongs to split/history orchestration, not this low-level builder.
    values = frame[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Sequence features must contain only finite values.")

    # A valid window needs T-1 rows before its final row. A larger target_start
    # intentionally skips earlier complete windows belonging to history.
    first_target = max(target_start, context_len - 1)
    indices = np.arange(first_target, len(frame), dtype=np.int64)
    if len(indices) == 0:
        return (empty_x, empty_idx) if return_indices else empty_x

    # Use exactly the same inclusive slicing rule as the labeled builder. Keeping
    # the two axes intact guarantees RNN/LSTM/GRU/Transformer input `(N, T, F)`.
    windows = np.stack(
        [values[index - context_len + 1 : index + 1] for index in indices],
        axis=0,
    ).astype(np.float32, copy=False)

    # Each positional index identifies the final source row of its sequence and
    # can therefore map predictions back to the correct chunk row.
    if len(windows) != len(indices):
        raise RuntimeError("Sequence features and indices are misaligned.")
    return (windows, indices) if return_indices else windows


def build_sequence_dataset_with_history(
    target_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    context_len: int,
    history_frame: pd.DataFrame | None = None,
    *,
    group_col: str | None = "ticker",
    date_col: str = "date",
    label_col: str = "Label_id",
    return_aligned_rows: bool = False,
):
    """Build target sequences using earlier split history without group leakage."""

    context_len = _validate_context_len(context_len)
    if isinstance(feature_columns, str):
        raise TypeError("feature_columns must be a sequence of column names.")
    columns = list(feature_columns)
    if not columns:
        raise ValueError("feature_columns must not be empty.")
    if len(columns) != len(set(columns)):
        raise ValueError("feature_columns must not contain duplicates.")

    # Work on copies because this function sorts rows and adds a harmless label
    # placeholder to historical rows. The caller's splits must remain untouched.
    target = target_frame.copy()
    history = None if history_frame is None else history_frame.copy()
    grouped = group_col is not None

    def require_columns(
        frame: pd.DataFrame,
        required: Sequence[str],
        frame_name: str,
    ) -> None:
        # Removing repeated names keeps errors readable if, for example, date_col
        # also appears in feature_columns.
        unique_required = tuple(dict.fromkeys(required))
        missing = [column for column in unique_required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing {frame_name} sequence columns: {missing}")

    target_required = [*columns, label_col, date_col]
    history_required = [*columns, date_col]
    if grouped:
        target_required.append(group_col)
        history_required.append(group_col)
    require_columns(target, target_required, "target")
    if history is not None:
        # Historical labels are not required. History provides feature context,
        # but none of its rows can become a returned training target.
        require_columns(history, history_required, "history")

    # These canonical shapes also cover an empty target or groups too short to
    # create a complete sequence.
    empty_x = np.empty((0, context_len, len(columns)), dtype=np.float32)
    empty_y = np.empty((0,), dtype=np.int64)

    def sort_and_validate_dates(
        frame: pd.DataFrame,
        frame_name: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return a chronological copy plus UTC dates used only for comparison."""

        work = frame.reset_index(drop=True).copy()
        parsed_dates = pd.to_datetime(work[date_col], errors="coerce", utc=True)
        if parsed_dates.isna().any():
            bad_rows = parsed_dates[parsed_dates.isna()].index.tolist()
            raise ValueError(f"Invalid {frame_name} dates at rows: {bad_rows}")
        if parsed_dates.duplicated().any():
            duplicate_dates = (
                parsed_dates[parsed_dates.duplicated(keep=False)]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            raise ValueError(
                f"Duplicate {frame_name} dates inside one series: {duplicate_dates}"
            )

        # Sort with normalized timestamps while preserving the original date
        # values in rows that will later be returned to the backtest.
        order = parsed_dates.sort_values(kind="mergesort").index
        sorted_frame = work.iloc[order].reset_index(drop=True)
        sorted_dates = parsed_dates.iloc[order].reset_index(drop=True)
        return sorted_frame, sorted_dates

    def build_one(
        one_target: pd.DataFrame,
        one_history: pd.DataFrame | None,
        series_name: str,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Build sequences for one ticker, or for the one ungrouped series."""

        sorted_target, target_dates = sort_and_validate_dates(
            one_target, f"target {series_name}"
        )
        if sorted_target.empty:
            return empty_x.copy(), empty_y.copy(), sorted_target

        if one_history is None or one_history.empty:
            sorted_history = sorted_target.iloc[0:0].copy()
        else:
            sorted_history, history_dates = sort_and_validate_dates(
                one_history, f"history {series_name}"
            )

            # This strict boundary prevents validation/test windows from seeing
            # an overlapping or future history row.
            first_target_date = target_dates.iloc[0]
            invalid_history = history_dates >= first_target_date
            if invalid_history.any():
                first_invalid = history_dates[invalid_history].iloc[0]
                raise ValueError(
                    f"History for {series_name} must end before target data; "
                    f"found {first_invalid} at/after {first_target_date}."
                )

        # A sequence ending on its target needs at most T-1 earlier rows. For
        # context_len=1, the target row alone forms the complete sequence.
        prefix = (
            sorted_history.iloc[0:0].copy()
            if context_len == 1
            else sorted_history.tail(context_len - 1).copy()
        )

        # The basic builder converts the complete label column to int64. History
        # may be unlabeled, so give prefix rows a placeholder that is never used
        # as a returned target because target_start skips the whole prefix.
        prefix[label_col] = 0
        source = pd.concat([prefix, sorted_target], ignore_index=True)

        X, y, indices = build_sequence_dataset(
            source,
            columns,
            context_len,
            target_start=len(prefix),
            return_indices=True,
            label_col=label_col,
        )

        # Positional indices point to the final row of each window. Using them on
        # the exact same source keeps labels, predictions, prices and dates aligned.
        if len(indices) and (indices < len(prefix)).any():
            raise RuntimeError("A generated sequence is aligned to a history row.")
        aligned = source.iloc[indices].copy().reset_index(drop=True)
        if not (len(X) == len(y) == len(aligned)):
            raise RuntimeError(
                "Sequence outputs and aligned rows have different sizes."
            )
        return X, y, aligned

    if not grouped:
        X, y, aligned = build_one(target, history, "series")
        return (X, y, aligned) if return_aligned_rows else (X, y)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    aligned_parts: list[pd.DataFrame] = []

    # Each ticker is processed separately. This is the key protection preventing
    # the last rows of one asset from becoming the context of another asset.
    for group_value, one_target in target.groupby(group_col, sort=False, dropna=False):
        one_history = None
        if history is not None:
            one_history = (
                history[history[group_col].isna()].copy()
                if pd.isna(group_value)
                else history[history[group_col] == group_value].copy()
            )
        X, y, aligned = build_one(
            one_target,
            one_history,
            f"{group_col}={group_value!r}",
        )
        if len(X):
            # Append all three outputs together so their ordering stays identical.
            X_parts.append(X)
            y_parts.append(y)
            aligned_parts.append(aligned)

    if not X_parts:
        empty_aligned = target.iloc[0:0].copy().reset_index(drop=True)
        return (
            (empty_x, empty_y, empty_aligned)
            if return_aligned_rows
            else (empty_x, empty_y)
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    aligned = pd.concat(aligned_parts, ignore_index=True)
    return (X, y, aligned) if return_aligned_rows else (X, y)


def build_context_dataset(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    context_len: int,
    target_start: int = 0,
    return_indices: bool = False,
    *,
    label_col: str = "Label_id",
):
    """Flatten rolling feature windows and align each window with its target."""

    context_len = _validate_context_len(context_len)
    columns = list(feature_columns)
    missing = [
        column for column in [*columns, label_col] if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing context columns: {missing}")
    feature_dim = len(columns)
    empty_x = np.empty((0, context_len * feature_dim), dtype=np.float32)
    empty_y = np.empty((0,), dtype=np.int64)
    empty_idx = np.empty((0,), dtype=np.int64)
    if len(frame) < context_len:
        return (empty_x, empty_y, empty_idx) if return_indices else (empty_x, empty_y)

    values = frame[columns].to_numpy(dtype=np.float32)
    labels = frame[label_col].to_numpy(dtype=np.int64)
    first_target = max(int(target_start), context_len - 1)
    indices = np.arange(first_target, len(frame), dtype=np.int64)
    if len(indices) == 0:
        return (empty_x, empty_y, empty_idx) if return_indices else (empty_x, empty_y)
    windows = np.asarray(
        [values[index - context_len + 1 : index + 1].reshape(-1) for index in indices],
        dtype=np.float32,
    )
    targets = labels[indices]
    return (windows, targets, indices) if return_indices else (windows, targets)


def build_context_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    context_len: int,
    target_start: int = 0,
    return_indices: bool = False,
):
    """Build inference windows without requiring a label column."""

    context_len = _validate_context_len(context_len)
    columns = list(feature_columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing context feature columns: {missing}")
    feature_dim = len(columns)
    empty_x = np.empty((0, context_len * feature_dim), dtype=np.float32)
    empty_idx = np.empty((0,), dtype=np.int64)
    if len(frame) < context_len:
        return (empty_x, empty_idx) if return_indices else empty_x
    values = frame[columns].to_numpy(dtype=np.float32)
    first_target = max(int(target_start), context_len - 1)
    indices = np.arange(first_target, len(frame), dtype=np.int64)
    if len(indices) == 0:
        return (empty_x, empty_idx) if return_indices else empty_x
    windows = np.asarray(
        [values[index - context_len + 1 : index + 1].reshape(-1) for index in indices],
        dtype=np.float32,
    )
    return (windows, indices) if return_indices else windows


def build_context_dataset_with_history(
    target_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    context_len: int,
    history_frame: pd.DataFrame | None = None,
    *,
    group_col: str | None = "ticker",
    date_col: str = "date",
    label_col: str = "Label_id",
    return_aligned_rows: bool = False,
):
    """Build target windows using preceding split history without group leakage."""

    context_len = _validate_context_len(context_len)
    target = target_frame.copy()
    history = None if history_frame is None else history_frame.copy()
    columns = list(feature_columns)

    def build_one(one_target: pd.DataFrame, one_history: pd.DataFrame | None):
        one_target = one_target.sort_values(date_col).copy()
        prefix = (
            one_target.iloc[0:0].copy()
            if one_history is None
            else one_history.sort_values(date_col).tail(context_len - 1).copy()
        )
        source = pd.concat([prefix, one_target], ignore_index=True)
        X, y, indices = build_context_dataset(
            source,
            columns,
            context_len,
            target_start=len(prefix),
            return_indices=True,
            label_col=label_col,
        )
        return X, y, source.iloc[indices].copy().reset_index(drop=True)

    grouped = group_col is not None and group_col in target.columns
    if not grouped:
        X, y, aligned = build_one(target, history)
        return (X, y, aligned) if return_aligned_rows else (X, y)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    aligned_parts: list[pd.DataFrame] = []
    for group_value, one_target in target.groupby(group_col, sort=False, dropna=False):
        one_history = None
        if history is not None and group_col in history.columns:
            one_history = (
                history[history[group_col].isna()].copy()
                if pd.isna(group_value)
                else history[history[group_col] == group_value].copy()
            )
        X, y, aligned = build_one(one_target, one_history)
        if len(X):
            X_parts.append(X)
            y_parts.append(y)
            aligned_parts.append(aligned)

    if not X_parts:
        empty_x = np.empty((0, context_len * len(columns)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64)
        empty_aligned = target.iloc[0:0].copy()
        return (
            (empty_x, empty_y, empty_aligned)
            if return_aligned_rows
            else (empty_x, empty_y)
        )
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    aligned = pd.concat(aligned_parts, ignore_index=True)
    return (X, y, aligned) if return_aligned_rows else (X, y)


__all__ = [
    "build_context_dataset",
    "build_context_dataset_with_history",
    "build_context_features",
    "build_sequence_dataset",
    "build_sequence_dataset_with_history",
    "build_sequence_features",
]
