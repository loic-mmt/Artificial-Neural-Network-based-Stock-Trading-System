"""Reusable trading-label generators and schema."""

from .breakout import (
    enforce_alternating_signals,
    generate_breakout_labels,
    generate_breakout_labels_by_ticker,
    label_statistics,
    labelling,
    labelling_all,
)
from .forward_return import build_forward_return_labels
from .schema import LABEL_ID_TO_NAME, LABEL_NAME_TO_ID, N_CLASSES, TradeLabel

__all__ = [
    "LABEL_ID_TO_NAME",
    "LABEL_NAME_TO_ID",
    "N_CLASSES",
    "TradeLabel",
    "build_forward_return_labels",
    "enforce_alternating_signals",
    "generate_breakout_labels",
    "generate_breakout_labels_by_ticker",
    "label_statistics",
    "labelling",
    "labelling_all",
]
