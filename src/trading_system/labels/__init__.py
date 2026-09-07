"""Reusable trading-label generators and schema."""

from .breakout import (
    build_breakout_label_result,
    enforce_alternating_signals,
    generate_breakout_labels,
    generate_breakout_labels_by_ticker,
    label_statistics,
    labelling,
    labelling_all,
)
from .config import LabelConfig, LabelObjective, LabelSemantics, normalize_label_method
from .forward_return import (
    build_forward_return_label_result,
    build_forward_return_labels,
    build_forward_return_labels_by_ticker,
)
from .registry import (
    LabelContext,
    LabelRegistry,
    LabelResult,
    create_default_label_registry,
)
from .schema import (
    LABEL_ID_TO_NAME,
    LABEL_NAME_TO_ID,
    N_CLASSES,
    POSITION_ID_TO_NAME,
    POSITION_NAME_TO_ID,
    PositionLabel,
    TradeLabel,
)
from .triple_barrier import (
    build_triple_barrier_label_result,
    first_barrier_touch,
    generate_triple_barrier_labels,
    generate_triple_barrier_labels_by_ticker,
    symmetric_cusum_events,
)
from .volatility_position import (
    build_persistent_positions,
    build_volatility_position_label_result,
    generate_volatility_position_labels,
    generate_volatility_position_labels_by_ticker,
)

__all__ = [
    "LABEL_ID_TO_NAME",
    "LABEL_NAME_TO_ID",
    "N_CLASSES",
    "POSITION_ID_TO_NAME",
    "POSITION_NAME_TO_ID",
    "PositionLabel",
    "TradeLabel",
    "LabelConfig",
    "LabelContext",
    "LabelObjective",
    "LabelRegistry",
    "LabelResult",
    "LabelSemantics",
    "build_breakout_label_result",
    "build_forward_return_label_result",
    "build_forward_return_labels",
    "build_forward_return_labels_by_ticker",
    "build_persistent_positions",
    "build_triple_barrier_label_result",
    "build_volatility_position_label_result",
    "enforce_alternating_signals",
    "generate_breakout_labels",
    "generate_breakout_labels_by_ticker",
    "generate_triple_barrier_labels",
    "generate_triple_barrier_labels_by_ticker",
    "generate_volatility_position_labels",
    "generate_volatility_position_labels_by_ticker",
    "label_statistics",
    "labelling",
    "labelling_all",
    "create_default_label_registry",
    "normalize_label_method",
    "first_barrier_touch",
    "symmetric_cusum_events",
]
