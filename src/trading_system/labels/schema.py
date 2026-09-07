from __future__ import annotations

from enum import IntEnum


class TradeLabel(IntEnum):
    SELL = 0
    HOLD = 1
    BUY = 2


class PositionLabel(IntEnum):
    SHORT = 0
    FLAT = 1
    LONG = 2


LABEL_ID_TO_NAME = {
    TradeLabel.SELL.value: "Sell",
    TradeLabel.HOLD.value: "Hold",
    TradeLabel.BUY.value: "Buy",
}
LABEL_NAME_TO_ID = {name: label_id for label_id, name in LABEL_ID_TO_NAME.items()}
POSITION_ID_TO_NAME = {
    PositionLabel.SHORT.value: "Short",
    PositionLabel.FLAT.value: "Flat",
    PositionLabel.LONG.value: "Long",
}
POSITION_NAME_TO_ID = {
    name: label_id for label_id, name in POSITION_ID_TO_NAME.items()
}
N_CLASSES = len(TradeLabel)


__all__ = [
    "LABEL_ID_TO_NAME",
    "LABEL_NAME_TO_ID",
    "N_CLASSES",
    "POSITION_ID_TO_NAME",
    "POSITION_NAME_TO_ID",
    "PositionLabel",
    "TradeLabel",
]
