from __future__ import annotations

from enum import IntEnum


class TradeLabel(IntEnum):
    SELL = 0
    HOLD = 1
    BUY = 2


LABEL_ID_TO_NAME = {
    TradeLabel.SELL.value: "Sell",
    TradeLabel.HOLD.value: "Hold",
    TradeLabel.BUY.value: "Buy",
}
LABEL_NAME_TO_ID = {name: label_id for label_id, name in LABEL_ID_TO_NAME.items()}
N_CLASSES = len(TradeLabel)


__all__ = ["LABEL_ID_TO_NAME", "LABEL_NAME_TO_ID", "N_CLASSES", "TradeLabel"]
