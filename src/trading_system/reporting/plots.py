from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from trading_system.evaluation.classification import compute_confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str] = ("Sell", "Hold", "Buy"),
):
    import matplotlib.pyplot as plt

    matrix = compute_confusion_matrix(y_true, y_pred)
    figure, axes = plt.subplots(figsize=(6, 5))
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes)
    axes.set_title("Confusion Matrix")
    axes.set_xlabel("Prediction")
    axes.set_ylabel("Ground truth")
    axes.set_xticks(range(len(label_names)), labels=label_names)
    axes.set_yticks(range(len(label_names)), labels=label_names)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    return figure


def plot_signals(
    frame: pd.DataFrame,
    window: int = 160,
    price_col: str = "adj_close",
):
    import matplotlib.pyplot as plt

    required = {"date", "Label", price_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing signal plot columns: {sorted(missing)}")
    plot_frame = frame.sort_values("date").tail(window).copy()
    buys = plot_frame[plot_frame["Label"] == "Buy"]
    sells = plot_frame[plot_frame["Label"] == "Sell"]
    figure, axes = plt.subplots(figsize=(12, 6))
    axes.plot(
        plot_frame["date"], plot_frame[price_col], label="Price", color="steelblue"
    )
    axes.scatter(
        buys["date"], buys[price_col], label="Buy", color="green", marker="^", s=90
    )
    axes.scatter(
        sells["date"], sells[price_col], label="Sell", color="red", marker="v", s=90
    )
    axes.set_title(f"Buy/Sell signals over last {len(plot_frame)} rows")
    axes.set_xlabel("Date")
    axes.set_ylabel("Price")
    axes.tick_params(axis="x", rotation=45)
    axes.legend()
    figure.tight_layout()
    return figure


def format_experiment_summary(
    metrics: Mapping[str, float],
    backtest: Mapping[str, float],
) -> str:
    return (
        f"accuracy={metrics['acc']:.3f} balanced_accuracy={metrics['bal_acc']:.3f} "
        f"macro_f1={metrics['macro_f1']:.3f}\n"
        f"model_pnl={backtest['model_pnl']:.2f} "
        f"buy_hold_pnl={backtest['buy_hold_pnl']:.2f} "
        f"outperformance={backtest['outperformance']:.2f}"
    )


__all__ = ["format_experiment_summary", "plot_confusion_matrix", "plot_signals"]
