from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trading_system.labels.schema import N_CLASSES

DEFAULT_LABELS = tuple(range(N_CLASSES))


def _paired_arrays(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must be same-length 1D arrays.")
    if len(actual) == 0:
        raise ValueError("Metrics require at least one observation.")
    return actual, predicted


def recall_for_label(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    actual, predicted = _paired_arrays(y_true, y_pred)
    true_positive = int(((actual == label) & (predicted == label)).sum())
    false_negative = int(((actual == label) & (predicted != label)).sum())
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator else 0.0


def precision_recall_f1_for_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: int,
) -> tuple[float, float, float]:
    actual, predicted = _paired_arrays(y_true, y_pred)
    true_positive = int(((actual == label) & (predicted == label)).sum())
    false_positive = int(((actual != label) & (predicted == label)).sum())
    false_negative = int(((actual == label) & (predicted != label)).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] = DEFAULT_LABELS,
) -> float:
    return float(np.mean([recall_for_label(y_true, y_pred, label) for label in labels]))


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] = DEFAULT_LABELS,
) -> float:
    scores = [
        precision_recall_f1_for_label(y_true, y_pred, label)[2] for label in labels
    ]
    return float(np.mean(scores))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    actual, predicted = _paired_arrays(y_true, y_pred)
    sell = precision_recall_f1_for_label(actual, predicted, 0)
    hold = precision_recall_f1_for_label(actual, predicted, 1)
    buy = precision_recall_f1_for_label(actual, predicted, 2)
    return {
        "acc": float((predicted == actual).mean()),
        "bal_acc": balanced_accuracy(actual, predicted),
        "macro_f1": macro_f1(actual, predicted),
        "precision_sell": sell[0],
        "recall_sell": sell[1],
        "precision_hold": hold[0],
        "recall_hold": hold[1],
        "precision_buy": buy[0],
        "recall_buy": buy[1],
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] = DEFAULT_LABELS,
) -> np.ndarray:
    actual, predicted = _paired_arrays(y_true, y_pred)
    label_to_index = {int(label): index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual_label, predicted_label in zip(actual, predicted):
        if (
            int(actual_label) not in label_to_index
            or int(predicted_label) not in label_to_index
        ):
            raise ValueError(
                "Confusion matrix received a label outside configured labels."
            )
        matrix[
            label_to_index[int(actual_label)], label_to_index[int(predicted_label)]
        ] += 1
    return matrix


__all__ = [
    "balanced_accuracy",
    "compute_confusion_matrix",
    "evaluate_predictions",
    "macro_f1",
    "precision_recall_f1_for_label",
    "recall_for_label",
]
