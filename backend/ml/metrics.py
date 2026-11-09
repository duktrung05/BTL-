from __future__ import annotations

from typing import Any


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    positions = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[positions[actual]][positions[predicted]] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(labels)) if row != index)
        false_negative = sum(matrix[index][column] for column in range(len(labels)) if column != index)
        support = sum(matrix[index])
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
    total = len(y_true)
    accuracy = sum(matrix[index][index] for index in range(len(labels))) / total if total else 0.0
    macro = {
        metric: sum(float(per_class[label][metric]) for label in labels) / len(labels)
        for metric in ("precision", "recall", "f1")
    }
    weighted = {
        metric: sum(float(per_class[label][metric]) * int(per_class[label]["support"]) for label in labels) / total
        for metric in ("precision", "recall", "f1")
    }
    return {
        "accuracy": round(accuracy, 6),
        "macro_precision": round(macro["precision"], 6),
        "macro_recall": round(macro["recall"], 6),
        "macro_f1": round(macro["f1"], 6),
        "weighted_f1": round(weighted["f1"], 6),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "class_order": labels,
        "total": total,
    }

