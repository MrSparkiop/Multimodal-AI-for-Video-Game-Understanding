"""Multi-label classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..config import GENRES

__all__ = [
    "METRIC_KEYS",
    "binarize",
    "multilabel_metrics",
    "per_class_table",
    "aggregate_seeds",
    "compare_models",
    "confusion_counts",
]

#: Headline metrics, in the order used by every table in the report.
METRIC_KEYS: tuple[str, ...] = (
    "micro_f1",
    "macro_f1",
    "micro_precision",
    "micro_recall",
    "macro_precision",
    "macro_recall",
    "mAP",
    "micro_ap",
    "hamming_loss",
    "subset_accuracy",
)


def _as_2d(array: Any, name: str, n_classes: int | None = None) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {values.shape}")
    if n_classes is not None and values.shape[1] != n_classes:
        raise ValueError(f"{name} has {values.shape[1]} columns, expected {n_classes}")
    return values


def binarize(
    probabilities: Any, threshold: float | Sequence[float] | np.ndarray = 0.5
) -> np.ndarray:
    """Threshold probabilities into ``{0, 1}`` predictions."""
    probs = _as_2d(probabilities, "probabilities")
    if np.isscalar(threshold) or (isinstance(threshold, np.ndarray) and threshold.ndim == 0):
        return (probs >= float(threshold)).astype(np.int8)
    vector = np.asarray(threshold, dtype=np.float64).ravel()
    if vector.shape[0] != probs.shape[1]:
        raise ValueError(
            f"per-class threshold has {vector.shape[0]} entries but there are "
            f"{probs.shape[1]} classes"
        )
    return (probs >= vector[None, :]).astype(np.int8)


def multilabel_metrics(
    y_true: Any,
    y_prob: Any,
    *,
    threshold: float | Sequence[float] | np.ndarray = 0.5,
    classes: Sequence[str] = GENRES,
    include_per_class: bool = True,
) -> dict[str, Any]:
    """Compute the full multi-label metric suite."""
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    n_classes = len(classes)
    truth = _as_2d(y_true, "y_true", n_classes).astype(np.int8)
    probs = _as_2d(y_prob, "y_prob", n_classes)
    if truth.shape[0] != probs.shape[0]:
        raise ValueError(
            f"y_true has {truth.shape[0]} rows but y_prob has {probs.shape[0]}"
        )
    predictions = binarize(probs, threshold)

    support = truth.sum(axis=0)
    metrics: dict[str, Any] = {
        "n_samples": int(truth.shape[0]),
        "n_classes": n_classes,
        "threshold": (
            float(threshold)
            if np.isscalar(threshold)
            else [round(float(t), 4) for t in np.asarray(threshold).ravel()]
        ),
        "micro_precision": float(precision_score(truth, predictions, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(truth, predictions, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(truth, predictions, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(truth, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(truth, predictions, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(truth, predictions)),
        "subset_accuracy": float((predictions == truth).all(axis=1).mean()),
        "label_cardinality_true": float(truth.sum(axis=1).mean()),
        "label_cardinality_pred": float(predictions.sum(axis=1).mean()),
    }

    # Threshold-free ranking quality.  Classes without positives are skipped
    # rather than silently scored as 0, which would understate the model.
    valid = support > 0
    if valid.any():
        metrics["mAP"] = float(
            average_precision_score(truth[:, valid], probs[:, valid], average="macro")
        )
        metrics["micro_ap"] = float(
            average_precision_score(truth[:, valid], probs[:, valid], average="micro")
        )
    else:  # pragma: no cover - degenerate split
        metrics["mAP"] = float("nan")
        metrics["micro_ap"] = float("nan")

    usable_auc = valid & (support < truth.shape[0])
    if usable_auc.any():
        try:
            metrics["macro_roc_auc"] = float(
                roc_auc_score(truth[:, usable_auc], probs[:, usable_auc], average="macro")
            )
        except ValueError:  # pragma: no cover
            metrics["macro_roc_auc"] = float("nan")
    else:  # pragma: no cover
        metrics["macro_roc_auc"] = float("nan")

    if include_per_class:
        per_class_f1 = f1_score(truth, predictions, average=None, zero_division=0)
        per_class_precision = precision_score(truth, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(truth, predictions, average=None, zero_division=0)
        per_class: dict[str, dict[str, float]] = {}
        for index, genre in enumerate(classes):
            entry = {
                "f1": float(per_class_f1[index]),
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "support": int(support[index]),
                "predicted_positive": int(predictions[:, index].sum()),
            }
            if support[index] > 0:
                entry["average_precision"] = float(
                    average_precision_score(truth[:, index], probs[:, index])
                )
            else:  # pragma: no cover
                entry["average_precision"] = float("nan")
            per_class[genre] = entry
        metrics["per_class"] = per_class
    return metrics


def per_class_table(metrics: dict[str, Any], *, classes: Sequence[str] = GENRES) -> "Any":
    """Return the ``per_class`` block of *metrics* as a tidy DataFrame."""
    import pandas as pd

    rows = []
    for genre in classes:
        entry = metrics.get("per_class", {}).get(genre, {})
        rows.append(
            {
                "genre": genre,
                "support": entry.get("support", 0),
                "precision": entry.get("precision", float("nan")),
                "recall": entry.get("recall", float("nan")),
                "f1": entry.get("f1", float("nan")),
                "average_precision": entry.get("average_precision", float("nan")),
                "predicted_positive": entry.get("predicted_positive", 0),
            }
        )
    return pd.DataFrame(rows)


def confusion_counts(
    y_true: Any,
    y_prob: Any,
    *,
    threshold: float | Sequence[float] | np.ndarray = 0.5,
    classes: Sequence[str] = GENRES,
) -> dict[str, dict[str, int]]:
    """Per-class TP / FP / FN / TN counts at a given threshold."""
    truth = _as_2d(y_true, "y_true", len(classes)).astype(np.int8)
    predictions = binarize(y_prob, threshold)
    out: dict[str, dict[str, int]] = {}
    for index, genre in enumerate(classes):
        true_column = truth[:, index]
        pred_column = predictions[:, index]
        out[genre] = {
            "tp": int(((true_column == 1) & (pred_column == 1)).sum()),
            "fp": int(((true_column == 0) & (pred_column == 1)).sum()),
            "fn": int(((true_column == 1) & (pred_column == 0)).sum()),
            "tn": int(((true_column == 0) & (pred_column == 0)).sum()),
        }
    return out


def aggregate_seeds(
    runs: Sequence[dict[str, Any]], *, keys: Sequence[str] = METRIC_KEYS
) -> dict[str, dict[str, float]]:
    """Aggregate the same metrics across repeated runs into mean / std / n."""
    aggregated: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [
            float(run[key])
            for run in runs
            if key in run and run[key] is not None and np.isfinite(float(run[key]))
        ]
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        aggregated[key] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "min": float(array.min()),
            "max": float(array.max()),
            "n": int(array.size),
        }
    return aggregated


def compare_models(
    metrics_by_model: dict[str, dict[str, Any]],
    *,
    keys: Sequence[str] = METRIC_KEYS,
    display_names: dict[str, str] | None = None,
) -> "Any":
    """Build the headline comparison table (one row per model)."""
    import pandas as pd

    rows = []
    for model_key, metrics in metrics_by_model.items():
        row: dict[str, Any] = {"model": (display_names or {}).get(model_key, model_key)}
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, dict) and "mean" in value:
                row[key] = value["mean"]
                row[f"{key}_std"] = value.get("std", 0.0)
            elif value is not None:
                row[key] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)
