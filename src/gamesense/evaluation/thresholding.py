"""Decision-threshold selection for multi-label prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..config import CONFIG, GENRES, EvaluationConfig
from ..utils import get_logger
from .metrics import binarize, multilabel_metrics

__all__ = [
    "threshold_sweep",
    "search_global_threshold",
    "search_per_class_thresholds",
    "optimize_thresholds",
]

LOGGER = get_logger("gamesense.evaluation.thresholding")


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    grid: Sequence[float] = CONFIG.evaluation.threshold_grid,
    classes: Sequence[str] = GENRES,
) -> "Any":
    """Evaluate every threshold in *grid* and return a tidy DataFrame."""
    import pandas as pd

    rows = []
    for threshold in grid:
        metrics = multilabel_metrics(
            y_true, y_prob, threshold=threshold, classes=classes, include_per_class=False
        )
        rows.append(
            {
                "threshold": float(threshold),
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "micro_precision": metrics["micro_precision"],
                "micro_recall": metrics["micro_recall"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "hamming_loss": metrics["hamming_loss"],
                "subset_accuracy": metrics["subset_accuracy"],
                "label_cardinality_pred": metrics["label_cardinality_pred"],
            }
        )
    return pd.DataFrame(rows)


def search_global_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    grid: Sequence[float] = CONFIG.evaluation.threshold_grid,
    metric: str = CONFIG.evaluation.threshold_metric,
    classes: Sequence[str] = GENRES,
) -> tuple[float, dict[str, Any]]:
    """Pick the single threshold that maximises *metric* on the given split.

    Returns ``(threshold, details)``. Call this with **validation** probabilities only.
    """
    best_threshold = float(CONFIG.evaluation.default_threshold)
    best_score = -np.inf
    scores: dict[str, float] = {}
    for threshold in grid:
        metrics = multilabel_metrics(
            y_true, y_prob, threshold=threshold, classes=classes, include_per_class=False
        )
        score = float(metrics.get(metric, -np.inf))
        scores[f"{threshold:.2f}"] = score
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    LOGGER.info("Best global threshold %.2f (%s = %.4f)", best_threshold, metric, best_score)
    return best_threshold, {
        "strategy": "global",
        "metric": metric,
        "grid": [float(t) for t in grid],
        "scores": scores,
        "best_threshold": best_threshold,
        "best_score": best_score,
    }


def search_per_class_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    grid: Sequence[float] = CONFIG.evaluation.threshold_grid,
    classes: Sequence[str] = GENRES,
    fallback: float = CONFIG.evaluation.default_threshold,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pick one threshold per genre, each maximising that genre's binary F1.

    Returns ``(thresholds, details)`` with one threshold per genre.
    """
    from sklearn.metrics import f1_score

    truth = np.asarray(y_true)
    probs = np.asarray(y_prob)
    thresholds = np.full(len(classes), float(fallback), dtype=np.float64)
    details: dict[str, Any] = {}

    for index, genre in enumerate(classes):
        column_true = truth[:, index]
        column_prob = probs[:, index]
        if column_true.sum() == 0:  # nothing to optimise against
            details[genre] = {"threshold": float(fallback), "f1": None, "support": 0}
            continue
        best_threshold, best_f1 = float(fallback), -np.inf
        for threshold in grid:
            score = float(
                f1_score(column_true, (column_prob >= threshold).astype(np.int8), zero_division=0)
            )
            if score > best_f1:
                best_f1, best_threshold = score, float(threshold)
        thresholds[index] = best_threshold
        details[genre] = {
            "threshold": best_threshold,
            "f1": best_f1,
            "support": int(column_true.sum()),
        }

    macro = multilabel_metrics(
        truth, probs, threshold=thresholds, classes=classes, include_per_class=False
    )
    LOGGER.info(
        "Per-class thresholds %s -> macro-F1 %.4f",
        np.round(thresholds, 2).tolist(),
        macro["macro_f1"],
    )
    return thresholds, {
        "strategy": "per_class",
        "metric": "f1_per_class",
        "grid": [float(t) for t in grid],
        "per_class": details,
        "thresholds": [float(t) for t in thresholds],
        "resulting_macro_f1": macro["macro_f1"],
        "resulting_micro_f1": macro["micro_f1"],
    }


def optimize_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    strategy: str | None = None,
    evaluation: EvaluationConfig | None = None,
    classes: Sequence[str] = GENRES,
) -> tuple[float | np.ndarray, dict[str, Any]]:
    """Dispatch to the configured threshold-selection strategy.

    Returns ``(threshold, details)``. Always call with **validation** probabilities.
    """
    cfg = evaluation or CONFIG.evaluation
    chosen = (strategy or cfg.threshold_strategy).lower()
    if chosen == "global":
        return search_global_threshold(
            y_true, y_prob, grid=cfg.threshold_grid, metric=cfg.threshold_metric, classes=classes
        )
    if chosen == "per_class":
        return search_per_class_thresholds(
            y_true, y_prob, grid=cfg.threshold_grid, classes=classes,
            fallback=cfg.default_threshold,
        )
    if chosen in ("none", "default", "fixed"):
        return float(cfg.default_threshold), {
            "strategy": "fixed",
            "best_threshold": float(cfg.default_threshold),
        }
    raise ValueError(f"unknown threshold strategy {strategy!r}")
