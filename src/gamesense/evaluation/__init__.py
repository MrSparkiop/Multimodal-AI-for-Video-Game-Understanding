"""Multi-label metrics, unified evaluation, threshold search and error analysis."""

from __future__ import annotations

from .error_analysis import (
    build_error_frame,
    classify_outcome,
    co_occurrence_confusion,
    compare_model_errors,
    confusion_pairs,
    enrich_with_image_stats,
    grouped_error_rates,
    image_statistics,
    per_class_error_summary,
    select_examples,
)
from .evaluator import (
    EvaluationResult,
    aggregate_over_seeds,
    build_comparison_table,
    collect_results,
    evaluate_model,
    evaluate_predictions,
    load_predictions,
    save_predictions,
)
from .metrics import (
    METRIC_KEYS,
    aggregate_seeds,
    binarize,
    compare_models,
    confusion_counts,
    multilabel_metrics,
    per_class_table,
)
from .thresholding import (
    optimize_thresholds,
    search_global_threshold,
    search_per_class_thresholds,
    threshold_sweep,
)

__all__ = [
    # metrics
    "METRIC_KEYS",
    "multilabel_metrics",
    "binarize",
    "per_class_table",
    "confusion_counts",
    "aggregate_seeds",
    "compare_models",
    # thresholding
    "threshold_sweep",
    "search_global_threshold",
    "search_per_class_thresholds",
    "optimize_thresholds",
    # evaluator
    "EvaluationResult",
    "evaluate_predictions",
    "evaluate_model",
    "save_predictions",
    "load_predictions",
    "collect_results",
    "aggregate_over_seeds",
    "build_comparison_table",
    # error analysis
    "classify_outcome",
    "build_error_frame",
    "select_examples",
    "per_class_error_summary",
    "confusion_pairs",
    "co_occurrence_confusion",
    "image_statistics",
    "enrich_with_image_stats",
    "grouped_error_rates",
    "compare_model_errors",
]
