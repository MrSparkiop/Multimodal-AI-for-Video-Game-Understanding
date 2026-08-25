"""Report figures: dataset EDA, training dynamics, comparison, explainability."""

from __future__ import annotations

from .plots import (
    FIGURE_DPI,
    HEATMAP_CMAP,
    OUTCOME_COLORS,
    PALETTE,
    apply_style,
    plot_class_imbalance,
    plot_confusion_heatmap,
    plot_description_length,
    plot_error_rate_by_group,
    plot_genre_cooccurrence,
    plot_genre_frequency,
    plot_gradcam,
    plot_gradcam_grid,
    plot_history_comparison,
    plot_labels_per_game,
    plot_learning_rate,
    plot_metric_comparison,
    plot_per_class_f1,
    plot_pr_curves,
    plot_precision_recall_scatter,
    plot_prediction_examples,
    plot_sample_images,
    plot_split_distribution,
    plot_threshold_analysis,
    plot_training_curves,
    save_figure,
)

__all__ = [
    # setup
    "FIGURE_DPI",
    "PALETTE",
    "OUTCOME_COLORS",
    "HEATMAP_CMAP",
    "apply_style",
    "save_figure",
    # dataset EDA
    "plot_genre_frequency",
    "plot_labels_per_game",
    "plot_description_length",
    "plot_genre_cooccurrence",
    "plot_class_imbalance",
    "plot_split_distribution",
    "plot_sample_images",
    # training
    "plot_training_curves",
    "plot_learning_rate",
    "plot_history_comparison",
    # model comparison
    "plot_metric_comparison",
    "plot_per_class_f1",
    "plot_precision_recall_scatter",
    "plot_threshold_analysis",
    "plot_pr_curves",
    "plot_confusion_heatmap",
    # explainability / qualitative
    "plot_gradcam",
    "plot_gradcam_grid",
    "plot_prediction_examples",
    "plot_error_rate_by_group",
]
