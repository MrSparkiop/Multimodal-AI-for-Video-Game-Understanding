"""Every figure that appears in the GameSense report, in one place."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..config import CONFIG, GENRES, MODEL_DISPLAY_NAMES
from ..utils import get_logger

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

LOGGER = get_logger("gamesense.visualization.plots")


# =========================================================================== #
# A. Setup: style, palette, saving
# =========================================================================== #
#: Resolution of every saved figure.  150 dpi keeps a 7x4.5 inch figure sharp in
#: a printed A4 report without producing multi-megabyte PNGs.
FIGURE_DPI: Final[int] = 150

# : Colour-blind-safe categorical palette (Okabe & Ito, 2008).
PALETTE: Final[tuple[str, ...]] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#7F7F7F",  # grey
    "#8C6D31",  # brown
)

#: Semantic colours for prediction quality, used by the qualitative figures.
OUTCOME_COLORS: Final[dict[str, str]] = {
    "correct": "#009E73",  # true genre that was predicted   (true positive)
    "missed": "#D55E00",  # true genre that was not predicted (false negative)
    "spurious": "#CC79A7",  # predicted genre that is not true (false positive)
    "absent": "#B0B0B0",  # correctly left unpredicted        (true negative)
}

#: Perceptually uniform colormap for every heatmap in the report.
HEATMAP_CMAP: Final[str] = "rocket_r"

#: Default figure geometries (inches).  Wide variants are used where genre names
#: sit on the x axis and need horizontal room.
DEFAULT_FIGSIZE: Final[tuple[float, float]] = (7.5, 4.5)
WIDE_FIGSIZE: Final[tuple[float, float]] = (10.0, 5.0)
SQUARE_FIGSIZE: Final[tuple[float, float]] = (6.5, 6.0)
PANEL_SIZE: Final[tuple[float, float]] = (5.0, 4.0)
IMAGE_TILE_SIZE: Final[tuple[float, float]] = (3.0, 2.6)

#: Layout / annotation constants (kept here so no function contains a bare number).
BAR_GROUP_WIDTH: Final[float] = 0.82
BAR_LABEL_PADDING: Final[float] = 2.0
HEADROOM: Final[float] = 1.18  # multiplier applied to axis limits for labels
GRID_ALPHA: Final[float] = 0.35
REFERENCE_LINE_ALPHA: Final[float] = 0.8
TICK_ROTATION: Final[int] = 40
TITLE_WRAP: Final[int] = 34
LEGEND_WRAP: Final[int] = 28
ANNOTATION_FONTSIZE: Final[int] = 8
# : Extra title padding (points) reserved for a one-row legend above the axes.
LEGEND_TITLE_PAD: Final[float] = 24.0
#: Upper y limit for score axes: a little above 1.0 so a bar labelled "0.967"
#: keeps its annotation inside the panel.
SCORE_CEILING: Final[float] = 1.08
#: Iso-F1 contours drawn behind the precision/recall scatter.
ISO_F1_LEVELS: Final[tuple[float, ...]] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
#: Placeholder tile used when a screenshot cannot be decoded.
UNREADABLE_COLOR: Final[str] = "#C8C8C8"
UNREADABLE_TEXT: Final[str] = "unreadable"

_STYLE_APPLIED = False

#: Axis labels for the metric keys produced by :mod:`gamesense.evaluation`.
_METRIC_LABELS: Final[dict[str, str]] = {
    "loss": "BCE loss (lower is better)",
    "micro_f1": "Micro-F1 (higher is better)",
    "macro_f1": "Macro-F1 (higher is better)",
    "micro_precision": "Micro-precision",
    "micro_recall": "Micro-recall",
    "macro_precision": "Macro-precision",
    "macro_recall": "Macro-recall",
    "weighted_f1": "Weighted F1",
    "samples_f1": "Sample-averaged F1",
    "map": "mAP (mean average precision)",
    "micro_ap": "Micro-averaged average precision",
    "macro_roc_auc": "Macro ROC-AUC",
    "hamming_loss": "Hamming loss (lower is better)",
    "subset_accuracy": "Subset (exact-match) accuracy",
    "mean_jaccard": "Mean Jaccard overlap of true vs predicted genre set",
    "exact_rate": "Fraction of samples with an exactly correct genre set",
    "wrong_rate": "Fraction of samples with no correct genre",
    "empty_rate": "Fraction of samples with an empty prediction",
    "mean_fp": "Mean number of spurious genres per sample",
    "mean_fn": "Mean number of missed genres per sample",
    "label_cardinality_pred": "Predicted genres per sample (mean)",
    "f1": "F1 (higher is better)",
    "precision": "Precision",
    "recall": "Recall",
    "average_precision": "Average precision",
}

#: Compact versions of the labels above, for legends, titles and tick labels.
_SHORT_LABELS: Final[dict[str, str]] = {
    "mean_jaccard": "Mean Jaccard overlap",
    "exact_rate": "Exact-match rate",
    "wrong_rate": "No-correct-genre rate",
    "empty_rate": "Empty-prediction rate",
    "mean_fp": "Spurious genres per sample",
    "mean_fn": "Missed genres per sample",
    "map": "mAP",
    "micro_ap": "Micro-AP",
    "f1": "F1",
    "average_precision": "Average precision",
}


# : Font fallback chain.
_FONT_STACK: Final[list[str]] = [
    "DejaVu Sans",
    "Segoe UI",
    "Arial",
    "Helvetica",
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "Malgun Gothic",
    "Microsoft YaHei",
    "Noto Sans CJK JP",
    "sans-serif",
]


def apply_style(*, context: str = "notebook", font_scale: float = 1.0) -> None:
    """Install the project-wide matplotlib/seaborn style."""
    global _STYLE_APPLIED
    sns.set_theme(
        style="whitegrid", context=context, font_scale=font_scale, palette=list(PALETTE)
    )
    mpl.rcParams.update(
        {
            "figure.dpi": 100,  # on-screen; saving uses FIGURE_DPI
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "figure.titlesize": "large",
            "axes.titleweight": "bold",
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "image.cmap": HEATMAP_CMAP,
            "font.family": "sans-serif",
            # Real Steam titles include Japanese, Korean and Cyrillic text, so the default
            # Latin-only face would render those figure labels as empty boxes.
            "font.sans-serif": _FONT_STACK,
        }
    )
    _STYLE_APPLIED = True


def _ensure_style() -> None:
    """Apply :func:`apply_style` once per interpreter session."""
    if not _STYLE_APPLIED:
        apply_style()


def save_figure(fig: Figure, path: str | Path, *, dpi: int = FIGURE_DPI) -> Path:
    """Write *fig* to *path*, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    return target


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _new_grid(
    ax: Axes | Sequence[Axes] | None,
    *,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **subplot_kwargs: Any,
) -> tuple[Figure, list[Axes], bool]:
    """Return ``(figure, flat_axes, owns_figure)`` for a requested grid."""
    _ensure_style()
    needed = nrows * ncols
    if ax is None:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize or DEFAULT_FIGSIZE, **subplot_kwargs
        )
        flat = list(np.atleast_1d(np.asarray(axes, dtype=object)).ravel())
        return fig, flat, True
    flat = [ax] if isinstance(ax, Axes) else list(np.asarray(ax, dtype=object).ravel())
    if len(flat) < needed:
        raise ValueError(f"this figure needs {needed} axes but {len(flat)} were provided")
    figure = flat[0].get_figure()
    if figure is None:  # pragma: no cover - defensive
        raise ValueError("the provided axes are not attached to a figure")
    return figure, flat[:needed], False


def _finalize(
    fig: Figure,
    axes: Sequence[Axes],
    *,
    save_path: str | Path | None,
    owns_figure: bool,
) -> Figure | Axes | list[Axes]:
    """Tight-layout, optionally save, and return the figure or the axes."""
    if owns_figure:
        fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path)
    if owns_figure:
        return fig
    return axes[0] if len(axes) == 1 else list(axes)


def _as_series(
    values: Mapping[str, float] | pd.Series | Sequence[float],
    *,
    classes: Sequence[str] | None = None,
) -> pd.Series:
    """Coerce counts/weights given as dict, Series or array into a Series."""
    if isinstance(values, pd.Series):
        series = values.astype(float)
    elif isinstance(values, Mapping):
        series = pd.Series(dict(values), dtype=float)
    else:
        array = np.asarray(list(values), dtype=float).ravel()
        index = list(classes) if classes is not None else list(range(array.size))
        if len(index) != array.size:
            raise ValueError(f"expected {len(index)} values, got {array.size}")
        series = pd.Series(array, index=index, dtype=float)
    if classes is not None:
        missing = [c for c in classes if c not in series.index]
        if missing:
            LOGGER.warning("no value supplied for %s - drawn as zero", ", ".join(missing))
        series = series.reindex(list(classes)).fillna(0.0)
    return series


def _scalar(value: Any) -> float:
    """Unwrap a metric value that may be a plain number or ``{"mean": ...}``."""
    if isinstance(value, Mapping):
        value = value.get("mean", np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric_label(key: str) -> str:
    """Human readable axis label for a metric/column key."""
    raw = str(key)
    prefix = ""
    for candidate, text in (("train_", "Training "), ("val_", "Validation "), ("test_", "Test ")):
        if raw.startswith(candidate):
            prefix, raw = text, raw[len(candidate) :]
            break
    label = _METRIC_LABELS.get(raw.lower(), raw.replace("_", " ").capitalize())
    if prefix:
        return prefix + label[0].lower() + label[1:]
    return label


def _short_label(key: str) -> str:
    """Compact metric name for legends, titles and tick labels."""
    raw = str(key).lower()
    for prefix in ("train_", "val_", "test_"):
        raw = raw[len(prefix) :] if raw.startswith(prefix) else raw
    if raw in _SHORT_LABELS:
        return _SHORT_LABELS[raw]
    return _metric_label(raw).split(" (")[0]


def _legend_above(
    axis: Axes,
    *,
    title_text: str,
    handles: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
    ncols: int = 1,
) -> None:
    """Draw the legend on one row above *axis* and lift the title above it."""
    common = {
        "loc": "lower center",
        "bbox_to_anchor": (0.5, 1.0),
        "ncols": max(1, ncols),
        "frameon": False,
        "fontsize": ANNOTATION_FONTSIZE,
    }
    if handles is None:
        axis.legend(**common)
    else:
        axis.legend(handles, list(labels or []), **common)
    axis.set_title(title_text, pad=LEGEND_TITLE_PAD)


def _pretty(name: str) -> str:
    """Turn a column name into a readable axis label."""
    return str(name).replace("_", " ").strip().capitalize()


def _wrap(text: str, width: int = TITLE_WRAP) -> str:
    """Wrap *text* so long game names do not overflow their panel."""
    return textwrap.fill(str(text), width=width)


def _colors(n: int) -> list[str]:
    """Return *n* palette colours, cycling if more series than hues exist."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def _rotate_xticks(ax: Axes, labels: Sequence[str], *, rotation: int = TICK_ROTATION) -> None:
    """Set rotated x tick labels without triggering a tick/label mismatch."""
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(list(labels), rotation=rotation, ha="right" if rotation else "center")


def _empty_panel(ax: Axes, message: str) -> None:
    """Render an explicit "no data" panel instead of an empty or fake plot."""
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_axis_off()


def _grouped_positions(n_groups: int, n_series: int) -> tuple[np.ndarray, float]:
    """Return ``(group_centres, bar_width)`` for a grouped bar chart."""
    width = BAR_GROUP_WIDTH / max(1, n_series)
    return np.arange(n_groups, dtype=float), width


def _load_image(path: str | Path, root: Path | None = None) -> np.ndarray | None:
    """Load an RGB image as a ``(H, W, 3)`` uint8 array, or ``None`` on failure."""
    from PIL import Image, UnidentifiedImageError

    candidate = Path(path)
    if root is not None and not candidate.is_absolute():
        candidate = Path(root) / candidate
    try:
        with Image.open(candidate) as handle:
            return np.asarray(handle.convert("RGB"), dtype=np.uint8)
    except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError) as error:
        LOGGER.warning("unreadable image %s (%s)", candidate, type(error).__name__)
        return None


def _show_image(ax: Axes, image: np.ndarray | None, *, title: str = "") -> None:
    """Draw *image* on *ax*, or a grey "unreadable" placeholder when ``None``."""
    ax.set_axis_off()
    if image is None:
        ax.imshow(np.full((1, 1, 3), mpl.colors.to_rgb(UNREADABLE_COLOR)))
        ax.text(
            0.5,
            0.5,
            UNREADABLE_TEXT,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=ANNOTATION_FONTSIZE,
            color="black",
        )
    else:
        ax.imshow(image)
    if title:
        ax.set_title(title, fontsize=ANNOTATION_FONTSIZE + 1)


def _history_dict(history: Any) -> dict[str, Any]:
    """Normalise a history argument into the dict of ``TrainingHistory.as_dict``."""
    if isinstance(history, Mapping):
        return dict(history)
    if hasattr(history, "as_dict"):
        return dict(history.as_dict())
    if isinstance(history, pd.DataFrame):
        return {column: history[column].tolist() for column in history.columns}
    raise TypeError(
        "history must be a mapping, a TrainingHistory (with .as_dict()) or a DataFrame"
    )


def _history_series(history: Mapping[str, Any], metric: str) -> dict[str, list[float]]:
    """Return the available ``{"train"/"val": values}`` series for *metric*."""
    key = str(metric).lower()
    out: dict[str, list[float]] = {}
    for split in ("train", "val"):
        values = history.get(f"{split}_{key}")
        if values is not None and len(values):
            out[split] = [float(value) for value in values]
    return out


def _history_epochs(history: Mapping[str, Any], length: int) -> list[int]:
    """Epoch numbers for the x axis, falling back to ``1..length``."""
    epochs = history.get("epochs")
    if epochs is not None and len(epochs) >= length:
        return [int(value) for value in list(epochs)[:length]]
    return list(range(1, length + 1))


def _best_epoch(history: Mapping[str, Any]) -> int | None:
    """Epoch selected by early stopping, or the argmax/argmin of the monitor."""
    meta = history.get("meta") or {}
    early = meta.get("early_stopping") or {}
    if early.get("best_epoch"):
        return int(early["best_epoch"])
    monitor = str(meta.get("monitor", CONFIG.training.monitor_metric)).lower()
    mode = str(meta.get("monitor_mode", CONFIG.training.monitor_mode)).lower()
    values = history.get(f"val_{monitor}") or history.get("val_loss")
    if not values:
        return None
    if history.get(f"val_{monitor}") is None:
        mode = "min"
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).any():
        return None
    index = int(np.nanargmax(array) if mode == "max" else np.nanargmin(array))
    return _history_epochs(history, array.size)[index]


def _gradcam_fields(result: Any) -> tuple[np.ndarray | None, np.ndarray | None, str, float]:
    """Extract ``(base_image, overlay, genre, probability)`` from a result."""
    def field(name: str) -> Any:
        if isinstance(result, Mapping):
            return result.get(name)
        return getattr(result, name, None)

    base = field("base_image")
    overlay = field("overlay")
    genre = str(field("genre") or "")
    probability = field("probability")
    return (
        None if base is None else np.asarray(base),
        None if overlay is None else np.asarray(overlay),
        genre,
        float(probability) if probability is not None else float("nan"),
    )


# =========================================================================== #
# B. Dataset EDA
# =========================================================================== #
def plot_genre_frequency(
    counts: Mapping[str, int] | pd.Series,
    *,
    total: int | None = None,
    classes: Sequence[str] | None = None,
    sort: bool = True,
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Horizontal bar chart of how many games carry each genre label."""
    series = _as_series(counts, classes=classes)
    series = series.sort_values(ascending=True) if sort else series.iloc[::-1]
    fig, axes, owns = _new_grid(ax, figsize=figsize or DEFAULT_FIGSIZE)
    axis = axes[0]

    positions = np.arange(len(series))
    axis.barh(positions, series.to_numpy(), color=_colors(len(series)), edgecolor="white")
    axis.set_yticks(positions)
    axis.set_yticklabels([str(name) for name in series.index])
    for position, value in zip(positions, series.to_numpy()):
        text = f"{int(round(value)):,}"
        if total:
            text += f"  ({value / total:.1%})"
        axis.text(
            value + max(series.max(), 1.0) * 0.01,
            position,
            text,
            va="center",
            fontsize=ANNOTATION_FONTSIZE,
        )

    axis.set_xlim(0, max(series.max(), 1.0) * HEADROOM)
    axis.set_xlabel("Games carrying the genre (positive labels)")
    axis.set_ylabel("Genre")
    suffix = f" -- {total:,} games in total" if total else ""
    axis.set_title(title or f"Genre label frequency{suffix}")
    axis.grid(axis="y", visible=False)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_labels_per_game(
    n_labels: Sequence[int] | pd.Series,
    *,
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Bar chart of the number of genres assigned to a game (label cardinality)."""
    values = np.asarray(pd.Series(list(n_labels)).dropna(), dtype=int)
    fig, axes, owns = _new_grid(ax, figsize=figsize or DEFAULT_FIGSIZE)
    axis = axes[0]
    if values.size == 0:
        _empty_panel(axis, "no games supplied")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)

    counts = pd.Series(values).value_counts().sort_index()
    bars = axis.bar(
        counts.index.astype(int),
        counts.to_numpy(),
        color=PALETTE[0],
        edgecolor="white",
        label="Games",
    )
    axis.bar_label(
        bars,
        labels=[f"{int(c):,}\n{c / values.size:.1%}" for c in counts.to_numpy()],
        padding=BAR_LABEL_PADDING,
        fontsize=ANNOTATION_FONTSIZE,
    )
    mean = float(values.mean())
    axis.axvline(
        mean,
        color=PALETTE[1],
        linestyle="--",
        alpha=REFERENCE_LINE_ALPHA,
        label=f"mean = {mean:.2f} genres/game",
    )
    axis.set_xticks(counts.index.astype(int))
    axis.set_ylim(0, counts.max() * HEADROOM)
    axis.set_xlabel("Genres assigned to a game (label cardinality)")
    axis.set_ylabel("Number of games")
    axis.set_title(title or f"How many genres does a game have? ({values.size:,} games)")
    axis.legend(loc="upper right")
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_description_length(
    word_counts: Sequence[int] | pd.Series,
    *,
    bins: int = 40,
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Histogram of store-description length in words, with the median marked."""
    values = np.asarray(pd.Series(list(word_counts)).dropna(), dtype=float)
    fig, axes, owns = _new_grid(ax, figsize=figsize or DEFAULT_FIGSIZE)
    axis = axes[0]
    if values.size == 0:
        _empty_panel(axis, "no descriptions supplied")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)

    axis.hist(values, bins=bins, color=PALETTE[0], edgecolor="white", label="Games")
    median = float(np.median(values))
    axis.axvline(
        median,
        color=PALETTE[1],
        linestyle="--",
        alpha=REFERENCE_LINE_ALPHA,
        label=f"median = {median:.0f} words",
    )
    axis.annotate(
        f"median {median:.0f}\nmean {values.mean():.0f}\nrange {values.min():.0f}-{values.max():.0f}",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "grey"},
    )
    axis.set_xlabel("Description length (words after cleaning)")
    axis.set_ylabel("Number of games")
    axis.set_title(title or f"Description length distribution ({values.size:,} games)")
    axis.legend(loc="upper center")
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_genre_cooccurrence(
    label_matrix: np.ndarray,
    classes: Sequence[str] = GENRES,
    *,
    normalize: bool = True,
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Annotated heatmap of how often two genres are attached to the same game."""
    matrix = np.asarray(label_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(classes):
        raise ValueError(
            f"label_matrix must be (n_games, {len(classes)}), got {matrix.shape}"
        )
    counts = matrix.T @ matrix
    support = np.diag(counts).copy()
    if normalize:
        counts = np.divide(
            counts, support[:, None], out=np.zeros_like(counts), where=support[:, None] > 0
        )

    fig, axes, owns = _new_grid(ax, figsize=figsize or SQUARE_FIGSIZE)
    axis = axes[0]
    sns.heatmap(
        pd.DataFrame(counts, index=list(classes), columns=list(classes)),
        annot=True,
        fmt=".2f" if normalize else ".0f",
        cmap=HEATMAP_CMAP,
        square=True,
        linewidths=0.5,
        annot_kws={"fontsize": ANNOTATION_FONTSIZE},
        cbar_kws={
            "label": "P(column | row)" if normalize else "Games with both genres",
            "shrink": 0.8,
        },
        ax=axis,
    )
    axis.set_xlabel("Co-occurring genre")
    axis.set_ylabel("Conditioning genre")
    default = (
        "Genre co-occurrence: P(column genre | row genre)"
        if normalize
        else "Genre co-occurrence counts (diagonal = support)"
    )
    axis.set_title(title or default)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=TICK_ROTATION, ha="right")
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_class_imbalance(
    counts: Mapping[str, int] | pd.Series,
    *,
    total: int | None = None,
    pos_weights: Mapping[str, float] | pd.Series | Sequence[float] | None = None,
    classes: Sequence[str] | None = None,
    clip: float | None = CONFIG.training.pos_weight_clip,
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Class imbalance: per-genre prevalence with the derived ``pos_weight``."""
    series = _as_series(counts, classes=classes)
    order = list(series.index)
    heights = series.to_numpy() / total * 100.0 if total else series.to_numpy()
    fig, axes, owns = _new_grid(ax, figsize=figsize or WIDE_FIGSIZE)
    axis = axes[0]

    bars = axis.bar(
        np.arange(len(order)),
        heights,
        color=_colors(len(order)),
        edgecolor="white",
        label="Prevalence" if total else "Positive games",
    )
    axis.bar_label(
        bars,
        labels=[f"{h:.1f}%" if total else f"{int(round(h)):,}" for h in heights],
        padding=BAR_LABEL_PADDING,
        fontsize=ANNOTATION_FONTSIZE,
    )
    axis.set_ylim(0, max(heights.max(), 1.0) * HEADROOM)
    axis.set_ylabel("Prevalence (% of games)" if total else "Games carrying the genre")
    axis.set_xlabel("Genre")
    _rotate_xticks(axis, order)
    axis.set_title(title or "Class imbalance across the eight target genres")
    handles, labels = axis.get_legend_handles_labels()

    if pos_weights is not None:
        weights = _as_series(pos_weights, classes=order)
        twin = axis.twinx()
        twin.grid(visible=False)
        (line,) = twin.plot(
            np.arange(len(order)),
            weights.to_numpy(),
            color="black",
            marker="o",
            linestyle="-",
            label="pos_weight = negatives / positives",
        )
        twin.set_ylabel("pos_weight applied to the BCE positive term")
        twin.set_ylim(0, max(float(weights.max()), 1.0) * HEADROOM)
        handles.append(line)
        labels.append(line.get_label())
        if clip is not None and float(weights.max()) >= float(clip) * 0.999:
            clip_line = twin.axhline(
                float(clip), color=PALETTE[1], linestyle=":", alpha=REFERENCE_LINE_ALPHA
            )
            handles.append(clip_line)
            labels.append(f"pos_weight clip = {float(clip):g}")
    axis.legend(handles, labels, loc="upper right", fontsize=ANNOTATION_FONTSIZE + 1)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_split_distribution(
    split_summary: Mapping[str, Any],
    *,
    classes: Sequence[str] | None = None,
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Grouped bars of per-genre prevalence in train / val / test."""
    splits = dict(split_summary.get("splits") or {})
    order = list(classes or split_summary.get("classes") or GENRES)
    fig, axes, owns = _new_grid(ax, figsize=figsize or WIDE_FIGSIZE)
    axis = axes[0]
    if not splits:
        _empty_panel(axis, "split_summary contains no 'splits' entry")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)

    centres, width = _grouped_positions(len(order), len(splits))
    for index, (name, payload) in enumerate(splits.items()):
        prevalence = _as_series(payload.get("genre_prevalence") or {}, classes=order) * 100.0
        n_games = payload.get("n_games")
        label = f"{name}" + (f" (n={int(n_games):,} games)" if n_games else "")
        axis.bar(
            centres + (index - (len(splits) - 1) / 2) * width,
            prevalence.to_numpy(),
            width=width,
            color=PALETTE[index % len(PALETTE)],
            edgecolor="white",
            label=label,
        )
    _rotate_xticks(axis, order)
    axis.set_xlabel("Genre")
    axis.set_ylabel("Prevalence within the split (% of its games)")
    seed = split_summary.get("seed")
    suffix = f" (iterative stratification, seed {seed})" if seed is not None else ""
    axis.set_title(title or f"Genre prevalence is preserved across splits{suffix}")
    axis.legend(title="Split", ncols=len(splits), fontsize=ANNOTATION_FONTSIZE + 1)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_sample_images(
    image_paths: Sequence[str | Path],
    titles: Sequence[str],
    *,
    ncols: int = 4,
    root: Path | None = None,
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Grid of screenshots with wrapped captions."""
    paths = list(image_paths)
    captions = list(titles) + [""] * max(0, len(paths) - len(titles))
    if not paths:
        fig, axes, owns = _new_grid(ax, figsize=figsize or DEFAULT_FIGSIZE)
        _empty_panel(axes[0], "no screenshots supplied")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)

    ncols = max(1, min(ncols, len(paths)))
    nrows = int(np.ceil(len(paths) / ncols))
    default_size = (IMAGE_TILE_SIZE[0] * ncols, IMAGE_TILE_SIZE[1] * nrows)
    fig, axes, owns = _new_grid(ax, nrows=nrows, ncols=ncols, figsize=figsize or default_size)

    for index, axis in enumerate(axes):
        if index >= len(paths):
            axis.set_axis_off()
            continue
        _show_image(axis, _load_image(paths[index], root), title=_wrap(captions[index]))
    if owns and title:
        fig.suptitle(title)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


# =========================================================================== #
# C. Training dynamics
# =========================================================================== #
def plot_training_curves(
    history: Any,
    *,
    metrics: Sequence[str] = ("loss", "micro_f1"),
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Train/validation curves per epoch, one panel per metric."""
    payload = _history_dict(history)
    keys = list(metrics) or ["loss"]
    default_size = (PANEL_SIZE[0] * len(keys), PANEL_SIZE[1])
    fig, axes, owns = _new_grid(ax, ncols=len(keys), figsize=figsize or default_size)
    best = _best_epoch(payload)

    for axis, metric in zip(axes, keys):
        series = _history_series(payload, metric)
        if not series:
            _empty_panel(axis, f"history has no '{metric}' series")
            continue
        for index, (split, values) in enumerate(series.items()):
            axis.plot(
                _history_epochs(payload, len(values)),
                values,
                marker="o",
                markersize=3,
                color=PALETTE[index],
                label=f"{split} {metric}",
            )
        if best is not None:
            axis.axvline(
                best,
                color="black",
                linestyle="--",
                alpha=REFERENCE_LINE_ALPHA,
                label=f"best epoch ({best})",
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(_metric_label(metric))
        axis.set_title(f"{_metric_label(metric)} per epoch")
        axis.legend(fontsize=ANNOTATION_FONTSIZE + 1)

    meta = payload.get("meta") or {}
    kind = meta.get("model_kind")
    seed = meta.get("seed")
    if owns:
        suffix = ""
        if kind:
            suffix = f" -- {MODEL_DISPLAY_NAMES.get(str(kind), str(kind))}"
            if seed is not None:
                suffix += f", seed {seed}"
        fig.suptitle(title or f"Training dynamics{suffix}")
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_learning_rate(
    history: Any,
    *,
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Learning-rate schedule actually executed, read back from the history."""
    payload = _history_dict(history)
    values = [float(v) for v in (payload.get("learning_rate") or [])]
    fig, axes, owns = _new_grid(ax, figsize=figsize or DEFAULT_FIGSIZE)
    axis = axes[0]
    if not values:
        _empty_panel(axis, "history has no 'learning_rate' series")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)

    epochs = _history_epochs(payload, len(values))
    axis.plot(epochs, values, marker="o", markersize=3, color=PALETTE[0], label="Head parameter group")
    positive = [v for v in values if v > 0]
    if positive and max(positive) / min(positive) > 10:
        axis.set_yscale("log")
    peak = int(np.argmax(values))
    axis.annotate(
        f"peak {values[peak]:.2e} at epoch {epochs[peak]}",
        xy=(epochs[peak], values[peak]),
        xytext=(0.5, 0.92),
        textcoords="axes fraction",
        fontsize=ANNOTATION_FONTSIZE,
        arrowprops={"arrowstyle": "->", "color": "grey"},
    )
    scheduler = (payload.get("meta") or {}).get("scheduler")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Learning rate (optimiser step size)")
    axis.set_title(title or f"Learning-rate schedule{f' ({scheduler})' if scheduler else ''}")
    axis.legend()
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_history_comparison(
    histories: Mapping[str, Any],
    *,
    metric: str = "val_micro_f1",
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """One learning curve per model, so convergence speed can be compared."""
    fig, axes, owns = _new_grid(ax, figsize=figsize or DEFAULT_FIGSIZE)
    axis = axes[0]
    drawn = 0
    higher_is_better = "loss" not in metric.lower()
    for index, (name, history) in enumerate(histories.items()):
        payload = _history_dict(history)
        values = [float(v) for v in (payload.get(metric) or [])]
        if not values:
            LOGGER.warning("history for %r has no %r series - skipped", name, metric)
            continue
        epochs = _history_epochs(payload, len(values))
        colour = PALETTE[index % len(PALETTE)]
        axis.plot(
            epochs,
            values,
            marker="o",
            markersize=3,
            color=colour,
            label=MODEL_DISPLAY_NAMES.get(str(name), str(name)),
        )
        best = int(np.nanargmax(values) if higher_is_better else np.nanargmin(values))
        axis.plot(epochs[best], values[best], marker="*", markersize=13, color=colour)
        drawn += 1

    if not drawn:
        _empty_panel(axis, f"no history contained a {metric!r} series")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(_metric_label(metric))
    axis.set_title(title or f"{_metric_label(metric)} per epoch, all models (star = best epoch)")
    axis.legend(fontsize=ANNOTATION_FONTSIZE + 1)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


# =========================================================================== #
# D. Model comparison
# =========================================================================== #
def plot_metric_comparison(
    table: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("micro_f1", "macro_f1", "mAP"),
    model_column: str = "model",
    error_suffix: str = "_std",
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Headline grouped bar chart: one bar group per metric, one bar per model."""
    if table is None or len(table) == 0:
        fig, axes, owns = _new_grid(ax, figsize=figsize or DEFAULT_FIGSIZE)
        _empty_panel(axes[0], "no model has been evaluated yet")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)

    available = [m for m in metrics if m in table.columns]
    if not available:
        raise KeyError(f"none of {list(metrics)} are columns of the comparison table")
    names = (
        [str(v) for v in table[model_column]]
        if model_column in table.columns
        else [str(v) for v in table.index]
    )
    fig, axes, owns = _new_grid(ax, figsize=figsize or WIDE_FIGSIZE)
    axis = axes[0]
    centres, width = _grouped_positions(len(available), len(names))
    ceiling = 0.0

    for index, name in enumerate(names):
        row = table.iloc[index]
        heights = np.array([_scalar(row.get(m)) for m in available], dtype=float)
        errors = np.array([_scalar(row.get(f"{m}{error_suffix}")) for m in available])
        yerr = None if not np.isfinite(errors).any() else np.nan_to_num(errors)
        bars = axis.bar(
            centres + (index - (len(names) - 1) / 2) * width,
            np.nan_to_num(heights),
            width=width,
            yerr=yerr,
            capsize=3,
            color=PALETTE[index % len(PALETTE)],
            edgecolor="white",
            label=_wrap(MODEL_DISPLAY_NAMES.get(name, name), LEGEND_WRAP),
        )
        axis.bar_label(
            bars,
            labels=[f"{h:.3f}" if np.isfinite(h) else "n/a" for h in heights],
            padding=BAR_LABEL_PADDING + (4 if yerr is not None else 0),
            fontsize=ANNOTATION_FONTSIZE,
        )
        ceiling = max(ceiling, float(np.nanmax(np.nan_to_num(heights) + np.nan_to_num(errors))))

    _rotate_xticks(axis, [_metric_label(m).split(" (")[0] for m in available], rotation=0)
    axis.set_xlabel("Metric (test split, threshold selected on validation)")
    axis.set_ylabel("Score (0-1, higher is better)")
    axis.set_ylim(0, min(1.0, max(ceiling, 0.1) * HEADROOM))
    axis.set_title(title or "Model comparison on the held-out test split")
    # With three models and three metric groups an in-axes legend overlaps the
    # tallest bars, so it goes underneath the axes instead.
    axis.legend(
        fontsize=ANNOTATION_FONTSIZE + 1,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=min(3, len(names)),
        frameon=False,
    )
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_per_class_f1(
    per_class_by_model: Mapping[str, Mapping[str, Mapping[str, Any]]],
    classes: Sequence[str] = GENRES,
    *,
    metric: str = "f1",
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Per-genre F1 for several models side by side."""
    order = list(classes)
    fig, axes, owns = _new_grid(ax, figsize=figsize or WIDE_FIGSIZE)
    axis = axes[0]
    if not per_class_by_model:
        _empty_panel(axis, "no per-class metrics supplied")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)

    centres, width = _grouped_positions(len(order), len(per_class_by_model))
    support: dict[str, Any] = {}
    for index, (name, per_class) in enumerate(per_class_by_model.items()):
        values = [_scalar((per_class.get(genre) or {}).get(metric)) for genre in order]
        for genre in order:
            support.setdefault(genre, (per_class.get(genre) or {}).get("support"))
        axis.bar(
            centres + (index - (len(per_class_by_model) - 1) / 2) * width,
            np.nan_to_num(values),
            width=width,
            color=PALETTE[index % len(PALETTE)],
            edgecolor="white",
            label=_wrap(MODEL_DISPLAY_NAMES.get(str(name), str(name)), LEGEND_WRAP),
        )

    labels = [
        f"{genre}\n(n={int(support[genre]):,})" if support.get(genre) is not None else genre
        for genre in order
    ]
    _rotate_xticks(axis, labels, rotation=0)
    axis.set_xlabel("Genre (n = positives in the evaluated split)")
    axis.set_ylabel(f"Per-genre {metric.replace('_', ' ')} (0-1, higher is better)")
    axis.set_ylim(0, 1.0)
    axis.set_title(title or f"Per-genre {metric.replace('_', ' ')} by model")
    axis.legend(fontsize=ANNOTATION_FONTSIZE + 1)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_precision_recall_scatter(
    metrics_by_model: Mapping[str, Mapping[str, Any]],
    *,
    precision_key: str = "micro_precision",
    recall_key: str = "micro_recall",
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Precision vs recall per model on iso-F1 contours."""
    fig, axes, owns = _new_grid(ax, figsize=figsize or SQUARE_FIGSIZE)
    axis = axes[0]
    recall_grid = np.linspace(0.01, 1.0, 200)
    for level in ISO_F1_LEVELS:
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = level * recall_grid / (2 * recall_grid - level)
        valid = (2 * recall_grid > level) & (precision <= 1.0) & (precision > 0)
        axis.plot(recall_grid[valid], precision[valid], color="grey", linestyle=":", linewidth=0.8)
        if valid.any():
            axis.annotate(
                f"F1={level:g}",
                xy=(recall_grid[valid][-1], precision[valid][-1]),
                fontsize=ANNOTATION_FONTSIZE - 1,
                color="grey",
                va="bottom",
                ha="right",
            )

    drawn = 0
    for index, (name, metrics) in enumerate(metrics_by_model.items()):
        recall = _scalar(metrics.get(recall_key))
        precision = _scalar(metrics.get(precision_key))
        if not (np.isfinite(recall) and np.isfinite(precision)):
            LOGGER.warning("model %r has no %s/%s - skipped", name, precision_key, recall_key)
            continue
        label = MODEL_DISPLAY_NAMES.get(str(name), str(name))
        axis.scatter(
            recall, precision, s=110, color=PALETTE[index % len(PALETTE)], edgecolor="black",
            zorder=3, label=_wrap(label, LEGEND_WRAP),
        )
        axis.annotate(
            f"{label.split(' (')[0]}\nP={precision:.3f} R={recall:.3f}",
            xy=(recall, precision),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=ANNOTATION_FONTSIZE,
        )
        drawn += 1

    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel(f"Recall ({recall_key.split('_')[0]}-averaged)")
    axis.set_ylabel(f"Precision ({precision_key.split('_')[0]}-averaged)")
    axis.set_title(title or "Precision / recall trade-off with iso-F1 contours")
    if drawn:
        axis.legend(loc="lower left", fontsize=ANNOTATION_FONTSIZE + 1)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_threshold_analysis(
    sweep: pd.DataFrame | Sequence[Mapping[str, float]],
    *,
    best_threshold: float | None = None,
    metrics: Sequence[str] = ("micro_f1", "macro_f1", "micro_precision", "micro_recall"),
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Metric-vs-threshold curves with the selected threshold marked."""
    frame = sweep if isinstance(sweep, pd.DataFrame) else pd.DataFrame(list(sweep))
    fig, axes, owns = _new_grid(ax, figsize=figsize or DEFAULT_FIGSIZE)
    axis = axes[0]
    if len(frame) == 0 or "threshold" not in frame.columns:
        _empty_panel(axis, "threshold sweep is empty (no probabilities evaluated yet)")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)

    frame = frame.sort_values("threshold")
    thresholds = frame["threshold"].to_numpy(dtype=float)
    available = [m for m in metrics if m in frame.columns]
    for index, metric in enumerate(available):
        axis.plot(
            thresholds,
            frame[metric].to_numpy(dtype=float),
            marker="o",
            markersize=3,
            color=PALETTE[index % len(PALETTE)],
            label=_metric_label(metric).split(" (")[0],
        )
    if best_threshold is not None:
        axis.axvline(
            float(best_threshold),
            color="black",
            linestyle="--",
            alpha=REFERENCE_LINE_ALPHA,
            label=f"selected threshold = {float(best_threshold):.2f}",
        )
        if available:
            nearest = int(np.argmin(np.abs(thresholds - float(best_threshold))))
            axis.annotate(
                f"{available[0]} = {float(frame[available[0]].to_numpy()[nearest]):.3f}",
                xy=(thresholds[nearest], float(frame[available[0]].to_numpy()[nearest])),
                xytext=(8, -14),
                textcoords="offset points",
                fontsize=ANNOTATION_FONTSIZE,
            )
    axis.set_xlabel("Decision threshold applied to the sigmoid outputs")
    axis.set_ylabel("Metric value on the validation split")
    axis.set_title(title or "Threshold sensitivity (selected on validation, never on test)")
    if available or best_threshold is not None:
        axis.legend(fontsize=ANNOTATION_FONTSIZE + 1)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_pr_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str] = GENRES,
    *,
    include_micro: bool = True,
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Per-genre precision-recall curves with average precision in the legend."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    truth = np.asarray(labels)
    probs = np.asarray(probabilities, dtype=float)
    if truth.shape != probs.shape:
        raise ValueError(f"labels {truth.shape} and probabilities {probs.shape} differ")
    fig, axes, owns = _new_grid(ax, figsize=figsize or SQUARE_FIGSIZE)
    axis = axes[0]

    drawn = 0
    for index, genre in enumerate(classes):
        support = int(truth[:, index].sum())
        if support == 0:
            LOGGER.warning("genre %s has no positives in this split - curve skipped", genre)
            continue
        precision, recall, _ = precision_recall_curve(truth[:, index], probs[:, index])
        score = float(average_precision_score(truth[:, index], probs[:, index]))
        axis.step(
            recall, precision, where="post", color=PALETTE[index % len(PALETTE)],
            label=f"{genre} (AP={score:.3f}, n={support:,})",
        )
        drawn += 1
    if include_micro and truth.sum() > 0:
        precision, recall, _ = precision_recall_curve(truth.ravel(), probs.ravel())
        score = float(average_precision_score(truth.ravel(), probs.ravel()))
        axis.step(
            recall, precision, where="post", color="black", linestyle="--", linewidth=1.6,
            label=f"micro-average (AP={score:.3f})",
        )
        drawn += 1
    if not drawn:
        _empty_panel(axis, "no genre has positive labels in this split")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)

    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Recall (fraction of the genre's positives retrieved)")
    axis.set_ylabel("Precision (fraction of positive predictions that are right)")
    axis.set_title(title or "Per-genre precision-recall curves (threshold-free)")
    axis.legend(loc="upper right", fontsize=ANNOTATION_FONTSIZE - 1)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_confusion_heatmap(
    frame: pd.DataFrame,
    *,
    title: str = "Cross-genre confusion (diagonal = recall, off-diagonal = spurious rate)",
    value_format: str | None = None,
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Annotated heatmap of the confusion frame from the error analysis."""
    fig, axes, owns = _new_grid(ax, figsize=figsize or SQUARE_FIGSIZE)
    axis = axes[0]
    if frame is None or frame.empty:
        _empty_panel(axis, "confusion frame is empty")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)

    values = frame.to_numpy(dtype=float)
    normalized = float(np.nanmax(np.abs(values))) <= 1.0
    fmt = value_format or (".2f" if normalized else ".0f")
    sns.heatmap(
        frame,
        annot=True,
        fmt=fmt,
        cmap=HEATMAP_CMAP,
        square=True,
        linewidths=0.5,
        annot_kws={"fontsize": ANNOTATION_FONTSIZE},
        cbar_kws={
            "label": "Rate (0-1)" if normalized else "Samples",
            "shrink": 0.8,
        },
        ax=axis,
    )
    axis.set_xlabel("Predicted genre")
    axis.set_ylabel("True genre present in the label set")
    axis.set_title(_wrap(title, width=64))
    axis.set_xticklabels(axis.get_xticklabels(), rotation=TICK_ROTATION, ha="right")
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


# =========================================================================== #
# E. Explainability and qualitative results
# =========================================================================== #
def plot_gradcam(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    *,
    title: str = "",
    colormap: str = "jet",
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Three-panel Grad-CAM figure: screenshot, heat map, blended overlay."""
    default_size = (IMAGE_TILE_SIZE[0] * 3, IMAGE_TILE_SIZE[1] * 1.25)
    fig, axes, owns = _new_grid(ax, ncols=3, figsize=figsize or default_size)
    _show_image(axes[0], None if base_image is None else np.asarray(base_image), title="Screenshot")
    image = axes[1].imshow(np.asarray(heatmap), cmap=colormap, vmin=0.0, vmax=1.0)
    axes[1].set_axis_off()
    axes[1].set_title("Grad-CAM heat map", fontsize=ANNOTATION_FONTSIZE + 1)
    fig.colorbar(
        image, ax=axes[1], fraction=0.046, pad=0.04, label="Normalised class evidence"
    )
    _show_image(axes[2], None if overlay is None else np.asarray(overlay), title="Overlay")
    if owns:
        fig.suptitle(title or "Grad-CAM: pixels that supported the predicted genre")
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_gradcam_grid(
    results: Sequence[Any],
    *,
    ncols: int = 3,
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Grid of original/overlay pairs, *ncols* explanations per row."""
    items = list(results)
    if not items:
        fig, axes, owns = _new_grid(ax, figsize=figsize or DEFAULT_FIGSIZE)
        _empty_panel(axes[0], "no Grad-CAM explanations supplied (model not trained yet?)")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)

    ncols = max(1, min(ncols, len(items)))
    nrows = int(np.ceil(len(items) / ncols))
    default_size = (IMAGE_TILE_SIZE[0] * 2 * ncols, IMAGE_TILE_SIZE[1] * nrows)
    fig, axes, owns = _new_grid(
        ax, nrows=nrows, ncols=2 * ncols, figsize=figsize or default_size
    )
    for index in range(nrows * ncols):
        left, right = axes[2 * index], axes[2 * index + 1]
        if index >= len(items):
            left.set_axis_off()
            right.set_axis_off()
            continue
        base, overlay, genre, probability = _gradcam_fields(items[index])
        _show_image(left, base, title="original")
        caption = genre or "predicted genre"
        if np.isfinite(probability):
            caption += f" (p={probability:.2f})"
        _show_image(right, overlay, title=_wrap(caption, LEGEND_WRAP))
    if owns:
        fig.suptitle(title or "Grad-CAM explanations: original screenshot vs class evidence")
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def _probability_colors(
    row: pd.Series, classes: Sequence[str]
) -> tuple[list[str], list[float]]:
    """Bar colours + probabilities for one row of an error frame."""
    true_set = {g for g in str(row.get("true_genres", "") or "").split("|") if g}
    pred_set = {g for g in str(row.get("pred_genres", "") or "").split("|") if g}
    colours, values = [], []
    for genre in classes:
        values.append(_scalar(row.get(f"p_{genre}")))
        if genre in true_set and genre in pred_set:
            colours.append(OUTCOME_COLORS["correct"])
        elif genre in true_set:
            colours.append(OUTCOME_COLORS["missed"])
        elif genre in pred_set:
            colours.append(OUTCOME_COLORS["spurious"])
        else:
            colours.append(OUTCOME_COLORS["absent"])
    return colours, values


def _example_title(row: pd.Series) -> str:
    """Panel caption naming the game, its true genres and the predicted ones."""
    name = row.get("name") or row.get("sample_id") or ""
    outcome = row.get("outcome")
    header = _wrap(str(name), LEGEND_WRAP + 8)
    truth = str(row.get("true_genres", "") or "-").replace("|", ", ")
    predicted = str(row.get("pred_genres", "") or "-").replace("|", ", ")
    suffix = f"  [{outcome}]" if outcome else ""
    return (
        f"{header}\ntrue: {_wrap(truth, LEGEND_WRAP + 12)}\n"
        f"pred: {_wrap(predicted, LEGEND_WRAP + 12)}{suffix}"
    )


def plot_prediction_examples(
    rows: pd.DataFrame,
    *,
    root: Path | None = None,
    n: int = 6,
    classes: Sequence[str] = GENRES,
    threshold: float | None = None,
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Qualitative panel: screenshot next to the predicted genre probabilities."""
    frame = rows.head(int(n)) if rows is not None else pd.DataFrame()
    if len(frame) == 0:
        fig, axes, owns = _new_grid(ax, figsize=figsize or DEFAULT_FIGSIZE)
        _empty_panel(axes[0], "no prediction examples supplied (nothing evaluated yet?)")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)

    order = list(classes)
    default_size = (IMAGE_TILE_SIZE[0] * 3.0, IMAGE_TILE_SIZE[1] * 1.3 * len(frame))
    fig, axes, owns = _new_grid(
        ax, nrows=len(frame), ncols=2, figsize=figsize or default_size
    )
    for index, (_, row) in enumerate(frame.iterrows()):
        image_axis, bar_axis = axes[2 * index], axes[2 * index + 1]
        path = row.get("image_path")
        _show_image(
            image_axis,
            _load_image(path, root) if isinstance(path, (str, Path)) else None,
            title=_example_title(row),
        )
        colours, values = _probability_colors(row, order)
        positions = np.arange(len(order))
        bar_axis.barh(positions, values, color=colours, edgecolor="white")
        bar_axis.set_yticks(positions)
        bar_axis.set_yticklabels(order, fontsize=ANNOTATION_FONTSIZE)
        bar_axis.invert_yaxis()
        bar_axis.set_xlim(0, 1.0)
        bar_axis.grid(axis="y", visible=False)
        for position, value in zip(positions, values):
            if np.isfinite(value):
                bar_axis.text(
                    min(value + 0.02, 0.98), position, f"{value:.2f}",
                    va="center", fontsize=ANNOTATION_FONTSIZE - 1,
                )
        if threshold is not None:
            bar_axis.axvline(float(threshold), color="black", linestyle="--", linewidth=1.0)
        if index == len(frame) - 1:
            bar_axis.set_xlabel("Predicted probability (sigmoid output)")

    handles = [Patch(facecolor=colour, label=name) for name, colour in OUTCOME_COLORS.items()]
    if threshold is not None:
        handles.append(
            Line2D([], [], color="black", linestyle="--", label=f"threshold {float(threshold):.2f}")
        )
    if owns:
        fig.suptitle(title or "Qualitative predictions: screenshot and per-genre probabilities")
        fig.legend(
            handles=handles,
            loc="lower center",
            ncols=len(handles),
            fontsize=ANNOTATION_FONTSIZE + 1,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        if save_path is not None:
            save_figure(fig, save_path)
        return fig
    bar_axis.legend(handles=handles, fontsize=ANNOTATION_FONTSIZE)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)


def plot_error_rate_by_group(
    frame: pd.DataFrame,
    *,
    x: str,
    metric: str = "mean_jaccard",
    kind: str = "bar",
    count_column: str = "n",
    ax: Axes | Sequence[Axes] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | Axes | list[Axes]:
    """Prediction quality per bucket of an explanatory variable."""
    fig, axes, owns = _new_grid(ax, figsize=figsize or DEFAULT_FIGSIZE)
    axis = axes[0]
    if frame is None or len(frame) == 0:
        _empty_panel(axis, "grouped error frame is empty")
        return _finalize(fig, axes, save_path=save_path, owns_figure=owns)
    for column in (x, metric):
        if column not in frame.columns:
            raise KeyError(f"column {column!r} is not in the grouped frame")

    labels = [str(value) for value in frame[x]]
    values = frame[metric].to_numpy(dtype=float)
    counts = (
        frame[count_column].to_numpy() if count_column in frame.columns else np.full(len(frame), np.nan)
    )
    positions = np.arange(len(labels))
    if kind == "line":
        axis.plot(
            positions, values, marker="o", color=PALETTE[0], label=_metric_label(metric).split(" (")[0]
        )
    else:
        bars = axis.bar(
            positions, values, color=PALETTE[0], edgecolor="white",
            label=_metric_label(metric).split(" (")[0],
        )
        axis.bar_label(
            bars, labels=[f"{value:.3f}" for value in values],
            padding=BAR_LABEL_PADDING, fontsize=ANNOTATION_FONTSIZE,
        )
    for position, value, count in zip(positions, values, counts):
        if np.isfinite(count):
            axis.annotate(
                f"n={int(count):,}",
                xy=(position, 0),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=ANNOTATION_FONTSIZE - 1,
                color="white" if kind == "bar" else "black",
            )
    _rotate_xticks(axis, labels, rotation=TICK_ROTATION if max(map(len, labels)) > 6 else 0)
    axis.set_ylim(0, max(float(np.nanmax(values)), 0.1) * HEADROOM)
    axis.set_xlabel(_pretty(x) + " (bucket)")
    axis.set_ylabel(_metric_label(metric))
    axis.set_title(title or f"{_metric_label(metric).split(' (')[0]} by {_pretty(x).lower()}")
    axis.legend(fontsize=ANNOTATION_FONTSIZE + 1)
    return _finalize(fig, axes, save_path=save_path, owns_figure=owns)
