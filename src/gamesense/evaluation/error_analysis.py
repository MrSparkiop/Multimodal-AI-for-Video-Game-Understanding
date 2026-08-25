"""Qualitative and quantitative error analysis."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..config import CONFIG, GENRES, GameSenseConfig
from ..utils import get_logger
from .metrics import binarize

__all__ = [
    "Outcome",
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

LOGGER = get_logger("gamesense.evaluation.error_analysis")

Outcome = Literal["exact", "partial", "wrong", "empty_prediction"]


def classify_outcome(true_row: np.ndarray, pred_row: np.ndarray) -> Outcome:
    """Label one prediction as an exact / partial / wrong / empty outcome."""
    true_set = set(np.flatnonzero(np.asarray(true_row) > 0).tolist())
    pred_set = set(np.flatnonzero(np.asarray(pred_row) > 0).tolist())
    if not pred_set:
        return "empty_prediction"
    if pred_set == true_set:
        return "exact"
    if pred_set & true_set:
        return "partial"
    return "wrong"


def build_error_frame(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_ids: Sequence[str],
    samples: pd.DataFrame | None = None,
    games: pd.DataFrame | None = None,
    threshold: float | Sequence[float] | np.ndarray = 0.5,
    classes: Sequence[str] = GENRES,
    text_column: str = "description_notitle",
) -> pd.DataFrame:
    """Assemble a per-sample analysis frame."""
    truth = np.asarray(labels)
    probs = np.asarray(probabilities)
    if truth.shape != probs.shape:
        raise ValueError(f"labels {truth.shape} and probabilities {probs.shape} differ in shape")
    predictions = binarize(probs, threshold)
    class_list = list(classes)

    records: list[dict[str, Any]] = []
    for row in range(truth.shape[0]):
        true_names = [class_list[i] for i in np.flatnonzero(truth[row] > 0)]
        pred_names = [class_list[i] for i in np.flatnonzero(predictions[row] > 0)]
        true_set, pred_set = set(true_names), set(pred_names)
        union = true_set | pred_set
        record: dict[str, Any] = {
            "sample_id": str(sample_ids[row]),
            "n_true": len(true_names),
            "n_pred": len(pred_names),
            "true_genres": "|".join(true_names),
            "pred_genres": "|".join(pred_names),
            "tp": len(true_set & pred_set),
            "fp": len(pred_set - true_set),
            "fn": len(true_set - pred_set),
            "missed_genres": "|".join(sorted(true_set - pred_set)),
            "spurious_genres": "|".join(sorted(pred_set - true_set)),
            "jaccard": len(true_set & pred_set) / len(union) if union else 1.0,
            "outcome": classify_outcome(truth[row], predictions[row]),
            "max_prob": float(probs[row].max()),
        }
        for index, genre in enumerate(class_list):
            record[f"p_{genre}"] = float(probs[row, index])
        records.append(record)

    frame = pd.DataFrame(records)

    if samples is not None:
        columns = [c for c in ("sample_id", "app_id", "image_path", "shot_index") if c in samples.columns]
        frame = frame.merge(samples[columns].astype({"sample_id": str}), on="sample_id", how="left")
    if games is not None and "app_id" in frame.columns:
        wanted = ["app_id", "name"]
        if text_column in games.columns:
            wanted.append(text_column)
        if "description_word_count" in games.columns:
            wanted.append("description_word_count")
        game_view = games[wanted].copy()
        game_view["app_id"] = game_view["app_id"].astype(str)
        frame["app_id"] = frame["app_id"].astype(str)
        frame = frame.merge(game_view.drop_duplicates("app_id"), on="app_id", how="left")
        if text_column in frame.columns:
            frame = frame.rename(columns={text_column: "description"})
    if "description" in frame.columns and "description_word_count" not in frame.columns:
        frame["description_word_count"] = frame["description"].fillna("").str.split().str.len()
    return frame


def select_examples(
    frame: pd.DataFrame,
    *,
    n_per_outcome: int = 4,
    outcomes: Sequence[Outcome] = ("exact", "partial", "wrong", "empty_prediction"),
    seed: int = CONFIG.training.seed,
    prefer_confident: bool = True,
) -> pd.DataFrame:
    """Sample a balanced set of qualitative examples."""
    picked: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    for outcome in outcomes:
        subset = frame[frame["outcome"] == outcome]
        if subset.empty:
            continue
        if prefer_confident and len(subset) > n_per_outcome:
            subset = subset.sort_values("max_prob", ascending=False).head(
                max(n_per_outcome * 4, n_per_outcome)
            )
        take = min(n_per_outcome, len(subset))
        indices = rng.choice(len(subset), size=take, replace=False)
        picked.append(subset.iloc[np.sort(indices)])
    if not picked:
        return frame.head(0)
    return pd.concat(picked, ignore_index=True)


def per_class_error_summary(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | Sequence[float] | np.ndarray = 0.5,
    classes: Sequence[str] = GENRES,
) -> pd.DataFrame:
    """Per-genre TP/FP/FN/TN plus precision, recall, F1 and error rates."""
    truth = np.asarray(labels)
    predictions = binarize(probabilities, threshold)
    rows = []
    for index, genre in enumerate(classes):
        true_column = truth[:, index]
        pred_column = predictions[:, index]
        tp = int(((true_column == 1) & (pred_column == 1)).sum())
        fp = int(((true_column == 0) & (pred_column == 1)).sum())
        fn = int(((true_column == 1) & (pred_column == 0)).sum())
        tn = int(((true_column == 0) & (pred_column == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "genre": genre,
                "support": tp + fn,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
                "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)


def confusion_pairs(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | Sequence[float] | np.ndarray = 0.5,
    classes: Sequence[str] = GENRES,
    normalize: bool = True,
) -> pd.DataFrame:
    """Cross-genre confusion: predicted genre *j* while the truth was genre *i*."""
    truth = np.asarray(labels)
    predictions = binarize(probabilities, threshold)
    n = len(classes)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        rows = np.flatnonzero(truth[:, i] == 1)
        if rows.size == 0:
            continue
        for j in range(n):
            if i == j:
                matrix[i, j] = predictions[rows, j].mean()
            else:
                spurious = (predictions[rows, j] == 1) & (truth[rows, j] == 0)
                denominator = max(1, int((truth[rows, j] == 0).sum())) if normalize else 1
                matrix[i, j] = spurious.sum() / denominator if normalize else spurious.sum()
    return pd.DataFrame(matrix, index=list(classes), columns=list(classes))


def co_occurrence_confusion(
    frame: pd.DataFrame, *, top_k: int = 10
) -> pd.DataFrame:
    """Most frequent ``(missed genre, spurious genre)`` substitution pairs."""
    counter: dict[tuple[str, str], int] = {}
    for _, row in frame.iterrows():
        missed = [g for g in str(row.get("missed_genres", "")).split("|") if g]
        spurious = [g for g in str(row.get("spurious_genres", "")).split("|") if g]
        for m in missed:
            for s in spurious:
                counter[(m, s)] = counter.get((m, s), 0) + 1
    rows = [
        {"true_genre_missed": pair[0], "predicted_instead": pair[1], "count": count}
        for pair, count in sorted(counter.items(), key=lambda kv: -kv[1])[:top_k]
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Image-side correlates
# --------------------------------------------------------------------------- #
def image_statistics(path: str | Path) -> dict[str, float]:
    """Cheap perceptual statistics for one screenshot."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as handle:
            rgb = handle.convert("RGB")
            grey = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
            hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
    except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError):
        return {"brightness": float("nan"), "contrast": float("nan"), "colorfulness": float("nan")}
    return {
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
        "colorfulness": float(hsv[..., 1].mean()),
    }


def enrich_with_image_stats(
    frame: pd.DataFrame,
    *,
    config: GameSenseConfig = CONFIG,
    limit: int | None = None,
    path_column: str = "image_path",
) -> pd.DataFrame:
    """Add brightness / contrast / colourfulness columns to an error frame."""
    if path_column not in frame.columns:
        LOGGER.warning("No %s column - skipping image statistics", path_column)
        return frame
    out = frame.copy()
    subset = out.index if limit is None else out.index[:limit]
    stats = {
        index: image_statistics(config.paths.root / str(out.at[index, path_column]))
        for index in subset
    }
    for key in ("brightness", "contrast", "colorfulness"):
        out[key] = [stats.get(index, {}).get(key, float("nan")) for index in out.index]
    return out


def grouped_error_rates(
    frame: pd.DataFrame,
    *,
    by: str,
    bins: Sequence[float] | int | None = None,
    labels: Sequence[str] | None = None,
    metric: str = "jaccard",
) -> pd.DataFrame:
    """Average prediction quality within buckets of an explanatory variable."""
    if by not in frame.columns:
        raise KeyError(f"column {by!r} not present in the frame")
    working = frame.dropna(subset=[by]).copy()
    if bins is None:
        working["_bucket"] = working[by]
    else:
        working["_bucket"] = pd.cut(working[by], bins=bins, labels=labels, include_lowest=True)

    grouped = working.groupby("_bucket", observed=True).agg(
        n=("sample_id", "count"),
        mean_metric=(metric, "mean"),
        exact_rate=("outcome", lambda values: float((values == "exact").mean())),
        wrong_rate=("outcome", lambda values: float((values == "wrong").mean())),
        empty_rate=("outcome", lambda values: float((values == "empty_prediction").mean())),
        mean_fp=("fp", "mean"),
        mean_fn=("fn", "mean"),
    )
    grouped = grouped.rename(columns={"mean_metric": f"mean_{metric}"})
    return grouped.reset_index().rename(columns={"_bucket": by})


def compare_model_errors(
    frames: dict[str, pd.DataFrame], *, key: str = "sample_id"
) -> pd.DataFrame:
    """Join per-sample outcomes from several models for a side-by-side view."""
    merged: pd.DataFrame | None = None
    for name, frame in frames.items():
        columns = [key, "true_genres", "pred_genres", "outcome", "jaccard", "tp", "fp", "fn"]
        view = frame[[c for c in columns if c in frame.columns]].copy()
        view = view.rename(
            columns={
                column: f"{column}_{name}"
                for column in view.columns
                if column not in (key, "true_genres")
            }
        )
        merged = view if merged is None else merged.merge(view, on=[key, "true_genres"], how="outer")
    return merged if merged is not None else pd.DataFrame()
