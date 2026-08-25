"""Unified evaluation: one code path for all three systems."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CONFIG, GENRES, MODEL_DISPLAY_NAMES, GameSenseConfig
from ..utils import get_logger, load_json, save_json
from .metrics import METRIC_KEYS, aggregate_seeds, compare_models, multilabel_metrics
from .thresholding import optimize_thresholds, threshold_sweep

__all__ = [
    "EvaluationResult",
    "evaluate_predictions",
    "evaluate_model",
    "save_predictions",
    "load_predictions",
    "collect_results",
    "build_comparison_table",
    "aggregate_over_seeds",
]

LOGGER = get_logger("gamesense.evaluation.evaluator")


@dataclass
class EvaluationResult:
    """Everything measured for one (model, seed) pair."""

    model_kind: str
    seed: int
    classes: tuple[str, ...]
    threshold_default: float
    threshold_selected: float | list[float]
    val_metrics_default: dict[str, Any] = field(default_factory=dict)
    val_metrics_selected: dict[str, Any] = field(default_factory=dict)
    test_metrics_default: dict[str, Any] = field(default_factory=dict)
    test_metrics_selected: dict[str, Any] = field(default_factory=dict)
    threshold_search: dict[str, Any] = field(default_factory=dict)
    threshold_sweep_val: list[dict[str, float]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def headline(self) -> dict[str, Any]:
        """Test metrics at the validation-selected threshold (the reported row)."""
        return self.test_metrics_selected

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "seed": self.seed,
            "classes": list(self.classes),
            "threshold_default": self.threshold_default,
            "threshold_selected": self.threshold_selected,
            "val_metrics_default": self.val_metrics_default,
            "val_metrics_selected": self.val_metrics_selected,
            "test_metrics_default": self.test_metrics_default,
            "test_metrics_selected": self.test_metrics_selected,
            "threshold_search": self.threshold_search,
            "threshold_sweep_val": self.threshold_sweep_val,
            "extra": self.extra,
        }

    def save(self, path: str | Path) -> Path:
        return save_json(self.as_dict(), path)


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def evaluate_predictions(
    *,
    model_kind: str,
    seed: int,
    val_labels: np.ndarray,
    val_probabilities: np.ndarray,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
    classes: Sequence[str] = GENRES,
    config: GameSenseConfig = CONFIG,
    threshold_strategy: str | None = None,
    extra: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Score one model from its raw probability outputs."""
    default_threshold = float(config.evaluation.default_threshold)
    selected, search = optimize_thresholds(
        val_labels,
        val_probabilities,
        strategy=threshold_strategy,
        evaluation=config.evaluation,
        classes=classes,
    )
    sweep = threshold_sweep(
        val_labels, val_probabilities, grid=config.evaluation.threshold_grid, classes=classes
    )

    result = EvaluationResult(
        model_kind=model_kind,
        seed=int(seed),
        classes=tuple(classes),
        threshold_default=default_threshold,
        threshold_selected=(
            float(selected) if np.isscalar(selected) else [float(t) for t in np.asarray(selected)]
        ),
        val_metrics_default=multilabel_metrics(
            val_labels, val_probabilities, threshold=default_threshold, classes=classes
        ),
        val_metrics_selected=multilabel_metrics(
            val_labels, val_probabilities, threshold=selected, classes=classes
        ),
        test_metrics_default=multilabel_metrics(
            test_labels, test_probabilities, threshold=default_threshold, classes=classes
        ),
        test_metrics_selected=multilabel_metrics(
            test_labels, test_probabilities, threshold=selected, classes=classes
        ),
        threshold_search=search,
        threshold_sweep_val=sweep.to_dict("records"),
        extra=extra or {},
    )
    LOGGER.info(
        "%s (seed %d): test micro-F1 %.4f / macro-F1 %.4f / mAP %.4f at threshold %s",
        model_kind,
        seed,
        result.test_metrics_selected["micro_f1"],
        result.test_metrics_selected["macro_f1"],
        result.test_metrics_selected["mAP"],
        result.threshold_selected if np.isscalar(selected) else "per-class",
    )
    return result


def save_predictions(
    path: str | Path,
    *,
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_ids: Sequence[str],
    app_ids: Sequence[str] | None = None,
) -> Path:
    """Persist raw probabilities so figures can be rebuilt without retraining."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "probabilities": np.asarray(probabilities, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int8),
        "sample_ids": np.asarray(list(sample_ids), dtype=object).astype("U"),
    }
    if app_ids is not None:
        payload["app_ids"] = np.asarray(list(app_ids), dtype=object).astype("U")
    np.savez_compressed(target, **payload)
    return target


def load_predictions(path: str | Path) -> dict[str, Any]:
    """Load an NPZ written by :func:`save_predictions`."""
    with np.load(Path(path), allow_pickle=False) as data:
        out = {
            "probabilities": data["probabilities"],
            "labels": data["labels"],
            "sample_ids": [str(value) for value in data["sample_ids"]],
        }
        if "app_ids" in data.files:
            out["app_ids"] = [str(value) for value in data["app_ids"]]
    return out


# --------------------------------------------------------------------------- #
# End-to-end evaluation from a checkpoint
# --------------------------------------------------------------------------- #
def evaluate_model(
    model_kind: str,
    *,
    seed: int | None = None,
    config: GameSenseConfig = CONFIG,
    bundle: Any | None = None,
    device: str | None = None,
    text_column: str | None = None,
    use_cached_features: bool | None = None,
    save: bool = True,
    progress: bool = True,
) -> EvaluationResult:
    """Load a trained checkpoint, run inference on val+test and score it."""
    from ..data.loader import (
        build_dataloaders,
        build_feature_dataloaders,
        load_bundle,
    )
    from ..data.preprocessing import TEXT_COLUMNS
    from ..models import load_model_from_checkpoint
    from ..training.trainer import Trainer

    seed = config.training.seed if seed is None else int(seed)
    text_column = text_column or TEXT_COLUMNS["no_title"]
    cached = config.training.cache_features if use_cached_features is None else use_cached_features
    bundle = bundle if bundle is not None else load_bundle(config)

    checkpoint = config.checkpoint_path(model_kind, seed)
    model, payload = load_model_from_checkpoint(
        checkpoint, kind=model_kind, config=config, device=device
    )
    modality = "text" if model_kind == "text_bilstm" else model_kind

    if cached and model_kind != "text_bilstm":
        loaders, _ = build_feature_dataloaders(
            bundle, modality=modality, config=config, text_column=text_column,
            device=device, seed=seed, progress=progress,
        )
    else:
        loaders = build_dataloaders(
            bundle, modality=modality, text_column=text_column, config=config,
            seed=seed, augment_train=False,
        )

    trainer = Trainer(
        model,
        config=config,
        train_labels=bundle.labels_for("train"),
        device=device,
        seed=seed,
    )
    val = trainer.predict(loaders["val"], progress=progress)
    test = trainer.predict(loaders["test"], progress=progress)

    result = evaluate_predictions(
        model_kind=model_kind,
        seed=seed,
        val_labels=val["labels"],
        val_probabilities=val["probabilities"],
        test_labels=test["labels"],
        test_probabilities=test["probabilities"],
        classes=bundle.classes,
        config=config,
        extra={
            "checkpoint": str(checkpoint),
            "text_column": text_column,
            "used_cached_features": bool(cached and model_kind != "text_bilstm"),
            "val_loss": val["loss"],
            "test_loss": test["loss"],
            "checkpoint_metadata": {
                key: payload.get("metadata", {}).get(key)
                for key in ("best_epoch", "best_val_score", "monitor")
            },
            "model": model.describe(),
        },
    )
    if save:
        result.save(config.metrics_path(model_kind, seed))
        save_predictions(
            config.predictions_path(model_kind, "val", seed),
            probabilities=val["probabilities"], labels=val["labels"],
            sample_ids=val["sample_ids"], app_ids=val["app_ids"],
        )
        save_predictions(
            config.predictions_path(model_kind, "test", seed),
            probabilities=test["probabilities"], labels=test["labels"],
            sample_ids=test["sample_ids"], app_ids=test["app_ids"],
        )
    return result


# --------------------------------------------------------------------------- #
# Aggregation across models and seeds
# --------------------------------------------------------------------------- #
def collect_results(
    *,
    model_kinds: Sequence[str] = ("image", "text", "multimodal"),
    seeds: Sequence[int] | None = None,
    config: GameSenseConfig = CONFIG,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Read every metrics JSON that exists on disk."""
    seeds = list(config.training.seeds if seeds is None else seeds)
    collected: dict[str, dict[int, dict[str, Any]]] = {}
    for kind in model_kinds:
        for seed in seeds:
            path = config.metrics_path(kind, seed)
            if not path.is_file():
                LOGGER.info("No metrics file for %s seed %d (%s) - skipping", kind, seed, path.name)
                continue
            collected.setdefault(kind, {})[seed] = load_json(path)
    return collected


def aggregate_over_seeds(
    collected: dict[str, dict[int, dict[str, Any]]],
    *,
    split: str = "test",
    threshold: str = "selected",
    keys: Sequence[str] = METRIC_KEYS,
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate ``mean +- std`` per model over the available seeds."""
    field_name = f"{split}_metrics_{threshold}"
    out: dict[str, dict[str, dict[str, float]]] = {}
    for kind, per_seed in collected.items():
        runs = [payload[field_name] for payload in per_seed.values() if field_name in payload]
        if runs:
            out[kind] = aggregate_seeds(runs, keys=keys)
    return out


def build_comparison_table(
    collected: dict[str, dict[int, dict[str, Any]]],
    *,
    split: str = "test",
    threshold: str = "selected",
    keys: Sequence[str] = METRIC_KEYS,
    aggregate: bool = True,
) -> "Any":
    """Build the headline model-comparison table from on-disk metrics."""
    field_name = f"{split}_metrics_{threshold}"
    if aggregate:
        aggregated = aggregate_over_seeds(collected, split=split, threshold=threshold, keys=keys)
        table = compare_models(aggregated, keys=keys, display_names=MODEL_DISPLAY_NAMES)
        table.insert(
            1,
            "n_seeds",
            [
                len(collected.get(kind, {}))
                for kind in aggregated
            ],
        )
        return table
    single = {
        kind: next(iter(per_seed.values()))[field_name]
        for kind, per_seed in collected.items()
        if per_seed and field_name in next(iter(per_seed.values()))
    }
    return compare_models(single, keys=keys, display_names=MODEL_DISPLAY_NAMES)
