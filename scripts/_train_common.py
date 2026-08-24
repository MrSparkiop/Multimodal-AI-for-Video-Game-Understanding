"""Shared training driver used by the three ``train_*.py`` scripts."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Any

import numpy as np

from gamesense.config import CONFIG, GENRES, GameSenseConfig
from gamesense.data.loader import build_dataloaders, build_feature_dataloaders, load_bundle
from gamesense.data.preprocessing import TEXT_COLUMNS
from gamesense.evaluation.evaluator import evaluate_predictions, save_predictions
from gamesense.models import build_model
from gamesense.training.trainer import Trainer
from gamesense.utils import describe_environment, get_logger, human_time, save_json, set_seed

LOGGER = get_logger("gamesense.train")

TEXT_COLUMN_CHOICES = {
    "notitle": TEXT_COLUMNS["no_title"],
    "original": TEXT_COLUMNS["original"],
    "masked": TEXT_COLUMNS["masked"],
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the arguments every training script shares."""
    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=CONFIG.training.epochs)
    training.add_argument("--batch-size", type=int, default=CONFIG.training.batch_size)
    training.add_argument("--head-lr", type=float, default=CONFIG.training.head_lr,
                          help="learning rate for the newly initialised head")
    training.add_argument("--backbone-lr", type=float, default=CONFIG.training.backbone_lr,
                          help="learning rate for unfrozen pretrained layers")
    training.add_argument("--weight-decay", type=float, default=CONFIG.training.weight_decay)
    training.add_argument("--dropout", type=float, default=CONFIG.model.dropout)
    training.add_argument("--patience", type=int, default=CONFIG.training.early_stopping_patience)
    training.add_argument("--scheduler", choices=("cosine", "plateau", "none"),
                          default=CONFIG.training.scheduler)
    training.add_argument("--class-weighting", choices=("none", "pos_weight"),
                          default=CONFIG.training.class_weighting,
                          help="how to compensate for genre imbalance in the loss")
    training.add_argument("--monitor", default=CONFIG.training.monitor_metric,
                          help="validation metric driving early stopping and checkpointing")

    experiment = parser.add_argument_group("experiment")
    experiment.add_argument("--seed", type=int, default=CONFIG.training.seed)
    experiment.add_argument("--seeds", type=int, nargs="+", default=None,
                            help="train once per seed and report mean +- std")
    experiment.add_argument("--text-column", choices=sorted(TEXT_COLUMN_CHOICES),
                            default="notitle",
                            help="description variant: notitle (default, leak-safe), "
                                 "original (titles kept), masked (genre words masked too)")
    experiment.add_argument("--threshold-strategy", choices=("global", "per_class", "fixed"),
                            default=CONFIG.evaluation.threshold_strategy)
    experiment.add_argument("--tag", default=None,
                            help="suffix for artefact names (e.g. an ablation label)")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default=CONFIG.device, help="auto | cpu | cuda | mps")
    runtime.add_argument("--max-samples", type=int, default=None,
                         help="truncate every split (smoke test)")
    runtime.add_argument("--no-cache-features", action="store_true",
                         help="run the encoders every epoch instead of caching embeddings")
    runtime.add_argument("--force-features", action="store_true",
                         help="recompute the frozen-encoder feature cache")
    runtime.add_argument("--augment", action="store_true",
                         help="enable train-time image augmentation (forces end-to-end mode)")
    runtime.add_argument("--no-progress", action="store_true", help="disable progress bars")
    return parser


def config_from_args(args: argparse.Namespace) -> GameSenseConfig:
    """Apply CLI overrides onto the project configuration."""
    return replace(
        CONFIG,
        device=args.device,
        model=replace(CONFIG.model, dropout=args.dropout),
        training=replace(
            CONFIG.training,
            epochs=args.epochs,
            batch_size=args.batch_size,
            head_lr=args.head_lr,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.patience,
            scheduler=args.scheduler,
            class_weighting=args.class_weighting,
            monitor_metric=args.monitor,
            seed=args.seed,
            max_samples=args.max_samples,
        ),
        evaluation=replace(
            CONFIG.evaluation,
            threshold_strategy=(
                args.threshold_strategy if args.threshold_strategy != "fixed" else "global"
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def _artifact_kind(model_kind: str, tag: str | None) -> str:
    return f"{model_kind}_{tag}" if tag else model_kind


def run_single_seed(
    model_kind: str,
    args: argparse.Namespace,
    *,
    seed: int,
    config: GameSenseConfig,
    bundle: Any,
    model_kwargs: dict[str, Any],
    use_cache: bool,
) -> dict[str, Any]:
    """Train and evaluate one model with one seed."""
    modality = "text" if model_kind == "text_bilstm" else model_kind
    text_column = TEXT_COLUMN_CHOICES[args.text_column]
    artifact_kind = _artifact_kind(model_kind, args.tag)
    progress = not args.no_progress

    set_seed(seed)
    model = build_model(model_kind, config=config, num_classes=len(GENRES), **model_kwargs)
    LOGGER.info("Model: %s", model.describe())

    if use_cache:
        loaders, _ = build_feature_dataloaders(
            bundle,
            modality=modality,
            config=config,
            text_column=text_column,
            device=args.device,
            batch_size=args.batch_size,
            seed=seed,
            force=args.force_features,
            progress=progress,
        )
    else:
        loaders = build_dataloaders(
            bundle,
            modality=modality,
            text_column=text_column,
            batch_size=args.batch_size,
            config=config,
            seed=seed,
            augment_train=args.augment,
        )

    trainer = Trainer(
        model,
        config=config,
        train_labels=bundle.labels_for("train"),
        device=args.device,
        seed=seed,
        class_weighting=args.class_weighting,
    )
    history = trainer.fit(
        loaders["train"],
        loaders["val"],
        epochs=args.epochs,
        patience=args.patience,
        monitor=args.monitor,
        progress=progress,
        model_kind=artifact_kind,
        checkpoint_path=config.checkpoint_path(artifact_kind, seed),
        history_path=config.history_path(artifact_kind, seed),
        extra_metadata={
            "model_init_kwargs": model_kwargs,
            "text_column": text_column,
            "used_cached_features": use_cache,
            "augmentation": bool(args.augment),
            "tag": args.tag,
            "cli": vars(args),
        },
    )

    val = trainer.predict(loaders["val"], progress=progress)
    test = trainer.predict(loaders["test"], progress=progress)
    result = evaluate_predictions(
        model_kind=artifact_kind,
        seed=seed,
        val_labels=val["labels"],
        val_probabilities=val["probabilities"],
        test_labels=test["labels"],
        test_probabilities=test["probabilities"],
        classes=bundle.classes,
        config=config,
        threshold_strategy=args.threshold_strategy,
        extra={
            "text_column": text_column,
            "used_cached_features": use_cache,
            "model": model.describe(),
            "best_epoch": history.meta.get("early_stopping", {}).get("best_epoch"),
            "epochs_run": history.meta.get("epochs_run"),
            "total_train_seconds": history.meta.get("total_seconds"),
            "val_loss": val["loss"],
            "test_loss": test["loss"],
            "environment": describe_environment(),
        },
    )
    result.save(config.metrics_path(artifact_kind, seed))
    for split, payload in (("val", val), ("test", test)):
        save_predictions(
            config.predictions_path(artifact_kind, split, seed),
            probabilities=payload["probabilities"],
            labels=payload["labels"],
            sample_ids=payload["sample_ids"],
            app_ids=payload["app_ids"],
        )
    return {"seed": seed, "history": history, "result": result}


def run_training(
    model_kind: str,
    args: argparse.Namespace,
    *,
    model_kwargs: dict[str, Any] | None = None,
    force_end_to_end: bool = False,
) -> int:
    """Train ``model_kind`` for every requested seed and report the outcome."""
    config = config_from_args(args)
    config.paths.ensure()
    model_kwargs = dict(model_kwargs or {})
    seeds = list(args.seeds) if args.seeds else [args.seed]

    use_cache = (
        config.training.cache_features
        and not args.no_cache_features
        and not args.augment
        and not force_end_to_end
        and model_kind != "text_bilstm"
    )
    if force_end_to_end and not args.no_cache_features:
        LOGGER.info("Feature caching disabled automatically (partial fine-tuning requested)")

    bundle = load_bundle(config, max_samples=args.max_samples)
    LOGGER.info(
        "Data: %s | text column: %s | cached features: %s",
        bundle.summary(),
        TEXT_COLUMN_CHOICES[args.text_column],
        use_cache,
    )

    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        LOGGER.info("=" * 72)
        LOGGER.info("Training %s with seed %d (%d/%d)", model_kind, seed, seeds.index(seed) + 1, len(seeds))
        LOGGER.info("=" * 72)
        runs.append(
            run_single_seed(
                model_kind, args, seed=seed, config=config, bundle=bundle,
                model_kwargs=model_kwargs, use_cache=use_cache,
            )
        )

    # -- summary ----------------------------------------------------------- #
    artifact_kind = _artifact_kind(model_kind, args.tag)
    headline = [run["result"].test_metrics_selected for run in runs]
    LOGGER.info("=" * 72)
    LOGGER.info("%s: %d run(s) in %s", artifact_kind, len(runs), human_time(time.perf_counter() - started))
    for run in runs:
        metrics = run["result"].test_metrics_selected
        LOGGER.info(
            "  seed %-5d test micro-F1 %.4f | macro-F1 %.4f | mAP %.4f | threshold %s",
            run["seed"], metrics["micro_f1"], metrics["macro_f1"], metrics["mAP"],
            run["result"].threshold_selected,
        )
    if len(runs) > 1:
        for key in ("micro_f1", "macro_f1", "mAP"):
            values = np.array([m[key] for m in headline], dtype=float)
            LOGGER.info("  %-9s mean %.4f +- %.4f (n=%d)", key, values.mean(),
                        values.std(ddof=1), values.size)
        save_json(
            {
                "model_kind": artifact_kind,
                "seeds": seeds,
                "test_metrics_selected": headline,
                "mean_std": {
                    key: {
                        "mean": float(np.mean([m[key] for m in headline])),
                        "std": float(np.std([m[key] for m in headline], ddof=1)),
                    }
                    for key in ("micro_f1", "macro_f1", "mAP", "micro_precision", "micro_recall")
                },
            },
            config.paths.metrics / f"seed_summary_{artifact_kind}.json",
        )
    LOGGER.info("Artefacts: %s | %s | %s",
                config.checkpoint_path(artifact_kind, seeds[0]).name,
                config.metrics_path(artifact_kind, seeds[0]).name,
                config.history_path(artifact_kind, seeds[0]).name)
    LOGGER.info("=" * 72)
    return 0
