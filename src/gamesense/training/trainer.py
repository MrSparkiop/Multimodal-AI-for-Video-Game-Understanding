"""Reusable training loop shared by all three GameSense models."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import CONFIG, GENRES, GameSenseConfig
from ..evaluation.metrics import multilabel_metrics
from ..models.model_utils import GameSenseModel, parameter_groups, save_checkpoint
from ..utils import (
    describe_environment,
    get_logger,
    human_time,
    resolve_device,
    save_json,
    set_seed,
)
from .early_stopping import EarlyStopping
from .losses import build_criterion, describe_criterion

__all__ = ["INPUT_KEYS", "TrainingHistory", "Trainer", "build_optimizer", "build_scheduler"]

LOGGER = get_logger("gamesense.training.trainer")

#: Batch keys that are forwarded to the model.  Everything else (ids, raw text)
#: is bookkeeping.
INPUT_KEYS: tuple[str, ...] = (
    "image",
    "input_ids",
    "attention_mask",
    "image_features",
    "text_features",
)


# --------------------------------------------------------------------------- #
# Optimiser / scheduler
# --------------------------------------------------------------------------- #
def build_optimizer(
    model: nn.Module,
    *,
    config: GameSenseConfig = CONFIG,
    head_lr: float | None = None,
    backbone_lr: float | None = None,
    weight_decay: float | None = None,
) -> torch.optim.Optimizer:
    """Create the optimiser with separate head / backbone parameter groups."""
    cfg = config.training
    groups = parameter_groups(
        model,
        head_lr=cfg.head_lr if head_lr is None else head_lr,
        backbone_lr=cfg.backbone_lr if backbone_lr is None else backbone_lr,
        weight_decay=cfg.weight_decay if weight_decay is None else weight_decay,
    )
    if not groups:
        raise ValueError("model has no trainable parameters")
    name = cfg.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups)
    if name == "adam":
        return torch.optim.Adam(groups)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    config: GameSenseConfig = CONFIG,
    steps_per_epoch: int = 1,
    epochs: int | None = None,
) -> tuple[Any | None, str]:
    """Create the LR schedule.

    Returns ``(scheduler, cadence)`` where cadence is ``"step"``, ``"epoch"`` or ``"none"``.
    """
    cfg = config.training
    total_epochs = cfg.epochs if epochs is None else epochs
    kind = cfg.scheduler.lower()

    if kind == "none":
        return None, "none"
    if kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.monitor_mode,
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
        )
        return scheduler, "epoch"
    if kind == "cosine":
        total_steps = max(1, steps_per_epoch * total_epochs)
        warmup_steps = max(1, int(round(total_steps * cfg.warmup_ratio)))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"
    raise ValueError(f"unknown scheduler {cfg.scheduler!r}")


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
@dataclass
class TrainingHistory:
    """Per-epoch record of everything worth plotting or auditing."""

    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_micro_f1: list[float] = field(default_factory=list)
    val_micro_f1: list[float] = field(default_factory=list)
    train_macro_f1: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)
    val_map: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def append(
        self,
        *,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        learning_rate: float,
        seconds: float,
    ) -> None:
        self.epochs.append(int(epoch))
        self.train_loss.append(float(train_loss))
        self.val_loss.append(float(val_loss))
        self.train_micro_f1.append(float(train_metrics.get("micro_f1", float("nan"))))
        self.val_micro_f1.append(float(val_metrics.get("micro_f1", float("nan"))))
        self.train_macro_f1.append(float(train_metrics.get("macro_f1", float("nan"))))
        self.val_macro_f1.append(float(val_metrics.get("macro_f1", float("nan"))))
        self.val_map.append(float(val_metrics.get("mAP", float("nan"))))
        self.learning_rate.append(float(learning_rate))
        self.epoch_seconds.append(float(seconds))

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_micro_f1": self.train_micro_f1,
            "val_micro_f1": self.val_micro_f1,
            "train_macro_f1": self.train_macro_f1,
            "val_macro_f1": self.val_macro_f1,
            "val_map": self.val_map,
            "learning_rate": self.learning_rate,
            "epoch_seconds": self.epoch_seconds,
            "meta": self.meta,
        }

    def to_frame(self) -> Any:
        import pandas as pd

        payload = {k: v for k, v in self.as_dict().items() if isinstance(v, list)}
        return pd.DataFrame(payload)

    def save(self, path: str | Path) -> Path:
        return save_json(self.as_dict(), path)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """Train one GameSense model with validation, early stopping and logging."""

    def __init__(
        self,
        model: GameSenseModel,
        *,
        config: GameSenseConfig = CONFIG,
        criterion: nn.Module | None = None,
        train_labels: np.ndarray | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str | None = None,
        classes: Sequence[str] = GENRES,
        seed: int | None = None,
        class_weighting: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self.config = config
        self.classes = tuple(classes)
        self.seed = config.training.seed if seed is None else int(seed)
        self.device = resolve_device(device if device is not None else config.device)
        self.threshold = config.evaluation.default_threshold if threshold is None else threshold

        set_seed(self.seed)
        self.model = model.to(self.device)
        self.criterion = criterion or build_criterion(
            train_labels,
            strategy=class_weighting,
            training=config.training,
            device=self.device,
        )
        self.optimizer = optimizer or build_optimizer(self.model, config=config)
        self.history = TrainingHistory()
        self.early_stopping: EarlyStopping | None = None

    # -- internals --------------------------------------------------------- #
    def _forward_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = {
            key: batch[key].to(self.device, non_blocking=True)
            for key in INPUT_KEYS
            if key in batch and torch.is_tensor(batch[key])
        }
        if not inputs:
            raise ValueError(
                f"batch contains no model inputs; keys were {sorted(batch)}"
            )
        targets = batch["labels"].to(self.device, non_blocking=True).float()
        return self.model(**inputs), targets

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        train: bool,
        description: str = "",
        progress: bool = True,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Run one pass over *loader*, returning ``(mean_loss, probs, targets)``.

        Returns ``(mean_loss, probabilities, labels)``.
        """
        self.model.train(train)
        total_loss, n_batches = 0.0, 0
        probabilities: list[np.ndarray] = []
        truths: list[np.ndarray] = []

        iterator: Iterable[Any] = loader
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(loader, desc=description or ("train" if train else "eval"),
                                leave=False, unit="batch")
            except ImportError:  # pragma: no cover
                iterator = loader

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in iterator:
                logits, targets = self._forward_batch(batch)
                loss = self.criterion(logits, targets)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip = self.config.training.grad_clip_norm
                    if clip:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad], clip
                        )
                    self.optimizer.step()
                    if self._scheduler is not None and self._scheduler_cadence == "step":
                        self._scheduler.step()
                total_loss += float(loss.detach().cpu())
                n_batches += 1
                probabilities.append(torch.sigmoid(logits.detach()).cpu().numpy())
                truths.append(targets.detach().cpu().numpy())

        mean_loss = total_loss / max(1, n_batches)
        probs = np.concatenate(probabilities, axis=0) if probabilities else np.zeros((0, len(self.classes)))
        labels = np.concatenate(truths, axis=0) if truths else np.zeros((0, len(self.classes)))
        return mean_loss, probs, labels

    # -- public API -------------------------------------------------------- #
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        epochs: int | None = None,
        patience: int | None = None,
        monitor: str | None = None,
        progress: bool = True,
        model_kind: str | None = None,
        checkpoint_path: str | Path | None = None,
        history_path: str | Path | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> TrainingHistory:
        """Train the model, returning its :class:`TrainingHistory`."""
        cfg = self.config.training
        n_epochs = cfg.epochs if epochs is None else int(epochs)
        monitor_key = monitor or cfg.monitor_metric
        kind = model_kind or self.model.modality

        self._scheduler, self._scheduler_cadence = build_scheduler(
            self.optimizer,
            config=self.config,
            steps_per_epoch=max(1, len(train_loader)),
            epochs=n_epochs,
        )
        self.early_stopping = EarlyStopping(
            patience=cfg.early_stopping_patience if patience is None else patience,
            min_delta=cfg.early_stopping_min_delta,
            mode=cfg.monitor_mode,
            restore_best_weights=True,
        )

        self.history.meta = {
            "model_kind": kind,
            "model": self.model.describe(),
            "seed": self.seed,
            "device": str(self.device),
            "epochs_requested": n_epochs,
            "monitor": monitor_key,
            "monitor_mode": cfg.monitor_mode,
            "batch_size": train_loader.batch_size,
            "optimizer": type(self.optimizer).__name__,
            "learning_rates": [group.get("lr") for group in self.optimizer.param_groups],
            "weight_decay": cfg.weight_decay,
            "scheduler": cfg.scheduler,
            "grad_clip_norm": cfg.grad_clip_norm,
            "criterion": describe_criterion(self.criterion),
            "threshold_used_during_training": self.threshold,
            "n_train_batches": len(train_loader),
            "n_val_batches": len(val_loader),
            "environment": describe_environment(self.device),
            **(extra_metadata or {}),
        }

        LOGGER.info(
            "Training %s for up to %d epochs on %s (%s trainable parameters)",
            kind,
            n_epochs,
            self.device,
            f"{self.model.describe()['parameters']['trainable']:,}",
        )
        started = time.perf_counter()
        for epoch in range(1, n_epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_probs, train_truth = self._run_epoch(
                train_loader, train=True, description=f"epoch {epoch}/{n_epochs} [train]",
                progress=progress,
            )
            val_loss, val_probs, val_truth = self._run_epoch(
                val_loader, train=False, description=f"epoch {epoch}/{n_epochs} [val]",
                progress=progress,
            )
            train_metrics = multilabel_metrics(
                train_truth, train_probs, threshold=self.threshold,
                classes=self.classes, include_per_class=False,
            )
            val_metrics = multilabel_metrics(
                val_truth, val_probs, threshold=self.threshold,
                classes=self.classes, include_per_class=False,
            )
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            elapsed = time.perf_counter() - epoch_start
            self.history.append(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=current_lr,
                seconds=elapsed,
            )

            score = float(val_metrics.get(monitor_key, -float("inf")))
            improved = self.early_stopping.step(score, epoch, self.model)
            LOGGER.info(
                "epoch %02d/%d | train loss %.4f f1 %.4f | val loss %.4f f1 %.4f mAP %.4f | "
                "lr %.2e | %s%s",
                epoch, n_epochs, train_loss, train_metrics["micro_f1"],
                val_loss, val_metrics["micro_f1"], val_metrics.get("mAP", float("nan")),
                current_lr, human_time(elapsed), "  <- best" if improved else "",
            )

            if self._scheduler is not None and self._scheduler_cadence == "epoch":
                self._scheduler.step(score)
            if self.early_stopping.should_stop:
                break

        self.early_stopping.restore(self.model)
        total_seconds = time.perf_counter() - started
        self.history.meta.update(
            {
                "epochs_run": len(self.history.epochs),
                "total_seconds": round(total_seconds, 2),
                "early_stopping": self.early_stopping.state(),
            }
        )
        LOGGER.info(
            "Finished %s in %s (best %s = %.4f at epoch %d)",
            kind, human_time(total_seconds), monitor_key,
            self.early_stopping.best_score if self.early_stopping.best_score is not None else float("nan"),
            self.early_stopping.best_epoch,
        )

        if checkpoint_path is not None:
            save_checkpoint(
                self.model,
                checkpoint_path,
                metadata={
                    "history_meta": self.history.meta,
                    "best_epoch": self.early_stopping.best_epoch,
                    "best_val_score": self.early_stopping.best_score,
                    "monitor": monitor_key,
                },
            )
        if history_path is not None:
            self.history.save(history_path)
        return self.history

    @torch.no_grad()
    def predict(
        self, loader: DataLoader, *, progress: bool = True
    ) -> dict[str, Any]:
        """Run inference over *loader*.

        Returns ``{"probabilities", "labels", "sample_ids", "app_ids", "loss"}``.
        """
        self.model.eval()
        probabilities: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        sample_ids: list[str] = []
        app_ids: list[str] = []
        total_loss, n_batches = 0.0, 0

        iterator: Iterable[Any] = loader
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(loader, desc="predict", leave=False, unit="batch")
            except ImportError:  # pragma: no cover
                iterator = loader

        for batch in iterator:
            logits, targets = self._forward_batch(batch)
            total_loss += float(self.criterion(logits, targets).detach().cpu())
            n_batches += 1
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            truths.append(targets.cpu().numpy())
            sample_ids.extend(batch.get("sample_id", []))
            app_ids.extend(batch.get("app_id", []))

        return {
            "probabilities": np.concatenate(probabilities, axis=0)
            if probabilities
            else np.zeros((0, len(self.classes)), dtype=np.float32),
            "labels": np.concatenate(truths, axis=0)
            if truths
            else np.zeros((0, len(self.classes)), dtype=np.float32),
            "sample_ids": sample_ids,
            "app_ids": app_ids,
            "loss": total_loss / max(1, n_batches),
        }

    # Scheduler slots -- assigned in fit(), declared here for clarity.
    _scheduler: Any | None = None
    _scheduler_cadence: str = "none"
