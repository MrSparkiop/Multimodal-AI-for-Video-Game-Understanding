"""Training infrastructure: trainer, multi-label losses and early stopping."""

from __future__ import annotations

from .early_stopping import EarlyStopping
from .losses import build_criterion, describe_criterion, pos_weight_from_labels
from .trainer import (
    INPUT_KEYS,
    Trainer,
    TrainingHistory,
    build_optimizer,
    build_scheduler,
)

__all__ = [
    "Trainer",
    "TrainingHistory",
    "build_optimizer",
    "build_scheduler",
    "build_criterion",
    "describe_criterion",
    "pos_weight_from_labels",
    "EarlyStopping",
    "INPUT_KEYS",
]
