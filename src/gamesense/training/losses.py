"""Loss functions for multi-label genre prediction."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from ..config import CONFIG, TrainingConfig
from ..data.preprocessing import class_weights
from ..utils import get_logger

__all__ = ["build_criterion", "pos_weight_from_labels", "describe_criterion"]

LOGGER = get_logger("gamesense.training.losses")


def pos_weight_from_labels(
    labels: np.ndarray | torch.Tensor, *, clip: float | None = None
) -> torch.Tensor:
    """Compute the ``pos_weight`` vector for :class:`~torch.nn.BCEWithLogitsLoss`."""
    array = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    return torch.from_numpy(class_weights(array, clip=clip))


def build_criterion(
    train_labels: np.ndarray | torch.Tensor | None = None,
    *,
    strategy: Literal["none", "pos_weight"] | None = None,
    training: TrainingConfig | None = None,
    device: torch.device | str = "cpu",
    reduction: str = "mean",
) -> nn.Module:
    """Build the multi-label loss."""
    cfg = training or CONFIG.training
    chosen = strategy or cfg.class_weighting
    if chosen not in ("none", "pos_weight"):
        raise ValueError(f"unknown class weighting strategy {chosen!r}")

    if chosen == "pos_weight":
        if train_labels is None:
            raise ValueError("pos_weight requires the training label matrix")
        weights = pos_weight_from_labels(train_labels, clip=cfg.pos_weight_clip).to(device)
        LOGGER.info(
            "BCEWithLogitsLoss with pos_weight (clipped at %.1f): %s",
            cfg.pos_weight_clip,
            np.round(weights.detach().cpu().numpy(), 2).tolist(),
        )
        return nn.BCEWithLogitsLoss(pos_weight=weights, reduction=reduction)

    LOGGER.info("BCEWithLogitsLoss without class weighting")
    return nn.BCEWithLogitsLoss(reduction=reduction)


def describe_criterion(criterion: nn.Module) -> dict[str, Any]:
    """JSON-serialisable description of the loss, stored with each experiment."""
    info: dict[str, Any] = {"class": type(criterion).__name__}
    weight = getattr(criterion, "pos_weight", None)
    if weight is not None:
        info["pos_weight"] = [round(float(value), 4) for value in weight.detach().cpu().flatten()]
    else:
        info["pos_weight"] = None
    info["reduction"] = getattr(criterion, "reduction", None)
    return info
