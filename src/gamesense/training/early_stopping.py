"""Early stopping and best-checkpoint tracking."""

from __future__ import annotations

import copy
from typing import Any, Literal

import torch
from torch import nn

from ..config import CONFIG
from ..utils import get_logger

__all__ = ["EarlyStopping"]

LOGGER = get_logger("gamesense.training.early_stopping")


class EarlyStopping:
    """Track the best monitored metric and decide when to stop."""

    def __init__(
        self,
        *,
        patience: int = CONFIG.training.early_stopping_patience,
        min_delta: float = CONFIG.training.early_stopping_min_delta,
        mode: Literal["max", "min"] = CONFIG.training.monitor_mode,
        restore_best_weights: bool = True,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        if patience < 0:
            raise ValueError("patience must be non-negative")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score: float | None = None
        self.best_epoch: int = -1
        self.epochs_without_improvement: int = 0
        self.should_stop: bool = False
        self._best_state: dict[str, torch.Tensor] | None = None

    # -- API ---------------------------------------------------------------- #
    def is_improvement(self, score: float) -> bool:
        """Whether *score* beats the current best by more than ``min_delta``."""
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float, epoch: int, model: nn.Module | None = None) -> bool:
        """Record one epoch's validation score.

        Returns True when this epoch is the new best.
        """
        improved = self.is_improvement(float(score))
        if improved:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
            if self.restore_best_weights and model is not None:
                self._best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
        else:
            self.epochs_without_improvement += 1
            if self.patience and self.epochs_without_improvement >= self.patience:
                self.should_stop = True
                LOGGER.info(
                    "Early stopping: no improvement for %d epochs (best %.4f at epoch %d)",
                    self.epochs_without_improvement,
                    self.best_score if self.best_score is not None else float("nan"),
                    self.best_epoch,
                )
        return improved

    def best_state(self) -> dict[str, torch.Tensor] | None:
        """Return a copy of the best state dict, if one was captured."""
        return copy.deepcopy(self._best_state) if self._best_state is not None else None

    def restore(self, model: nn.Module) -> bool:
        """Load the best weights back into *model*.  Returns whether it happened."""
        if self._best_state is None:
            return False
        model.load_state_dict(self._best_state)
        LOGGER.info("Restored best weights from epoch %d", self.best_epoch)
        return True

    def state(self) -> dict[str, Any]:
        """JSON-serialisable summary stored with the training history."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped_early": self.should_stop,
        }
