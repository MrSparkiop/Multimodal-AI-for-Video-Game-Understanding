"""Grad-CAM for the vision branch."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import nn

from ..config import CONFIG, GENRES, GameSenseConfig
from ..utils import get_logger, resolve_device

__all__ = ["GradCAM", "gradcam_heatmap", "overlay_heatmap", "denormalize_image"]

LOGGER = get_logger("gamesense.explainability.gradcam")


@contextmanager
def _temporarily_trainable(module: nn.Module) -> Iterator[None]:
    """Re-enable ``requires_grad`` inside the block, restore it afterwards."""
    previous = [(p, p.requires_grad) for p in module.parameters()]
    try:
        for parameter, _ in previous:
            parameter.requires_grad_(True)
        yield
    finally:
        for parameter, flag in previous:
            parameter.requires_grad_(flag)


class GradCAM:
    """Grad-CAM explainer bound to one model and one convolutional layer."""

    def __init__(
        self,
        model: nn.Module,
        *,
        target_layer: nn.Module | None = None,
        config: GameSenseConfig = CONFIG,
        classes: Sequence[str] = GENRES,
    ) -> None:
        layer = target_layer if target_layer is not None else getattr(model, "gradcam_target_layer", None)
        if layer is None:
            raise ValueError(
                "model does not expose 'gradcam_target_layer'; pass target_layer explicitly"
            )
        self.model = model
        self.target_layer = layer
        self.config = config
        self.classes = tuple(classes)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            layer.register_forward_hook(self._save_activations),
            layer.register_full_backward_hook(self._save_gradients),
        ]

    # -- hooks -------------------------------------------------------------- #
    def _save_activations(self, _module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        self._activations = output

    def _save_gradients(
        self, _module: nn.Module, _grad_input: Any, grad_output: tuple[torch.Tensor, ...]
    ) -> None:
        self._gradients = grad_output[0]

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        """Remove the forward/backward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- main API ----------------------------------------------------------- #
    def __call__(
        self,
        image: torch.Tensor,
        *,
        class_index: int | str | None = None,
        model_inputs: dict[str, Any] | None = None,
        normalize: bool = True,
    ) -> tuple[np.ndarray, int, float]:
        """Compute the heat map for one image."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[0] != 1:
            raise ValueError("Grad-CAM explains one image at a time")

        device = next(self.model.parameters()).device
        image = image.to(device).requires_grad_(True)
        extras = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in (model_inputs or {}).items()
        }

        was_training = self.model.training
        self.model.eval()
        self._activations = self._gradients = None

        with _temporarily_trainable(self.model), torch.enable_grad():
            logits = self.model(image=image, **extras)
            probabilities = torch.sigmoid(logits)
            index = self._resolve_index(class_index, probabilities)
            score = logits[0, index]
            self.model.zero_grad(set_to_none=True)
            score.backward()

        if was_training:
            self.model.train()

        if self._activations is None or self._gradients is None:  # pragma: no cover
            raise RuntimeError(
                "Grad-CAM captured no activations/gradients -- is the target layer part of "
                "the forward pass for this input?"
            )

        activations = self._activations.detach()[0]  # (C, h, w)
        gradients = self._gradients.detach()[0]  # (C, h, w)
        weights = gradients.mean(dim=(1, 2))  # alpha^c_k
        cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))

        cam = torch.nn.functional.interpolate(
            cam[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        heatmap = cam.cpu().numpy().astype(np.float32)
        if normalize:
            span = float(heatmap.max() - heatmap.min())
            heatmap = (heatmap - heatmap.min()) / span if span > 1e-12 else np.zeros_like(heatmap)
        return heatmap, int(index), float(probabilities[0, index].detach().cpu())

    def _resolve_index(self, class_index: int | str | None, probabilities: torch.Tensor) -> int:
        if class_index is None:
            return int(torch.argmax(probabilities[0]).item())
        if isinstance(class_index, str):
            if class_index not in self.classes:
                raise ValueError(f"unknown genre {class_index!r}; expected one of {self.classes}")
            return self.classes.index(class_index)
        index = int(class_index)
        if not 0 <= index < probabilities.shape[1]:
            raise IndexError(f"class index {index} out of range")
        return index


# --------------------------------------------------------------------------- #
# Convenience helpers
# --------------------------------------------------------------------------- #
def gradcam_heatmap(
    model: nn.Module,
    image: torch.Tensor,
    *,
    class_index: int | str | None = None,
    model_inputs: dict[str, Any] | None = None,
    config: GameSenseConfig = CONFIG,
) -> tuple[np.ndarray, int, float]:
    """One-shot Grad-CAM that registers and removes its hooks automatically.

    Returns ``(heatmap, class_index, probability)``; the heatmap is ``(H, W)`` in ``[0, 1]``.
    """
    with GradCAM(model, config=config) as explainer:
        return explainer(image, class_index=class_index, model_inputs=model_inputs)


def denormalize_image(
    tensor: torch.Tensor, *, config: GameSenseConfig = CONFIG
) -> np.ndarray:
    """Undo ImageNet normalisation and return an ``(H, W, 3)`` array in ``[0, 1]``."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    mean = torch.tensor(config.image.normalize_mean).view(3, 1, 1)
    std = torch.tensor(config.image.normalize_std).view(3, 1, 1)
    restored = (tensor.detach().cpu() * std + mean).clamp(0, 1)
    return restored.permute(1, 2, 0).numpy()


def overlay_heatmap(
    image: np.ndarray | torch.Tensor,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
    colormap: str = "jet",
    config: GameSenseConfig = CONFIG,
) -> np.ndarray:
    """Blend a heat map over an image and return an ``(H, W, 3)`` RGB array."""
    import matplotlib as mpl

    if torch.is_tensor(image):
        base = denormalize_image(image, config=config)
    else:
        base = np.asarray(image, dtype=np.float32)
        if base.max() > 1.5:
            base = base / 255.0
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=2)

    if heatmap.shape != base.shape[:2]:
        from PIL import Image

        resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (base.shape[1], base.shape[0]), Image.Resampling.BILINEAR
        )
        heatmap = np.asarray(resized, dtype=np.float32) / 255.0

    coloured = mpl.colormaps[colormap](np.clip(heatmap, 0.0, 1.0))[..., :3]
    blended = (1.0 - alpha) * base + alpha * coloured
    return np.clip(blended, 0.0, 1.0)
