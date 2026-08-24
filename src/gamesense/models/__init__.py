"""The three GameSense architectures plus shared building blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..config import CONFIG, GENRES, GameSenseConfig
from ..utils import get_logger, resolve_device
from .image_model import ImageEncoder, ImageOnlyClassifier, build_image_model
from .model_utils import (
    GameSenseModel,
    MLPHead,
    encode_split,
    freeze_module,
    l2_normalize,
    load_checkpoint,
    parameter_groups,
    save_checkpoint,
    unfreeze_module,
)
from .multimodal_model import MultimodalClassifier, build_multimodal_model
from .text_model import (
    BiLSTMTextClassifier,
    BiLSTMTextEncoder,
    TextEncoder,
    TextOnlyClassifier,
    build_text_model,
    masked_mean_pool,
)

__all__ = [
    "GameSenseModel",
    "MLPHead",
    "ImageEncoder",
    "ImageOnlyClassifier",
    "TextEncoder",
    "TextOnlyClassifier",
    "BiLSTMTextEncoder",
    "BiLSTMTextClassifier",
    "MultimodalClassifier",
    "build_image_model",
    "build_text_model",
    "build_multimodal_model",
    "build_model",
    "load_model_from_checkpoint",
    "masked_mean_pool",
    "freeze_module",
    "unfreeze_module",
    "l2_normalize",
    "parameter_groups",
    "save_checkpoint",
    "load_checkpoint",
    "encode_split",
]


def build_model(
    kind: str,
    *,
    config: GameSenseConfig = CONFIG,
    num_classes: int = len(GENRES),
    **kwargs: Any,
) -> GameSenseModel:
    """Factory returning one of the three systems (plus the BiLSTM baseline)."""
    key = kind.lower().strip()
    if key in ("image", "image_only", "vision"):
        return build_image_model(config=config, num_classes=num_classes, **kwargs)
    if key in ("text", "text_only", "distilbert"):
        return build_text_model(
            config=config, num_classes=num_classes, architecture="distilbert", **kwargs
        )
    if key in ("text_bilstm", "bilstm", "lstm"):
        return build_text_model(
            config=config, num_classes=num_classes, architecture="bilstm", **kwargs
        )
    if key in ("multimodal", "fusion", "both"):
        return build_multimodal_model(config=config, num_classes=num_classes, **kwargs)
    raise ValueError(
        f"unknown model kind {kind!r}; expected image | text | multimodal | text_bilstm"
    )


def load_model_from_checkpoint(
    path: str | Path,
    *,
    kind: str | None = None,
    config: GameSenseConfig = CONFIG,
    device: torch.device | str | None = None,
    strict_kwargs: bool = True,
    **overrides: Any,
) -> tuple[GameSenseModel, dict[str, Any]]:
    """Rebuild a trained model from a checkpoint written by :func:`save_checkpoint`.

    Returns ``(model, payload)``; encoders are rebuilt from published weights, then the trained
    head is loaded.
    """
    logger = get_logger("gamesense.models")
    payload = load_checkpoint(path, map_location="cpu")
    model_kind = kind or payload.get("modality") or "multimodal"
    metadata = payload.get("metadata", {}) or {}
    recorded = metadata.get("model_init_kwargs")
    if recorded is None:
        # Training writes the kwargs through Trainer.fit(extra_metadata=...), which
        # nests them one level deeper.  Accept both layouts.
        recorded = metadata.get("history_meta", {}).get("model_init_kwargs", {})
    init_kwargs: dict[str, Any] = dict(recorded or {})
    if not strict_kwargs:
        init_kwargs = {}
    init_kwargs.update(overrides)

    model = build_model(
        model_kind,
        config=config,
        num_classes=int(payload.get("num_classes", len(GENRES))),
        **init_kwargs,
    )
    report = model.load_exported_state_dict(payload["state_dict"])
    if report["missing"]:
        logger.warning("Checkpoint %s is missing %d tensors", Path(path).name, len(report["missing"]))
    model.to(resolve_device(device if device is not None else config.device)).eval()
    return model, payload
