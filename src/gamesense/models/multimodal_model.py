"""MODEL C -- the multimodal system (late fusion by concatenation)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ..config import CONFIG, GENRES, GameSenseConfig
from ..utils import get_logger
from .image_model import ImageEncoder
from .model_utils import GameSenseModel, MLPHead, l2_normalize
from .text_model import TextEncoder

__all__ = ["MultimodalClassifier", "build_multimodal_model"]

LOGGER = get_logger("gamesense.models.multimodal")


class MultimodalClassifier(GameSenseModel):
    """MODEL C: concatenate the visual and textual embeddings, then classify."""

    modality = "multimodal"
    pretrained_children = ("image_encoder", "text_encoder")
    _frozen_children = ("image_encoder", "text_encoder")

    def __init__(
        self,
        *,
        config: GameSenseConfig = CONFIG,
        num_classes: int = len(GENRES),
        freeze_image_backbone: bool | None = None,
        freeze_text_encoder: bool | None = None,
        unfreeze_image_stages: int | None = None,
        unfreeze_text_layers: int | None = None,
        dropout: float | None = None,
        fusion_hidden_dims: Sequence[int] | None = None,
        normalize_embeddings: bool | None = None,
        modality_dropout: float = 0.0,
        pretrained: bool = True,
        image_encoder: ImageEncoder | None = None,
        text_encoder: TextEncoder | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.image_encoder = image_encoder or ImageEncoder(
            config=config,
            freeze=freeze_image_backbone,
            unfreeze_stages=unfreeze_image_stages,
            pretrained=pretrained,
        )
        self.text_encoder = text_encoder or TextEncoder(
            config=config,
            freeze=freeze_text_encoder,
            unfreeze_layers=unfreeze_text_layers,
            pretrained=pretrained,
        )
        self.image_dim = self.image_encoder.embedding_dim
        self.text_dim = self.text_encoder.embedding_dim
        self.fused_dim = self.image_dim + self.text_dim
        self.normalize_embeddings = (
            config.model.normalize_embeddings_before_fusion
            if normalize_embeddings is None
            else normalize_embeddings
        )
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        self.modality_dropout = float(modality_dropout)

        self.fusion_head = MLPHead(
            self.fused_dim,
            num_classes,
            hidden_dims=(
                fusion_hidden_dims
                if fusion_hidden_dims is not None
                else config.model.fusion_hidden_dims
            ),
            dropout=config.model.dropout if dropout is None else dropout,
        )
        LOGGER.info(
            "MultimodalClassifier(image=%d + text=%d -> fused=%d, hidden=%s)",
            self.image_dim,
            self.text_dim,
            self.fused_dim,
            tuple(fusion_hidden_dims or config.model.fusion_hidden_dims),
        )

    # -- encoding ---------------------------------------------------------- #
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        trainable = any(p.requires_grad for p in self.image_encoder.parameters())
        if trainable:
            return self.image_encoder(image)
        with torch.no_grad():
            return self.image_encoder(image)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        trainable = any(p.requires_grad for p in self.text_encoder.parameters())
        if trainable:
            return self.text_encoder(input_ids, attention_mask)
        with torch.no_grad():
            return self.text_encoder(input_ids, attention_mask)

    def fuse(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate (optionally L2-normalised) embeddings into one vector."""
        if image_features.shape[0] != text_features.shape[0]:
            raise ValueError(
                "image and text batches must correspond to the same games: got "
                f"{image_features.shape[0]} vs {text_features.shape[0]} rows"
            )
        if self.normalize_embeddings:
            image_features = l2_normalize(image_features)
            text_features = l2_normalize(text_features)
        if self.training and self.modality_dropout > 0:
            image_features, text_features = self._apply_modality_dropout(
                image_features, text_features
            )
        return torch.cat([image_features, text_features], dim=1)

    def _apply_modality_dropout(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly zero one branch per sample (never both)."""
        batch = image_features.shape[0]
        device = image_features.device
        drop_image = torch.rand(batch, 1, device=device) < self.modality_dropout
        drop_text = (torch.rand(batch, 1, device=device) < self.modality_dropout) & (~drop_image)
        return image_features * (~drop_image), text_features * (~drop_text)

    # -- forward paths ----------------------------------------------------- #
    def forward(
        self,
        image: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        image_features: torch.Tensor | None = None,
        text_features: torch.Tensor | None = None,
        **_ignored: Any,
    ) -> torch.Tensor:
        """Return ``(batch, num_classes)`` logits."""
        if image_features is None:
            if image is None:
                raise ValueError("provide either 'image' or 'image_features'")
            image_features = self.encode_image(image)
        if text_features is None:
            if input_ids is None or attention_mask is None:
                raise ValueError(
                    "provide either 'input_ids' + 'attention_mask' or 'text_features'"
                )
            text_features = self.encode_text(input_ids, attention_mask)
        return self.fusion_head(self.fuse(image_features, text_features))

    def forward_from_features(
        self, *, image_features: torch.Tensor, text_features: torch.Tensor, **_ignored: Any
    ) -> torch.Tensor:
        return self.fusion_head(self.fuse(image_features, text_features))

    # -- explainability ---------------------------------------------------- #
    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Grad-CAM attaches to the vision branch's last conv stage."""
        return self.image_encoder.layer4

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "backbone": self.image_encoder.backbone_name,
                "text_model": self.text_encoder.model_name,
                "image_dim": self.image_dim,
                "text_dim": self.text_dim,
                "fused_dim": self.fused_dim,
                "fusion": "concatenation + MLP",
                "normalize_embeddings_before_fusion": self.normalize_embeddings,
                "modality_dropout": self.modality_dropout,
            }
        )
        return info


def build_multimodal_model(
    *,
    config: GameSenseConfig = CONFIG,
    num_classes: int = len(GENRES),
    **kwargs: Any,
) -> MultimodalClassifier:
    """Convenience constructor mirroring :func:`gamesense.models.build_model`."""
    return MultimodalClassifier(config=config, num_classes=num_classes, **kwargs)
