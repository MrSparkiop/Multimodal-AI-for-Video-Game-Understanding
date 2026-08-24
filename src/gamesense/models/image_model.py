"""MODEL A -- image-only multi-label genre classifier (transfer learning)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ..config import CONFIG, GENRES, GameSenseConfig
from ..utils import get_logger
from .model_utils import GameSenseModel, MLPHead, freeze_module, unfreeze_module

__all__ = ["ImageEncoder", "ImageOnlyClassifier", "build_image_model"]

LOGGER = get_logger("gamesense.models.image")

#: Backbone name -> (torchvision constructor, weights enum attribute, feature dim)
_SUPPORTED_BACKBONES: dict[str, tuple[str, str, int]] = {
    "resnet18": ("resnet18", "ResNet18_Weights", 512),
    "resnet34": ("resnet34", "ResNet34_Weights", 512),
    "resnet50": ("resnet50", "ResNet50_Weights", 2048),
}


class ImageEncoder(nn.Module):
    """ResNet backbone truncated after global average pooling."""

    def __init__(
        self,
        *,
        config: GameSenseConfig = CONFIG,
        freeze: bool | None = None,
        unfreeze_stages: int | None = None,
        pretrained: bool | None = None,
    ) -> None:
        super().__init__()
        from torchvision import models as tv_models

        name = config.image.backbone
        if name not in _SUPPORTED_BACKBONES:
            raise ValueError(
                f"unsupported backbone {name!r}; expected one of {sorted(_SUPPORTED_BACKBONES)}"
            )
        constructor_name, weights_attr, feature_dim = _SUPPORTED_BACKBONES[name]
        use_pretrained = config.image.pretrained if pretrained is None else pretrained

        weights = None
        if use_pretrained:
            weights = getattr(tv_models, weights_attr).IMAGENET1K_V1
        backbone = getattr(tv_models, constructor_name)(weights=weights)

        self.backbone_name = name
        self.pretrained = bool(use_pretrained)
        self.embedding_dim = feature_dim
        # Keep the named stages so Grad-CAM and selective unfreezing can address
        # them directly instead of indexing into an anonymous Sequential.
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)

        do_freeze = config.model.freeze_image_backbone if freeze is None else freeze
        stages = config.model.unfreeze_image_stages if unfreeze_stages is None else unfreeze_stages
        if do_freeze:
            freeze_module(self)
        if stages:
            self.unfreeze_last_stages(stages)
        LOGGER.info(
            "ImageEncoder(%s, pretrained=%s, frozen=%s, unfrozen_stages=%d)",
            name,
            self.pretrained,
            do_freeze,
            stages or 0,
        )

    # -- API --------------------------------------------------------------- #
    def unfreeze_last_stages(self, n_stages: int) -> list[str]:
        """Re-enable gradients for the last *n_stages* residual stages.

        Counted from the end: 1 opens ``layer4``, 2 opens ``layer4`` and ``layer3``.
        """
        order = ["layer4", "layer3", "layer2", "layer1", "stem"]
        opened = order[: max(0, int(n_stages))]
        for name in opened:
            unfreeze_module(getattr(self, name))
        return opened

    def feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """Return the last convolutional feature map ``(batch, C, H, W)``."""
        hidden = self.stem(images)
        hidden = self.layer1(hidden)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)
        return self.layer4(hidden)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into ``(batch, embedding_dim)`` embeddings."""
        if images.dim() != 4:
            raise ValueError(f"expected a 4-D image batch, got shape {tuple(images.shape)}")
        maps = self.feature_map(images)
        return torch.flatten(self.pool(maps), 1)


class ImageOnlyClassifier(GameSenseModel):
    """MODEL A: ResNet18 visual embedding + multi-label MLP head."""

    modality = "image"
    pretrained_children = ("encoder",)
    _frozen_children = ("encoder",)

    def __init__(
        self,
        *,
        config: GameSenseConfig = CONFIG,
        num_classes: int = len(GENRES),
        freeze_backbone: bool | None = None,
        unfreeze_stages: int | None = None,
        dropout: float | None = None,
        hidden_dims: Sequence[int] | None = None,
        pretrained: bool | None = None,
        encoder: ImageEncoder | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.encoder = encoder or ImageEncoder(
            config=config,
            freeze=freeze_backbone,
            unfreeze_stages=unfreeze_stages,
            pretrained=pretrained,
        )
        self.head = MLPHead(
            self.encoder.embedding_dim,
            num_classes,
            hidden_dims=(
                hidden_dims
                if hidden_dims is not None
                else ((config.model.head_hidden_dim,) if config.model.head_hidden_dim else ())
            ),
            dropout=config.model.dropout if dropout is None else dropout,
        )
        self.embedding_dim = self.encoder.embedding_dim

    # -- forward paths ----------------------------------------------------- #
    def forward(
        self,
        image: torch.Tensor | None = None,
        *,
        image_features: torch.Tensor | None = None,
        **_ignored: Any,
    ) -> torch.Tensor:
        """Return ``(batch, num_classes)`` logits."""
        if image_features is None and image is None:
            raise ValueError("provide either 'image' or 'image_features'")
        if image_features is None:
            image_features = self.encode(image)  # type: ignore[arg-type]
        return self.head(image_features)

    def forward_from_features(
        self, *, image_features: torch.Tensor, **_ignored: Any
    ) -> torch.Tensor:
        return self.head(image_features)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Visual embedding, with gradients only if the backbone is unfrozen."""
        trainable = any(p.requires_grad for p in self.encoder.parameters())
        if trainable:
            return self.encoder(image)
        with torch.no_grad():
            return self.encoder(image)

    # -- explainability hooks ---------------------------------------------- #
    @property
    def gradcam_target_layer(self) -> nn.Module:
        """The layer Grad-CAM attaches to (last conv stage of the backbone)."""
        return self.encoder.layer4

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "backbone": self.encoder.backbone_name,
                "backbone_pretrained": self.encoder.pretrained,
                "embedding_dim": self.embedding_dim,
                "image_size": self.config.image.image_size,
            }
        )
        return info


def build_image_model(
    *,
    config: GameSenseConfig = CONFIG,
    num_classes: int = len(GENRES),
    **kwargs: Any,
) -> ImageOnlyClassifier:
    """Convenience constructor mirroring :func:`gamesense.models.build_model`."""
    return ImageOnlyClassifier(config=config, num_classes=num_classes, **kwargs)
