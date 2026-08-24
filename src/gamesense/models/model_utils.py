"""Shared model building blocks, freezing helpers, checkpoint I/O and encoders."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from ..config import CONFIG, GENRES, GameSenseConfig
from ..utils import count_parameters, get_logger, resolve_device

__all__ = [
    "MLPHead",
    "FrozenAwareModule",
    "GameSenseModel",
    "freeze_module",
    "unfreeze_module",
    "l2_normalize",
    "parameter_groups",
    "save_checkpoint",
    "load_checkpoint",
    "encode_split",
]

LOGGER = get_logger("gamesense.models")


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def freeze_module(module: nn.Module) -> nn.Module:
    """Disable gradients for every parameter of *module* and put it in eval mode."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    return module


def unfreeze_module(module: nn.Module) -> nn.Module:
    """Re-enable gradients for every parameter of *module*."""
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    return module


def l2_normalize(tensor: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise L2 normalisation."""
    return tensor / tensor.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def parameter_groups(
    model: nn.Module,
    *,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    pretrained_prefixes: Sequence[str] | None = None,
    head_keywords: Sequence[str] = ("head", "fusion", "classifier", "projection"),
) -> list[dict[str, Any]]:
    """Split trainable parameters into a head group and a pretrained group.

    Pretrained tensors get *backbone_lr*, everything else *head_lr*; biases and norms skip
    weight decay.
    """
    if pretrained_prefixes is None:
        pretrained_prefixes = getattr(model, "pretrained_children", None)

    def _is_pretrained(name: str) -> bool:
        if pretrained_prefixes is None:
            # Legacy heuristic: anything not obviously a head is a backbone.
            return not any(keyword in name for keyword in head_keywords)
        return any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in pretrained_prefixes
        )

    groups: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        pretrained = _is_pretrained(name)
        no_decay = name.endswith(".bias") or ".norm" in name.lower() or "layernorm" in name.lower()
        key = f"{'pretrained' if pretrained else 'head'}_{'nodecay' if no_decay else 'decay'}"
        group = groups.setdefault(
            key,
            {
                "params": [],
                "lr": backbone_lr if pretrained else head_lr,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": key,
            },
        )
        group["params"].append(parameter)
    return [group for group in groups.values() if group["params"]]


# --------------------------------------------------------------------------- #
# Head
# --------------------------------------------------------------------------- #
class MLPHead(nn.Module):
    """Multi-label classification head: ``Dropout -> [Linear-LN-ReLU-Dropout]* -> Linear``."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = len(GENRES),
        *,
        hidden_dims: Sequence[int] = (256,),
        dropout: float = 0.3,
        input_dropout: float | None = None,
        activation: Literal["relu", "gelu"] = "relu",
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        act_layer = nn.ReLU if activation == "relu" else nn.GELU
        layers: list[nn.Module] = []
        drop_in = dropout if input_dropout is None else input_dropout
        if drop_in > 0:
            layers.append(nn.Dropout(drop_in))
        current = in_features
        for width in [w for w in hidden_dims if w and w > 0]:
            layers.extend(
                [nn.Linear(current, width), nn.LayerNorm(width), act_layer(), nn.Dropout(dropout)]
            )
            current = width
        layers.append(nn.Linear(current, num_classes))
        self.net = nn.Sequential(*layers)
        self.in_features = in_features
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, in_features)`` embeddings to ``(batch, num_classes)`` logits."""
        return self.net(features)


# --------------------------------------------------------------------------- #
# Base classes
# --------------------------------------------------------------------------- #
class FrozenAwareModule(nn.Module):
    """Base class that keeps fully frozen sub-modules in ``eval`` mode."""

    _frozen_children: tuple[str, ...] = ()

    def train(self, mode: bool = True) -> "FrozenAwareModule":  # noqa: D102 - see nn.Module
        super().train(mode)
        for name in self._frozen_children:
            child = getattr(self, name, None)
            if isinstance(child, nn.Module) and not any(
                p.requires_grad for p in child.parameters()
            ):
                child.eval()
        return self


class GameSenseModel(FrozenAwareModule):
    """Common interface for the three systems compared in the study."""

    modality: str = "unknown"
    num_classes: int = len(GENRES)
    #: Names of child modules that hold pretrained weights.  Used to keep
    #: checkpoints small when the encoder was not fine-tuned.
    pretrained_children: tuple[str, ...] = ()

    # -- inference helpers ------------------------------------------------- #
    @torch.inference_mode()
    def predict_proba(self, **inputs: Any) -> torch.Tensor:
        """Return per-genre probabilities via an element-wise sigmoid."""
        was_training = self.training
        self.eval()
        logits = self(**inputs)
        if was_training:
            self.train()
        return torch.sigmoid(logits)

    def forward_from_features(self, **features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # -- checkpointing ----------------------------------------------------- #
    def frozen_pretrained_prefixes(self) -> tuple[str, ...]:
        """Prefixes of pretrained children that are *entirely* frozen.

        Those tensors are reproducible from published weights, so checkpoints omit them.
        """
        prefixes: list[str] = []
        for name in self.pretrained_children:
            child = getattr(self, name, None)
            if isinstance(child, nn.Module) and not any(
                p.requires_grad for p in child.parameters()
            ):
                prefixes.append(name + ".")
        return tuple(prefixes)

    def exportable_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict without the fully frozen pretrained encoder tensors."""
        drop = self.frozen_pretrained_prefixes()
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not any(key.startswith(prefix) for prefix in drop)
        }

    def load_exported_state_dict(self, state: dict[str, torch.Tensor]) -> dict[str, list[str]]:
        """Load an exported state dict, tolerating the omitted encoder tensors."""
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        drop = self.frozen_pretrained_prefixes()
        missing = [
            key
            for key in incompatible.missing_keys
            if not any(key.startswith(prefix) for prefix in drop)
        ]
        if unexpected or missing:
            LOGGER.warning(
                "Checkpoint mismatch - missing: %s | unexpected: %s", missing[:8], unexpected[:8]
            )
        return {"missing": missing, "unexpected": unexpected}

    # -- introspection ----------------------------------------------------- #
    def describe(self) -> dict[str, Any]:
        """Human-readable summary used in logs, the notebook and the app."""
        counts = count_parameters(self)
        return {
            "class": type(self).__name__,
            "modality": self.modality,
            "num_classes": self.num_classes,
            "parameters": counts,
            "trainable_fraction": round(counts["trainable"] / max(1, counts["total"]), 5),
            "frozen_pretrained": list(self.frozen_pretrained_prefixes()),
        }


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(
    model: GameSenseModel,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    full: bool = False,
) -> Path:
    """Persist a trained model.

    By default stores only what training could change, keeping checkpoints a few MB rather than
    ~300 MB.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_class": type(model).__name__,
        "modality": model.modality,
        "num_classes": model.num_classes,
        "classes": list(GENRES),
        "full_state": bool(full),
        "state_dict": model.state_dict() if full else model.exportable_state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, target)
    LOGGER.info("Saved checkpoint %s (%.2f MB)", target.name, target.stat().st_size / 1e6)
    return target


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a checkpoint payload written by :func:`save_checkpoint`."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"checkpoint not found: {target}")
    return torch.load(target, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------- #
# Frozen-encoder feature extraction
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def encode_split(
    bundle: Any,
    split: str,
    kind: Literal["image", "text"],
    *,
    config: GameSenseConfig = CONFIG,
    text_column: str | None = None,
    device: torch.device | str | None = None,
    batch_size: int | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Run a frozen encoder over one split and return ``(features, sample_ids)``.

    Returns ``(features, sample_ids)``. Augmentation is off and the encoder is in eval mode, so
    it is cacheable.
    """
    from ..data.dataset import build_image_transforms
    from ..data.loader import build_dataloader, build_dataset, get_tokenizer
    from ..data.preprocessing import TEXT_COLUMNS

    text_column = text_column or TEXT_COLUMNS["no_title"]
    device = resolve_device(device)
    batch_size = batch_size or config.training.eval_batch_size

    if kind == "image":
        from .image_model import ImageEncoder

        encoder: nn.Module = ImageEncoder(config=config, freeze=True).to(device).eval()
        tokenizer = None
    elif kind == "text":
        from .text_model import TextEncoder

        encoder = TextEncoder(config=config, freeze=True).to(device).eval()
        tokenizer = get_tokenizer(config.text.model_name)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown encoder kind {kind!r}")

    dataset = build_dataset(
        bundle, split, modality=kind, text_column=text_column, augment=False, config=config
    )
    if kind == "image":
        dataset.transform = build_image_transforms(train=False, image=config.image)
    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        modality=kind,
        tokenizer=tokenizer,
        num_workers=config.training.num_workers,
        config=config,
    )

    iterator: Iterable[Any] = loader
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc=f"encode {kind}/{split}", unit="batch")
        except ImportError:  # pragma: no cover
            iterator = loader

    chunks: list[np.ndarray] = []
    sample_ids: list[str] = []
    for batch in iterator:
        if kind == "image":
            embeddings = encoder(batch["image"].to(device))
        else:
            embeddings = encoder(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
        chunks.append(embeddings.detach().cpu().numpy().astype(np.float32))
        sample_ids.extend(batch["sample_id"])

    features = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.zeros((0, getattr(encoder, "embedding_dim", 0)), dtype=np.float32)
    )
    LOGGER.info("Encoded %s/%s -> %s", kind, split, features.shape)
    return features, sample_ids
