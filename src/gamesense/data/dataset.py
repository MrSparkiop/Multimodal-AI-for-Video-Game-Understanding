"""PyTorch datasets, image transforms and batch collation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import CONFIG, GENRES, GameSenseConfig, ImageConfig
from ..utils import get_logger
from .preprocessing import TEXT_COLUMNS, label_matrix

__all__ = [
    "Modality",
    "build_image_transforms",
    "GameSenseDataset",
    "FeatureDataset",
    "MultiLabelCollator",
    "FeatureCollator",
]

LOGGER = get_logger("gamesense.data.dataset")

Modality = Literal["image", "text", "multimodal"]


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_image_transforms(
    *, train: bool, image: ImageConfig | None = None
) -> "torch.nn.Module":
    """Build the torchvision transform pipeline for one split."""
    from torchvision import transforms

    cfg = image or CONFIG.image
    size = cfg.image_size
    normalize = transforms.Normalize(mean=list(cfg.normalize_mean), std=list(cfg.normalize_std))

    if not train:
        return transforms.Compose(
            [
                transforms.Resize(int(size * 1.14)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    stages: list[Any] = []
    if cfg.train_random_crop:
        stages.append(transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.25)))
    else:
        stages.extend([transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size)])
    if cfg.train_horizontal_flip_p > 0:
        stages.append(transforms.RandomHorizontalFlip(p=cfg.train_horizontal_flip_p))
    if cfg.train_color_jitter > 0:
        jitter = cfg.train_color_jitter
        stages.append(
            transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter)
        )
    stages.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(stages)


# --------------------------------------------------------------------------- #
# Main dataset
# --------------------------------------------------------------------------- #
class GameSenseDataset(Dataset):
    """Multi-label dataset over ``(screenshot, description)`` pairs."""

    def __init__(
        self,
        samples: pd.DataFrame,
        games: pd.DataFrame,
        *,
        modality: Modality = "multimodal",
        train: bool = False,
        text_column: str = TEXT_COLUMNS["no_title"],
        classes: Sequence[str] = GENRES,
        config: GameSenseConfig = CONFIG,
        transform: Any | None = None,
        on_image_error: Literal["zeros", "raise"] = "zeros",
    ) -> None:
        if modality not in ("image", "text", "multimodal"):
            raise ValueError(f"unknown modality {modality!r}")
        required = {"sample_id", "app_id"}
        missing = required - set(samples.columns)
        if missing:
            raise KeyError(f"samples frame is missing columns: {sorted(missing)}")
        if modality in ("image", "multimodal") and "image_path" not in samples.columns:
            raise KeyError("samples frame must contain 'image_path' for image modalities")

        self.modality: Modality = modality
        self.train = train
        self.classes = tuple(classes)
        self.config = config
        self.text_column = text_column
        self.on_image_error = on_image_error
        self.n_image_errors = 0
        self._logged_errors = 0

        games_indexed = games.copy()
        games_indexed["app_id"] = games_indexed["app_id"].astype(str)
        if text_column not in games_indexed.columns:
            raise KeyError(
                f"games frame has no text column {text_column!r}; "
                f"available: {sorted(set(TEXT_COLUMNS.values()) & set(games_indexed.columns))}"
            )
        games_indexed = games_indexed.drop_duplicates(subset=["app_id"]).set_index("app_id")

        self.samples = samples.reset_index(drop=True).copy()
        self.samples["app_id"] = self.samples["app_id"].astype(str)
        unknown = set(self.samples["app_id"]) - set(games_indexed.index)
        if unknown:
            raise KeyError(f"{len(unknown)} sample app_ids are absent from the games frame")

        aligned = games_indexed.loc[self.samples["app_id"]]
        self.texts: list[str] = [str(value) for value in aligned[text_column].to_numpy()]
        self.names: list[str] = [str(value) for value in aligned["name"].to_numpy()]
        self.labels = torch.from_numpy(label_matrix(aligned, classes=self.classes))
        self.image_paths: list[Path | None] = (
            [config.paths.root / str(p) for p in self.samples["image_path"]]
            if "image_path" in self.samples.columns
            else [None] * len(self.samples)
        )
        self.transform = transform if transform is not None else build_image_transforms(
            train=train, image=config.image
        )

    # -- protocol --------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "index": index,
            "sample_id": str(self.samples.at[index, "sample_id"]),
            "app_id": str(self.samples.at[index, "app_id"]),
            "labels": self.labels[index],
        }
        if self.modality in ("text", "multimodal"):
            item["text"] = self.texts[index]
        if self.modality in ("image", "multimodal"):
            image, ok = self._load_image(index)
            item["image"] = image
            item["image_ok"] = ok
        return item

    # -- helpers ---------------------------------------------------------- #
    def _load_image(self, index: int) -> tuple[torch.Tensor, bool]:
        from PIL import Image, UnidentifiedImageError

        path = self.image_paths[index]
        try:
            if path is None:
                raise FileNotFoundError("no image_path for this sample")
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            return self.transform(image), True
        except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError) as exc:
            if self.on_image_error == "raise":
                raise
            self.n_image_errors += 1
            if self._logged_errors < 5:
                self._logged_errors += 1
                LOGGER.warning("Unreadable image %s (%s) - substituting zeros", path, type(exc).__name__)
            size = self.config.image.image_size
            return torch.zeros(3, size, size, dtype=torch.float32), False

    def label_array(self) -> np.ndarray:
        """Return the ``(n_samples, n_classes)`` label matrix as numpy."""
        return self.labels.numpy()

    def describe(self) -> dict[str, Any]:
        """Small summary used in logs, tests and the notebook."""
        matrix = self.label_array()
        return {
            "n_samples": len(self),
            "n_games": int(self.samples["app_id"].nunique()),
            "modality": self.modality,
            "train_transforms": bool(self.train),
            "text_column": self.text_column,
            "positives_per_class": {
                genre: int(count) for genre, count in zip(self.classes, matrix.sum(axis=0))
            },
            "labels_per_sample_mean": round(float(matrix.sum(axis=1).mean()), 3) if len(self) else 0.0,
        }


# --------------------------------------------------------------------------- #
# Cached-feature dataset (frozen encoders)
# --------------------------------------------------------------------------- #
class FeatureDataset(Dataset):
    """Dataset over pre-extracted frozen-encoder embeddings."""

    def __init__(
        self,
        *,
        labels: np.ndarray,
        image_features: np.ndarray | None = None,
        text_features: np.ndarray | None = None,
        sample_ids: Sequence[str] | None = None,
        app_ids: Sequence[str] | None = None,
    ) -> None:
        if image_features is None and text_features is None:
            raise ValueError("at least one feature matrix must be provided")
        self.labels = torch.as_tensor(np.asarray(labels, dtype=np.float32))
        self.image_features = (
            torch.as_tensor(np.asarray(image_features, dtype=np.float32))
            if image_features is not None
            else None
        )
        self.text_features = (
            torch.as_tensor(np.asarray(text_features, dtype=np.float32))
            if text_features is not None
            else None
        )
        for name, tensor in (("image", self.image_features), ("text", self.text_features)):
            if tensor is not None and tensor.shape[0] != self.labels.shape[0]:
                raise ValueError(
                    f"{name} features have {tensor.shape[0]} rows but labels have "
                    f"{self.labels.shape[0]}"
                )
        n = self.labels.shape[0]
        self.sample_ids = list(sample_ids) if sample_ids is not None else [str(i) for i in range(n)]
        self.app_ids = list(app_ids) if app_ids is not None else [str(i) for i in range(n)]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "index": index,
            "labels": self.labels[index],
            "sample_id": self.sample_ids[index],
            "app_id": self.app_ids[index],
        }
        if self.image_features is not None:
            item["image_features"] = self.image_features[index]
        if self.text_features is not None:
            item["text_features"] = self.text_features[index]
        return item

    def label_array(self) -> np.ndarray:
        return self.labels.numpy()


# --------------------------------------------------------------------------- #
# Collators
# --------------------------------------------------------------------------- #
class MultiLabelCollator:
    """Collate :class:`GameSenseDataset` items into a batch dictionary."""

    def __init__(
        self,
        *,
        tokenizer: Any | None = None,
        max_length: int = CONFIG.text.max_length,
        modality: Modality = "multimodal",
    ) -> None:
        if modality in ("text", "multimodal") and tokenizer is None:
            raise ValueError(f"modality {modality!r} requires a tokenizer")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.modality = modality

    def __call__(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "labels": torch.stack([item["labels"] for item in items]).float(),
            "sample_id": [item["sample_id"] for item in items],
            "app_id": [item["app_id"] for item in items],
            "index": torch.tensor([item["index"] for item in items], dtype=torch.long),
        }
        if self.modality in ("image", "multimodal"):
            batch["image"] = torch.stack([item["image"] for item in items])
            batch["image_ok"] = torch.tensor(
                [bool(item.get("image_ok", True)) for item in items], dtype=torch.bool
            )
        if self.modality in ("text", "multimodal"):
            encoded = self.tokenizer(
                [item["text"] for item in items],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch["input_ids"] = encoded["input_ids"]
            batch["attention_mask"] = encoded["attention_mask"]
            batch["text"] = [item["text"] for item in items]
        return batch


class FeatureCollator:
    """Collate :class:`FeatureDataset` items."""

    def __call__(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "labels": torch.stack([item["labels"] for item in items]).float(),
            "sample_id": [item["sample_id"] for item in items],
            "app_id": [item["app_id"] for item in items],
            "index": torch.tensor([item["index"] for item in items], dtype=torch.long),
        }
        if "image_features" in items[0]:
            batch["image_features"] = torch.stack([item["image_features"] for item in items])
        if "text_features" in items[0]:
            batch["text_features"] = torch.stack([item["text_features"] for item in items])
        return batch
