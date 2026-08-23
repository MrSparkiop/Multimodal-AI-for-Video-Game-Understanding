"""Loading processed data, building dataloaders and managing the feature cache."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import CONFIG, GENRES, GameSenseConfig
from ..utils import get_logger, seed_worker, torch_generator
from .dataset import (
    FeatureCollator,
    FeatureDataset,
    GameSenseDataset,
    Modality,
    MultiLabelCollator,
)
from .preprocessing import TEXT_COLUMNS, label_matrix
from .splitting import SPLIT_NAMES

__all__ = [
    "DataBundle",
    "load_games",
    "load_samples",
    "load_split",
    "load_bundle",
    "get_tokenizer",
    "build_dataset",
    "build_dataloader",
    "build_dataloaders",
    "feature_cache_path",
    "load_or_extract_features",
    "build_feature_datasets",
    "build_feature_dataloaders",
]

LOGGER = get_logger("gamesense.data.loader")

# Columns that must be present in games.csv for the project to work.
_REQUIRED_GAME_COLUMNS = ("app_id", "name", "genres")


@dataclass
class DataBundle:
    """Everything the training scripts need after ``prepare_data.py`` has run."""

    games: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    classes: tuple[str, ...] = GENRES

    @property
    def train(self) -> pd.DataFrame:
        return self.splits["train"]

    @property
    def val(self) -> pd.DataFrame:
        return self.splits["val"]

    @property
    def test(self) -> pd.DataFrame:
        return self.splits["test"]

    def games_for(self, split: str) -> pd.DataFrame:
        """Game-level rows belonging to *split*."""
        app_ids = set(self.splits[split]["app_id"].astype(str))
        return self.games[self.games["app_id"].astype(str).isin(app_ids)].reset_index(drop=True)

    def labels_for(self, split: str) -> np.ndarray:
        """Sample-level label matrix for *split* (aligned with the split frame)."""
        indexed = self.games.drop_duplicates("app_id").set_index(self.games["app_id"].astype(str))
        aligned = indexed.loc[self.splits[split]["app_id"].astype(str)]
        return label_matrix(aligned, classes=self.classes)

    def summary(self) -> dict[str, Any]:
        return {
            "n_games": int(len(self.games)),
            "classes": list(self.classes),
            "splits": {
                name: {
                    "n_samples": int(len(frame)),
                    "n_games": int(frame["app_id"].nunique()),
                }
                for name, frame in self.splits.items()
            },
        }


# --------------------------------------------------------------------------- #
# Reading processed artefacts
# --------------------------------------------------------------------------- #
def _read_csv(path: Path, what: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{what} not found at {path}.\nRun:  python scripts/prepare_data.py"
        )
    return pd.read_csv(path, dtype={"app_id": str, "sample_id": str}, keep_default_na=False)


def load_games(config: GameSenseConfig = CONFIG) -> pd.DataFrame:
    """Read ``data/processed/games.csv`` (one row per game)."""
    frame = _read_csv(config.paths.games_csv, "games.csv")
    missing = [column for column in _REQUIRED_GAME_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"games.csv is missing required columns: {missing}")
    return frame


def load_samples(config: GameSenseConfig = CONFIG) -> pd.DataFrame:
    """Read ``data/processed/samples.csv`` (one row per screenshot)."""
    return _read_csv(config.paths.samples_csv, "samples.csv")


def load_split(split: str, config: GameSenseConfig = CONFIG) -> pd.DataFrame:
    """Read one split CSV."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    return _read_csv(config.paths.split_csv(split), f"{split}.csv")


def load_bundle(
    config: GameSenseConfig = CONFIG, *, max_samples: int | None = None
) -> DataBundle:
    """Load games + all three splits."""
    games = load_games(config)
    splits = {name: load_split(name, config) for name in SPLIT_NAMES}
    if max_samples is not None:
        splits = {name: frame.head(max_samples).copy() for name, frame in splits.items()}
        kept = set()
        for frame in splits.values():
            kept |= set(frame["app_id"].astype(str))
        games = games[games["app_id"].astype(str).isin(kept)].reset_index(drop=True)
    bundle = DataBundle(games=games, splits=splits)
    LOGGER.info("Loaded data bundle: %s", bundle.summary())
    return bundle


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def get_tokenizer(model_name: str = CONFIG.text.model_name) -> Any:
    """Return (and cache) the Hugging Face tokenizer for *model_name*."""
    from transformers import AutoTokenizer

    LOGGER.info("Loading tokenizer '%s'", model_name)
    return AutoTokenizer.from_pretrained(model_name)


# --------------------------------------------------------------------------- #
# Datasets / dataloaders over raw inputs
# --------------------------------------------------------------------------- #
def build_dataset(
    bundle: DataBundle,
    split: str,
    *,
    modality: Modality = "multimodal",
    text_column: str = TEXT_COLUMNS["no_title"],
    augment: bool | None = None,
    config: GameSenseConfig = CONFIG,
) -> GameSenseDataset:
    """Build a :class:`GameSenseDataset` for one split."""
    train = (split == "train") if augment is None else augment
    return GameSenseDataset(
        bundle.splits[split],
        bundle.games,
        modality=modality,
        train=train,
        text_column=text_column,
        classes=bundle.classes,
        config=config,
    )


def build_dataloader(
    dataset: GameSenseDataset | FeatureDataset,
    *,
    batch_size: int,
    shuffle: bool,
    modality: Modality = "multimodal",
    tokenizer: Any | None = None,
    num_workers: int = CONFIG.training.num_workers,
    seed: int = CONFIG.training.seed,
    config: GameSenseConfig = CONFIG,
) -> DataLoader:
    """Wrap a dataset in a reproducible :class:`~torch.utils.data.DataLoader`."""
    if isinstance(dataset, FeatureDataset):
        collate: Any = FeatureCollator()
    else:
        collate = MultiLabelCollator(
            tokenizer=tokenizer, max_length=config.text.max_length, modality=modality
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=torch_generator(seed) if shuffle else None,
        drop_last=False,
        pin_memory=False,
    )


def build_dataloaders(
    bundle: DataBundle,
    *,
    modality: Modality = "multimodal",
    text_column: str = TEXT_COLUMNS["no_title"],
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    tokenizer: Any | None = None,
    seed: int = CONFIG.training.seed,
    config: GameSenseConfig = CONFIG,
    augment_train: bool = True,
) -> dict[str, DataLoader]:
    """Build train/val/test loaders over raw images and text."""
    if modality in ("text", "multimodal") and tokenizer is None:
        tokenizer = get_tokenizer(config.text.model_name)
    train_bs = batch_size or config.training.batch_size
    eval_bs = eval_batch_size or config.training.eval_batch_size
    loaders: dict[str, DataLoader] = {}
    for split in SPLIT_NAMES:
        dataset = build_dataset(
            bundle,
            split,
            modality=modality,
            text_column=text_column,
            augment=(split == "train" and augment_train),
            config=config,
        )
        loaders[split] = build_dataloader(
            dataset,
            batch_size=train_bs if split == "train" else eval_bs,
            shuffle=(split == "train"),
            modality=modality,
            tokenizer=tokenizer,
            num_workers=config.training.num_workers,
            seed=seed,
            config=config,
        )
    return loaders


# --------------------------------------------------------------------------- #
# Frozen-encoder feature cache
# --------------------------------------------------------------------------- #
def _cache_key(kind: Literal["image", "text"], split: str, config: GameSenseConfig, text_column: str) -> str:
    """Hash everything that influences the embeddings into a short key."""
    if kind == "image":
        payload = f"image|{config.image.backbone}|{config.image.pretrained}|{config.image.image_size}"
    else:
        payload = f"text|{config.text.model_name}|{config.text.max_length}|{config.text.pooling}|{text_column}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{kind}_{split}_{digest}"


def feature_cache_path(
    kind: Literal["image", "text"],
    split: str,
    *,
    config: GameSenseConfig = CONFIG,
    text_column: str = TEXT_COLUMNS["no_title"],
) -> Path:
    """Return the ``.npz`` cache location for one ``(kind, split)`` pair."""
    return config.paths.feature_cache / f"{_cache_key(kind, split, config, text_column)}.npz"


def load_or_extract_features(
    bundle: DataBundle,
    split: str,
    kind: Literal["image", "text"],
    *,
    config: GameSenseConfig = CONFIG,
    text_column: str = TEXT_COLUMNS["no_title"],
    device: torch.device | str | None = None,
    batch_size: int | None = None,
    force: bool = False,
    progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Return frozen-encoder embeddings for one split, extracting them if needed.

    Returns ``(features, sample_ids)``; the cache is rejected if its ids do not match the
    current split.
    """
    from ..models.model_utils import encode_split  # local import avoids a cycle

    path = feature_cache_path(kind, split, config=config, text_column=text_column)
    expected_ids = [str(value) for value in bundle.splits[split]["sample_id"]]

    if path.is_file() and not force:
        with np.load(path, allow_pickle=False) as data:
            cached_ids = [str(value) for value in data["sample_ids"]]
            if cached_ids == expected_ids:
                LOGGER.info("Feature cache hit: %s (%s)", path.name, data["features"].shape)
                return data["features"], cached_ids
        LOGGER.warning("Feature cache %s does not match the current split - re-extracting", path.name)

    features, sample_ids = encode_split(
        bundle,
        split,
        kind,
        config=config,
        text_column=text_column,
        device=device,
        batch_size=batch_size,
        progress=progress,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, sample_ids=np.array(sample_ids, dtype=object).astype("U"))
    LOGGER.info("Wrote feature cache %s %s", path.name, features.shape)
    return features, sample_ids


def build_feature_datasets(
    bundle: DataBundle,
    *,
    modality: Modality = "multimodal",
    config: GameSenseConfig = CONFIG,
    text_column: str = TEXT_COLUMNS["no_title"],
    device: torch.device | str | None = None,
    force: bool = False,
    progress: bool = True,
    splits: Sequence[str] = SPLIT_NAMES,
) -> dict[str, FeatureDataset]:
    """Build :class:`FeatureDataset` objects for the requested splits."""
    datasets: dict[str, FeatureDataset] = {}
    for split in splits:
        frame = bundle.splits[split]
        image_features = text_features = None
        if modality in ("image", "multimodal"):
            image_features, _ = load_or_extract_features(
                bundle, split, "image", config=config, text_column=text_column,
                device=device, force=force, progress=progress,
            )
        if modality in ("text", "multimodal"):
            text_features, _ = load_or_extract_features(
                bundle, split, "text", config=config, text_column=text_column,
                device=device, force=force, progress=progress,
            )
        datasets[split] = FeatureDataset(
            labels=bundle.labels_for(split),
            image_features=image_features,
            text_features=text_features,
            sample_ids=[str(v) for v in frame["sample_id"]],
            app_ids=[str(v) for v in frame["app_id"]],
        )
    return datasets


def build_feature_dataloaders(
    bundle: DataBundle,
    *,
    modality: Modality = "multimodal",
    config: GameSenseConfig = CONFIG,
    text_column: str = TEXT_COLUMNS["no_title"],
    device: torch.device | str | None = None,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    seed: int = CONFIG.training.seed,
    force: bool = False,
    progress: bool = True,
) -> tuple[dict[str, DataLoader], dict[str, FeatureDataset]]:
    """Build loaders over cached embeddings (the fast frozen-encoder path).

    Returns ``(loaders, datasets)`` keyed by split name.
    """
    datasets = build_feature_datasets(
        bundle, modality=modality, config=config, text_column=text_column,
        device=device, force=force, progress=progress,
    )
    train_bs = batch_size or config.training.batch_size
    eval_bs = eval_batch_size or config.training.eval_batch_size
    loaders = {
        split: build_dataloader(
            dataset,
            batch_size=train_bs if split == "train" else eval_bs,
            shuffle=(split == "train"),
            modality=modality,
            num_workers=0,
            seed=seed,
            config=config,
        )
        for split, dataset in datasets.items()
    }
    return loaders, datasets
