"""Tests for :mod:`gamesense.data.dataset`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from gamesense.config import CONFIG, GENRES, label_column
from gamesense.data.dataset import (
    FeatureCollator,
    FeatureDataset,
    GameSenseDataset,
    MultiLabelCollator,
    build_image_transforms,
)
from gamesense.data.preprocessing import TEXT_COLUMNS, positive_counts

#: Item keys each modality must produce (and only these).
EXPECTED_ITEM_KEYS: dict[str, set[str]] = {
    "image": {"index", "sample_id", "app_id", "labels", "image", "image_ok"},
    "text": {"index", "sample_id", "app_id", "labels", "text"},
    "multimodal": {
        "index",
        "sample_id",
        "app_id",
        "labels",
        "text",
        "image",
        "image_ok",
    },
}

#: Screenshots per game in the synthetic fixtures.
SHOTS_PER_GAME = 2

#: Size of the small batches assembled for the collator tests.
BATCH_SIZE = 4

#: Index of the first sample whose image file is neither corrupt nor missing
#: (the fixture sabotages the first two rows on purpose).
FIRST_HEALTHY_INDEX = 2


def _noise_image(height: int = 96, width: int = 128, *, seed: int = 0) -> Any:
    """Return a small non-uniform RGB :class:`PIL.Image.Image`."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def _image_dataset(
    fixture: tuple[Any, pd.DataFrame, pd.DataFrame, str, str],
    *,
    modality: str = "multimodal",
    rows: slice | None = None,
    **kwargs: Any,
) -> GameSenseDataset:
    """Build a dataset over the on-disk fixture, optionally on a slice of rows."""
    config, games, samples, _, _ = fixture
    subset = samples if rows is None else samples.iloc[rows].reset_index(drop=True)
    return GameSenseDataset(subset, games, modality=modality, config=config, **kwargs)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def test_eval_transform_is_deterministic_and_correctly_shaped() -> None:
    """Two evaluations of one checkpoint must see byte-identical inputs."""
    transform = build_image_transforms(train=False)
    image = _noise_image()
    first, second = transform(image), transform(image)

    size = CONFIG.image.image_size
    assert tuple(first.shape) == (3, size, size)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)


def test_train_transform_augments_but_stays_seed_reproducible() -> None:
    """Augmentation must actually vary, yet remain reproducible from a seed."""
    transform = build_image_transforms(train=True)
    image = _noise_image()

    torch.manual_seed(0)
    first = transform(image)
    second = transform(image)  # no reseeding: a different draw
    torch.manual_seed(0)
    replay = transform(image)  # same seed as `first`

    size = CONFIG.image.image_size
    assert tuple(first.shape) == (3, size, size)
    assert not torch.equal(first, second), "the train pipeline is not augmenting"
    assert torch.equal(first, replay), "the train pipeline is not seed-reproducible"


# --------------------------------------------------------------------------- #
# GameSenseDataset
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("modality", sorted(EXPECTED_ITEM_KEYS))
def test_dataset_item_keys_and_shapes_per_modality(
    image_dataset_on_disk, modality: str
) -> None:
    dataset = _image_dataset(image_dataset_on_disk, modality=modality)
    item = dataset[FIRST_HEALTHY_INDEX]

    assert set(item) == EXPECTED_ITEM_KEYS[modality]
    assert item["labels"].shape == (len(GENRES),)
    assert item["labels"].dtype == torch.float32
    assert isinstance(item["sample_id"], str) and item["sample_id"]
    assert isinstance(item["app_id"], str) and item["app_id"]

    if "image" in item:
        size = CONFIG.image.image_size
        assert tuple(item["image"].shape) == (3, size, size)
        assert item["image_ok"] is True
    if "text" in item:
        assert isinstance(item["text"], str) and item["text"].strip()


def test_labels_are_multi_hot_over_the_full_label_space(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    dataset = GameSenseDataset(synthetic_samples, synthetic_games, modality="text")
    labels = dataset.label_array()

    assert labels.shape == (len(synthetic_samples), len(GENRES))
    assert set(np.unique(labels)).issubset({0.0, 1.0})
    assert (labels.sum(axis=1) >= 1).all()  # every synthetic game has a genre


def test_labels_are_joined_from_the_game_frame(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    """Both screenshots of a game must carry byte-identical labels and text."""
    dataset = GameSenseDataset(synthetic_samples, synthetic_games, modality="text")
    by_game: dict[str, list[int]] = {}
    for index in range(len(dataset)):
        by_game.setdefault(dataset[index]["app_id"], []).append(index)

    assert by_game, "the fixture must contain at least one game"
    for app_id, indices in by_game.items():
        assert len(indices) == SHOTS_PER_GAME
        reference = dataset[indices[0]]
        for index in indices[1:]:
            assert torch.equal(dataset[index]["labels"], reference["labels"]), app_id
            assert dataset[index]["text"] == reference["text"]

    # ... and they agree, column by column, with the game frame they came from.
    expected = {
        str(row["app_id"]): [float(row[label_column(genre)]) for genre in GENRES]
        for row in synthetic_games.to_dict("records")
    }
    for app_id, indices in by_game.items():
        assert dataset[indices[0]]["labels"].tolist() == expected[app_id]


def test_unreadable_images_become_zeros_and_are_counted(image_dataset_on_disk) -> None:
    """A truncated or absent screenshot must not abort an epoch."""
    _, _, samples, corrupt_id, missing_id = image_dataset_on_disk
    dataset = _image_dataset(image_dataset_on_disk, modality="image")
    positions = {
        str(sample_id): index
        for index, sample_id in enumerate(dataset.samples["sample_id"].astype(str))
    }

    assert dataset.n_image_errors == 0
    for bad_id in (corrupt_id, missing_id):
        item = dataset[positions[str(bad_id)]]
        assert item["image_ok"] is False
        assert torch.count_nonzero(item["image"]) == 0
        size = CONFIG.image.image_size
        assert tuple(item["image"].shape) == (3, size, size)
    assert dataset.n_image_errors == 2

    # A healthy neighbour is still decoded normally.
    healthy = dataset[FIRST_HEALTHY_INDEX]
    assert healthy["image_ok"] is True
    assert torch.count_nonzero(healthy["image"]) > 0
    assert dataset.n_image_errors == 2


def test_unreadable_images_can_be_made_fatal(image_dataset_on_disk) -> None:
    """``on_image_error="raise"`` is the strict mode used while preparing data."""
    _, _, _, corrupt_id, missing_id = image_dataset_on_disk
    dataset = _image_dataset(
        image_dataset_on_disk, modality="image", on_image_error="raise"
    )
    positions = {
        str(sample_id): index
        for index, sample_id in enumerate(dataset.samples["sample_id"].astype(str))
    }
    for bad_id in (corrupt_id, missing_id):
        with pytest.raises(OSError):
            dataset[positions[str(bad_id)]]


# --------------------------------------------------------------------------- #
# Constructor validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("column", ["sample_id", "app_id"])
def test_missing_sample_columns_raise(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame, column: str
) -> None:
    with pytest.raises(KeyError, match="missing columns"):
        GameSenseDataset(
            synthetic_samples.drop(columns=[column]), synthetic_games, modality="text"
        )


def test_image_modalities_require_an_image_path_column(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    with pytest.raises(KeyError, match="image_path"):
        GameSenseDataset(
            synthetic_samples.drop(columns=["image_path"]),
            synthetic_games,
            modality="image",
        )


def test_unknown_text_column_raises(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    with pytest.raises(KeyError, match="description_does_not_exist"):
        GameSenseDataset(
            synthetic_samples,
            synthetic_games,
            modality="text",
            text_column="description_does_not_exist",
        )


def test_unknown_app_id_raises(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    orphan = synthetic_samples.iloc[[0]].copy()
    orphan["app_id"] = "not-a-real-game"
    with pytest.raises(KeyError, match="absent from the games frame"):
        GameSenseDataset(
            pd.concat([synthetic_samples, orphan], ignore_index=True),
            synthetic_games,
            modality="text",
        )


def test_unknown_modality_raises(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="unknown modality"):
        GameSenseDataset(synthetic_samples, synthetic_games, modality="audio")  # type: ignore[arg-type]


def test_describe_reports_the_dataset_composition(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    dataset = GameSenseDataset(
        synthetic_samples, synthetic_games, modality="text", train=True
    )
    summary = dataset.describe()

    assert summary["n_samples"] == len(synthetic_samples)
    assert summary["n_games"] == len(synthetic_games)
    assert summary["modality"] == "text"
    assert summary["train_transforms"] is True
    assert summary["text_column"] == TEXT_COLUMNS["no_title"]

    game_positives = positive_counts(synthetic_games)
    for genre in GENRES:
        assert summary["positives_per_class"][genre] == SHOTS_PER_GAME * game_positives[genre]
    assert summary["labels_per_sample_mean"] > 0


# --------------------------------------------------------------------------- #
# MultiLabelCollator
# --------------------------------------------------------------------------- #
def test_collator_pads_to_the_batch_maximum(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame, tokenizer
) -> None:
    """Dynamic padding is what makes CPU training affordable -- pin it down."""
    dataset = GameSenseDataset(synthetic_samples, synthetic_games, modality="text")
    lengths = [
        len(tokenizer(text, truncation=True, max_length=CONFIG.text.max_length)["input_ids"])
        for text in dataset.texts
    ]
    order = np.argsort(lengths)
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    picked_lengths = [lengths[index] for index in picks]
    assert min(picked_lengths) < max(picked_lengths), "fixture texts are all the same length"

    collator = MultiLabelCollator(tokenizer=tokenizer, modality="text")
    batch = collator([dataset[index] for index in picks])

    assert tuple(batch["input_ids"].shape) == (len(picks), max(picked_lengths))
    assert batch["input_ids"].shape[1] < CONFIG.text.max_length
    assert batch["attention_mask"].sum(dim=1).tolist() == picked_lengths
    assert tuple(batch["labels"].shape) == (len(picks), len(GENRES))
    assert batch["labels"].dtype == torch.float32
    assert batch["sample_id"] == [dataset[index]["sample_id"] for index in picks]
    assert batch["index"].tolist() == picks


def test_collator_stacks_images_and_metadata(image_dataset_on_disk) -> None:
    dataset = _image_dataset(
        image_dataset_on_disk,
        modality="image",
        rows=slice(FIRST_HEALTHY_INDEX, FIRST_HEALTHY_INDEX + BATCH_SIZE),
    )
    batch = MultiLabelCollator(modality="image")([dataset[i] for i in range(len(dataset))])

    size = CONFIG.image.image_size
    assert tuple(batch["image"].shape) == (BATCH_SIZE, 3, size, size)
    assert tuple(batch["labels"].shape) == (BATCH_SIZE, len(GENRES))
    assert batch["image_ok"].dtype == torch.bool
    assert bool(batch["image_ok"].all())
    assert len(batch["app_id"]) == BATCH_SIZE
    assert "input_ids" not in batch  # image modality carries no tokens


@pytest.mark.parametrize("modality", ["text", "multimodal"])
def test_collator_requires_a_tokenizer_for_text_modalities(modality: str) -> None:
    with pytest.raises(ValueError, match="requires a tokenizer"):
        MultiLabelCollator(modality=modality)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# FeatureDataset / FeatureCollator
# --------------------------------------------------------------------------- #
def _feature_arrays(n_rows: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(labels, image_features, text_features)`` of matching height."""
    rng = np.random.default_rng(0)
    labels = (rng.random((n_rows, len(GENRES))) < 0.3).astype(np.float32)
    image_features = rng.normal(size=(n_rows, CONFIG.image.embedding_dim)).astype(np.float32)
    text_features = rng.normal(size=(n_rows, CONFIG.text.embedding_dim)).astype(np.float32)
    return labels, image_features, text_features


def test_feature_dataset_exposes_both_feature_blocks() -> None:
    labels, image_features, text_features = _feature_arrays()
    dataset = FeatureDataset(
        labels=labels,
        image_features=image_features,
        text_features=text_features,
        sample_ids=[f"s{i}" for i in range(len(labels))],
        app_ids=[f"g{i}" for i in range(len(labels))],
    )

    assert len(dataset) == len(labels)
    item = dataset[0]
    assert tuple(item["image_features"].shape) == (CONFIG.image.embedding_dim,)
    assert tuple(item["text_features"].shape) == (CONFIG.text.embedding_dim,)
    assert tuple(item["labels"].shape) == (len(GENRES),)
    assert item["labels"].dtype == torch.float32
    assert item["sample_id"] == "s0" and item["app_id"] == "g0"
    assert dataset.label_array().shape == labels.shape


def test_feature_dataset_omits_a_missing_modality() -> None:
    labels, _, text_features = _feature_arrays()
    dataset = FeatureDataset(labels=labels, text_features=text_features)
    item = dataset[0]
    assert "text_features" in item and "image_features" not in item
    assert item["sample_id"] == "0"  # generated placeholders


def test_feature_dataset_rejects_mismatched_row_counts() -> None:
    labels, image_features, _ = _feature_arrays()
    with pytest.raises(ValueError, match="rows"):
        FeatureDataset(labels=labels, image_features=image_features[:-1])


def test_feature_dataset_requires_at_least_one_feature_matrix() -> None:
    labels, _, _ = _feature_arrays()
    with pytest.raises(ValueError, match="at least one feature matrix"):
        FeatureDataset(labels=labels)


def test_feature_collator_stacks_every_present_block() -> None:
    labels, image_features, text_features = _feature_arrays()
    dataset = FeatureDataset(
        labels=labels, image_features=image_features, text_features=text_features
    )
    batch = FeatureCollator()([dataset[i] for i in range(BATCH_SIZE)])

    assert tuple(batch["image_features"].shape) == (BATCH_SIZE, CONFIG.image.embedding_dim)
    assert tuple(batch["text_features"].shape) == (BATCH_SIZE, CONFIG.text.embedding_dim)
    assert tuple(batch["labels"].shape) == (BATCH_SIZE, len(GENRES))
    assert batch["index"].tolist() == list(range(BATCH_SIZE))
    assert len(batch["sample_id"]) == BATCH_SIZE


def test_feature_collator_skips_absent_blocks() -> None:
    labels, image_features, _ = _feature_arrays()
    dataset = FeatureDataset(labels=labels, image_features=image_features)
    batch = FeatureCollator()([dataset[i] for i in range(BATCH_SIZE)])
    assert "image_features" in batch and "text_features" not in batch
