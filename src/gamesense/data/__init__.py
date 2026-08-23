"""Data acquisition, cleaning, splitting and PyTorch dataset plumbing."""

from __future__ import annotations

from .acquisition import (
    DownloadStats,
    download_raw_dataset,
    download_screenshots,
    expand_to_samples,
    load_raw_games,
    select_games,
)
from .dataset import (
    FeatureDataset,
    GameSenseDataset,
    Modality,
    MultiLabelCollator,
    build_image_transforms,
)
from .loader import (
    DataBundle,
    build_dataloaders,
    build_feature_dataloaders,
    get_tokenizer,
    load_bundle,
    load_games,
    load_or_extract_features,
    load_samples,
)
from .preprocessing import (
    TEXT_COLUMNS,
    CleaningReport,
    add_text_variants,
    adult_content_mask,
    class_weights,
    clean_description,
    clean_games,
    decode_labels,
    encode_labels,
    label_matrix,
    mask_genre_terms,
    normalize_genre,
    normalize_genres,
    positive_counts,
    remove_title_mentions,
)
from .splitting import SPLIT_NAMES, SplitResult, make_splits, verify_no_group_leakage

__all__ = [
    # acquisition
    "download_raw_dataset",
    "load_raw_games",
    "select_games",
    "expand_to_samples",
    "download_screenshots",
    "DownloadStats",
    # preprocessing
    "clean_description",
    "clean_games",
    "add_text_variants",
    "adult_content_mask",
    "normalize_genre",
    "normalize_genres",
    "remove_title_mentions",
    "mask_genre_terms",
    "encode_labels",
    "decode_labels",
    "label_matrix",
    "positive_counts",
    "class_weights",
    "CleaningReport",
    "TEXT_COLUMNS",
    # splitting
    "make_splits",
    "verify_no_group_leakage",
    "SplitResult",
    "SPLIT_NAMES",
    # datasets / loaders
    "GameSenseDataset",
    "FeatureDataset",
    "MultiLabelCollator",
    "build_image_transforms",
    "Modality",
    "DataBundle",
    "load_bundle",
    "load_games",
    "load_samples",
    "get_tokenizer",
    "build_dataloaders",
    "build_feature_dataloaders",
    "load_or_extract_features",
]
