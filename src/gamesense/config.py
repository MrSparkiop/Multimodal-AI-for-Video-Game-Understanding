"""Centralised configuration for the GameSense project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

__all__ = [
    "PROJECT_ROOT",
    "GENRES",
    "NUM_CLASSES",
    "EXCLUDED_GENRE_REASONS",
    "Paths",
    "DatasetConfig",
    "ImageConfig",
    "TextConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "GameSenseConfig",
    "CONFIG",
    "MODEL_KINDS",
    "MODEL_DISPLAY_NAMES",
    "genre_to_index",
    "label_column",
    "label_columns",
]


# --------------------------------------------------------------------------- #
# Project root discovery
# --------------------------------------------------------------------------- #
def _find_project_root(start: Path) -> Path:
    """Walk upwards from *start* until a directory containing ``pyproject.toml``."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start.parents[2] if len(start.parents) >= 3 else start


PROJECT_ROOT: Final[Path] = _find_project_root(Path(__file__).resolve())


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:  # pragma: no cover - defensive
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else default


# --------------------------------------------------------------------------- #
# Target label space
# --------------------------------------------------------------------------- #
# The genres GameSense predicts.
GENRES: Final[tuple[str, ...]] = (
    "Action",
    "Adventure",
    "Casual",
    "RPG",
    "Racing",
    "Simulation",
    "Sports",
    "Strategy",
)

NUM_CLASSES: Final[int] = len(GENRES)

# Steam genre strings that are never used as prediction targets, with the
# reason.  Kept in the config so the notebook can render the table directly.
EXCLUDED_GENRE_REASONS: Final[dict[str, str]] = {
    "Indie": "studio-size / business descriptor, not gameplay (present in ~66% of rows)",
    "Free To Play": "monetisation model, not gameplay",
    "Early Access": "release state, not gameplay",
    "Massively Multiplayer": "multiplayer mode rather than a genre; also very low support",
    "Utilities": "non-game software category",
    "Design & Illustration": "non-game software category",
    "Animation & Modeling": "non-game software category",
    "Video Production": "non-game software category",
    "Audio Production": "non-game software category",
    "Photo Editing": "non-game software category",
    "Game Development": "non-game software category",
    "Software Training": "non-game software category",
    "Web Publishing": "non-game software category",
    "Education": "non-game software category",
    "Accounting": "non-game software category",
    "Nudity": "content warning, excluded for academic appropriateness",
    "Sexual Content": "content warning, excluded for academic appropriateness",
    "Violent": "content warning, not a genre",
    "Gore": "content warning, not a genre",
}


def genre_to_index() -> dict[str, int]:
    """Return the canonical ``genre -> column index`` mapping."""
    return {genre: idx for idx, genre in enumerate(GENRES)}


def label_column(genre: str) -> str:
    """Return the tabular column name storing the binary label for *genre*."""
    slug = genre.lower().replace("&", "and")
    slug = "_".join(part for part in slug.replace("-", " ").split() if part)
    return f"y_{slug}"


def label_columns(genres: tuple[str, ...] | list[str] | None = None) -> list[str]:
    """Return label column names for *genres* (defaults to :data:`GENRES`)."""
    return [label_column(g) for g in (GENRES if genres is None else genres)]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Paths:
    """All filesystem locations used by the project, relative to the root."""

    root: Path = PROJECT_ROOT

    # -- directories ------------------------------------------------------- #
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def splits(self) -> Path:
        return self.data / "splits"

    @property
    def images(self) -> Path:
        return self.data / "images"

    @property
    def checkpoints(self) -> Path:
        return self.root / "models"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def metrics(self) -> Path:
        return self.results / "metrics"

    @property
    def predictions(self) -> Path:
        return self.results / "predictions"

    @property
    def logs(self) -> Path:
        return self.results / "logs"

    @property
    def feature_cache(self) -> Path:
        return self.processed / "features"

    # -- concrete files ---------------------------------------------------- #
    @property
    def raw_parquet(self) -> Path:
        """Local copy of the upstream Steam dataset."""
        return self.raw / "steam_games.parquet"

    @property
    def games_csv(self) -> Path:
        """One row per game after cleaning."""
        return self.processed / "games.csv"

    @property
    def samples_csv(self) -> Path:
        """One row per (game, screenshot) training sample after cleaning."""
        return self.processed / "samples.csv"

    @property
    def cleaning_report(self) -> Path:
        return self.processed / "cleaning_report.json"

    @property
    def label_space(self) -> Path:
        return self.processed / "label_space.json"

    @property
    def split_summary(self) -> Path:
        return self.splits / "split_summary.json"

    def split_csv(self, split: str) -> Path:
        return self.splits / f"{split}.csv"

    def ensure(self) -> None:
        """Create every output directory (idempotent)."""
        for directory in (
            self.raw,
            self.processed,
            self.splits,
            self.images,
            self.checkpoints,
            self.figures,
            self.metrics,
            self.predictions,
            self.logs,
            self.feature_cache,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dataset / acquisition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetConfig:
    """Where the raw data comes from and how much of it we keep."""

    # Hugging Face dataset repository (MIT licensed), built from Steam's public
    # Web API by https://github.com/FronkonGames/Steam-Games-Scraper
    hf_repo_id: str = "FronkonGames/steam-games-dataset"
    hf_repo_type: str = "dataset"
    hf_filename: str = "data/train-00000-of-00001.parquet"

    # Maximum number of games kept after cleaning + stratified subsampling.
    max_games: int = _env_int("GAMESENSE_MAX_GAMES", 8_000)
    # Screenshots downloaded per game.  Multiple screenshots per game is exactly
    # why splitting must happen at game level (see gamesense.data.splitting).
    screenshots_per_game: int = _env_int("GAMESENSE_SHOTS_PER_GAME", 2)

    # Longest side (pixels) at which images are stored on disk.  Larger than
    # ImageConfig.image_size so random crops still have some freedom.
    stored_image_max_side: int = 360
    stored_image_quality: int = 88

    # Politeness settings for the image downloader.
    download_workers: int = 8
    download_timeout_connect: float = 10.0
    download_timeout_read: float = 30.0
    download_retries: int = 3
    download_retry_backoff: float = 1.5
    # Small pause per worker between requests (seconds) to stay well below any
    # plausible rate limit on the public Steam CDN.
    download_delay: float = 0.05
    user_agent: str = "GameSense-Academic-Research/1.0 (university deep learning project)"

    # Cleaning thresholds.
    min_description_words: int = 15
    max_description_words: int = 400
    min_ascii_ratio: float = 0.95
    # Minimum number of games a genre must have (after subsampling) to be kept.
    min_genre_support: int = 150
    # Minimum stored-image side; smaller downloads are treated as corrupt.
    min_image_side: int = 64


# --------------------------------------------------------------------------- #
# Modality configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ImageConfig:
    """Vision branch settings."""

    backbone: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224
    # ImageNet statistics: required because we use ImageNet pretrained weights.
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    embedding_dim: int = 512  # ResNet18 penultimate feature width
    # Light augmentation, training split only.
    train_random_crop: bool = True
    train_horizontal_flip_p: float = 0.5
    train_color_jitter: float = 0.1


@dataclass(frozen=True)
class TextConfig:
    """Language branch settings."""

    model_name: str = _env_str("GAMESENSE_TEXT_MODEL", "distilbert-base-uncased")
    max_length: int = 128
    embedding_dim: int = 768  # DistilBERT hidden size
    pooling: Literal["cls", "mean"] = "mean"
    # BiLSTM baseline (optional additional experiment).
    lstm_vocab_size: int = 30_522
    lstm_embedding_dim: int = 128
    lstm_hidden_dim: int = 128
    lstm_layers: int = 1


@dataclass(frozen=True)
class ModelConfig:
    """Classifier heads and fusion."""

    dropout: float = 0.3
    # Hidden width of the image / text classification heads (0 = linear head).
    head_hidden_dim: int = 256
    # Hidden widths of the multimodal fusion MLP.
    fusion_hidden_dims: tuple[int, ...] = (512, 256)
    # Freeze the pretrained encoders by default (transfer learning as feature
    # extraction).  Scripts can request partial unfreezing via CLI flags.
    freeze_image_backbone: bool = True
    freeze_text_encoder: bool = True
    # How many trailing ResNet stages / transformer layers to unfreeze when
    # partial fine-tuning is requested.
    unfreeze_image_stages: int = 0
    unfreeze_text_layers: int = 0
    # L2-normalise each embedding before fusion so neither modality dominates
    # purely because of feature scale.
    normalize_embeddings_before_fusion: bool = True


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation settings shared by all three models."""

    seed: int = _env_int("GAMESENSE_SEED", 42)
    # Seeds used for the repeated-runs experiment (statistical validity).
    seeds: tuple[int, ...] = (42, 123, 2026)

    batch_size: int = _env_int("GAMESENSE_BATCH_SIZE", 32)
    eval_batch_size: int = 64
    epochs: int = _env_int("GAMESENSE_EPOCHS", 40)

    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    head_lr: float = 1e-3
    backbone_lr: float = 2e-5  # only used when a backbone is unfrozen
    weight_decay: float = 1e-2
    # Gradient clipping (max L2 norm); None disables it.
    grad_clip_norm: float | None = 1.0

    scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    warmup_ratio: float = 0.05
    plateau_factor: float = 0.5
    plateau_patience: int = 2

    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    # Metric watched by early stopping / checkpoint selection (validation split).
    monitor_metric: str = "micro_f1"
    monitor_mode: Literal["max", "min"] = "max"

    # Class-imbalance handling: "none" | "pos_weight".
    class_weighting: Literal["none", "pos_weight"] = "none"
    # Upper bound applied to pos_weight so rare classes cannot destabilise
    # the loss.
    pos_weight_clip: float = 10.0

    num_workers: int = _env_int("GAMESENSE_NUM_WORKERS", 0)
    # Cache frozen-encoder embeddings to disk.
    cache_features: bool = True
    # Limit the number of samples (debug / smoke tests).  None = use all.
    max_samples: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    """Metric and threshold settings."""

    default_threshold: float = 0.50
    # Grid searched on the validation split only (never on test).
    threshold_grid: tuple[float, ...] = (
        0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    )
    # "global" = one threshold for all classes, "per_class" = one per class.
    threshold_strategy: Literal["global", "per_class"] = "global"
    # Metric optimised during threshold search.
    threshold_metric: str = "micro_f1"
    # Number of examples exported for qualitative error analysis.
    n_error_examples: int = 12


@dataclass(frozen=True)
class GameSenseConfig:
    """Root configuration object."""

    paths: Paths = field(default_factory=Paths)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    text: TextConfig = field(default_factory=TextConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # "auto" resolves to CUDA when available, else CPU.
    device: str = _env_str("GAMESENSE_DEVICE", "auto")

    @property
    def genres(self) -> tuple[str, ...]:
        return GENRES

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    def checkpoint_path(self, model_kind: str, seed: int | None = None) -> Path:
        """Return the checkpoint file for ``model_kind``."""
        seed = self.training.seed if seed is None else seed
        suffix = "" if seed == self.training.seed else f"_seed{seed}"
        return self.paths.checkpoints / f"{model_kind}_model{suffix}.pt"

    def history_path(self, model_kind: str, seed: int | None = None) -> Path:
        seed = self.training.seed if seed is None else seed
        return self.paths.logs / f"history_{model_kind}_seed{seed}.json"

    def metrics_path(self, model_kind: str, seed: int | None = None) -> Path:
        seed = self.training.seed if seed is None else seed
        return self.paths.metrics / f"metrics_{model_kind}_seed{seed}.json"

    def predictions_path(
        self, model_kind: str, split: str, seed: int | None = None
    ) -> Path:
        seed = self.training.seed if seed is None else seed
        return self.paths.predictions / f"pred_{model_kind}_{split}_seed{seed}.npz"


#: The configuration instance used across the project.
CONFIG: Final[GameSenseConfig] = GameSenseConfig()

#: The three systems compared in the main research question.
MODEL_KINDS: Final[tuple[str, ...]] = ("image", "text", "multimodal")

#: Human readable names used in tables and figures.
MODEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "image": "Image Only (ResNet18)",
    "text": "Text Only (DistilBERT)",
    "multimodal": "Multimodal (ResNet18 + DistilBERT)",
    "text_bilstm": "Text Only (BiLSTM baseline)",
}
