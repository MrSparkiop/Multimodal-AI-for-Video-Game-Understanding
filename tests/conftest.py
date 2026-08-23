"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SRC))

from gamesense.config import GENRES, CONFIG, Paths, label_column  # noqa: E402
from gamesense.data.preprocessing import clean_games  # noqa: E402

RNG_SEED = 20260821


# --------------------------------------------------------------------------- #
# Raw / cleaned frames
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def raw_games_frame() -> pd.DataFrame:
    """A small hand-written frame with the same schema as the Steam parquet."""
    def shots(app_id: int, n: int) -> list[str]:
        return [f"https://example.invalid/{app_id}/ss_{i}.jpg" for i in range(n)]

    rows = [
        {
            "appID": "100",
            "name": "Dragon Quest of Aethel",
            "short_description": (
                "<p>Dragon Quest of Aethel is a sprawling open world role playing game. "
                "Explore ruins, fight monsters, craft weapons and level up your party "
                "across dozens of hours of quests.</p>"
            ),
            "genres": ["RPG", "Adventure", "Indie"],
            "screenshots": shots(100, 4),
        },
        {
            "appID": "101",
            "name": "Turbo Circuit 2",
            "short_description": (
                "Race exotic cars around twenty licensed circuits. Tune your engine, "
                "master the drift and beat the clock in this arcade racing experience."
            ),
            "genres": ["Racing", "Sports", "Casual"],
            "screenshots": shots(101, 3),
        },
        {
            "appID": "102",
            "name": "Colony Architect: Deep Space",
            "short_description": (
                "Build and manage an orbital colony. Balance oxygen, power and morale "
                "while planning production chains in this management simulation game."
            ),
            "genres": ["Simulation", "Strategy"],
            "screenshots": shots(102, 5),
        },
        {
            "appID": "103",
            "name": "Neon Reflex",
            "short_description": (
                "A fast paced action shooter set in a neon city. Dodge bullets, chain "
                "combos and climb the global leaderboards in short arcade runs."
            ),
            "genres": ["Action", "Casual", "Early Access"],
            "screenshots": shots(103, 2),
        },
        {
            "appID": "104",
            "name": "Grand Tactics: Iron Front",
            "short_description": (
                "Command divisions across a historically detailed campaign map. Plan "
                "supply lines, choose doctrines and win turn based battles."
            ),
            "genres": ["Strategy", "Simulation"],
            "screenshots": shots(104, 6),
        },
        # -- rows the pipeline must reject -------------------------------- #
        {
            "appID": "104",  # duplicate appID
            "name": "Grand Tactics: Iron Front (duplicate row)",
            "short_description": "A duplicate row that must be removed by the cleaner entirely.",
            "genres": ["Strategy"],
            "screenshots": shots(104, 6),
        },
        {
            "appID": "105",
            "name": "No Description Game",
            "short_description": "",
            "genres": ["Action"],
            "screenshots": shots(105, 3),
        },
        {
            "appID": "106",
            "name": "Tiny Blurb",
            "short_description": "Short blurb.",
            "genres": ["Action"],
            "screenshots": shots(106, 3),
        },
        {
            "appID": "107",
            "name": "Nihongo Bouken",
            "short_description": "これは日本語だけで書かれた説明です。ゲームの内容を詳しく説明しています。",
            "genres": ["Adventure"],
            "screenshots": shots(107, 3),
        },
        {
            "appID": "108",
            "name": "Spreadsheet Helper Pro",
            "short_description": (
                "A productivity utility for accountants that generates ledgers and "
                "balance sheets from your existing spreadsheet exports."
            ),
            "genres": ["Utilities", "Accounting"],
            "screenshots": shots(108, 3),
        },
        {
            "appID": "109",
            "name": "One Screenshot Only",
            "short_description": (
                "An atmospheric adventure through a decaying city where every choice "
                "reshapes the story and the people you meet along the way."
            ),
            "genres": ["Adventure"],
            "screenshots": shots(109, 1),
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def clean_games_frame(raw_games_frame: pd.DataFrame) -> pd.DataFrame:
    """The synthetic raw frame after the real cleaning pipeline."""
    frame, _ = clean_games(raw_games_frame, require_screenshots=2)
    return frame


@pytest.fixture
def cleaning_report(raw_games_frame: pd.DataFrame):
    """The :class:`CleaningReport` produced for the synthetic raw frame."""
    _, report = clean_games(raw_games_frame, require_screenshots=2)
    return report


# --------------------------------------------------------------------------- #
# Synthetic games + samples + images on a tmp path
# --------------------------------------------------------------------------- #
def _make_synthetic_games(n_games: int, *, seed: int = RNG_SEED) -> pd.DataFrame:
    """Build ``n_games`` synthetic games with plausible multi-hot labels."""
    rng = np.random.default_rng(seed)
    prevalence = np.linspace(0.45, 0.08, len(GENRES))
    rows = []
    for index in range(n_games):
        app_id = f"{9000 + index}"
        labels = (rng.random(len(GENRES)) < prevalence).astype(int)
        if labels.sum() == 0:  # every game must have at least one genre
            labels[rng.integers(len(GENRES))] = 1
        names = [genre for genre, flag in zip(GENRES, labels) if flag]
        row = {
            "app_id": app_id,
            "name": f"Synthetic Game {index}",
            "description_raw": f"Synthetic Game {index} is a demo entry number {index}.",
            "description_clean": (
                f"Synthetic Game {index} is a demonstration entry used by the automated "
                f"tests. It mentions {', '.join(names)} so the text has some signal in it."
            ),
            "description_notitle": (
                f"this game is a demonstration entry used by the automated tests. "
                f"It mentions {', '.join(names)} so the text has some signal in it."
            ),
            "description_masked": (
                "this game is a demonstration entry used by the automated tests. "
                "It mentions [GENRE] so the text has some signal in it."
            ),
            "description_word_count": 24,
            "genres": "|".join(names),
            "n_genres": int(labels.sum()),
            "n_screenshots": 2,
            "screenshot_urls": "\t".join(
                [f"https://example.invalid/{app_id}/a.jpg", f"https://example.invalid/{app_id}/b.jpg"]
            ),
        }
        for genre, flag in zip(GENRES, labels):
            row[label_column(genre)] = int(flag)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_games() -> pd.DataFrame:
    """120 synthetic games (game-level frame with ``y_*`` label columns)."""
    return _make_synthetic_games(120)


@pytest.fixture
def synthetic_samples(synthetic_games: pd.DataFrame) -> pd.DataFrame:
    """Two samples per synthetic game (image paths are not created here)."""
    rows = []
    for app_id in synthetic_games["app_id"]:
        for shot_index in range(2):
            rows.append(
                {
                    "sample_id": f"{app_id}_{shot_index}",
                    "app_id": app_id,
                    "shot_index": shot_index,
                    "screenshot_url": f"https://example.invalid/{app_id}/{shot_index}.jpg",
                    "image_path": f"data/images/{app_id}_{shot_index}.jpg",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_config(tmp_path: Path):
    """A :class:`GameSenseConfig` rooted at ``tmp_path`` with all dirs created."""
    config = replace(CONFIG, paths=Paths(root=tmp_path))
    config.paths.ensure()
    return config


@pytest.fixture
def image_dataset_on_disk(tmp_config, synthetic_games, synthetic_samples):
    """Write small real JPEGs for every sample plus one corrupt and one missing file."""
    from PIL import Image

    rng = np.random.default_rng(RNG_SEED)
    samples = synthetic_samples.copy()
    corrupt_id = samples.at[0, "sample_id"]
    missing_id = samples.at[1, "sample_id"]

    for row in samples.to_dict("records"):
        destination = tmp_config.paths.root / row["image_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if row["sample_id"] == missing_id:
            continue  # deliberately absent
        if row["sample_id"] == corrupt_id:
            destination.write_bytes(b"this is not a valid JPEG payload")
            continue
        pixels = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(destination, format="JPEG", quality=70)

    return tmp_config, synthetic_games, samples, corrupt_id, missing_id


@pytest.fixture
def written_bundle(image_dataset_on_disk):
    """A full on-disk project: games.csv, samples.csv and the three split CSVs."""
    from gamesense.data.splitting import make_splits

    config, games, samples, _, _ = image_dataset_on_disk
    games.to_csv(config.paths.games_csv, index=False)
    samples.to_csv(config.paths.samples_csv, index=False)
    result = make_splits(samples, games, seed=RNG_SEED, config=config)
    for name, frame in result.frames.items():
        frame.to_csv(config.paths.split_csv(name), index=False)
    return config, result


# --------------------------------------------------------------------------- #
# Predictions / tokenizer
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_predictions():
    """``(labels, probabilities)`` where the probabilities are informative."""
    rng = np.random.default_rng(RNG_SEED)
    n_samples = 400
    labels = (rng.random((n_samples, len(GENRES))) < 0.3).astype(np.int8)
    noise = rng.normal(0.0, 0.9, size=labels.shape)
    logits = 2.0 * labels - 1.0 + noise
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return labels, probabilities.astype(np.float32)


@pytest.fixture(scope="session")
def tokenizer():
    """The real DistilBERT tokenizer, or a skip when it is unavailable offline."""
    try:
        from gamesense.data.loader import get_tokenizer

        return get_tokenizer()
    except Exception as exc:  # pragma: no cover - offline machine
        pytest.skip(f"tokenizer unavailable ({type(exc).__name__}: {exc})")
