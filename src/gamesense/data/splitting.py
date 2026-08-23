"""Train / validation / test splitting with game-level leakage prevention."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import CONFIG, GENRES, GameSenseConfig
from ..utils import get_logger
from .preprocessing import label_matrix

__all__ = [
    "SPLIT_NAMES",
    "SplitResult",
    "iterative_stratified_split",
    "split_games",
    "make_splits",
    "verify_no_group_leakage",
    "split_report",
]

LOGGER = get_logger("gamesense.data.splitting")

#: Canonical split names / order used everywhere in the project.
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")

#: Default proportions (70 / 15 / 15).
DEFAULT_PROPORTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)


@dataclass
class SplitResult:
    """Sample-level splits plus the game-level assignment and a summary."""

    frames: dict[str, pd.DataFrame]
    game_assignment: dict[str, str]
    summary: dict[str, Any]

    def __getitem__(self, split: str) -> pd.DataFrame:
        return self.frames[split]

    @property
    def train(self) -> pd.DataFrame:
        return self.frames["train"]

    @property
    def val(self) -> pd.DataFrame:
        return self.frames["val"]

    @property
    def test(self) -> pd.DataFrame:
        return self.frames["test"]


# --------------------------------------------------------------------------- #
# Iterative stratification
# --------------------------------------------------------------------------- #
def iterative_stratified_split(
    labels: np.ndarray,
    proportions: Sequence[float],
    *,
    seed: int = CONFIG.training.seed,
) -> list[np.ndarray]:
    """Distribute multi-label rows across folds keeping per-label ratios.

    Returns one index array per fold, in the order of *proportions* (Sechidis et al., 2011).
    """
    matrix = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("labels must be a 2-D matrix")
    ratios = np.asarray(proportions, dtype=np.float64)
    if not np.isclose(ratios.sum(), 1.0, atol=1e-6):
        raise ValueError(f"proportions must sum to 1, got {ratios.sum()}")

    rng = np.random.default_rng(seed)
    n_rows, n_labels = matrix.shape
    n_folds = len(ratios)

    # Desired number of rows / per-label positives per fold.
    desired_rows = ratios * n_rows
    desired_positives = np.outer(ratios, matrix.sum(axis=0)).astype(np.float64)

    remaining = np.ones(n_rows, dtype=bool)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    remaining_positives = matrix.sum(axis=0).astype(np.float64)

    while remaining.any():
        # Label with fewest remaining positives (ignoring exhausted labels).
        candidate_labels = np.flatnonzero(remaining_positives > 0)
        if candidate_labels.size == 0:
            # Only label-free rows remain: give them to the neediest folds.
            leftovers = np.flatnonzero(remaining)
            rng.shuffle(leftovers)
            for row in leftovers:
                fold = int(np.argmax(desired_rows))
                folds[fold].append(int(row))
                desired_rows[fold] -= 1
                remaining[row] = False
            break

        label = int(candidate_labels[np.argmin(remaining_positives[candidate_labels])])
        rows = np.flatnonzero(remaining & (matrix[:, label] == 1))
        rng.shuffle(rows)
        for row in rows:
            # Fold that most needs a positive of this label; ties broken by the
            # fold that most needs rows overall, then randomly.
            need = desired_positives[:, label]
            best = np.flatnonzero(need == need.max())
            if best.size > 1:
                row_need = desired_rows[best]
                best = best[np.flatnonzero(row_need == row_need.max())]
            fold = int(best[0]) if best.size == 1 else int(rng.choice(best))

            folds[fold].append(int(row))
            remaining[row] = False
            desired_rows[fold] -= 1
            positives = matrix[row]
            desired_positives[fold] -= positives
            remaining_positives -= positives

        remaining_positives[label] = max(0.0, remaining_positives[label])

    return [np.array(sorted(fold), dtype=np.int64) for fold in folds]


# --------------------------------------------------------------------------- #
# Game-level splitting
# --------------------------------------------------------------------------- #
def split_games(
    games: pd.DataFrame,
    *,
    proportions: Sequence[float] = DEFAULT_PROPORTIONS,
    seed: int = CONFIG.training.seed,
    classes: Sequence[str] = GENRES,
    stratify: bool = True,
) -> dict[str, str]:
    """Assign every game to exactly one split, returning ``{app_id: split_name}``."""
    if "app_id" not in games.columns:
        raise KeyError("games frame must contain an 'app_id' column")
    frame = games.reset_index(drop=True)

    if stratify:
        folds = iterative_stratified_split(
            label_matrix(frame, classes=classes), proportions, seed=seed
        )
    else:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(frame))
        sizes = (np.asarray(proportions, dtype=float) * len(frame)).astype(int)
        sizes[-1] = len(frame) - sizes[:-1].sum()
        folds, start = [], 0
        for size in sizes:
            folds.append(np.sort(permutation[start : start + size]))
            start += size

    assignment: dict[str, str] = {}
    for split_name, indices in zip(SPLIT_NAMES, folds):
        for index in indices:
            assignment[str(frame.at[int(index), "app_id"])] = split_name
    return assignment


def make_splits(
    samples: pd.DataFrame,
    games: pd.DataFrame,
    *,
    proportions: Sequence[float] = DEFAULT_PROPORTIONS,
    seed: int = CONFIG.training.seed,
    classes: Sequence[str] = GENRES,
    stratify: bool = True,
    config: GameSenseConfig = CONFIG,
) -> SplitResult:
    """Produce sample-level train/val/test frames split by ``app_id``."""
    assignment = split_games(
        games, proportions=proportions, seed=seed, classes=classes, stratify=stratify
    )
    frame = samples.copy()
    frame["split"] = frame["app_id"].astype(str).map(assignment)
    unassigned = int(frame["split"].isna().sum())
    if unassigned:
        raise RuntimeError(
            f"{unassigned} samples reference an app_id that is absent from the games frame"
        )

    frames = {
        split: frame[frame["split"] == split].reset_index(drop=True) for split in SPLIT_NAMES
    }
    leakage = verify_no_group_leakage(frames)
    if not leakage["ok"]:  # pragma: no cover - guarded by tests
        raise RuntimeError(f"game-level leakage detected: {leakage}")

    summary = split_report(
        frames, games=games, proportions=proportions, seed=seed, classes=classes, config=config
    )
    summary["leakage_check"] = leakage
    LOGGER.info(
        "Splits (games / samples): train %d/%d, val %d/%d, test %d/%d",
        summary["splits"]["train"]["n_games"],
        summary["splits"]["train"]["n_samples"],
        summary["splits"]["val"]["n_games"],
        summary["splits"]["val"]["n_samples"],
        summary["splits"]["test"]["n_games"],
        summary["splits"]["test"]["n_samples"],
    )
    return SplitResult(frames=frames, game_assignment=assignment, summary=summary)


# --------------------------------------------------------------------------- #
# Verification / reporting
# --------------------------------------------------------------------------- #
def verify_no_group_leakage(
    frames: dict[str, pd.DataFrame], *, group_column: str = "app_id"
) -> dict[str, Any]:
    """Assert that no ``group_column`` value appears in more than one split.

    Returns a report whose ``"ok"`` key is False if any group or sample id spans two splits.
    """
    groups = {name: set(frame[group_column].astype(str)) for name, frame in frames.items()}
    overlaps: dict[str, list[str]] = {}
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = groups[left] & groups[right]
            if shared:
                overlaps[f"{left}|{right}"] = sorted(shared)[:20]

    sample_overlaps: dict[str, int] = {}
    if all("sample_id" in frame.columns for frame in frames.values()):
        ids = {name: set(frame["sample_id"].astype(str)) for name, frame in frames.items()}
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                shared = ids[left] & ids[right]
                if shared:
                    sample_overlaps[f"{left}|{right}"] = len(shared)

    return {
        "ok": not overlaps and not sample_overlaps,
        "group_column": group_column,
        "n_groups": {name: len(value) for name, value in groups.items()},
        "group_overlaps": overlaps,
        "sample_id_overlaps": sample_overlaps,
    }


def split_report(
    frames: dict[str, pd.DataFrame],
    *,
    games: pd.DataFrame,
    proportions: Sequence[float],
    seed: int,
    classes: Sequence[str] = GENRES,
    config: GameSenseConfig = CONFIG,
) -> dict[str, Any]:
    """Build the JSON summary stored next to the split CSVs."""
    game_labels = games.set_index(games["app_id"].astype(str))
    report: dict[str, Any] = {
        "seed": seed,
        "proportions": list(proportions),
        "classes": list(classes),
        "n_games_total": int(len(games)),
        "n_samples_total": int(sum(len(f) for f in frames.values())),
        "screenshots_per_game": config.dataset.screenshots_per_game,
        "splits": {},
    }
    for name, frame in frames.items():
        app_ids = frame["app_id"].astype(str).unique()
        subset = game_labels.loc[app_ids]
        matrix = label_matrix(subset, classes=classes)
        n_games = len(app_ids)
        report["splits"][name] = {
            "n_games": int(n_games),
            "n_samples": int(len(frame)),
            "game_fraction": round(n_games / max(1, len(games)), 4),
            "genre_counts": {
                genre: int(count) for genre, count in zip(classes, matrix.sum(axis=0))
            },
            "genre_prevalence": {
                genre: round(float(count) / max(1, n_games), 4)
                for genre, count in zip(classes, matrix.sum(axis=0))
            },
            "labels_per_game_mean": round(float(matrix.sum(axis=1).mean()) if n_games else 0.0, 3),
        }
    return report
