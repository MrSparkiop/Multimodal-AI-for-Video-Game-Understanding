"""Tests for :mod:`gamesense.data.splitting`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamesense.config import GENRES
from gamesense.data.preprocessing import label_matrix, positive_counts
from gamesense.data.splitting import (
    DEFAULT_PROPORTIONS,
    SPLIT_NAMES,
    iterative_stratified_split,
    make_splits,
    split_games,
    verify_no_group_leakage,
)

#: Seed used for the split tests; any fixed value makes the run reproducible.
SEED = 20260821

# : Rows in the purpose-built stratification matrix.
N_STRATIFY_ROWS = 300

#: Per-label prevalence of that matrix; the last entry is deliberately rare so
#: the prevalence assertions are actually testing something hard.
STRATIFY_PREVALENCE = (0.50, 0.40, 0.30, 0.20, 0.10, 0.03)

#: Tolerance on the fold sizes as a fraction of the requested proportion.
SIZE_TOLERANCE = 0.02

#: Tolerance on the per-label prevalence difference between a fold and the whole.
PREVALENCE_TOLERANCE = 0.05

#: A genre needs roughly ``1 / min(proportions)`` positives before *every* split
#: can be expected to contain one; 10 comfortably exceeds 1 / 0.15.
MIN_SUPPORT_FOR_EVERY_SPLIT = 10


def _label_matrix_with_a_rare_class(
    *, n_rows: int = N_STRATIFY_ROWS, seed: int = SEED
) -> np.ndarray:
    """Build a synthetic multi-label matrix with a deliberately rare class."""
    rng = np.random.default_rng(seed)
    draws = rng.random((n_rows, len(STRATIFY_PREVALENCE)))
    return (draws < np.asarray(STRATIFY_PREVALENCE)).astype(np.int64)


# --------------------------------------------------------------------------- #
# iterative_stratified_split
# --------------------------------------------------------------------------- #
def test_iterative_split_is_an_exact_partition() -> None:
    """Every row must appear in exactly one fold -- nothing lost or duplicated."""
    matrix = _label_matrix_with_a_rare_class()
    folds = iterative_stratified_split(matrix, DEFAULT_PROPORTIONS, seed=SEED)

    combined = np.concatenate(folds)
    assert np.array_equal(np.sort(combined), np.arange(len(matrix)))
    assert len(combined) == len(np.unique(combined))  # no duplicates
    for left in range(len(folds)):
        for right in range(left + 1, len(folds)):
            assert not set(folds[left].tolist()) & set(folds[right].tolist())


def test_iterative_split_sizes_follow_the_requested_proportions() -> None:
    matrix = _label_matrix_with_a_rare_class()
    folds = iterative_stratified_split(matrix, DEFAULT_PROPORTIONS, seed=SEED)
    for fold, proportion in zip(folds, DEFAULT_PROPORTIONS):
        assert len(fold) / len(matrix) == pytest.approx(proportion, abs=SIZE_TOLERANCE)


def test_iterative_split_preserves_per_label_prevalence() -> None:
    """Including for the rare class -- that is the point of stratifying."""
    matrix = _label_matrix_with_a_rare_class()
    global_prevalence = matrix.mean(axis=0)
    folds = iterative_stratified_split(matrix, DEFAULT_PROPORTIONS, seed=SEED)

    worst = 0.0
    for fold in folds:
        deviation = np.abs(matrix[fold].mean(axis=0) - global_prevalence)
        worst = max(worst, float(deviation.max()))
    assert worst <= PREVALENCE_TOLERANCE, f"max prevalence drift {worst:.4f}"
    # The rare class must actually be present in every fold, not merely close.
    rare = len(STRATIFY_PREVALENCE) - 1
    assert all(matrix[fold][:, rare].sum() >= 1 for fold in folds)


def test_iterative_split_rejects_proportions_that_do_not_sum_to_one() -> None:
    matrix = _label_matrix_with_a_rare_class()
    with pytest.raises(ValueError, match="sum to 1"):
        iterative_stratified_split(matrix, (0.5, 0.4), seed=SEED)


def test_iterative_split_rejects_a_non_matrix() -> None:
    with pytest.raises(ValueError, match="2-D"):
        iterative_stratified_split(np.array([1, 0, 1]), DEFAULT_PROPORTIONS, seed=SEED)


def test_iterative_split_is_deterministic_per_seed() -> None:
    """Same seed -> identical folds; a different seed -> a different assignment."""
    matrix = _label_matrix_with_a_rare_class()
    first = iterative_stratified_split(matrix, DEFAULT_PROPORTIONS, seed=SEED)
    again = iterative_stratified_split(matrix, DEFAULT_PROPORTIONS, seed=SEED)
    other = iterative_stratified_split(matrix, DEFAULT_PROPORTIONS, seed=SEED + 1)

    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other))


# --------------------------------------------------------------------------- #
# The critical leakage guarantee
# --------------------------------------------------------------------------- #
def test_no_game_or_sample_leakage_across_splits(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    """No ``app_id`` and no ``sample_id`` may appear in two splits."""
    result = make_splits(synthetic_samples, synthetic_games, seed=SEED)

    assert verify_no_group_leakage(result.frames)["ok"] is True

    app_ids = {name: set(frame["app_id"].astype(str)) for name, frame in result.frames.items()}
    sample_ids = {
        name: set(frame["sample_id"].astype(str)) for name, frame in result.frames.items()
    }
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            assert not app_ids[left] & app_ids[right], f"game leak {left}/{right}"
            assert not sample_ids[left] & sample_ids[right], f"sample leak {left}/{right}"

    # Nothing was lost or duplicated on the way, either.
    assert sum(len(ids) for ids in sample_ids.values()) == len(synthetic_samples)
    assert set().union(*app_ids.values()) == set(synthetic_games["app_id"].astype(str))
    # Both screenshots of a game are in the same split, so #samples == 2 x #games.
    for name in SPLIT_NAMES:
        assert len(sample_ids[name]) == 2 * len(app_ids[name])


def test_verify_no_group_leakage_detects_an_injected_leak(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    """The auditing helper must fail loudly when leakage really is present."""
    result = make_splits(synthetic_samples, synthetic_games, seed=SEED)
    poisoned = dict(result.frames)
    poisoned["test"] = pd.concat(
        [result.frames["test"], result.frames["train"].head(1)], ignore_index=True
    )

    report = verify_no_group_leakage(poisoned)
    assert report["ok"] is False
    assert report["group_overlaps"] or report["sample_id_overlaps"]


def test_make_splits_rejects_a_sample_with_an_unknown_app_id(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    orphan = synthetic_samples.iloc[[0]].copy()
    orphan["app_id"] = "does-not-exist"
    orphan["sample_id"] = "orphan_0"
    polluted = pd.concat([synthetic_samples, orphan], ignore_index=True)

    with pytest.raises(RuntimeError, match="absent from the games frame"):
        make_splits(polluted, synthetic_games, seed=SEED)


def test_split_games_requires_an_app_id_column(synthetic_games: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="app_id"):
        split_games(synthetic_games.drop(columns=["app_id"]), seed=SEED)


# --------------------------------------------------------------------------- #
# Summary structure
# --------------------------------------------------------------------------- #
def test_split_summary_has_the_documented_structure(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    summary = make_splits(synthetic_samples, synthetic_games, seed=SEED).summary

    assert set(summary) >= {
        "seed",
        "proportions",
        "classes",
        "n_games_total",
        "n_samples_total",
        "screenshots_per_game",
        "splits",
        "leakage_check",
    }
    assert summary["seed"] == SEED
    assert summary["classes"] == list(GENRES)
    assert summary["n_games_total"] == len(synthetic_games)
    assert summary["n_samples_total"] == len(synthetic_samples)
    assert set(summary["splits"]) == set(SPLIT_NAMES)

    for name in SPLIT_NAMES:
        entry = summary["splits"][name]
        assert set(entry) >= {
            "n_games",
            "n_samples",
            "game_fraction",
            "genre_counts",
            "genre_prevalence",
            "labels_per_game_mean",
        }
        assert set(entry["genre_counts"]) == set(GENRES)
        assert entry["n_games"] > 0 and entry["n_samples"] > 0


def test_split_summary_counts_reconcile_with_the_input(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    """Per-split games / samples / genre counts must add up to the global ones."""
    summary = make_splits(synthetic_samples, synthetic_games, seed=SEED).summary
    splits = summary["splits"]

    assert sum(splits[name]["n_games"] for name in SPLIT_NAMES) == len(synthetic_games)
    assert sum(splits[name]["n_samples"] for name in SPLIT_NAMES) == len(synthetic_samples)

    global_counts = positive_counts(synthetic_games)
    for genre in GENRES:
        assert sum(splits[name]["genre_counts"][genre] for name in SPLIT_NAMES) == (
            global_counts[genre]
        )


def test_split_frames_and_summary_agree(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    result = make_splits(synthetic_samples, synthetic_games, seed=SEED)
    for name in SPLIT_NAMES:
        frame = result.frames[name]
        assert len(frame) == result.summary["splits"][name]["n_samples"]
        assert frame["app_id"].nunique() == result.summary["splits"][name]["n_games"]
        assert set(frame["split"]) == {name}
    # The dataclass accessors must point at the same frames.
    assert result.train is result["train"] and result.val is result["val"]
    assert result.test is result.frames["test"]


# --------------------------------------------------------------------------- #
# Reproducibility and rare-genre coverage
# --------------------------------------------------------------------------- #
def test_game_assignment_is_reproducible_from_the_seed(
    synthetic_games: pd.DataFrame,
) -> None:
    first = split_games(synthetic_games, seed=SEED)
    again = split_games(synthetic_games, seed=SEED)
    other = split_games(synthetic_games, seed=SEED + 1)

    assert first == again
    assert first != other
    assert set(first) == set(synthetic_games["app_id"].astype(str))
    assert set(first.values()) == set(SPLIT_NAMES)


def test_make_splits_is_reproducible_from_the_seed(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    first = make_splits(synthetic_samples, synthetic_games, seed=SEED)
    again = make_splits(synthetic_samples, synthetic_games, seed=SEED)
    other = make_splits(synthetic_samples, synthetic_games, seed=SEED + 1)

    assert first.game_assignment == again.game_assignment
    assert first.game_assignment != other.game_assignment
    for name in SPLIT_NAMES:
        assert first.frames[name]["sample_id"].tolist() == (
            again.frames[name]["sample_id"].tolist()
        )


def test_every_split_has_positives_for_every_well_supported_genre(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    """Stratification exists so macro-F1 is computable on every split."""
    result = make_splits(synthetic_samples, synthetic_games, seed=SEED, stratify=True)
    global_counts = positive_counts(synthetic_games)
    supported = [
        genre for genre, count in global_counts.items() if count >= MIN_SUPPORT_FOR_EVERY_SPLIT
    ]
    assert supported, "the synthetic fixture should support several genres"

    for name in SPLIT_NAMES:
        counts = result.summary["splits"][name]["genre_counts"]
        for genre in supported:
            assert counts[genre] >= 1, f"{genre} has no positives in {name}"


def test_unstratified_splitting_still_partitions_the_games(
    synthetic_samples: pd.DataFrame, synthetic_games: pd.DataFrame
) -> None:
    """``stratify=False`` is the ablation baseline; it must stay leakage-free."""
    result = make_splits(synthetic_samples, synthetic_games, seed=SEED, stratify=False)
    assert verify_no_group_leakage(result.frames)["ok"] is True
    assert sum(len(frame) for frame in result.frames.values()) == len(synthetic_samples)
    matrix = label_matrix(synthetic_games)
    assert matrix.shape[0] == result.summary["n_games_total"]
