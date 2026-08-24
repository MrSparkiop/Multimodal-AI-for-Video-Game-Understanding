"""Build the GameSense dataset: acquire, clean, subsample, download, split."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import pandas as pd

from gamesense.config import CONFIG, EXCLUDED_GENRE_REASONS, GENRES
from gamesense.data.acquisition import (
    download_raw_dataset,
    download_screenshots,
    expand_to_samples,
    load_raw_games,
    select_games,
)
from gamesense.data.preprocessing import (
    add_text_variants,
    clean_games,
    image_average_hash,
    positive_counts,
)
from gamesense.data.splitting import DEFAULT_PROPORTIONS, make_splits
from gamesense.utils import get_logger, human_time, save_json, set_seed

LOGGER = get_logger("gamesense.prepare_data")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the GameSense dataset (metadata + screenshots + splits).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-games", type=int, default=CONFIG.dataset.max_games,
                        help="maximum number of games to keep after subsampling")
    parser.add_argument("--shots-per-game", type=int, default=CONFIG.dataset.screenshots_per_game,
                        help="screenshots downloaded per game")
    parser.add_argument("--seed", type=int, default=CONFIG.training.seed,
                        help="random seed for subsampling and splitting")
    parser.add_argument("--rare-boost", type=float, default=0.30,
                        help="fraction of the sampling budget spent raising rare-genre support")
    parser.add_argument("--description-field", default="short_description",
                        choices=("short_description", "detailed_description"),
                        help="raw column used as the game description")
    parser.add_argument("--proportions", type=float, nargs=3, default=list(DEFAULT_PROPORTIONS),
                        metavar=("TRAIN", "VAL", "TEST"), help="split proportions (must sum to 1)")
    parser.add_argument("--workers", type=int, default=CONFIG.dataset.download_workers,
                        help="parallel screenshot downloads")
    parser.add_argument("--skip-download", action="store_true",
                        help="do not fetch screenshots (metadata-only dry run)")
    parser.add_argument("--force-download", action="store_true",
                        help="re-download the raw parquet even if present")
    parser.add_argument("--no-stratify", action="store_true",
                        help="use a plain random split instead of iterative stratification")
    parser.add_argument("--keep-duplicate-images", action="store_true",
                        help="skip perceptual-hash de-duplication of screenshots")
    parser.add_argument("--keep-orphan-images", action="store_true",
                        help="keep cached images that the final dataset no longer references")
    return parser.parse_args(argv)


def prune_orphan_images(samples: pd.DataFrame, *, image_dir: Path, root: Path) -> dict[str, Any]:
    """Delete cached screenshots the final dataset no longer references."""
    referenced = {(root / str(path)).resolve() for path in samples["image_path"]}
    removed, freed = 0, 0
    for path in image_dir.glob("*.jpg"):
        if path.resolve() in referenced:
            continue
        freed += path.stat().st_size
        path.unlink()
        removed += 1
    for leftover in image_dir.glob("*.part"):
        leftover.unlink(missing_ok=True)
    report = {
        "referenced": len(referenced),
        "orphans_removed": removed,
        "megabytes_freed": round(freed / 1e6, 2),
    }
    LOGGER.info("Image cache pruned: %s", report)
    return report


# --------------------------------------------------------------------------- #
# Image validation
# --------------------------------------------------------------------------- #
def validate_images(
    samples: pd.DataFrame,
    *,
    root: Path,
    deduplicate: bool = True,
    workers: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop missing, corrupt and near-duplicate images; returns ``(surviving, report)``."""
    paths = [root / str(path) for path in samples["image_path"]]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        hashes = list(pool.map(image_average_hash, paths))

    unreadable = [index for index, value in enumerate(hashes) if value is None]
    keep = np.ones(len(samples), dtype=bool)
    keep[unreadable] = False

    duplicates: list[int] = []
    if deduplicate:
        seen: dict[str, int] = {}
        for index, value in enumerate(hashes):
            if value is None:
                continue
            if value in seen:
                duplicates.append(index)
                keep[index] = False
            else:
                seen[value] = index

    surviving = samples.loc[keep].reset_index(drop=True)
    surviving = surviving.assign(
        image_hash=[hashes[i] for i in np.flatnonzero(keep)]
    )
    report = {
        "checked": int(len(samples)),
        "unreadable_or_missing": len(unreadable),
        "near_duplicates_removed": len(duplicates),
        "surviving": int(len(surviving)),
        "deduplicated": bool(deduplicate),
        "seconds": round(time.perf_counter() - started, 1),
    }
    LOGGER.info("Image validation: %s", report)
    return surviving, report


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not np.isclose(sum(args.proportions), 1.0, atol=1e-6):
        LOGGER.error("Split proportions must sum to 1 (got %s)", args.proportions)
        return 2

    config = replace(
        CONFIG,
        dataset=replace(
            CONFIG.dataset,
            max_games=args.max_games,
            screenshots_per_game=args.shots_per_game,
            download_workers=args.workers,
        ),
    )
    config.paths.ensure()
    set_seed(args.seed)
    started = time.perf_counter()
    steps: dict[str, Any] = {}

    # -- 1. raw metadata --------------------------------------------------- #
    LOGGER.info("STEP 1/7  Acquire raw metadata")
    raw_path = download_raw_dataset(config=config, force=args.force_download)
    raw = load_raw_games(raw_path, config=config)
    steps["raw"] = {
        "path": str(raw_path),
        "megabytes": round(raw_path.stat().st_size / 1e6, 1),
        "rows": int(len(raw)),
        "columns": list(raw.columns),
    }

    # -- 2. cleaning ------------------------------------------------------- #
    LOGGER.info("STEP 2/7  Clean and normalise")
    # The leakage-controlled description variants are the most expensive step in the pipeline,
    # so they are computed after subsampling (step 3) rather than on all ~124k raw rows.
    cleaned, cleaning = clean_games(
        raw,
        dataset=config.dataset,
        description_field=args.description_field,
        require_screenshots=args.shots_per_game,
        compute_text_variants=False,
    )
    steps["cleaning"] = cleaning.as_dict()
    if cleaned.empty:
        LOGGER.error("Cleaning removed every row - nothing to do.")
        return 1

    # -- 3. subsampling ---------------------------------------------------- #
    LOGGER.info("STEP 3/7  Subsample games")
    selected, sampling = select_games(
        cleaned,
        max_games=args.max_games,
        seed=args.seed,
        rare_boost_fraction=args.rare_boost,
    )
    selected, dropped_by_variants = add_text_variants(selected, dataset=config.dataset)
    sampling["text_variants"] = dropped_by_variants
    sampling["selected_after_text_variants"] = int(len(selected))
    steps["sampling"] = sampling

    # -- 4. expand to samples --------------------------------------------- #
    LOGGER.info("STEP 4/7  Expand games to (game, screenshot) samples")
    samples = expand_to_samples(selected, shots_per_game=args.shots_per_game, config=config)

    # -- 5. screenshots ---------------------------------------------------- #
    if args.skip_download:
        LOGGER.warning("STEP 5/7  Skipping screenshot download (--skip-download)")
        steps["download"] = {"skipped": True}
        steps["image_validation"] = {"skipped": True}
    else:
        LOGGER.info("STEP 5/7  Download %d screenshots", len(samples))
        samples, download_stats = download_screenshots(samples, config=config, workers=args.workers)
        steps["download"] = download_stats.as_dict()
        samples, image_report = validate_images(
            samples,
            root=config.paths.root,
            deduplicate=not args.keep_duplicate_images,
            workers=args.workers,
        )
        steps["image_validation"] = image_report

    if samples.empty:
        LOGGER.error("No usable screenshots remain - cannot build an image dataset.")
        return 1

    # -- 6. reconcile games with surviving samples ------------------------- #
    LOGGER.info("STEP 6/7  Reconcile and write processed tables")
    surviving_ids = set(samples["app_id"].astype(str))
    before = len(selected)
    games = selected[selected["app_id"].astype(str).isin(surviving_ids)].reset_index(drop=True)
    steps["reconciliation"] = {
        "games_before": int(before),
        "games_after": int(len(games)),
        "games_dropped_without_images": int(before - len(games)),
        "samples": int(len(samples)),
        "samples_per_game_mean": round(len(samples) / max(1, len(games)), 3),
    }

    support = positive_counts(games)
    under_supported = {
        genre: count
        for genre, count in support.items()
        if count < config.dataset.min_genre_support
    }
    if under_supported:
        LOGGER.warning(
            "These genres fall below min_genre_support=%d: %s. "
            "Increase --max-games / --rare-boost, or remove them from config.GENRES.",
            config.dataset.min_genre_support,
            under_supported,
        )

    games.to_csv(config.paths.games_csv, index=False)
    samples.to_csv(config.paths.samples_csv, index=False)
    save_json(
        {
            "classes": list(GENRES),
            "support_games": support,
            "prevalence_games": {
                genre: round(count / max(1, len(games)), 4) for genre, count in support.items()
            },
            "min_genre_support": config.dataset.min_genre_support,
            "under_supported": under_supported,
            "excluded_genres": EXCLUDED_GENRE_REASONS,
            "description_field": args.description_field,
        },
        config.paths.label_space,
    )

    # -- 7. splits --------------------------------------------------------- #
    LOGGER.info("STEP 7/7  Split by game id (no screenshot of a game crosses splits)")
    split_result = make_splits(
        samples,
        games,
        proportions=tuple(args.proportions),
        seed=args.seed,
        stratify=not args.no_stratify,
        config=config,
    )
    for name, frame in split_result.frames.items():
        frame.to_csv(config.paths.split_csv(name), index=False)
    save_json(split_result.summary, config.paths.split_summary)

    steps["splits"] = split_result.summary
    if args.keep_orphan_images or args.skip_download:
        steps["image_cache_pruning"] = {"skipped": True}
    else:
        steps["image_cache_pruning"] = prune_orphan_images(
            samples, image_dir=config.paths.images, root=config.paths.root
        )
    steps["arguments"] = vars(args)
    steps["total_seconds"] = round(time.perf_counter() - started, 1)
    save_json(steps, config.paths.cleaning_report)

    # -- summary ----------------------------------------------------------- #
    LOGGER.info("=" * 72)
    LOGGER.info("Dataset ready in %s", human_time(steps["total_seconds"]))
    LOGGER.info("  games   : %d", len(games))
    LOGGER.info("  samples : %d (%.2f screenshots/game)", len(samples),
                steps["reconciliation"]["samples_per_game_mean"])
    for name in ("train", "val", "test"):
        info = split_result.summary["splits"][name]
        LOGGER.info("  %-7s : %5d games / %5d samples", name, info["n_games"], info["n_samples"])
    LOGGER.info("  genre support (games): %s", support)
    LOGGER.info("  leakage check       : %s", split_result.summary["leakage_check"]["ok"])
    LOGGER.info("  report              : %s", config.paths.cleaning_report)
    LOGGER.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
