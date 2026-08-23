"""Dataset acquisition: metadata download, subsampling and screenshot fetching."""

from __future__ import annotations

import io
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import CONFIG, GENRES, DatasetConfig, GameSenseConfig
from ..utils import get_logger
from .preprocessing import label_matrix

__all__ = [
    "RAW_COLUMNS",
    "DownloadStats",
    "download_raw_dataset",
    "load_raw_games",
    "select_games",
    "expand_to_samples",
    "download_screenshots",
]

LOGGER = get_logger("gamesense.data.acquisition")

#: Only the columns the project actually needs are read from the 190 MB Parquet
#: file -- reading all 41 columns would waste memory for no benefit.
RAW_COLUMNS: tuple[str, ...] = (
    "appID",
    "name",
    "release_date",
    "short_description",
    "detailed_description",
    "genres",
    "categories",
    "screenshots",
    "required_age",
    "notes",
    "header_image",
    "positive",
    "negative",
)


@dataclass
class DownloadStats:
    """Outcome of a screenshot download pass."""

    requested: int = 0
    already_present: int = 0
    downloaded: int = 0
    failed: int = 0
    bytes_written: int = 0
    seconds: float = 0.0
    errors: dict[str, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "already_present": self.already_present,
            "downloaded": self.downloaded,
            "failed": self.failed,
            "megabytes_written": round(self.bytes_written / 1e6, 2),
            "seconds": round(self.seconds, 1),
            "errors_by_type": self.errors or {},
        }


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def download_raw_dataset(
    *,
    config: GameSenseConfig = CONFIG,
    force: bool = False,
) -> Path:
    """Ensure the raw Parquet snapshot exists locally and return its path."""
    target = config.paths.raw_parquet
    if target.is_file() and not force:
        LOGGER.info("Raw dataset already present: %s (%.1f MB)", target, target.stat().st_size / 1e6)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    cfg = config.dataset
    LOGGER.info("Downloading %s / %s ...", cfg.hf_repo_id, cfg.hf_filename)
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            repo_id=cfg.hf_repo_id,
            repo_type=cfg.hf_repo_type,
            filename=cfg.hf_filename,
        )
        shutil.copy(cached, target)
    except Exception as exc:  # network failure, missing dependency, ...
        raise FileNotFoundError(
            f"Could not obtain the raw dataset automatically ({exc!r}).\n"
            f"Download '{cfg.hf_filename}' from "
            f"https://huggingface.co/datasets/{cfg.hf_repo_id} manually and save it as\n"
            f"  {target}\n"
            "then re-run this script (see data/README.md)."
        ) from exc
    LOGGER.info("Saved raw dataset to %s (%.1f MB)", target, target.stat().st_size / 1e6)
    return target


def load_raw_games(
    path: str | Path | None = None,
    *,
    config: GameSenseConfig = CONFIG,
    columns: Sequence[str] = RAW_COLUMNS,
) -> pd.DataFrame:
    """Read the raw Parquet snapshot into a dataframe (selected columns only)."""
    source = Path(path) if path is not None else config.paths.raw_parquet
    if not source.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {source}. Run scripts/prepare_data.py "
            "or see data/README.md."
        )
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(source).schema_arrow.names)
    wanted = [column for column in columns if column in available]
    missing = [column for column in columns if column not in available]
    if missing:
        LOGGER.warning("Columns absent from the Parquet file and skipped: %s", missing)
    frame = pq.read_table(source, columns=wanted).to_pandas()
    LOGGER.info("Loaded %d raw rows x %d columns from %s", len(frame), frame.shape[1], source.name)
    return frame


# --------------------------------------------------------------------------- #
# Subsampling
# --------------------------------------------------------------------------- #
def select_games(
    games: pd.DataFrame,
    *,
    max_games: int | None = None,
    seed: int = CONFIG.training.seed,
    rare_boost_fraction: float = 0.30,
    classes: Sequence[str] = GENRES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Draw a subsample of at most *max_games* games.

    Returns ``(subsample, info)``, where *info* records both the pool and the realised genre
    distribution.
    """
    budget = max_games if max_games is not None else CONFIG.dataset.max_games
    rng = np.random.default_rng(seed)
    frame = games.reset_index(drop=True)
    pool_counts = {
        genre: int(count)
        for genre, count in zip(classes, label_matrix(frame, classes=classes).sum(axis=0))
    }

    if budget >= len(frame):
        info = {
            "strategy": "all_eligible_games_kept",
            "requested": budget,
            "eligible_pool": int(len(frame)),
            "selected": int(len(frame)),
            "pool_genre_counts": pool_counts,
            "selected_genre_counts": pool_counts,
            "rare_boost_fraction": 0.0,
            "seed": seed,
        }
        return frame, info

    n_random = int(round(budget * (1.0 - rare_boost_fraction)))
    n_boost = budget - n_random

    # ---- part 1: uniform random draw ----------------------------------- #
    permutation = rng.permutation(len(frame))
    chosen: list[int] = list(permutation[:n_random])
    chosen_set = set(chosen)

    # ---- part 2: rarest-class-first round robin ------------------------ #
    matrix = label_matrix(frame, classes=classes)
    per_class_pool: dict[str, list[int]] = {}
    for index, genre in enumerate(classes):
        candidates = np.flatnonzero(matrix[:, index] > 0)
        rng.shuffle(candidates)
        per_class_pool[genre] = [int(i) for i in candidates if int(i) not in chosen_set]

    cursor: dict[str, int] = defaultdict(int)
    selected_counts = Counter()
    for index, genre in enumerate(classes):
        selected_counts[genre] = int(matrix[chosen, index].sum()) if chosen else 0

    added = 0
    stalled = 0
    while added < n_boost and stalled < len(classes):
        # Always extend the currently rarest class first.
        order = sorted(classes, key=lambda g: (selected_counts[g], pool_counts[g]))
        progressed = False
        for genre in order:
            pool = per_class_pool[genre]
            while cursor[genre] < len(pool):
                candidate = pool[cursor[genre]]
                cursor[genre] += 1
                if candidate in chosen_set:
                    continue
                chosen.append(candidate)
                chosen_set.add(candidate)
                for j, other in enumerate(classes):
                    if matrix[candidate, j] > 0:
                        selected_counts[other] += 1
                added += 1
                progressed = True
                break
            if added >= n_boost:
                break
        stalled = 0 if progressed else stalled + 1

    subsample = frame.iloc[sorted(chosen)].reset_index(drop=True)
    realised = {
        genre: int(count)
        for genre, count in zip(classes, label_matrix(subsample, classes=classes).sum(axis=0))
    }
    info = {
        "strategy": "hybrid_random_plus_rare_class_round_robin",
        "requested": budget,
        "eligible_pool": int(len(frame)),
        "selected": int(len(subsample)),
        "n_uniform_random": n_random,
        "n_rare_boost": int(added),
        "rare_boost_fraction": rare_boost_fraction,
        "seed": seed,
        "pool_genre_counts": pool_counts,
        "pool_genre_fractions": {
            genre: round(count / max(1, len(frame)), 4) for genre, count in pool_counts.items()
        },
        "selected_genre_counts": realised,
        "selected_genre_fractions": {
            genre: round(count / max(1, len(subsample)), 4) for genre, count in realised.items()
        },
    }
    LOGGER.info(
        "Selected %d / %d eligible games (%d uniform + %d rare-boosted)",
        len(subsample),
        len(frame),
        n_random,
        added,
    )
    return subsample, info


# --------------------------------------------------------------------------- #
# Screenshots
# --------------------------------------------------------------------------- #
def _sample_id(app_id: str, shot_index: int) -> str:
    return f"{app_id}_{shot_index}"


def expand_to_samples(
    games: pd.DataFrame,
    *,
    shots_per_game: int | None = None,
    config: GameSenseConfig = CONFIG,
) -> pd.DataFrame:
    """Expand one row per game into one row per ``(game, screenshot)`` sample.

    ``app_id`` is carried through because it is the grouping key that keeps a game inside one
    split.
    """
    limit = shots_per_game if shots_per_game is not None else config.dataset.screenshots_per_game
    rows: list[dict[str, Any]] = []
    image_root = config.paths.images
    for record in games.to_dict("records"):
        urls = [u for u in str(record.get("screenshot_urls", "")).split("\t") if u]
        for shot_index, url in enumerate(urls[:limit]):
            sample_id = _sample_id(record["app_id"], shot_index)
            rows.append(
                {
                    "sample_id": sample_id,
                    "app_id": record["app_id"],
                    "shot_index": shot_index,
                    "screenshot_url": url,
                    "image_path": (image_root / f"{sample_id}.jpg")
                    .relative_to(config.paths.root)
                    .as_posix(),
                }
            )
    frame = pd.DataFrame(rows)
    LOGGER.info("Expanded %d games into %d image samples", len(games), len(frame))
    return frame


def _download_one(
    url: str,
    destination: Path,
    *,
    session: Any,
    cfg: DatasetConfig,
) -> tuple[str, int, str | None]:
    """Fetch, decode, down-scale and store one screenshot.

    Returns ``(status, bytes_written, error_type)`` with status ``"present"``, ``"ok"`` or
    ``"fail"``.
    """
    from PIL import Image, UnidentifiedImageError

    if destination.is_file() and destination.stat().st_size > 0:
        return "present", 0, None

    last_error: str | None = None
    for attempt in range(cfg.download_retries):
        try:
            response = session.get(
                url, timeout=(cfg.download_timeout_connect, cfg.download_timeout_read)
            )
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as image:
                image = image.convert("RGB")
                if min(image.size) < cfg.min_image_side:
                    return "fail", 0, "image_too_small"
                image.thumbnail(
                    (cfg.stored_image_max_side, cfg.stored_image_max_side),
                    Image.Resampling.LANCZOS,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".part")
                image.save(temporary, format="JPEG", quality=cfg.stored_image_quality, optimize=True)
            temporary.replace(destination)
            if cfg.download_delay:
                time.sleep(cfg.download_delay)
            return "ok", destination.stat().st_size, None
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            last_error = type(exc).__name__
        except Exception as exc:  # requests errors, timeouts, ...
            last_error = type(exc).__name__
        if attempt < cfg.download_retries - 1:
            time.sleep(cfg.download_retry_backoff * (attempt + 1))
    return "fail", 0, last_error or "unknown"


def download_screenshots(
    samples: pd.DataFrame,
    *,
    config: GameSenseConfig = CONFIG,
    workers: int | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, DownloadStats]:
    """Download every screenshot referenced by *samples*.

    Returns ``(surviving_samples, stats)``; rows whose image could not be fetched are dropped.
    Resumable.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise ImportError("The 'requests' package is required to download screenshots.") from exc

    cfg = config.dataset
    n_workers = workers if workers is not None else cfg.download_workers
    stats = DownloadStats(requested=int(len(samples)), errors={})
    root = config.paths.root

    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent, "Accept": "image/*"})
    adapter = requests.adapters.HTTPAdapter(pool_connections=n_workers, pool_maxsize=n_workers)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    tasks = [(row["screenshot_url"], root / row["image_path"]) for row in samples.to_dict("records")]

    iterator: Any
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results_iter = pool.map(
            lambda task: _download_one(task[0], task[1], session=session, cfg=cfg), tasks
        )
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(results_iter, total=len(tasks), desc="screenshots", unit="img")
            except ImportError:  # pragma: no cover
                iterator = results_iter
        else:
            iterator = results_iter
        results = list(iterator)
    stats.seconds = time.perf_counter() - started

    keep = np.ones(len(samples), dtype=bool)
    error_counter: Counter[str] = Counter()
    for index, (status, written, error) in enumerate(results):
        if status == "present":
            stats.already_present += 1
        elif status == "ok":
            stats.downloaded += 1
            stats.bytes_written += written
        else:
            stats.failed += 1
            keep[index] = False
            error_counter[error or "unknown"] += 1
    stats.errors = dict(error_counter)

    surviving = samples.loc[keep].reset_index(drop=True)
    LOGGER.info(
        "Screenshots: %d requested, %d already present, %d downloaded, %d failed (%.1f s)",
        stats.requested,
        stats.already_present,
        stats.downloaded,
        stats.failed,
        stats.seconds,
    )
    return surviving, stats
