"""Cross-cutting helpers: reproducibility, device selection, logging and I/O."""

from __future__ import annotations

import datetime
import decimal
import json
import logging
import os
import platform
import random
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch

from .config import CONFIG

__all__ = [
    "set_seed",
    "seed_worker",
    "torch_generator",
    "resolve_device",
    "describe_environment",
    "get_logger",
    "save_json",
    "load_json",
    "count_parameters",
    "human_time",
    "timed",
    "chunked",
    "ensure_src_on_path",
]

T = TypeVar("T")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = CONFIG.training.seed, *, deterministic: bool = True) -> None:
    """Seed every RNG the project touches."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CPU CI
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for :class:`torch.utils.data.DataLoader`."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_generator(seed: int = CONFIG.training.seed) -> torch.Generator:
    """Return a seeded generator for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a device specification to a concrete :class:`torch.device`."""
    if isinstance(device, torch.device):
        return device
    requested = (device or CONFIG.device or "auto").lower()

    if requested in {"auto", ""}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested.startswith("cuda") and not torch.cuda.is_available():
        get_logger(__name__).warning("CUDA requested but unavailable - falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        get_logger(__name__).warning("MPS requested but unavailable - falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def describe_environment(device: torch.device | None = None) -> dict[str, Any]:
    """Return a JSON-serialisable description of the runtime environment."""
    device = resolve_device(device) if device is None else device
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:  # optional dependency in some environments
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:  # pragma: no cover
        info["torchvision"] = None
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover
        info["transformers"] = None
    if torch.cuda.is_available():  # pragma: no cover - CPU CI
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "gamesense", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, attaching a stream handler exactly once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    """Make numpy / pathlib / torch objects JSON serialisable."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, torch.device):
        return str(obj)
    # pandas scalars that legitimately end up in reports: the bucket labels produced by pd.cut
    # (Interval) and any date-like index value.
    if type(obj).__name__ in {"Interval", "Timestamp", "Period", "Timedelta"}:
        return str(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write *data* as UTF-8 JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, default=_json_default, ensure_ascii=False)
    return path


def load_json(path: str | Path) -> Any:
    """Read UTF-8 JSON from *path*."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def count_parameters(module: torch.nn.Module) -> dict[str, int]:
    """Return ``{"total": ..., "trainable": ..., "frozen": ...}`` parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def human_time(seconds: float) -> str:
    """Format a duration as ``1h 02m 03s`` / ``2m 03s`` / ``3.4s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Context manager logging how long a block took."""
    log = logger or get_logger("gamesense.timing")
    start = time.perf_counter()
    try:
        yield
    finally:
        log.info("%s finished in %s", label, human_time(time.perf_counter() - start))


def chunked(items: Sequence[T] | Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield consecutive chunks of at most *size* elements."""
    if size <= 0:
        raise ValueError("size must be positive")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_src_on_path() -> Path:
    """Add ``<project root>/src`` to :data:`sys.path` if not already importable."""
    src = CONFIG.paths.root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src
