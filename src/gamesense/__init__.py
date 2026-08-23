"""GameSense - Multimodal AI for Video Game Understanding."""

from __future__ import annotations

from .config import CONFIG, GENRES, MODEL_KINDS, NUM_CLASSES, GameSenseConfig
from .utils import get_logger, resolve_device, set_seed

__author__ = "Blagoy Hristov"
__version__ = "1.0.0"

__all__ = [
    "__author__",
    "__version__",
    "CONFIG",
    "GENRES",
    "NUM_CLASSES",
    "MODEL_KINDS",
    "GameSenseConfig",
    "set_seed",
    "resolve_device",
    "get_logger",
]
