"""Make ``src/`` importable when a script is run directly."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

__all__ = ["PROJECT_ROOT", "SRC"]
