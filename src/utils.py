"""Utility helpers for the AutoML pipeline project."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> None:
    """Configure application-wide logging.

    Args:
        log_level: Logging level name.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directories(*paths: str | Path) -> None:
    """Create output directories if they do not exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
