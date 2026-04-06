"""Small logging helper so every script prints consistently."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Create a logger with a compact format suitable for CLI scripts."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )
    return logging.getLogger("synthetic_road_generator")
