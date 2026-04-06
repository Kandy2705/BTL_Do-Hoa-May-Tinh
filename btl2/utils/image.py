"""Image export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    """Write an RGB uint8 image to disk."""
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def save_grayscale(path: str | Path, grayscale: np.ndarray) -> None:
    """Write a single-channel image as an 8-bit PNG."""
    Image.fromarray(grayscale.astype(np.uint8), mode="L").save(path)


def save_mask(path: str | Path, mask_rgb: np.ndarray) -> None:
    """Write a color-encoded segmentation mask."""
    Image.fromarray(mask_rgb.astype(np.uint8), mode="RGB").save(path)

