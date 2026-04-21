"""Helper ghi ảnh PNG cho các artifact của BTL 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    """Ghi ảnh RGB uint8 ra đĩa."""
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def save_grayscale(path: str | Path, grayscale: np.ndarray) -> None:
    """Ghi ảnh một kênh thành PNG 8-bit."""
    Image.fromarray(grayscale.astype(np.uint8), mode="L").save(path)


def save_mask(path: str | Path, mask_rgb: np.ndarray) -> None:
    """Ghi segmentation mask đã mã hóa màu RGB."""
    Image.fromarray(mask_rgb.astype(np.uint8), mode="RGB").save(path)
